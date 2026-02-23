# PILON-R Improvement Plan: Exploiting What Makes PILON Different

**Date:** February 23, 2026  
**Status:** Phase B Complete -> Phase B.5 (Structural Advantages)  
**Progress:** B.5a complete (compute path fixes landed and validated on February 23, 2026).  
**Hardware:** RTX 4070 (12GB VRAM)

---

## The Problem

PILON-R has been benchmarked by dense FFN rules: same scale, same VRAM budget, same training recipe. Under those rules, dense wins—it has decades of optimization behind it. The 1.22× convergence gap and higher VRAM at 48M params are expected outcomes when you race a new architecture on the incumbent's home turf.

**The fix isn't closing the gap at 48M. It's testing at a scale and in a mode where PILON's structural properties create advantages dense doesn't have.**

Three structural properties are unique to PILON and completely untested:

1. **Primitives are independent and swappable** — you can load subsets without loading all
2. **Computation is factored** — the same primitives compose differently per layer/token, so skipping computation doesn't waste unique parameters
3. **Top-k is native** — dense activates 100% of FFN params per token; PILON already activates only top-k

---

## Codebase Issues Found

Before any new features, these issues in the current code need attention:

### Issue 1: `compute_all_outputs()` defeats sparsity

In `primitives.py` line 338-382, `compute_all_outputs()` computes ALL primitive outputs for every token, then selects top-k afterward. The `forward_fast()` path (line 239) calls this. This means even with top_k=8 out of 48 primitives, you compute all 48 outputs and throw away 40. Dense FFN doesn't waste compute this way.

**Fix:** The `forward_topk_fused()` path (line 165) already solves this—it builds a single concatenated low-rank map from only the top-k primitives and does two GEMMs. But `MoECompositionalFFN.forward()` (ffn.py line 646) calls `compute_all_outputs()` in the dense expert path, and even the sparse expert path (line 672) calls it per-expert. The MoE forward needs rewriting to use fused paths.

**Impact:** This is likely a significant contributor to PILON being slower than dense. You're doing 48 primitive computations when you only need 8.

### Issue 2: MoE forward is O(experts × primitives)

In `ffn.py` line 574-693, the MoE forward computes primitive outputs, then for each expert, combines them with that expert's composition weights. The sparse path (line 662-680) reshapes to `(B*S*K, 1, d_ff)` and calls `compute_all_outputs` again—this is computing all primitive outputs K times per token per layer. With 8 experts, top-2 routing, and 48 primitives, you're doing 96 primitive computations per token when dense does 1 matrix multiply.

**Fix:** Each expert's composition weights select a subset of primitives. Pre-fuse each expert into a single low-rank map (like `forward_topk_fused` does for static composition), then you only do 2 GEMMs per expert, not P.

### Issue 3: No VRAM tracking in training

The training loop (train.py) logs VRAM but doesn't compare it against the dense baseline in the same run. For the efficiency thesis to hold, every experiment needs side-by-side VRAM numbers.

### Issue 4: 360M/500M configs exist but never ran

`model_360m.py` and `model_500m.py` are fully configured but untested. The 360M config targets the crossover scale we discussed. This is the most important untested experiment.

---

## Plan: 5 Phases, 3 Weeks

### Phase B.5a: Fix the Compute Path (Days 1-2)

**Goal:** Make PILON's actual FLOPS match its theoretical FLOPS.
**Status:** DONE (completed February 23, 2026).

Right now, top-k=8 with 48 primitives should do 8/48 = 16.7% of the compute. It actually does 100% because of `compute_all_outputs()`. Fix this first because every subsequent measurement is wrong until this is fixed.

**Changes to `primitives.py`:**

```python
# NEW: Sparse-only forward that never touches non-selected primitives
def forward_sparse(
    self,
    x: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    active_rank: Optional[int] = None,
    top_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Truly sparse forward: only loads and computes top-k primitives.
    Never touches the other (P - top_k) primitives.
    """
    if top_indices is None:
        _, top_indices = torch.topk(weights, top_k, sorted=False)
    top_weights = weights[top_indices]
    top_weights = top_weights / (top_weights.sum() + 1e-8)

    # Only index into the k primitives we need
    A_sel = self.A[top_indices]  # (k, d_in, rank)
    B_sel = self.B[top_indices]  # (k, rank, d_out)
    scale = self.latent_scale[:active_rank] if active_rank else self.latent_scale
    bias = self.latent_bias[:active_rank] if active_rank else self.latent_bias

    batch, seq, _ = x.shape
    x_flat = x.reshape(-1, self.d_in)

    # k separate rank-r GEMMs (can be batched via einsum)
    U = torch.einsum("td,kdr->tkr", x_flat, A_sel)
    U = U * scale + bias
    Y = torch.einsum("tkr,kro->tko", U, B_sel)

    # Weighted sum across k primitives
    out = torch.einsum("tko,k->to", Y, top_weights)
    return out.view(batch, seq, self.d_out)
```

**Changes to `ffn.py` MoECompositionalFFN:**

Replace `compute_all_outputs()` calls with per-expert fused computation:

```python
def _expert_forward_fused(self, x, fc1_bank, fc2_bank, expert_idx):
    """Single expert forward using only its selected primitives."""
    fc1_w, fc2_w = self.expert_compositions.get_expert_weights(expert_idx)
    h = fc1_bank.forward_topk_fused(x, fc1_w, self.top_k_primitives)
    h = self.activation(h)
    out = fc2_bank.forward_topk_fused(h, fc2_w, self.top_k_primitives)
    return out
```

**Validation:** Before and after timing on the existing 48M model. Measure:
- tok/s (should increase substantially)
- VRAM peak during forward pass (should decrease)
- Numerical equivalence within bf16 tolerance (verify outputs match old path)

**Success gate:** tok/s improves by >20% with top_k=8/48. If not, the overhead is elsewhere and this diagnosis was wrong.

---

### Phase B.5b: VRAM-Efficient Primitive Loading (Days 3-5)

**Goal:** Demonstrate that PILON can run with fewer primitives resident in VRAM than total primitives exist.

This is the core structural advantage. Dense FFN must load ALL weights. PILON only needs the top-k active primitives in VRAM at any moment.

**New class: `TieredPrimitiveBank`**

```python
class TieredPrimitiveBank(nn.Module):
    """
    Primitive bank with hot/warm/cold tiers.

    - Hot (VRAM): top-k most-used primitives, always resident
    - Warm (RAM): remaining primitives, loaded on-demand
    - Cold (disk): future - for massive primitive libraries

    During forward pass, only hot primitives participate.
    Periodically (every N steps), usage stats update which
    primitives are hot vs warm.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_primitives: int,
        rank: int,
        n_hot: int,       # Primitives kept in VRAM
        swap_interval: int = 100,  # Steps between hot/warm swaps
    ):
        super().__init__()
        self.n_hot = n_hot
        self.n_primitives = n_primitives
        self.swap_interval = swap_interval

        # Hot tier: on GPU (these are the nn.Parameters)
        self.A_hot = nn.Parameter(torch.randn(n_hot, d_in, rank))
        self.B_hot = nn.Parameter(torch.randn(n_hot, rank, d_out))

        # Warm tier: on CPU (pinned memory for fast transfer)
        self.register_buffer('A_warm', torch.randn(
            n_primitives - n_hot, d_in, rank
        ).pin_memory(), persistent=True)
        self.register_buffer('B_warm', torch.randn(
            n_primitives - n_hot, rank, d_out
        ).pin_memory(), persistent=True)

        # Usage tracking
        self.register_buffer('usage_counts',
            torch.zeros(n_primitives))
        self.register_buffer('hot_indices',
            torch.arange(n_hot))

    def swap_tiers(self):
        """Promote most-used warm primitives, demote least-used hot."""
        # ... usage-based swap logic
```

**Why this matters:** At 360M scale with d_model=1024, d_ff=4096, rank=80:
- Each primitive (fc1 + fc2) = 80 × (1024×80 + 80×4096 + 4096×80 + 80×1024) × 2 bytes ≈ 200KB
- 80 primitives × 3 bands × 2 (fc1+fc2) = 480 banks total = ~96MB
- With n_hot=16 per bank: ~19MB in VRAM (5× reduction in primitive VRAM)
- Dense FFN at same scale: 24 layers × 2 × 1024 × 4096 × 2 bytes = ~402MB (no reduction possible)

**Training consideration:** During training, gradients only flow through the hot primitives that were active. Warm primitives don't need optimizer states. This means:
- Adam states (2× model size) only needed for hot primitives
- At n_hot=16 out of 80: optimizer memory for primitives drops 5×

**Implementation steps:**

1. Build `TieredPrimitiveBank` that wraps existing `PrimitiveBank`
2. Add hot/warm/cold designation with usage-based promotion
3. Modify `BandPrimitiveBanks` to optionally use tiered banks
4. Add `--n-hot` CLI argument to `train.py`
5. Add VRAM comparison logging (PILON tiered vs PILON full vs dense)

**Validation:** Compare at 360M scale:
- PILON-360M with all primitives in VRAM (current behavior)
- PILON-360M with n_hot=16 (tiered)
- Dense-360M baseline
- Measure: peak VRAM, tok/s, and quality (loss after 1K steps)

**Success gate:** PILON-tiered uses measurably less VRAM than dense-360M while maintaining loss within 1.3× of dense at 1K steps.

---

### Phase B.5c: Early Exit for Easy Tokens (Days 6-9)

**Goal:** Reduce average compute per token by letting easy tokens skip later layers.

In a dense transformer, skipping layer 20 means skipping the unique weights in that layer forever—they're not accessible to any other layer. In PILON, primitives are shared across layers within a band. Skipping layer 20 doesn't lose access to those primitives—layer 19 or 21 already uses the same bank. The information capacity is preserved even when computation is skipped.

**Architecture change to `model.py`:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, ..., enable_early_exit: bool = False,
                 exit_threshold: float = 0.1):
        # ... existing init ...
        self.enable_early_exit = enable_early_exit
        self.exit_threshold = exit_threshold

        if enable_early_exit:
            # Lightweight confidence head: predicts whether this
            # token needs more processing
            self.exit_gate = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )

    def forward(self, x, attention_mask=None, ...):
        # Attention (always runs - needed for token interaction)
        h = self.norm1(x)
        h = self.attention(h, attention_mask)
        x = x + self.dropout(h)

        if self.enable_early_exit and not self.training:
            # Check if tokens are confident enough to skip FFN
            confidence = self.exit_gate(x)  # (B, S, 1)
            skip_mask = (confidence > self.exit_threshold)  # (B, S, 1)

            if skip_mask.all():
                # ALL tokens skip FFN at this layer
                return x

            if skip_mask.any():
                # Mixed: only compute FFN for uncertain tokens
                h = self.norm2(x)
                # Only run FFN on tokens that need it
                needs_compute = ~skip_mask.squeeze(-1)  # (B, S)
                h_subset = h[needs_compute]  # (N, d_model)

                if h_subset.numel() > 0:
                    ffn_out = self.ffn(h_subset.unsqueeze(0))
                    if isinstance(ffn_out, dict):
                        ffn_out = ffn_out["output"]
                    ffn_out = ffn_out.squeeze(0)
                    h_full = torch.zeros_like(h)
                    h_full[needs_compute] = ffn_out
                    x = x + self.dropout(h_full)
                return x

        # Standard path (training or early exit disabled)
        h = self.norm2(x)
        ffn_result = self.ffn(h)
        if isinstance(ffn_result, dict):
            h = ffn_result["output"]
            self._last_aux_loss = ffn_result.get("aux_loss", 0.0)
        else:
            h = ffn_result
        x = x + self.dropout(h)
        return x
```

**Why this is PILON-specific:** The exit gate only matters for inference speed. During training, all layers run normally. At inference, the key insight is that PILON layers within a band share primitives, so a token that exits at layer 12 (middle band) still had access to the same primitive knowledge as if it ran through layers 12-15. It just didn't apply a different composition. Dense models lose unique weights when skipping layers.

**Training the exit gate:**
- Phase 1: Train the model normally (gates disabled)
- Phase 2: Freeze the model, train only the exit gates
- Gate training signal: per-layer KL divergence between current layer's output logits and final layer's output logits. If KL is low, the token can exit safely.

**Metrics to track:**
- `skip_ratio_per_layer`: fraction of tokens skipping FFN at each layer
- `avg_layers_per_token`: average number of FFN computations per token
- `quality_delta`: generation quality with/without early exit

**Success gate:** Average layers per token drops below 70% of total layers (e.g., average 17 out of 24 for 360M) with <2% quality degradation on eval loss.

---

### Phase B.5d: 360M Crossover Experiment (Days 10-14)

**Goal:** Find the scale where PILON + structural advantages beats dense on efficiency.

This is the critical experiment. Everything above is infrastructure. This is the test.

**Setup:**

| Config | Params | Description |
|--------|--------|-------------|
| Dense-360M | ~337M | Standard transformer, all params in VRAM |
| PILON-360M-full | ~332M | All primitives in VRAM (current behavior) |
| PILON-360M-tiered | ~332M | n_hot=16 per bank, rest in RAM |
| PILON-360M-exit | ~332M | Tiered + early exit at inference |

All four use identical: attention, embeddings, tokenizer, data, optimizer settings, training steps.

**Training plan:**

```bash
# Dense baseline
python -m pilon_r.train --model-size 360m --ffn-type standard \
  --output-dir outputs/360m_dense --total-tokens 1000000000

# PILON full (current behavior)
python -m pilon_r.train --model-size 360m --ffn-type compositional \
  --output-dir outputs/360m_pilon_full --total-tokens 1000000000 \
  --phase1-sparse --phase1-top-k 8

# PILON tiered
python -m pilon_r.train --model-size 360m --ffn-type compositional \
  --output-dir outputs/360m_pilon_tiered --total-tokens 1000000000 \
  --phase1-sparse --phase1-top-k 8 --n-hot 16

# PILON exit (train normally, add exit gates after)
# ... post-training exit gate fitting ...
```

**1B tokens at 360M on a 4070:**
- Effective batch = 4 × 16 = 64 sequences × 2048 tokens = 131K tokens/step
- 1B tokens ÷ 131K = ~7,630 steps
- At ~200k tok/s (Phase B measured): ~5,000 seconds ≈ 1.4 hours per model
- Total for 4 configs: ~6 hours

**What we measure:**

| Metric | What it tests |
|--------|---------------|
| Peak VRAM during training | Does tiered reduce memory? |
| Peak VRAM during inference | Can PILON run where dense can't? |
| Training tok/s | Is sparse compute actually faster? |
| Inference tok/s (with early exit) | Token generation speed |
| Val loss at 1B tokens | Quality |
| Val loss per VRAM-GB | Efficiency ratio (the real metric) |

**The crossover test:** If PILON-tiered achieves quality within 1.25× of dense while using <8GB VRAM (vs dense needing ~10-11GB for 360M with optimizer), PILON wins on efficiency at this scale. If it also generates tokens faster due to early exit and sparse computation, that's a double win.

**If it doesn't cross over at 360M:** Try 500M (configs already exist). Dense-500M at ~505M params will be very tight on 12GB VRAM. PILON-500M with tiered loading might fit comfortably. The scale advantage should be even more pronounced.

---

### Phase B.5e: Benchmarking and Documentation (Days 15-16)

**Goal:** Produce honest, reproducible results with proper baselines.

**Benchmark suite** (add to `benchmark.py`):

1. **VRAM efficiency curve:** Plot quality (val loss) vs VRAM usage for dense and PILON at 48M, 360M, 500M. This should show the crossover point.

2. **Compute efficiency:** Plot quality vs actual FLOPS (not step count). PILON with sparse computation does fewer FLOPS per step—measure whether those saved FLOPS translate to better quality-per-FLOP.

3. **Inference profile:**
   - Tokens per second (generation)
   - VRAM during generation
   - Average layers computed per token (with early exit)
   - Time-to-first-token

4. **Quality:**
   - Perplexity on held-out OpenWebText
   - HellaSwag (if time permits)
   - Generation samples (qualitative)

**Documentation:** Update PROGRESS.md and PHASE_PLAN with:
- Compute path fix results (before/after)
- Tiered loading VRAM numbers
- Early exit skip ratios
- 360M crossover data
- Honest assessment of whether the efficiency thesis holds

---

## Summary: What Each Phase Tests

| Phase | Question | If YES | If NO |
|-------|----------|--------|-------|
| B.5a: Compute fix (DONE) | Is PILON wasting FLOPS? | tok/s jumps, confirms overhead was artificial | Overhead is elsewhere (memory bandwidth?) |
| B.5b: Tiered loading | Can PILON use less VRAM? | Core structural advantage confirmed | Swapping overhead kills benefit |
| B.5c: Early exit | Can PILON skip layers cheaply? | Inference speed advantage | Exit gates can't predict well enough |
| B.5d: 360M crossover | Does PILON win on efficiency at scale? | **Thesis confirmed** — PILON is more efficient | PILON's overhead outweighs structural advantages |
| B.5e: Benchmarks | Is this reproducible? | Publishable results | Need more iteration |

**The whole plan stands or falls on B.5d.** Everything before it is infrastructure to make B.5d a fair test. If PILON-360M with tiered loading and early exit doesn't beat dense-360M on quality-per-VRAM-GB, the architecture isn't pulling its weight at currently testable scales.

---

## Files to Modify

| File | Changes |
|------|---------|
| `primitives.py` | Add `forward_sparse()`, add `TieredPrimitiveBank` |
| `ffn.py` | Rewrite MoE forward to use fused paths, fix `compute_all_outputs` usage |
| `model.py` | Add early exit gate to `TransformerBlock`, add skip logic |
| `config.py` | Add `n_hot`, `enable_early_exit`, `exit_threshold` to configs |
| `model_360m.py` | Add tiered config variant |
| `train.py` | Add `--n-hot` flag, VRAM comparison logging, exit gate training mode |
| `benchmark.py` | Add VRAM efficiency curve, inference profiling |
| `evaluate.py` | Add early-exit-aware generation |

## New Files

| File | Purpose |
|------|---------|
| `tiered_bank.py` | `TieredPrimitiveBank` with hot/warm/cold management |
| `early_exit.py` | Exit gate module, training loop, and analysis tools |
| `benchmark_efficiency.py` | VRAM vs quality curves, FLOPS measurement |

---

## Risk Assessment

**Highest risk: Phase B.5b (tiered loading).**
CPU→GPU transfer latency for warm primitives could be too slow, even with pinned memory. NVMe bandwidth is ~7GB/s, but the real bottleneck is PCIe latency per transfer. Mitigation: prefetch warm→hot transfers asynchronously during attention computation (attention doesn't use primitives, so there's a natural compute window).

**Medium risk: Phase B.5c (early exit).**
The exit gate might not be predictive enough, or the quality degradation might exceed 2%. Mitigation: start with inference-only (no training changes), use conservative thresholds, and measure quality at multiple skip ratios.

**Lowest risk: Phase B.5a (compute fix).**
This is a straightforward code optimization. The current path provably does unnecessary work. The only question is how much speedup it yields.
