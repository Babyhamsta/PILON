# PILON Phase C: Attention Experiments — Claude Code Task Spec

## Context

PILON has a proven compositional FFN (shared low-rank primitives, band sharing, ternary 
quantization, top-k selection). The attention is standard multi-head attention — functional 
but unoptimized for PILON's design philosophy.

HoloTern experiments showed:
- Standard softmax attention has Q/K gradient starvation (17-21x weaker from epoch 1)
- HRR attention has great gradient flow but zero selectivity → fails on diverse data
- The FFN is easy to compress; attention is where the real challenge lives

**Goal:** Find or design an attention mechanism that complements PILON's compositional FFN.
Not "make attention look like PILON" — make attention WORK WITH PILON.

**Strategy:** Run 4-5 attention variants on PILON's existing 48M config + FineWeb-Edu dataset.
Compare val loss, PPL, gradient health, and training dynamics. Each run takes 3-5 hours on 
the RTX 4070. Total experiment time: ~1-2 days.

**Framework:** All experiments build on the existing PILON codebase. We add new attention 
modules alongside the existing `MultiHeadAttention`. The model config gets a new 
`attention_type` field to select between them. Everything else stays the same.

---

## CRITICAL RULES

1. **Do not modify existing code that works.** Add new files, extend configs. Don't touch 
   `primitives.py`, `ffn.py`, or the existing `MultiHeadAttention` class.
2. **All new attention modules must match the existing interface:**
   ```python
   def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
               past_kv=None, use_cache=False) -> torch.Tensor:
   ```
3. **Windows paths.** No `/mnt/`, no `.sh` scripts. Use `python -m pilon_r.train` commands.
4. **torch.compile compatible.** Avoid Python-level control flow in the forward pass that 
   would cause graph breaks. No `.item()` calls in the forward path.
5. **Run existing tests first** to make sure nothing is broken before adding new code.
6. **Every new module gets tests.** At minimum: output shape, gradient flow, causal behavior, 
   no NaN.

---

## INTEGRATION POINT

The swap happens in `TransformerBlock.__init__()` in `model.py` (line ~262-269):

```python
# Current code:
self.attention = MultiHeadAttention(
    d_model=d_model, n_heads=n_heads, d_head=d_head,
    dropout=dropout, max_seq_len=max_seq_len
)
```

This becomes a factory call based on config:

```python
self.attention = create_attention(
    attention_type=attention_type,
    d_model=d_model, n_heads=n_heads, d_head=d_head,
    dropout=dropout, max_seq_len=max_seq_len,
    # New params for compositional/recurrent variants:
    primitive_banks=attn_primitive_banks,  # None for standard
    n_primitives=n_attn_primitives,
    top_k=attn_top_k,
    layer_idx=layer_idx,
)
```

Add `attention_type` to `ModelConfig`:
```python
attention_type: str = "standard_mha"  # "standard_mha", "compositional_mha", 
                                       # "gated_recurrence", "hybrid_recurrent_mha"
```

---

## NEW FILES TO CREATE

```
pilon_r/core/
    attention.py              # All new attention modules + factory function
    attention_primitives.py   # Primitive banks for attention projections (if needed)

tests/
    test_attention.py         # Tests for all new attention modules
```

---

## EXPERIMENT 1: Compositional Attention Projections

### Hypothesis
If FFN weights are over-duplicated across layers (PILON proved this), Q/K/V projection 
weights might be too. Sharing projection primitives across layers in a band could reduce 
parameters while preserving standard attention's selectivity.

### Architecture

Replace `nn.Linear` Q/K/V projections with PILON-style compositional projections:

```python
class CompositionalMHA(nn.Module):
    """
    Multi-head attention with compositional Q/K/V projections.
    
    Instead of separate nn.Linear for Q, K, V per layer, uses shared 
    low-rank primitive banks (per band) with per-layer composition weights.
    
    The attention MECHANISM is unchanged — still softmax(QK^T/sqrt(d)) @ V.
    Only the projection weights are shared/composed.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int, 
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        # Compositional params
        n_primitives: int = 16,     # Fewer than FFN since projections are smaller
        rank: int = 32,
        top_k: int = 4,
        layer_idx: int = 0,
        # Shared primitive bank (set externally via BandPrimitiveBanks)
        q_bank: Optional[PrimitiveBank] = None,
        k_bank: Optional[PrimitiveBank] = None,
        v_bank: Optional[PrimitiveBank] = None,
    ):
```

**Key design decisions:**
- Q, K, V each get their OWN primitive bank (different roles, shouldn't share)
- Banks are shared across layers within a band (same as FFN)
- Attention mechanism itself is unchanged (softmax, SDPA, causal mask)
- Output projection stays as standard `nn.Linear` (it's a single matrix, 
  not worth composing)
- Top-k is lower than FFN (4 vs 8) since projections are smaller

**What this tests:** Can attention projections be shared across layers 
without losing quality? If yes, this immediately reduces attention 
parameter count by ~60-70%.

### Config

```yaml
attention_type: compositional_mha
n_attn_primitives: 16
attn_rank: 32
attn_top_k: 4
# Band sharing same as FFN: early(0-2), middle(3-5), late(6-7)
```

---

## EXPERIMENT 2: Gated Linear Recurrence (RWKV-inspired)

### Hypothesis
A selective recurrent mechanism preserves the "what matters now" routing ability 
while eliminating quadratic attention cost and KV cache. If paired with PILON's 
compositional FFN, this could be the parameter-efficient + memory-efficient combo.

### Architecture

Inspired by RWKV-7's time-mixing but simplified for our testing purposes:

```python
class GatedLinearRecurrence(nn.Module):
    """
    Gated linear recurrence for token mixing.
    
    For each head, maintains a recurrent state that is updated per-token:
      state_t = decay_t * state_t-1 + gate_t * (k_t outer v_t)
      output_t = q_t @ state_t
    
    Where:
      - decay_t is data-dependent (learned from input) — THIS provides selectivity
      - gate_t controls what new information enters the state
      - q_t queries the accumulated state
    
    Key properties:
      - O(T) time and space (no attention matrix)
      - No KV cache (state is fixed-size)
      - Data-dependent decay provides selectivity (unlike HRR's uniform cumsum)
      - Causal by construction (state only accumulates past)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        
        # Projections for recurrence components
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        
        # Data-dependent decay: sigmoid(linear(x)) → per-head, per-position decay
        # This is the KEY difference from HRR — decay provides selectivity
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True)
        
        # Input gate: controls what enters the state
        self.gate_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        
        # Output gate: controls what leaves
        self.output_gate = nn.Linear(d_model, n_heads * d_head, bias=False)
        
        # Initialize decay bias to ~0.9 (conservative, mostly remember)
        nn.init.constant_(self.decay_proj.bias, 2.0)  # sigmoid(2.0) ≈ 0.88
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head)
        
        # Data-dependent decay per head per position
        decay = torch.sigmoid(self.decay_proj(x))  # (B, T, n_heads)
        decay = decay.unsqueeze(-1)  # (B, T, n_heads, 1)
        
        # Input gate
        gate = torch.sigmoid(self.gate_proj(x).view(B, T, self.n_heads, self.d_head))
        
        # Recurrent computation
        # state: (B, n_heads, d_head, d_head) — outer product accumulator
        # For efficiency, use d_head x 1 state instead of outer product:
        # state_t = decay_t * state_t-1 + gate_t * k_t * v_t (element-wise)
        # output_t = q_t * state_t (element-wise, then sum over d_head)
        #
        # Actually, for proper expressivity use a (d_head,) state per head:
        # state_t = decay_t * state_t-1 + gate_t * (k_t * v_t)
        # output_t = sum(q_t * state_t)
        #
        # This is a simplified linear attention / linear recurrence.
        
        kv = gate * k * v  # (B, T, n_heads, d_head) — gated key-value product
        
        # Parallel scan for the linear recurrence:
        # state_t = decay_t * state_t-1 + kv_t
        # 
        # For training efficiency, use cumulative operations:
        # This can be parallelized via associative scan, but for prototype
        # use a simple loop or the log-space cumsum trick.
        
        # Simple sequential implementation (correct but slow for long T):
        states = torch.zeros(B, self.n_heads, self.d_head, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(T):
            states = decay[:, t, :, :].squeeze(1) * states + kv[:, t, :, :]  # removed extra squeeze
            # Correctly index: states is (B, n_heads, d_head)
            out_t = (q[:, t, :, :] * states).sum(dim=-1, keepdim=True)  # wrong - need per d_head
            # Actually: output should be (B, n_heads, d_head), then project
            out_t = q[:, t, :, :] * states  # (B, n_heads, d_head)
            outputs.append(out_t)
        
        output = torch.stack(outputs, dim=1)  # (B, T, n_heads, d_head)
        
        # Output gate
        out_gate = torch.sigmoid(self.output_gate(x).view(B, T, self.n_heads, self.d_head))
        output = output * out_gate
        
        # Reshape and project
        output = output.reshape(B, T, self.n_heads * self.d_head)
        output = self.out_proj(output)
        
        return output
```

**IMPORTANT: The sequential loop is a prototype.** For production, this needs a parallel 
scan implementation. But for testing at seq_len=512 on a 48M model, the loop is fine. 
We can optimize later if the approach shows promise.

**The critical difference from HRR:** The `decay` is DATA-DEPENDENT. Each position 
computes its own decay rate from the input. This means the model can learn "forget 
everything before this sentence boundary" or "hold onto this variable name." HRR had 
no such mechanism — everything was accumulated equally.

### Config

```yaml
attention_type: gated_recurrence
# Uses same n_heads, d_head as standard — no new hyperparameters
```

---

## EXPERIMENT 3: Compositional Gated Recurrence (The Combination)

### Hypothesis
Combine Experiments 1 and 2: use PILON-style compositional projections WITH gated 
linear recurrence. This tests whether the PILON primitive sharing philosophy works 
for attention projections when paired with a non-softmax mixing mechanism.

### Architecture

Same as Experiment 2 (GatedLinearRecurrence) but Q/K/V/gate/decay projections use 
compositional primitive banks instead of `nn.Linear`:

```python
class CompositionalGatedRecurrence(nn.Module):
    """
    Gated linear recurrence with compositional projections.
    
    Combines PILON's shared primitives (for parameter efficiency)
    with gated recurrence (for memory efficiency and selectivity).
    """
```

This is the full "PILON philosophy applied to attention" experiment — shared 
primitives for ALL projections, recurrent mixing instead of softmax.

### Config

```yaml
attention_type: compositional_gated_recurrence
n_attn_primitives: 16
attn_rank: 32
attn_top_k: 4
```

---

## EXPERIMENT 4: Hybrid — Recurrence Early, Standard Attention Late

### Hypothesis
Early layers may benefit from broad mixing (recurrence, cheap) while late layers 
need precise routing (attention, selective). This tests the "different layers need 
different routing" idea without fully committing to one mechanism.

### Architecture

```python
class HybridAttention(nn.Module):
    """
    Routes to different attention mechanisms based on layer depth.
    
    Early layers (within early band): GatedLinearRecurrence
    Late layers (within late band): Standard MultiHeadAttention
    Middle layers: configurable (default: recurrence)
    """
```

Implementation: just a thin wrapper that instantiates the right module based on 
`layer_idx` and band assignments.

### Config

```yaml
attention_type: hybrid_recurrent_mha
hybrid_early_type: gated_recurrence    # Layers 0-2
hybrid_middle_type: gated_recurrence   # Layers 3-5
hybrid_late_type: standard_mha         # Layers 6-7
```

---

## EXPERIMENT RUNS

All runs use the existing PILON 48M config:
- d_model=512, n_layers=8, n_heads=8, d_ff=2048
- Compositional FFN with 48 primitives, rank 48, top-k 8
- Ternary primitives + SubLN + SqReLU (best PILON variant)
- FineWeb-Edu dataset, 500M tokens
- torch.compile enabled
- seed=42

### Run Matrix

| Run | Attention | FFN | Label | Purpose |
|-----|-----------|-----|-------|---------|
| C0 | Standard MHA | PILON Ternary | baseline | Already exists from prior PILON runs. Reuse. |
| C1 | Compositional MHA | PILON Ternary | comp_attn | Does sharing Q/K/V projections work? |
| C2 | Gated Recurrence | PILON Ternary | gated_rec | Does selective recurrence beat softmax? |
| C3 | Compositional Gated Rec | PILON Ternary | comp_gated_rec | Full compositional routing + PILON FFN |
| C4 | Hybrid (rec early, MHA late) | PILON Ternary | hybrid | Best of both worlds? |

Also run dense baseline for reference:
| C0-dense | Standard MHA | Standard Dense | dense_baseline | Reuse existing. |

### Commands

```
REM C0 — Already exists, reuse prior PILON ternary run results

REM C1 — Compositional attention projections
python -m pilon_r.train --model-size 48m --ffn-type compositional --ternary --use-subln --use-squared-relu --compile --attention-type compositional_mha --n-attn-primitives 16 --attn-rank 32 --attn-top-k 4 --phase1-sparse --phase1-top-k 8 --freeze-primitives-phase2 --total-tokens 500000000 --batch-size 8 --grad-accum 8 --seq-len 512 --dataset HuggingFaceFW/fineweb-edu --output-dir outputs/48m_comp_attn --log-comp-stats

REM C2 — Gated linear recurrence
python -m pilon_r.train --model-size 48m --ffn-type compositional --ternary --use-subln --use-squared-relu --compile --attention-type gated_recurrence --phase1-sparse --phase1-top-k 8 --freeze-primitives-phase2 --total-tokens 500000000 --batch-size 8 --grad-accum 8 --seq-len 512 --dataset HuggingFaceFW/fineweb-edu --output-dir outputs/48m_gated_rec --log-comp-stats

REM C3 — Compositional gated recurrence
python -m pilon_r.train --model-size 48m --ffn-type compositional --ternary --use-subln --use-squared-relu --compile --attention-type compositional_gated_recurrence --n-attn-primitives 16 --attn-rank 32 --attn-top-k 4 --phase1-sparse --phase1-top-k 8 --freeze-primitives-phase2 --total-tokens 500000000 --batch-size 8 --grad-accum 8 --seq-len 512 --dataset HuggingFaceFW/fineweb-edu --output-dir outputs/48m_comp_gated_rec --log-comp-stats

REM C4 — Hybrid
python -m pilon_r.train --model-size 48m --ffn-type compositional --ternary --use-subln --use-squared-relu --compile --attention-type hybrid_recurrent_mha --phase1-sparse --phase1-top-k 8 --freeze-primitives-phase2 --total-tokens 500000000 --batch-size 8 --grad-accum 8 --seq-len 512 --dataset HuggingFaceFW/fineweb-edu --output-dir outputs/48m_hybrid_attn --log-comp-stats
```

---

## IMPLEMENTATION ORDER

### Step 1: Add attention factory + config extensions

Extend `ModelConfig` in `config.py` with attention fields:
```python
# In ModelConfig:
attention_type: str = "standard_mha"
n_attn_primitives: int = 16
attn_rank: int = 32
attn_top_k: int = 4
```

Extend `TransformerBlock` to use factory for attention creation.
Extend `PILONTransformer` to create attention primitive banks when needed.
Extend CLI in `train.py` to accept new attention flags.

**Test:** Verify existing standard_mha path is unchanged — run smoke test, confirm 
identical output with same seed as before changes.

### Step 2: Implement CompositionalMHA (Experiment 1)

Create `pilon_r/core/attention.py` with:
- `CompositionalMHA` class
- Reuses existing `PrimitiveBank` and `LayerCompositionWeights` from `primitives.py`
- The Q/K/V projections call `bank(x, weights, top_k=k)` instead of `nn.Linear(x)`
- Everything else (SDPA, causal mask, multi-head reshape) stays identical to standard MHA

**Tests:**
- Output shape matches standard MHA
- Causal: changing input at position t doesn't affect output at positions < t
- Gradients flow to primitive bank parameters AND composition weight parameters
- No NaN for random inputs
- Parameter count is lower than standard MHA (the whole point)

### Step 3: Implement GatedLinearRecurrence (Experiment 2)

Add to `pilon_r/core/attention.py`:
- `GatedLinearRecurrence` class
- Sequential loop implementation (correct, not fast)
- Proper initialization (decay bias = 2.0 for conservative initial retention)

**Tests:**
- Output shape matches standard MHA
- Causal by construction (verify: changing input at t doesn't affect output at < t)
- Gradient flow through all projections including decay_proj
- No NaN
- State doesn't explode over long sequences (test with T=512)

**Performance note:** The sequential loop will be slower than SDPA attention. 
That's fine for testing. If the approach works, we optimize with parallel scan later.
Log ms/step so we know the overhead.

### Step 4: Implement CompositionalGatedRecurrence (Experiment 3)

Add to `pilon_r/core/attention.py`:
- `CompositionalGatedRecurrence` class  
- Same as GatedLinearRecurrence but Q/K/V/gate/decay use primitive bank projections
- Shares attention primitive banks across layers via band structure

**Tests:** Same as Steps 2 and 3 combined.

### Step 5: Implement HybridAttention (Experiment 4)

Add to `pilon_r/core/attention.py`:
- `HybridAttention` — thin wrapper that delegates to the right module per layer
- Takes `layer_idx` and band config, instantiates appropriate attention type

**Tests:**
- Early layers produce GatedLinearRecurrence-style output
- Late layers produce standard MHA-style output
- Causal across all layers

### Step 6: Run all experiments

Execute C1-C4 sequentially. C0 baseline reuses existing results.
Each run: ~3-5 hours on RTX 4070.
Total: ~12-20 hours (overnight).

---

## METRICS TO COLLECT

For each run, report at the end:

| Metric | How |
|--------|-----|
| Final val loss | Standard eval |
| Final val PPL | exp(val_loss) |
| vs Dense baseline | PPL ratio |
| vs PILON baseline (C0) | PPL ratio |
| Attention params | Count params in attention modules only |
| Total params | All params |
| Attention param ratio vs C0 | C_x attn params / C0 attn params |
| ms/step (compiled) | Wall clock |
| Primitive entropy (if compositional attn) | Same metric as FFN entropy |
| Training stability | Any NaN, loss spikes, divergence |

Also log per-run:
- Q/K gradient norms at step 1000, 5000, 10000 (compare with HoloTern findings)
- Loss curve (for overlay plot of all 5 runs)

---

## GATE CRITERIA

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| C1 matches C0 within 5% | Attention projections CAN be shared | Scale C1 to 360M |
| C2 beats C0 | Recurrence is better than softmax for PILON | Scale C2, consider dropping attention entirely |
| C2 worse than C0 by >10% | Recurrence too weak, softmax selectivity matters | Keep standard attn, focus on C1 |
| C3 beats C2 | Compositional projections help recurrence | Best combined architecture |
| C4 beats all | Different layers want different routing | Design layer-adaptive routing |
| Nothing beats C0 | Standard attention is fine for PILON | Accept it, focus on scaling FFN story |

**"Nothing beats C0" is a valid outcome.** It means PILON's contribution is purely 
in the FFN, and the attention is fine as-is. That's still a paper — just a different one.

---

## DELIVERABLES

1. `pilon_r/core/attention.py` — All new attention modules
2. `tests/test_attention.py` — Tests for all modules  
3. Updated `config.py` with attention config fields
4. Updated `model.py` TransformerBlock to use attention factory
5. Updated `train.py` CLI with attention flags
6. Results table comparing all runs
7. Loss curve overlay plot
8. Gradient norm comparison table
9. Written analysis: which attention variant (if any) complements PILON best
10. Updated EXPERIMENTS.md or PROGRESS.md with Phase C results

---

## KNOWN RISKS

**Sequential loop in GatedLinearRecurrence will be slow.** At seq_len=512, that's 
512 sequential steps per layer per forward pass. This could make C2/C3 significantly 
slower than C0/C1. Log the time and factor it into analysis — a 2x slower model that's 
5% better in quality is still interesting if the slowness is fixable with parallel scan.

**torch.compile may struggle with the sequential loop.** The per-step state update 
creates a dynamic computation graph. If compile breaks, fall back to eager mode 
for C2/C3 and note the performance difference.

**Compositional attention might collapse.** With only 16 primitives and top-k=4, 
the projections might not be expressive enough. If C1 is significantly worse than C0, 
try increasing to 24 primitives / top-k=6 before concluding it doesn't work.

**Two-phase training interaction.** PILON's Phase 1 (train all primitives) → Phase 2 
(freeze primitives, train composition) applies to FFN. For attention primitives, the 
same schedule should work, but verify that attention primitive entropy doesn't collapse 
in Phase 2.
