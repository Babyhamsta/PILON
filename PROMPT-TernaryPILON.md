# Claude Code Prompt: Ternary PILON-R (BitNet × PILON Blend)

## Context

PILON-R is a research project that replaces dense FFN layers in transformers with shared low-rank primitive banks combined via learned per-layer composition weights. It has been proven to work at 48M parameter scale — the architecture trains stably, primitives don't collapse, and the model learns language with dramatically lower VRAM usage and ~200k tok/s throughput.

We want to integrate BitNet b1.58's ternary quantization (weights constrained to {-1, 0, +1}) directly into PILON-R's primitive banks. This is NOT post-training quantization — we are modifying the architecture so primitives are natively ternary during training, using the straight-through estimator for gradient flow. The composition weights stay in full precision (they're tiny).

The goal: compound three compression mechanisms — structural sharing (PILON), arithmetic simplification (ternary), and activation sparsity (top-k) — into a single architecture. Nobody has published this combination.

## What BitLinear Does (the BitNet mechanism)

BitLinear replaces `nn.Linear` with a layer that:

1. **Forward pass**: Quantizes weights to {-1, 0, +1} using absmean quantization:
   ```python
   # Weight quantization (ternary)
   scale_w = weights.abs().mean()
   weights_ternary = (weights / (scale_w + eps)).round().clamp(-1, 1)
   
   # Activation quantization (INT8, per-token)
   scale_x = x.abs().max(dim=-1, keepdim=True).values
   x_quant = (x * (127.0 / (scale_x + eps))).round().clamp(-128, 127)
   
   # Forward: matmul with quantized weights and activations
   # Then rescale: output = (matmul_result * scale_w * scale_x) / 127.0
   ```

2. **Backward pass**: Uses the straight-through estimator (STE) — gradients flow through the quantization as if it weren't there, updating the latent full-precision "shadow" weights. The shadow weights are what get quantized each forward pass.

3. **Normalization**: BitNet uses SubLN (sub-layer normalization) before quantized projections and RMSNorm before the BitLinear layer to control activation variance. This is critical for training stability.

4. **Activation function**: BitNet uses Squared ReLU (`relu(x)^2`) instead of SiLU/GELU. This produces sparser activations which pairs well with ternary weights. However, PILON already uses SiLU in its FFN. We should test both — start with SiLU to maintain comparability with existing PILON baselines, then try Squared ReLU as an experiment.

## Architecture Overview

Current PILON architecture (what exists now):
```
PrimitiveBank:
  A: (n_primitives, d_in, rank)    # fp16 — left factor of low-rank decomposition
  B: (n_primitives, rank, d_out)   # fp16 — right factor of low-rank decomposition
  
  forward(x, weights, top_k):
    # weights = softmax(composition_logits)  [per-layer, learned]
    # Select top-k primitives
    # For each selected primitive: output += weight_i * (x @ A_i @ B_i)
    # Return combined output

LayerCompositionWeights:
  logits_fc1: (n_primitives,)  # fp32 — per-layer learned composition
  logits_fc2: (n_primitives,)  # fp32
  temperature: float
```

Target ternary PILON architecture:
```
TernaryPrimitiveBank:
  A_shadow: (n_primitives, d_in, rank)    # fp32 — latent weights for gradient updates
  B_shadow: (n_primitives, rank, d_out)   # fp32 — latent weights for gradient updates
  
  forward(x, weights, top_k):
    # Quantize A and B to ternary on the fly
    A_ternary = ternary_quantize(A_shadow)  # {-1, 0, +1}
    B_ternary = ternary_quantize(B_shadow)  # {-1, 0, +1}
    # Quantize activations to INT8
    x_quant = activation_quantize(x)
    # Select top-k primitives (unchanged from current PILON)
    # For each selected: output += weight_i * (x_quant @ A_ternary_i @ B_ternary_i)
    # Rescale output using stored scales
    # Return combined output

LayerCompositionWeights:
  # UNCHANGED — stays fp32, these are tiny and need precision
  logits_fc1: (n_primitives,)
  logits_fc2: (n_primitives,)
  temperature: float
```

## Implementation Plan — Phased

### Phase 1: Core BitLinear Integration (DO THIS FIRST)

**Files to modify**: `pilon_r/core/primitives.py`

1. **Create a `ternary_quantize` function:**
   ```python
   def ternary_quantize(w):
       """Quantize weights to {-1, 0, +1} using absmean scaling."""
       scale = w.abs().mean() + 1e-5
       w_scaled = w / scale
       w_ternary = w_scaled.round().clamp(-1, 1)
       # Straight-through estimator: 
       # detach the quantized version but keep gradient path through original
       return (w_ternary - w_scaled).detach() + w_scaled, scale
   ```

2. **Create an `activation_quantize` function:**
   ```python
   def activation_quantize(x, bits=8):
       """Quantize activations to INT8 per-token using absmax."""
       Qb = 2 ** (bits - 1) - 1  # 127 for 8-bit
       scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
       x_quant = (x * Qb / scale).round().clamp(-Qb, Qb)
       # STE: same pattern
       return (x_quant - x * Qb / scale).detach() + x * Qb / scale, scale, Qb
   ```

3. **Modify `PrimitiveBank` class** to support a `ternary=True` mode:
   - Add a `ternary: bool = False` parameter to `__init__`
   - When `ternary=True`:
     - Store the A and B parameters in fp32 (these become the shadow weights)
     - Add an `RMSNorm` layer for input normalization before quantization
     - In `forward()`, `forward_fast()`, `forward_sparse()`, and `forward_topk_fused()`:
       - Apply `ternary_quantize()` to A and B before the matmul
       - Apply `activation_quantize()` to the input x
       - Rescale output appropriately: `output = raw_output * (scale_w_A * scale_w_B * scale_x) / Qb`
     - IMPORTANT: The quantization must happen INSIDE the forward pass, not precomputed, because the STE needs the gradient path
   - When `ternary=False`: behavior is identical to current code (no changes)

4. **Do NOT modify `LayerCompositionWeights` or `ExpertCompositionBank`** — composition weights stay fp32. They're a few hundred parameters per layer and need precision for the softmax routing to work correctly.

5. **Do NOT modify the `BandPrimitiveBanks` container** — it just manages which layers share which banks. The ternary logic is entirely inside `PrimitiveBank`.

**Files to modify**: `pilon_r/core/ffn.py`

6. **Add optional SubLN normalization** to `CompositionalFFN`:
   - Add a `use_subln: bool = False` parameter
   - When enabled, add an RMSNorm before the output of each FFN projection (before the residual add)
   - This stabilizes training when weights are ternary
   - Wire this to the config so it's easy to toggle

**Files to modify**: `pilon_r/core/config.py`

7. **Add ternary config options** to the model config dataclass:
   ```python
   # In ModelConfig or equivalent:
   ternary_primitives: bool = False      # Enable ternary quantization in primitive banks
   activation_bits: int = 8              # Activation quantization bitwidth
   use_subln: bool = False               # SubLN normalization for ternary stability
   use_squared_relu: bool = False        # Squared ReLU instead of SiLU (experiment later)
   ```

### Phase 2: Training Adjustments

**Files to modify**: `pilon_r/train.py`

8. **Parameter group handling**: The shadow weights (A_shadow, B_shadow) should be in fp32 even if the rest of the model uses bf16 mixed precision. Make sure the parameter groups for primitives use fp32 master weights. This may already be partially handled by your mixed precision setup, but verify — ternary training is very sensitive to gradient precision.

9. **Learning rate considerations**: BitNet research has conflicting findings on LR:
   - Microsoft's original work used higher LR for ternary weights
   - Later research (Nielsen et al.) found smaller LR works better for small models
   - **Recommendation**: Start with your existing PILON LR settings unchanged. If training is unstable, try halving the primitive LR. If it converges too slowly, try 2x. Log this clearly.

10. **Gradient clipping**: Add or verify gradient clipping is active. Ternary training can produce gradient spikes due to the STE. A max_norm of 1.0 is a good starting point.

11. **The two-phase curriculum (Phase 1: primitives, Phase 2: compositions) should stay unchanged.** The ternary constraint doesn't change when things should train — it changes how the primitives represent information. Keep the curriculum and compare against the fp16 PILON baseline directly.

### Phase 3: First Experiment (48M Smoke Test)

**Goal**: Determine if ternary PILON trains at all and how it compares to fp16 PILON at the same scale.

**Run configuration** (should match existing 48M PILON config exactly, except ternary=True):
```
d_model=512, n_layers=8, n_heads=8, d_ff=2048
48 primitives, rank=48, top_k=8
3 bands: early (0-2), middle (3-5), late (6-7)
ternary_primitives=True, activation_bits=8, use_subln=True
Dataset: same as existing 48M baseline (likely TinyStories or OpenWebText-100k)
```

**What to log and compare**:
- Training loss curves (overlay ternary vs fp16 PILON vs dense baseline)
- Primitive entropy over training (are all primitives being used, or does ternary cause collapse?)
- Composition weight distributions (are the softmax outputs still diverse?)
- Final perplexity
- Training throughput (tok/s) — ternary forward passes may actually be faster due to simpler arithmetic in PyTorch, but the STE overhead might cancel it out
- VRAM usage (shadow weights are fp32, so VRAM might actually increase slightly during training — that's fine, the win is at inference)

**Success criteria**:
- If ternary PILON converges within ~1.5x the loss of fp16 PILON at the same step count: strong positive signal
- If ternary PILON converges to the SAME loss: exceptional result, publish-worthy
- If ternary PILON diverges or is >2x worse: investigate SubLN, LR, gradient clipping before concluding it doesn't work

### Phase 4: Squared ReLU Experiment (only after Phase 3 succeeds)

**Files to modify**: `pilon_r/core/ffn.py`

Replace SiLU activation with Squared ReLU when `use_squared_relu=True`:
```python
def squared_relu(x):
    return torch.relu(x).square()
```

Run the same 48M config but with `use_squared_relu=True`. Squared ReLU produces sparser activations which compound with ternary weights and top-k sparsity. This could be the configuration that really unlocks efficiency.

### Phase 5: Inference Optimization (only after Phase 3/4 succeed)

At inference time, the ternary primitives can be stored as actual INT2 (packed ternary) instead of fp32 shadow weights. This is where the real memory savings appear:

- Each primitive weight is {-1, 0, +1} = 1.58 bits = ~2 bits packed
- A bank of 48 primitives at rank 48, d_model 512:
  - FP16: 48 × 512 × 48 × 2 bytes = 2.36 MB per bank
  - Ternary packed: 48 × 512 × 48 × 0.25 bytes = 0.29 MB per bank
  - **8x memory reduction** on the primitive banks alone

For inference, create a `pack_ternary()` function that converts the trained shadow weights into packed INT2 format, and a corresponding inference-only forward pass that unpacks on the fly. This is the precursor to any eventual C++ inference engine.

## Critical Design Decisions

1. **Shadow weights in fp32, not fp16**: The STE passes tiny gradient signals through the quantization boundary. In fp16, these can vanish. Microsoft uses fp32 shadow weights for BitNet training. Match this.

2. **Quantize inside forward(), not in a hook or separate step**: The STE requires that `w_ternary` is computed as a function of the shadow weights within the autograd graph. If you quantize outside the graph, gradients won't flow.

3. **Do NOT change the composition mechanism**: The softmax over composition logits, the top-k selection, the temperature — all of this stays fp32 and unchanged. The ternary constraint is ONLY on the primitive bank contents (A and B matrices). This is a key architectural decision: the "what to compose" is ternary, the "how to compose" is precise.

4. **Initialization**: Keep the existing QR-orthogonalized initialization for the shadow weights. The STE will push them toward ternary-friendly distributions during training. Do not initialize directly to ternary values — let training find the right quantization.

5. **Do NOT implement custom CUDA kernels yet**: Use standard PyTorch operations. The ternary matmuls won't be faster in PyTorch because PyTorch doesn't have optimized ternary kernels. The speed/memory wins come at inference time with packed formats. During training, we just want correctness and convergence.

## File Modification Summary

| File | Changes | Risk |
|------|---------|------|
| `core/primitives.py` | Add ternary_quantize, activation_quantize, modify PrimitiveBank forward passes | **High** — this is the core change, test thoroughly |
| `core/ffn.py` | Add SubLN option, add Squared ReLU option | **Low** — additive changes behind flags |
| `core/config.py` | Add ternary config fields | **Low** — just new dataclass fields with defaults |
| `train.py` | Ensure fp32 shadow weights, verify gradient clipping | **Low** — mostly verification |
| `evaluate.py` | No changes needed for Phase 3 | None |
| `benchmark.py` | Later: add packed ternary inference path | Phase 5 only |

## What NOT To Do

- Do NOT rewrite the training loop or curriculum for ternary. Start with the exact same setup.
- Do NOT implement a C++ inference engine. That's premature.
- Do NOT modify the attention layers. Only FFN primitives go ternary.
- Do NOT add any new dependencies. This uses only PyTorch operations.
- Do NOT remove or deprecate any existing code paths. The `ternary=False` default means all existing functionality is preserved.
- Do NOT change how bands work, how layers are grouped, or how composition weights are shared. The ternary change is strictly inside PrimitiveBank's forward computation.
- Do NOT attempt to quantize the embedding layer or LM head. Only FFN primitive banks.

## Quick Validation Checklist

After implementation, before running any training:

1. **Gradient flow test**: Create a tiny model (2 layers, 4 primitives, rank 8). Run one forward + backward pass. Verify that `A_shadow.grad` and `B_shadow.grad` are non-zero and finite. If gradients are zero, the STE is broken.

2. **Ternary verification**: After the forward pass, check that the actually-used weights are in {-1, 0, +1}: `assert ternary_weights.unique().sort().values.tolist() == [-1.0, 0.0, 1.0]` (or a subset).

3. **Composition weight independence**: Verify that composition weights (logits_fc1, logits_fc2) are NOT affected by ternary quantization and remain in their original precision.

4. **Numerical stability**: Run 100 training steps on TinyStories with the tiny model. Check for NaN/Inf in loss, gradients, and outputs. If you see NaN, the RMSNorm or SubLN is likely needed.

5. **Baseline equivalence**: Run the same tiny model with `ternary=False`. Verify it produces identical results to the existing codebase (no accidental changes to the non-ternary path).
