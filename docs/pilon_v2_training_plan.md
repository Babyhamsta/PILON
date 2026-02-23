# PILON v2 Training Plan

This document describes a **forced‑improvement training strategy** for PILON that directly addresses its biggest weakness (slow/unstable training) while preserving its proven strengths (dramatically lower inference compute and VRAM).

The core idea is:

> **Train PILON like a dense FFN early, then gradually remove the dense scaffold while progressively sparsifying PILON.**

This leverages knobs that dense FFNs do *not* have: **rank, top‑k, and compositional routing**, and uses them as a curriculum.

---

## 1. Core Strategy Overview

### Three mechanisms combined

1. **Hybrid Dense Scaffold**  
   A temporary dense FFN path stabilizes optimization early and provides a strong learning signal.

2. **Progressive Sparsity**  
   PILON starts close to dense behavior and is gradually sparsified (rank ↓, top‑k ↓).

3. **Staged Parameter Training**  
   Primitives learn first, compositions specialize later.

The final model is **pure PILON**. The dense path is removed entirely before training ends.

---

## 2. Preconditions (Mandatory Fixes)

Before changing training, these must already be true.

### 2.1 Vectorized primitive computation

Primitive outputs must be computed **once** and mixed **afterward**.

Required tensor layout:

- A: `[P, d_in, r]`
- B: `[P, r, d_out]`

Computation:

- `Z = einsum(x, A)` → `[batch, P, r]`
- `Y = einsum(Z, B)` → `[batch, P, d_out]`
- Mix once with routing weights

**No Python loops. No per‑primitive modules.**

Target throughput improvement:
- ~14k tok/s → **18k+ tok/s**

If this is not achieved, fix kernels before proceeding.

---

### 2.2 Runtime overrides

Each PILON block must support runtime overrides:

- `active_rank`
- `top_k`
- `dense_scaffold_alpha`
- `full_mix_enabled` (disable top‑k)

These values must be changeable per training step.

---

## 3. Hybrid Dense Scaffold

### 3.1 Forward equation

For each PILON FFN block:

```
y = α(t) * y_dense + (1 − α(t)) * y_pilon
```

Where:
- `y_dense` = standard FFN (SwiGLU / GeLU)
- `y_pilon` = compositional primitive FFN
- `α(t)` ∈ [0,1] is annealed over training

### 3.2 Dense path options

**Option A — Inline dense FFN (simplest)**
- Dense FFN exists only during training
- Removed entirely for inference

**Option B — External frozen teacher (recommended)**
- Dense teacher provides target activations/logits
- Student PILON block matches teacher output
- No extra dense compute inside the student

Option B reduces training compute if a teacher checkpoint is available.

---

## 4. Progressive Sparsity Curriculum

### 4.1 Rank schedule

Let `rank_final` be the desired inference rank.

- `rank_start = max(8, rank_final // 4)`
- Rank increases monotonically during training

Example:

| Phase | active_rank |
|------|-------------|
| Early | 16 |
| Mid | 48 |
| Final | 64 |

---

### 4.2 Top‑k schedule

Let `top_k_final` be the desired inference top‑k.

- Start with **full mixing** (top‑k disabled)
- Gradually enable sparsity

Example (`P=32`, `top_k_final=8`):

| Phase | top_k |
|------|-------|
| Early | full (32) |
| Mid | 16 |
| Final | 8 |

This prevents early router collapse and dead primitives.

---

## 5. Training Phases

### Phase 1 — Basis Formation

**Goal:** Learn a stable primitive basis under dense guidance.

**Settings:**
- `α(t)`: 1.0 → 0.3
- `top_k`: full mix
- `active_rank`: low → mid

**Training:**
- Train primitives
- Freeze or very low LR for compositions

**Learning rates:**
- Primitives: `lr = base_lr × 2.0`, `wd = 0.1`
- Compositions: `lr = 0`
- Backbone: `lr = base_lr`, `wd = 0.1`

**Losses:**
- Language modeling loss
- Optional hidden‑state MSE (teacher → student)
- Optional cosine scale regularization

---

### Phase 2 — Transition

**Goal:** Transfer capacity to compositions and introduce sparsity.

**Settings:**
- `α(t)`: 0.3 → 0.0
- `top_k`: anneal toward target
- `active_rank`: reach full

**Training:**
- Enable composition learning
- Reduce primitive LR

**Learning rates:**
- Compositions: `lr = base_lr × 1.0` (or ×2.0 if previously frozen)
- Primitives: `lr = base_lr × 0.25–0.5`
- Backbone: `lr = base_lr × 0.75`

**Losses:**
- Language modeling loss
- Optional KL(logits_teacher || logits_student)
- Optional entropy / load‑balancing loss

---

### Phase 3 — Sparse Finalization

**Goal:** Lock in inference configuration and recover quality.

**Settings:**
- `α(t) = 0`
- `top_k = top_k_final`
- `active_rank = rank_final`

**Training:**
- Train all parameters
- Low learning rate

**Losses:**
- Language modeling loss
- Very weak balance regularization (optional)

---

## 6. Progressive Layer Unfreezing (Optional)

To further stabilize early training:

| Phase | Trainable layers |
|------|------------------|
| Phase 1 | Lower 1/3 |
| Early Phase 2 | Lower + middle |
| Late Phase 2+ | All layers |

This reduces simultaneous drift across the entire model.

---

## 7. Schedules (Concrete Defaults)

Assume total training steps = `T`.

### Scaffold α(t)

- `t ∈ [0, 0.2T]`: α = linear(1.0 → 0.3)
- `t ∈ (0.2T, 0.5T]`: α = linear(0.3 → 0.0)
- `t > 0.5T`: α = 0.0

### active_rank(t)

- Phase 1: low → mid
- Phase 2: mid → full
- Phase 3: full

### top_k(t)

- Phase 1: full mix
- Phase 2: full → mid → target
- Phase 3: target

---

## 8. Metrics to Log

### Performance
- Tokens/sec
- Step time
- Peak VRAM

### Optimization health
- Primitive entropy
- % of primitives used in top‑k
- Norms: A, B, mix logits

### Quality
- Loss vs wall‑clock (not steps)
- PPL gap vs dense baseline

**Success criteria:**
- ≥30% throughput improvement
- Convergence gap <1.2× vs dense

---

## 9. Required Ablations

Run at least:

1. Baseline PILON
2. + Vectorized primitives only
3. + Staged training only
4. + Full hybrid scaffold + progressive sparsity

If #2 does not improve throughput, fix kernels first.

---

## 10. Final Notes

- PILON is not failing on expressivity. It is failing on optimization.
- Dense FFNs do not allow curriculum on capacity. PILON does.
- Use rank and top‑k as **training controls**, not static hyperparameters.

This plan is designed to **force PILON to converge faster than dense**, not merely match it.

---

**Next steps:**
- Fill in concrete values for `P`, `rank_final`, `top_k_final`, and `T`
- Implement α(t), rank, and top‑k schedulers
- Run a small‑scale validation (100–300M params) before scaling

