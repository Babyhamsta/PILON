# PILON-R Phase B.5: 48M Crossover Experiment Results

**Date:** February 24, 2026
**Hardware:** RTX 4070 (12GB VRAM), Windows 11
**Dataset:** HuggingFaceFW/fineweb-edu (streaming)
**Tokens:** 500M per config (15,258 steps @ 32,768 tokens/step)

---

## Experiment Overview

Four configurations trained head-to-head on 500M tokens to validate PILON's structural advantages at 48M scale before committing to the longer 360M run.

| # | Config | Description |
|---|--------|-------------|
| 1 | Dense-48M | Standard FFN baseline (no PILON) |
| 2 | PILON-48M-full | Compositional FFN, all 48 primitives in VRAM |
| 3 | PILON-48M-tiered | Compositional FFN, n_hot=12 (25% of primitives in VRAM) |
| 4 | PILON-48M-exit | Tiered model + post-hoc trained early exit gates |

### PILON Training Flags

All PILON configs used proper training flags:
```
--freeze-primitives-phase2    # Phase 1 learns primitives, Phase 2 composes
--topk-cache-steps 10         # Cache router decisions for speed
--comp-lr-mult 4.0            # Higher LR for small composition weights
--comp-entropy-weight 0.001   # Prevent primitive collapse
```

### Phase Schedule (PILON configs)

| Phase | Steps | Description |
|-------|-------|-------------|
| Phase 1 (20%) | 0 - 3,051 | Primitives train, compositions frozen |
| Phase 2 (50%) | 3,052 - 7,629 | Compositions train, primitives frozen |
| Phase 3 (30%) | 7,630 - 15,258 | All parameters train |

---

## Results Summary

### Final Metrics (step 15,000)

| Config | Val Loss | Val PPL | Peak VRAM | Wall Clock | Throughput |
|--------|----------|---------|-----------|------------|------------|
| Dense-48M | **4.165** | **64.42** | 3.62 GB | 3.45 hr | ~42k tok/s |
| PILON-48M-full | 4.691 | 108.94 | 3.82 GB | 3.89 hr | ~38k tok/s |
| PILON-48M-tiered | 4.693 | 109.17 | 3.61 GB | 4.23 hr | ~32k tok/s |
| PILON-48M-exit | — | — | — | 24 min (gate training) | — |

### Loss Ratios vs Dense Baseline

| Config | Loss Ratio | PPL Ratio |
|--------|-----------|-----------|
| PILON-full / Dense | 1.126x | 1.691x |
| PILON-tiered / Dense | 1.127x | 1.695x |

### Gate Training (Config 4)

Trained exit gates on tiered checkpoint for 3 epochs (9,765 sequences, ~5M tokens):

| Epoch | Avg Loss | Time |
|-------|----------|------|
| 1 | 0.5270 | 7.7 min |
| 2 | 0.5227 | 7.7 min |
| 3 | 0.5219 | 7.7 min |

Model saved: `outputs/48m_crossover/pilon_48m_exit/model_with_gates.pt`

---

## Comparison with Old 48M Results (Phase B)

| Metric | Old Phase B (97K steps, 1B tok) | New Crossover (15K steps, 500M tok) |
|--------|--------------------------------|-------------------------------------|
| PILON loss gap vs dense | ~1.22x | ~1.13x |
| PILON throughput | ~200k tok/s (reported) | ~38k tok/s (measured) |
| Dense throughput | — | ~42k tok/s (measured) |
| Training flags | phase1-sparse, top-k=4, comp-lr-mult, entropy | phase1-sparse, top-k=8, comp-lr-mult, entropy, freeze-phase2 |
| Steps | 97,000 (10,000 default) | 15,258 |
| Dataset | Elriggs/openwebtext-100k | HuggingFaceFW/fineweb-edu |

**Key differences:**
- Old Phase B used `top_k=4`; crossover uses `top_k=8`
- Old Phase B trained on smaller dataset (openwebtext-100k, ~50M tokens repeated)
- Old Phase B ran far more steps (97K vs 15K)
- Old throughput of ~200k tok/s appears to have been an optimistic measurement; actual sustained throughput on RTX 4070 is ~38-42k tok/s

---

## VRAM Analysis

| Config | Peak VRAM | Checkpoint Size | Notes |
|--------|-----------|-----------------|-------|
| Dense-48M | 3.62 GB | 588 MB | Standard FFN weights |
| PILON-48M-full | 3.82 GB | 801 MB | All 48 primitives + composition weights |
| PILON-48M-tiered | 3.61 GB | 599 MB | Only 12/48 primitives in VRAM |
| PILON-48M-exit | — | 269 MB | Tiered + exit gates (smallest checkpoint) |

**Observation:** Tiered config achieves same VRAM as dense while storing 4x more primitive knowledge on disk. Exit model has the smallest checkpoint due to only saving active primitives + gates.

---

## Training Dynamics

### Dense-48M
- Smooth, monotonic loss decrease
- Final train_loss=4.406, val_loss=4.165
- Consistent ~42k tok/s throughput

### PILON-48M-full
- Phase 1 (primitives only): Loss starts higher, drops fast
- Phase 2 (compositions only): Brief loss spike, then rapid recovery
- Phase 3 (all): Gradual convergence
- Final train_loss=4.957, val_loss=4.691
- Composition entropy stable at ~2.0 (healthy, no collapse)
- Gini coefficient ~0.855 (expected sparsity from top-k=8)

### PILON-48M-tiered
- Nearly identical to full config (val_loss 4.693 vs 4.691)
- Tiered loading adds overhead: ~32k vs ~38k tok/s (16% slower)
- But uses less VRAM (3.61 vs 3.82 GB)
- Same composition statistics (entropy ~2.0, gini ~0.862)

### PILON-48M-exit (Gate Training)
- Gate loss converged from 0.583 → 0.522 over 3 epochs
- Gates learn which tokens are "easy" (low KL divergence between intermediate and final logits)
- Inference speedup to be measured via benchmark suite

---

## Key Findings

### 1. Dense Still Wins on Raw Loss at 500M Tokens
Dense-48M achieves 4.165 val_loss vs PILON's 4.691 — a 1.13x gap. This is actually better than the old Phase A gap (1.5x) and the old Phase B gap (1.22x), likely due to the training flag improvements.

### 2. Tiered Loading Works with Negligible Quality Loss
Full vs tiered: 4.691 vs 4.693 (0.04% difference). The tiered bank successfully maintains quality while keeping only 25% of primitives in VRAM.

### 3. Throughput is Not 200k tok/s
Actual sustained throughput on RTX 4070 is ~38-42k tok/s for 48M models. The previously reported 200k figure was likely a measurement artifact. Dense still edges out PILON in throughput (~42k vs ~38k tok/s).

### 4. Gate Training Works
Exit gates converge in 3 short epochs. The BCE loss drops from 0.583 to 0.522, indicating gates learn meaningful skip predictions. Full inference speedup measurement pending.

### 5. Training Flags Matter
Using `--freeze-primitives-phase2`, `--comp-lr-mult 4.0`, and `--comp-entropy-weight 0.001` is essential for PILON. Without these flags, PILON trains like dense (no staged learning, no composition specialization) and performs worse.

---

## Checkpoints

```
outputs/48m_crossover/
├── dense_48m/final_model.pt           (588 MB)
├── pilon_48m_full/final_model.pt      (801 MB)
├── pilon_48m_tiered/final_model.pt    (599 MB)
└── pilon_48m_exit/model_with_gates.pt (269 MB)
```

## Next Steps

1. **Run benchmarks** on all 4 checkpoints:
   ```bash
   python -m pilon_r.benchmark outputs/48m_crossover/dense_48m/final_model.pt --device cuda
   python -m pilon_r.benchmark outputs/48m_crossover/pilon_48m_full/final_model.pt --device cuda
   python -m pilon_r.benchmark outputs/48m_crossover/pilon_48m_tiered/final_model.pt --device cuda
   python -m pilon_r.benchmark outputs/48m_crossover/pilon_48m_exit/model_with_gates.pt --device cuda
   ```

2. **360M crossover experiment** — same 4 configs at larger scale (1B tokens, ~6 hours)

3. **Investigate loss gap** — PILON's 1.13x gap may close with:
   - More tokens (Chinchilla-optimal is ~960M for 48M params)
   - Hyperparameter tuning (comp-lr-mult, entropy weight, top-k)
   - Different rank/primitive count ratios
