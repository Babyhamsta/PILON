# PILON-R

**Program-Induced Linear Operator Network with Reasoning**

PILON-R replaces dense FFN weight matrices in transformers with shared low-rank primitives combined via learned per-layer composition weights. Instead of storing full `(d_model, d_ff)` matrices per layer, a small bank of low-rank primitives is shared across layers within a band, and each layer learns *which* primitives to combine and *how* to weight them.

The result: competitive language modeling quality at a fraction of the FFN parameter cost, with structural knobs (rank scheduling, progressive unfreezing, tiered VRAM loading, early exit) that dense FFNs simply cannot offer.

## How It Works

### Dense FFN (standard transformer)
```
x -> [d_model x d_ff] -> activation -> [d_ff x d_model] -> out
       ^                                  ^
       Two full dense matrices per layer (millions of params each)
```

### PILON Compositional FFN
```
                    Shared Primitive Bank (per band)
                    ┌─────────────────────────────┐
                    │  P0: A0 (d_in, r) @ B0 (r, d_out)  │
                    │  P1: A1 (d_in, r) @ B1 (r, d_out)  │
                    │  ...                                 │
                    │  P47: A47 (d_in, r) @ B47 (r, d_out)│
                    └─────────────────────────────┘
                                 |
                        Top-k selection (k=8)
                                 |
                    ┌─────────────────────────────┐
                    │  Gather 8 primitives          │
                    │  Concatenate into fused map   │
                    │  2 GEMMs (x @ A_cat, U @ B_cat) │
                    │  Weighted sum via sqrt(w)      │
                    └─────────────────────────────┘
                                 |
                              output
```

Layers within a **band** (e.g., layers 0-2 = "early") share the same primitive bank but learn independent composition weights. This means early layers can share low-level feature extractors while late layers share task-specific transforms.

### Ternary Quantization (BitNet b1.58)

Primitive weights can be constrained to `{-1, 0, +1}` using absmean scaling with straight-through estimation (STE). Combined with 8-bit activation quantization and SubLN normalization, this produces extremely compact models with minimal quality loss.

## Architecture Overview

| Component | Description |
|-----------|-------------|
| **PrimitiveBank** | Stores `n` low-rank primitives as packed `A (n, d_in, rank)` and `B (n, rank, d_out)` tensors with learned latent scale/bias |
| **BandPrimitiveBanks** | Groups layers into bands that share primitive banks (separate fc1/fc2 banks) |
| **LayerCompositionWeights** | Per-layer learned logits over primitives, softmax-normalized, top-k selected |
| **CompositionalFFN** | Fused forward path: top-k select, concatenate, 2 GEMMs, weighted sum |
| **MoECompositionalFFN** | Token-dependent routing: each token picks different expert compositions |
| **TieredPrimitiveBank** | VRAM-efficient: only `n_hot` primitives in GPU memory, rest in CPU pinned memory |
| **ExitGate** | Per-layer gate that skips FFN computation for "easy" tokens during inference |

## Model Configurations

### 48M (Development / Ablation)
```
d_model=512, n_layers=8, n_heads=8, d_ff=2048
n_primitives=48, rank=48, top_k=8
3 bands: early(0-2), middle(3-5), late(6-7)
```

### 360M (Scale Validation)
```
d_model=1024, n_layers=24, n_heads=16, d_ff=4096
n_primitives=80, rank=80, top_k=8
3 bands: early(0-7), middle(8-15), late(16-23)
```

## Results

### Training Stability and Quality (48M, 500M tokens)

| Model | Final Val Loss | vs Dense Baseline |
|-------|---------------|-------------------|
| Dense-48M (baseline) | ~3.8 | 1.00x |
| PILON-48M (fp16) | ~4.1 | ~1.08x |
| PILON-48M (ternary + SubLN + SqReLU) | ~4.2 | ~1.10x |

- Training is fully stable: no NaN, no divergence, no primitive collapse
- Primitive entropy stays healthy (~3.4+) throughout training
- The gap is convergence speed, not a ceiling — loss continues improving with more tokens

### Throughput (RTX 4070, batch=8, seq=512, fwd+bwd)

| Config | Eager (ms) | Compiled (ms) | tok/s (compiled) |
|--------|-----------|---------------|-----------------|
| Dense-48M | 54 | 49 | ~84k |
| PILON-48M-Ternary | 101 | 54 | ~76k |
| **Ratio** | 1.88x | **1.10x** | - |

`torch.compile` fuses PILON's many small elementwise kernels (ternary quantization, RMSNorm, sqrt scaling, etc.) into a handful of Triton kernels, closing the throughput gap almost entirely. Without compile, PILON suffers from ~560 tiny CUDA kernel launches per iteration vs ~32 for dense.

### Compute Math (Why PILON Should Be Efficient)

Per token, per layer at 48M config:

| | Multiplies | % of Dense |
|---|-----------|------------|
| Dense FFN | ~2.1M | 100% |
| PILON FFN (top-8 of 48, rank 48) | ~1.0M | **48%** |

PILON does roughly half the FLOPs of a dense FFN. The compiled profiler confirms this: PILON matmul time (67ms) < Dense matmul time (72ms) across identical workloads.

## Quickstart

### Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For torch.compile support (highly recommended):
pip install triton-windows
```

### Setup (Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke Test
```bash
python -m pilon_r.train --smoke-test --device cuda
```

### Train 48M Ternary PILON
```bash
python -m pilon_r.train \
    --model-size 48m \
    --ffn-type compositional \
    --phase1-sparse \
    --phase1-top-k 8 \
    --ternary \
    --use-subln \
    --use-squared-relu \
    --compile \
    --freeze-primitives-phase2 \
    --topk-cache-steps 10 \
    --comp-lr-mult 2.0 \
    --forward-fast-mode on \
    --forward-fast-min-topk 1 \
    --band-diversity-weight 0.01 \
    --no-checkpoint-ffn \
    --total-tokens 500000000 \
    --batch-size 8 \
    --grad-accum 8 \
    --seq-len 512 \
    --dataset HuggingFaceFW/fineweb-edu \
    --output-dir outputs/48m_ternary \
    --log-comp-stats
```

Or use the provided scripts:
```bash
bash scripts/run_48m_ternary_crossover.sh   # 48M, 500M tokens
bash scripts/run_360m_ternary_crossover.sh  # 360M, 1B tokens, torch.compile
```

### Profile Throughput
```bash
python scripts/profile_pilon.py
```

### Evaluate / Generate
```bash
python -m pilon_r.evaluate outputs/48m_ternary/final_model.pt --device cuda
```

### SFT Fine-tuning
```bash
python -m pilon_r.sft outputs/48m_ternary/final_model.pt \
    --epochs 1 --output-dir outputs/48m_ternary_sft --device cuda
```

## Key Training Features

- **Two-phase training**: Phase 1 trains primitives with all active (no top-k), Phase 2 freezes primitives and trains composition weights with top-k sparsity
- **Separate parameter groups**: Primitives get higher LR (2x) + weight decay; compositions get lower LR (0.5x) + zero weight decay
- **Rank scheduling**: Start with low effective rank, increase to full rank during warmup
- **Progressive band unfreezing**: Early band trains first, then middle, then late
- **Ternary weight caching**: Pre-quantize all primitives once per optimizer step, reuse via index_select across gradient accumulation micro-batches
- **`torch.compile`**: ~1.9x speedup on PILON by fusing elementwise ops into Triton kernels

## Key CLI Flags

| Flag | Description |
|------|-------------|
| `--model-size {48m,360m,500m}` | Model scale |
| `--ffn-type {compositional,standard}` | PILON vs dense baseline |
| `--ternary` | Enable ternary weight quantization |
| `--use-subln` | SubLN normalization (ternary stability) |
| `--use-squared-relu` | Squared ReLU activation |
| `--compile` | Enable torch.compile |
| `--phase1-sparse` | Use top-k in phase 1 (skip dense warmup) |
| `--freeze-primitives-phase2` | Freeze primitive banks in phase 2 |
| `--checkpoint-ffn` / `--no-checkpoint-ffn` | Gradient checkpointing for FFN (VRAM vs speed) |
| `--log-comp-stats` | Log composition weight statistics |

## Project Structure

```
pilon_r/
  train.py                     Training loop
  evaluate.py                  Evaluation + generation
  sft.py                       Supervised fine-tuning
  benchmark.py                 Inference benchmarking
  benchmark_efficiency.py      VRAM / compute / quality benchmarks

pilon_r/core/
  model.py                     PILONTransformer
  primitives.py                PrimitiveBank, ternary quantization, RMSNorm
  ffn.py                       CompositionalFFN, MoECompositionalFFN
  tiered_bank.py               TieredPrimitiveBank (hot/warm VRAM tiering)
  early_exit.py                ExitGate + gate training
  config.py                    All configuration dataclasses
  data.py                      Streaming data loading
  metrics.py                   Metric utilities

pilon_r/configs/
  model_360m.py                360M model configurations
  model_500m.py                500M model configurations

scripts/
  run_48m_ternary_crossover.sh   48M ternary experiment
  run_360m_ternary_crossover.sh  360M ternary experiment
  run_48m_crossover.sh           48M fp16 crossover
  run_360m_crossover.sh          360M fp16 crossover
  profile_pilon.py               Throughput profiler

docs/
  PROGRESS.md                  Detailed research progress log
  PHASE_PLAN_v2.1.md           Development plan
  commands.md                  Common commands reference
```

## Outputs

Training runs write to `outputs/` by default. Each run saves:
- `final_model.pt` — Final checkpoint
- `metrics.jsonl` — Per-step training metrics
- `config.json` — Full model + training configuration
- Periodic checkpoints at configurable intervals

## Research Status

| Phase | Status | Outcome |
|-------|--------|---------|
| Phase 0: Representation Viability | Complete | Low-rank primitives can represent FFN structure |
| Phase A: Training From Scratch | Complete | Stable training, learns language, no collapse |
| Phase B: Optimization & Throughput | Complete | ~200k tok/s, 1.22x convergence gap |
| Phase B.5: Structural Advantages | Complete | Tiered banks, early exit, sparse compute path |
| Ternary Quantization (BitNet b1.58) | Complete | {-1,0,1} weights, 1.10x compiled throughput ratio |
| Phase C: SSM/MLA Integration | Planned | Long context, memory efficiency |
| Phase D: Reasoning Integration | Planned | R1-style inference-time reasoning |

## License

MIT License (see `LICENSE`).
