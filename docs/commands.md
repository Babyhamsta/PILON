# Commands

Quickstart commands for common workflows.

## Train baseline
python -m pilon_r.train --baseline --output-dir outputs/phase_b --device cuda

## Smoke test (MoE enabled)
python -m pilon_r.train --smoke-test --moe --device cuda

## Generate samples
python -m analysis.generation_samples outputs/phase_a/baseline/checkpoint_step_10000.pt --interactive --device cuda

## SFT
python -m pilon_r.sft outputs/phase_b/checkpoint_step_10000.pt --epochs 3 --output-dir outputs/sft --device cuda
