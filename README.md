# PILON-R

PILON-R (Program-Induced Linear Operator Network with Reasoning) explores a compositional weight parameterization for transformer FFN layers. The goal is to replace dense FFN matrices with shared low-rank primitives plus learned composition weights.

## Structure
- `pilon_r/` Package entry points (train/eval/sft/compress)
- `pilon_r/core/` Shared modules (model, data, metrics, etc.)
- `analysis/` One-off analysis scripts
- `docs/` Project notes and commands
- `requirements.txt` Python dependencies

## Quickstart (Windows)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Smoke test:
```
python -m pilon_r.train --smoke-test --device cuda
```

If you do not have CUDA:
```
python -m pilon_r.train --smoke-test --device cpu
```

## Common commands
See `docs/commands.md`.

## Outputs
Training runs write to `outputs/` by default. Checkpoints and metrics are stored under that directory.

## License
MIT License (see `LICENSE`).
