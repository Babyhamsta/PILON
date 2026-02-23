"""
PILON-R v2 Training

Implements the training curriculum described in docs/pilon_v2_training_plan.md:
- Dense scaffold via optional frozen teacher (distillation)
- Progressive sparsity (top_k) and rank schedules
- Staged parameter training for primitives vs compositions

This script keeps the core model unchanged and applies runtime overrides
(top_k, active_rank) plus scaffold loss scheduling in the training loop.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from .core.config import (
    ModelConfig,
    TrainingConfig,
    get_compression_config,
    get_all_compression_levels,
)
from .core.model import PILONTransformer, create_model, create_baseline_model
from .core.ffn import CompositionalFFN
from .core.data import load_text_dataset, get_tokenizer, create_dataloader
from .core.metrics import Logger, TrainingMetrics
from .configs.model_360m import get_360m_config, get_360m_training_config
from .configs.model_500m import get_500m_config, get_500m_training_config


# ---------------------------
# Utility helpers
# ---------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_str(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device)


def sync_if_cuda(device) -> None:
    if device_str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_autocast_dtype(precision: str, device) -> Optional[torch.dtype]:
    dev = device_str(device)
    if not dev.startswith("cuda"):
        return None
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "fp16":
        return torch.float16
    return None


def get_autocast_context(amp_dtype: Optional[torch.dtype], device):
    dev = device_str(device)
    if amp_dtype is None or not dev.startswith("cuda"):
        return nullcontext()
    return autocast(device_type="cuda", dtype=amp_dtype)


try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return False


# ---------------------------
# Schedule helpers
# ---------------------------

def linear_schedule(step: int, start_step: int, end_step: int, start_val: float, end_val: float) -> float:
    if step <= start_step:
        return start_val
    if step >= end_step:
        return end_val
    progress = (step - start_step) / max(1, end_step - start_step)
    return start_val + progress * (end_val - start_val)


def get_phase(step: int, phase1_end: int, phase2_end: int) -> int:
    if step < phase1_end:
        return 1
    if step < phase2_end:
        return 2
    return 3


def get_scaffold_alpha(step: int, total_steps: int, phase1_frac: float, phase2_frac: float) -> float:
    phase1_end = int(total_steps * phase1_frac)
    phase2_end = int(total_steps * phase2_frac)
    if step <= phase1_end:
        return linear_schedule(step, 0, phase1_end, 1.0, 0.3)
    if step <= phase2_end:
        return linear_schedule(step, phase1_end, phase2_end, 0.3, 0.0)
    return 0.0


def get_active_rank(
    step: int,
    total_steps: int,
    rank_start: int,
    rank_mid: int,
    rank_final: int,
    phase1_frac: float,
    phase2_frac: float
) -> int:
    phase1_end = int(total_steps * phase1_frac)
    phase2_end = int(total_steps * phase2_frac)
    if step <= phase1_end:
        return int(round(linear_schedule(step, 0, phase1_end, rank_start, rank_mid)))
    if step <= phase2_end:
        return int(round(linear_schedule(step, phase1_end, phase2_end, rank_mid, rank_final)))
    return rank_final


def get_active_primitives(
    step: int,
    total_steps: int,
    prim_start: int,
    prim_mid: int,
    prim_final: int,
    phase1_frac: float,
    phase2_frac: float
) -> int:
    phase1_end = int(total_steps * phase1_frac)
    phase2_end = int(total_steps * phase2_frac)
    if step <= phase1_end:
        return int(round(linear_schedule(step, 0, phase1_end, prim_start, prim_mid)))
    if step <= phase2_end:
        return int(round(linear_schedule(step, phase1_end, phase2_end, prim_mid, prim_final)))
    return prim_final


def get_runtime_top_k(
    step: int,
    total_steps: int,
    n_primitives: int,
    top_k_final: int,
    top_k_mid: Optional[int],
    phase1_frac: float,
    phase2_frac: float,
    phase1_sparse: bool = False,
    phase1_top_k: Optional[int] = None
) -> Optional[int]:
    phase1_end = int(total_steps * phase1_frac)
    phase2_end = int(total_steps * phase2_frac)

    if step <= phase1_end:
        if phase1_sparse:
            start_k = phase1_top_k or top_k_final
            start_k = int(max(1, min(n_primitives, start_k)))
            return int(max(1, min(n_primitives, start_k)))
        return None  # full mix

    if step <= phase2_end:
        start_k = n_primitives
        if phase1_sparse:
            start_k = phase1_top_k or top_k_final
        start_k = int(max(1, min(n_primitives, start_k)))

        if top_k_mid is not None:
            phase2_mid = phase1_end + max(1, (phase2_end - phase1_end) // 2)
            if step <= phase2_mid:
                top_k = linear_schedule(step, phase1_end, phase2_mid, start_k, top_k_mid)
            else:
                top_k = linear_schedule(step, phase2_mid, phase2_end, top_k_mid, top_k_final)
        else:
            top_k = linear_schedule(step, phase1_end, phase2_end, start_k, top_k_final)

        return int(max(1, min(n_primitives, round(top_k))))

    return int(max(1, min(n_primitives, top_k_final)))


def compute_base_lr(step: int, config: TrainingConfig) -> float:
    if config.total_steps <= 1:
        return config.lr
    if step < config.warmup_steps:
        return config.lr * (step / max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.min_lr + (config.lr - config.min_lr) * cosine


# ---------------------------
# Training helpers
# ---------------------------

def set_composition_requires_grad(model: PILONTransformer, requires_grad: bool) -> None:
    for layer in model.layers:
        ffn = layer.ffn
        if hasattr(ffn, "composition_weights"):
            ffn.composition_weights.requires_grad_(requires_grad)


def set_primitive_requires_grad(model: PILONTransformer, requires_grad: bool) -> None:
    if getattr(model, "primitive_banks", None) is not None:
        model.primitive_banks.requires_grad_(requires_grad)


def apply_runtime_overrides(model: PILONTransformer, active_rank: int, runtime_top_k: Optional[int]) -> None:
    def _resolve_runtime_k(ffn: nn.Module, requested_top_k: Optional[int]) -> Optional[int]:
        if not hasattr(ffn, "n_experts"):
            return requested_top_k
        n_experts = int(getattr(ffn, "n_experts"))
        default_k = getattr(ffn, "top_k_experts", None)
        if default_k is None:
            default_k = n_experts
        default_k = int(max(1, min(n_experts, int(default_k))))
        if requested_top_k is None:
            return default_k
        requested = int(requested_top_k)
        if requested > n_experts:
            return default_k
        return int(max(1, min(n_experts, requested)))

    for layer in model.layers:
        ffn = layer.ffn
        if hasattr(ffn, "active_rank"):
            ffn.active_rank = active_rank
        if hasattr(ffn, "runtime_top_k"):
            ffn.runtime_top_k = _resolve_runtime_k(ffn, runtime_top_k)


def apply_runtime_overrides_extended(
    model: PILONTransformer,
    active_rank: int,
    runtime_top_k: Optional[int],
    runtime_top_k_fc1: Optional[int],
    runtime_top_k_fc2: Optional[int],
    active_primitives: Optional[int],
    uniform_topk: bool
) -> None:
    def _resolve_runtime_k(ffn: nn.Module, requested_top_k: Optional[int]) -> Optional[int]:
        if not hasattr(ffn, "n_experts"):
            return requested_top_k
        n_experts = int(getattr(ffn, "n_experts"))
        default_k = getattr(ffn, "top_k_experts", None)
        if default_k is None:
            default_k = n_experts
        default_k = int(max(1, min(n_experts, int(default_k))))
        if requested_top_k is None:
            return default_k
        requested = int(requested_top_k)
        if requested > n_experts:
            return default_k
        return int(max(1, min(n_experts, requested)))

    for layer in model.layers:
        ffn = layer.ffn
        if hasattr(ffn, "active_rank"):
            ffn.active_rank = active_rank
        if hasattr(ffn, "runtime_top_k"):
            ffn.runtime_top_k = _resolve_runtime_k(ffn, runtime_top_k)
        if hasattr(ffn, "runtime_top_k_fc1"):
            ffn.runtime_top_k_fc1 = runtime_top_k_fc1
        if hasattr(ffn, "runtime_top_k_fc2"):
            ffn.runtime_top_k_fc2 = runtime_top_k_fc2
        if hasattr(ffn, "active_primitives"):
            ffn.active_primitives = active_primitives
        if hasattr(ffn, "runtime_uniform_topk"):
            ffn.runtime_uniform_topk = uniform_topk


def apply_runtime_step_and_cache(
    model: PILONTransformer,
    step: int,
    topk_cache_steps: Optional[int]
) -> None:
    for layer in model.layers:
        ffn = layer.ffn
        if hasattr(ffn, "runtime_step"):
            ffn.runtime_step = step
        if topk_cache_steps is not None and hasattr(ffn, "topk_cache_steps"):
            ffn.topk_cache_steps = topk_cache_steps


def apply_progressive_unfreeze(model: PILONTransformer, step: int, total_steps: int, phase1_frac: float, phase2_frac: float) -> None:
    phase1_end = int(total_steps * phase1_frac)
    phase2_end = int(total_steps * phase2_frac)

    n_layers = len(model.layers)
    if step < phase1_end:
        max_idx = max(0, int(round(n_layers * (1.0 / 3.0))) - 1)
    elif step < phase2_end:
        max_idx = max(0, int(round(n_layers * (2.0 / 3.0))) - 1)
    else:
        max_idx = n_layers - 1

    for i, layer in enumerate(model.layers):
        requires = i <= max_idx
        for param in layer.parameters():
            param.requires_grad = requires


def _gini(weights: torch.Tensor) -> torch.Tensor:
    sorted_weights, _ = torch.sort(weights)
    n = sorted_weights.numel()
    cumulative = torch.cumsum(sorted_weights, dim=0)
    denom = cumulative[-1] + 1e-8
    gini = (n + 1 - 2 * cumulative.sum() / denom) / n
    return gini


def _topk_entropy(weights: torch.Tensor, top_k: int) -> torch.Tensor:
    k = min(top_k, weights.numel())
    top_weights, _ = torch.topk(weights, k)
    top_weights = top_weights / (top_weights.sum() + 1e-8)
    return -(top_weights * torch.log(top_weights + 1e-8)).sum()


def compute_composition_stats(model: PILONTransformer, top_k: int) -> Optional[Dict[str, float]]:
    max_weights = []
    logit_stds = []
    ginis = []
    topk_entropies = []

    for layer in model.layers:
        if not isinstance(layer.ffn, CompositionalFFN):
            continue
        comp = layer.ffn.composition_weights
        temp = comp.temperature
        for logits in (comp.fc1_logits, comp.fc2_logits):
            weights = F.softmax(logits / temp, dim=0)
            max_weights.append(weights.max())
            logit_stds.append(logits.float().std())
            ginis.append(_gini(weights))
            topk_entropies.append(_topk_entropy(weights, top_k))

    if not max_weights:
        return None

    return {
        "comp_max_w": torch.stack(max_weights).mean().item(),
        "comp_logit_std": torch.stack(logit_stds).mean().item(),
        "comp_gini": torch.stack(ginis).mean().item(),
        "comp_topk_entropy": torch.stack(topk_entropies).mean().item(),
    }


def compute_composition_entropy_loss(model: PILONTransformer) -> Optional[torch.Tensor]:
    entropies = []
    for layer in model.layers:
        if not isinstance(layer.ffn, CompositionalFFN):
            continue
        comp = layer.ffn.composition_weights
        temp = comp.temperature
        for logits in (comp.fc1_logits, comp.fc2_logits):
            probs = F.softmax(logits / temp, dim=0)
            log_probs = F.log_softmax(logits / temp, dim=0)
            entropies.append(-(probs * log_probs).sum())
    if not entropies:
        return None
    return torch.stack(entropies).mean()


def build_optimizer(model: PILONTransformer, base_lr: float) -> AdamW:
    primitive_params = []
    composition_params = []
    base_params = []

    for name, param in model.named_parameters():
        if "primitive_banks" in name:
            primitive_params.append(param)
        elif "composition_weights" in name:
            composition_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if primitive_params:
        param_groups.append({"params": primitive_params, "lr": base_lr, "weight_decay": 0.1, "name": "primitives"})
    if composition_params:
        param_groups.append({"params": composition_params, "lr": base_lr, "weight_decay": 0.0, "name": "composition"})
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr, "weight_decay": 0.1, "name": "base"})

    return AdamW(param_groups, betas=(0.9, 0.95))


def apply_phase_lrs(optimizer: AdamW, base_lr: float, phase: int) -> None:
    if phase == 1:
        # Give composition a tiny LR even in Phase 1 to prevent state freeze
        multipliers = {"primitives": 2.0, "composition": 0.01, "base": 1.0}
    elif phase == 2:
        multipliers = {"primitives": 0.5, "composition": 1.0, "base": 0.75}
    else:
        multipliers = {"primitives": 0.25, "composition": 0.5, "base": 0.5}

    for group in optimizer.param_groups:
        name = group.get("name", "base")
        group["lr"] = base_lr * multipliers.get(name, 1.0)


def apply_phase_lrs_with_override(
    optimizer: AdamW,
    base_lr: float,
    phase: int,
    composition_lr_mult: Optional[float]
) -> None:
    apply_phase_lrs(optimizer, base_lr, phase)
    if composition_lr_mult is None or phase == 1:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "composition":
            group["lr"] = base_lr * composition_lr_mult


def kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def save_checkpoint(
    path: Path,
    model: PILONTransformer,
    optimizer: AdamW,
    step: int,
    config: TrainingConfig,
    scaler: Optional[GradScaler] = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,  # Save ModelConfig for evaluation
        "train_config": config,  # Save TrainingConfig for resumption/info
    }
    if scaler is not None:
        try:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            pass
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    # Atomic replace to avoid partial checkpoints on interruption
    import os
    os.replace(tmp_path, path)

def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap model from torch.compile or DDP wrapper."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    if hasattr(model, "module"):
        return model.module
    return model

# ---------------------------
# Training loop
# ---------------------------

def train_v2(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir, name="trainv2")

    # Get model config based on size and ffn_type
    model_config = get_model_config(args.model_size, args.ffn_type)
    if args.baseline:
        model_config.ffn_type = "standard"

    if model_config.ffn_type == "compositional":
        if args.compression_level is not None:
            compression = get_compression_config(args.compression_level)
            model_config.primitive_config.n_primitives = compression["n_primitives"]
            model_config.primitive_config.rank = compression["rank"]
            model_config.primitive_config.top_k = compression["top_k"]
            model_config.primitive_config.top_k_fc1 = compression["top_k"]
            model_config.primitive_config.top_k_fc2 = compression["top_k"]

        if args.n_primitives is not None:
            model_config.primitive_config.n_primitives = args.n_primitives
        if args.rank is not None:
            model_config.primitive_config.rank = args.rank
        if args.top_k is not None:
            model_config.primitive_config.top_k = args.top_k
            if args.top_k_fc1 is None:
                model_config.primitive_config.top_k_fc1 = args.top_k
            if args.top_k_fc2 is None:
                model_config.primitive_config.top_k_fc2 = args.top_k
        if args.top_k_fc1 is not None:
            model_config.primitive_config.top_k_fc1 = args.top_k_fc1
        if args.top_k_fc2 is not None:
            model_config.primitive_config.top_k_fc2 = args.top_k_fc2
        if args.composition_temp is not None:
            model_config.primitive_config.temperature = args.composition_temp

        pc = model_config.primitive_config
        pc.n_primitives = max(1, int(pc.n_primitives))
        pc.rank = max(1, int(pc.rank))
        pc.top_k = int(max(1, min(pc.n_primitives, int(pc.top_k))))
        if pc.top_k_fc1 is None:
            pc.top_k_fc1 = pc.top_k
        else:
            pc.top_k_fc1 = int(max(1, min(pc.n_primitives, int(pc.top_k_fc1))))
        if pc.top_k_fc2 is None:
            pc.top_k_fc2 = pc.top_k
        else:
            pc.top_k_fc2 = int(max(1, min(pc.n_primitives, int(pc.top_k_fc2))))

    # Override sequence length if specified
    if args.seq_len is not None:
        model_config.max_seq_len = args.seq_len

    # Get training config (use 360m-specific if applicable)
    seq_len = args.seq_len or model_config.max_seq_len
    if args.model_size == "360m":
        train_config = get_360m_training_config(
            total_tokens=args.total_tokens or 1_000_000_000,
            micro_batch_size=args.batch_size or 4,
            gradient_accumulation=args.grad_accum or 16,
            max_seq_len=seq_len,
        )
    elif args.model_size == "500m":
        train_config = get_500m_training_config(
            total_tokens=args.total_tokens or 1_000_000_000,
            micro_batch_size=args.batch_size or 2,
            gradient_accumulation=args.grad_accum or 32,
            max_seq_len=seq_len,
        )
    else:
        train_config = TrainingConfig()
        if args.seq_len is not None:
            train_config.max_seq_len = args.seq_len

    if args.steps is not None:
        train_config.total_steps = args.steps
    if args.batch_size is not None:
        train_config.micro_batch_size = args.batch_size
    if args.grad_accum is not None:
        train_config.gradient_accumulation = args.grad_accum
    if args.dataset is not None:
        train_config.dataset = args.dataset
    if args.save_every is not None:
        train_config.save_every = args.save_every

    # Store start time for wall-clock tracking
    training_start_time = time.time()

    # Allow both compositional and standard FFN for comparison experiments
    is_compositional = model_config.ffn_type == "compositional"
    if not is_compositional:
        logger.info("Training with standard FFN (dense baseline)")

    # Build model
    model = create_model(model_config)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    model.to(device)

    if device.startswith("cuda") and args.tf32:
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True

    # Optional dense teacher (scaffold)
    teacher = None
    if args.teacher_ckpt:
        teacher = create_baseline_model(model_config)
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        if "model_state_dict" in ckpt:
            teacher.load_state_dict(ckpt["model_state_dict"])
        else:
            teacher.load_state_dict(ckpt)
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        logger.info(f"Loaded teacher from {args.teacher_ckpt}")

    # Dataloaders
    tokenizer = get_tokenizer(getattr(args, 'tokenizer_path', None))
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    train_dataset = load_text_dataset(
        dataset_name=train_config.dataset,
        tokenizer=tokenizer,
        max_seq_len=train_config.max_seq_len,
        max_tokens=train_config.tokens,
        split="train",
        streaming=True
    )
    val_dataset = load_text_dataset(
        dataset_name=train_config.dataset,
        tokenizer=tokenizer,
        max_seq_len=train_config.max_seq_len,
        max_tokens=2_000_000,
        split="validation",
        streaming=True
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        persistent_workers=train_config.persistent_workers
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        persistent_workers=train_config.persistent_workers
    )

    if args.compile:
        try:
            model = torch.compile(model)
            logger.info("Enabled torch.compile for model")
        except Exception as exc:
            logger.warning(f"torch.compile failed: {exc}")

    # Optimizer (created AFTER compile to ensure we capture the right parameters)
    base_lr = compute_base_lr(0, train_config)
    optimizer = build_optimizer(model, base_lr)

    # AMP
    amp_dtype = resolve_autocast_dtype(train_config.precision, device)
    use_scaler = (amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    # Schedule params (only for compositional)
    is_moe = False
    router_n_experts = 1
    router_top_k_final = 1
    if is_compositional:
        moe_cfg = model_config.primitive_config.moe_config
        is_moe = moe_cfg is not None
        n_primitives = model_config.primitive_config.n_primitives
        top_k_final = model_config.primitive_config.top_k
        top_k_final_fc1 = model_config.primitive_config.top_k_fc1 or top_k_final
        top_k_final_fc2 = model_config.primitive_config.top_k_fc2 or top_k_final
        if is_moe:
            router_n_experts = max(1, int(moe_cfg.n_experts))
            router_top_k_final = int(max(1, min(router_n_experts, int(moe_cfg.top_k))))
        else:
            router_n_experts = n_primitives
            router_top_k_final = top_k_final
        rank_final = model_config.primitive_config.rank
        rank_start = args.rank_start or max(8, rank_final // 4)
        rank_mid = args.rank_mid or max(rank_start, int(round(rank_final * 0.75)))
    else:
        n_primitives = 1
        top_k_final = 1
        top_k_final_fc1 = 1
        top_k_final_fc2 = 1
        router_n_experts = 1
        router_top_k_final = 1
        rank_final = 1
        rank_start = 1
        rank_mid = 1

    phase1_frac = args.phase1_frac
    phase2_frac = args.phase2_frac

    train_iter = iter(train_loader)
    metrics = TrainingMetrics()
    start_step = 0

    # Resume from checkpoint if provided
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_scaler and ckpt.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                logger.warning("Could not restore GradScaler state; continuing without it.")
        resume_step = int(ckpt.get("step", -1))
        if resume_step >= 0:
            start_step = min(resume_step + 1, train_config.total_steps)
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = TrainingMetrics.load(metrics_path)
            except Exception:
                logger.warning("Could not load metrics.json; starting fresh metrics.")
        logger.info(f"Resumed from {resume_path} at step {resume_step}. Next step: {start_step}.")
        if start_step >= train_config.total_steps:
            logger.info("Checkpoint already at or beyond total_steps. Exiting.")
            return

    logger.info("Starting trainv2")

    log_timing = args.log_timing
    log_comp_stats = args.log_comp_stats
    topk_cache_steps = args.topk_cache_steps
    if topk_cache_steps is not None and topk_cache_steps <= 0:
        topk_cache_steps = None
    freeze_primitives_phase2 = train_config.freeze_primitives_in_phase2
    if args.freeze_primitives_phase2 is not None:
        freeze_primitives_phase2 = args.freeze_primitives_phase2
    comp_entropy_weight = args.comp_entropy_weight
    comp_lr_mult = args.comp_lr_mult

    step_time = time.time()
    tokens_since_log = 0
    grad_accum_steps = max(1, int(train_config.gradient_accumulation))
    moe_aux_loss_weight = 0.0
    if is_compositional and model_config.primitive_config.moe_config is not None:
        moe_aux_loss_weight = float(model_config.primitive_config.moe_config.aux_loss_weight)

    for step in range(start_step, train_config.total_steps):
        data_s = 0.0
        fwd_s = 0.0
        bwd_s = 0.0
        opt_s = 0.0
        step_s = 0.0
        if log_timing:
            sync_if_cuda(device)
            step_start = time.perf_counter()
        # Schedules
        phase = get_phase(step, int(train_config.total_steps * phase1_frac), int(train_config.total_steps * phase2_frac))
        scaffold_alpha = get_scaffold_alpha(step, train_config.total_steps, phase1_frac, phase2_frac)
        active_rank = get_active_rank(step, train_config.total_steps, rank_start, rank_mid, rank_final, phase1_frac, phase2_frac)
        runtime_top_k = get_runtime_top_k(
            step,
            train_config.total_steps,
            router_n_experts,
            router_top_k_final,
            args.top_k_mid,
            phase1_frac,
            phase2_frac,
            phase1_sparse=args.phase1_sparse,
            phase1_top_k=args.phase1_top_k
        )
        if is_moe and runtime_top_k is None:
            # In MoE, phase-1 "full mix" means using all experts.
            runtime_top_k = router_n_experts
        runtime_top_k_fc1 = get_runtime_top_k(
            step,
            train_config.total_steps,
            n_primitives,
            top_k_final_fc1,
            args.top_k_mid,
            phase1_frac,
            phase2_frac,
            phase1_sparse=args.phase1_sparse,
            phase1_top_k=args.phase1_top_k
        )
        runtime_top_k_fc2 = get_runtime_top_k(
            step,
            train_config.total_steps,
            n_primitives,
            top_k_final_fc2,
            args.top_k_mid,
            phase1_frac,
            phase2_frac,
            phase1_sparse=args.phase1_sparse,
            phase1_top_k=args.phase1_top_k
        )

        active_primitives = None
        if args.prims_start is not None or args.prims_mid is not None or args.prims_final is not None:
            prim_final = args.prims_final or n_primitives
            prim_start = args.prims_start
            if prim_start is None:
                prim_start = max(4, min(prim_final, max(top_k_final_fc1, top_k_final_fc2)))
            prim_mid = args.prims_mid or max(prim_start, int(round(prim_final * 0.5)))
            prim_final = max(1, min(n_primitives, prim_final))
            prim_start = max(1, min(prim_final, prim_start))
            prim_mid = max(prim_start, min(prim_final, prim_mid))
            active_primitives = get_active_primitives(
                step,
                train_config.total_steps,
                prim_start,
                prim_mid,
                prim_final,
                phase1_frac,
                phase2_frac
            )

        # Unwrap model for attribute/param manipulation
        raw_model = unwrap_model(model)

        # Compositional-specific runtime overrides
        if is_compositional:
            apply_runtime_overrides_extended(
                raw_model,
                active_rank,
                runtime_top_k,
                runtime_top_k_fc1,
                runtime_top_k_fc2,
                active_primitives,
                uniform_topk=(phase == 1 and args.uniform_topk_phase1)
            )
            apply_runtime_step_and_cache(raw_model, step, topk_cache_steps)
            raw_model.update_caches()

            # Phase-specific parameter training
            if phase == 1:
                set_composition_requires_grad(raw_model, False)
                set_primitive_requires_grad(raw_model, True)
            else:
                set_composition_requires_grad(raw_model, True)
                set_primitive_requires_grad(raw_model, not freeze_primitives_phase2)

            # Diagnostic: check if grads are enabled and flowing
            if step % 100 == 0:
                comp_params = [(n, p) for n, p in raw_model.named_parameters() if "composition_weights" in n]
                if comp_params:
                    any_req = any(p.requires_grad for n, p in comp_params)
                    # Take first one as representative
                    name, p = comp_params[0]
                    p_max = p.max().item()
                    p_std = p.std().item()
                    gnorm = p.grad.norm().item() if p.grad is not None else 0.0
                    logger.info(f"Step {step} (Phase {phase}): {name} req_grad={any_req} max={p_max:.4f} std={p_std:.4f} gnorm={gnorm:.4e}")

            # Optional progressive unfreeze (applies after phase-specific flags)
            if args.progressive_unfreeze:
                apply_progressive_unfreeze(raw_model, step, train_config.total_steps, phase1_frac, phase2_frac)

        # LR schedule
        base_lr = compute_base_lr(step, train_config)
        if is_compositional:
            apply_phase_lrs_with_override(optimizer, base_lr, phase, comp_lr_mult)
        else:
            # Simple LR for dense baseline
            for group in optimizer.param_groups:
                group["lr"] = base_lr

        tokens_in_step = 0
        lm_loss_accum = 0.0
        distill_loss_accum = 0.0
        distill_loss_count = 0
        comp_entropy_loss_accum = 0.0
        comp_entropy_loss_count = 0
        aux_loss_raw_accum = 0.0
        aux_loss_scaled_accum = 0.0
        aux_loss_count = 0
        grad_norm_value = 0.0

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            # Get micro-batch
            if log_timing:
                data_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if log_timing:
                sync_if_cuda(device)
                data_s += time.perf_counter() - data_start

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            tokens_in_step += input_ids.numel()

            # Forward
            if log_timing:
                sync_if_cuda(device)
                fwd_start = time.perf_counter()
            with get_autocast_context(amp_dtype, device):
                outputs = model(input_ids, labels=labels)
                lm_loss = outputs["loss"]
                total_loss = lm_loss

                aux_loss_tensor = None
                aux_candidate = outputs.get("aux_loss")
                if aux_candidate is not None:
                    if not torch.is_tensor(aux_candidate):
                        aux_candidate = lm_loss.new_tensor(float(aux_candidate))
                    aux_loss_tensor = aux_candidate.to(dtype=lm_loss.dtype)
                    if moe_aux_loss_weight > 0.0:
                        total_loss = total_loss + moe_aux_loss_weight * aux_loss_tensor

                distill_loss = None
                if teacher is not None and scaffold_alpha > 0.0:
                    with torch.no_grad():
                        teacher_out = teacher(input_ids)
                    student_logits = outputs["logits"][:, :-1, :].contiguous()
                    teacher_logits = teacher_out["logits"][:, :-1, :].contiguous()
                    distill_loss = kl_divergence(student_logits, teacher_logits, temperature=args.distill_temp)
                    total_loss = total_loss + scaffold_alpha * distill_loss

                comp_entropy_loss = None
                if is_compositional and comp_entropy_weight > 0 and phase >= 2:
                    comp_entropy_loss = compute_composition_entropy_loss(raw_model)
                    if comp_entropy_loss is not None:
                        total_loss = total_loss + comp_entropy_weight * comp_entropy_loss

            if log_timing:
                sync_if_cuda(device)
                fwd_s += time.perf_counter() - fwd_start

            lm_loss_accum += float(lm_loss.detach().item())
            if distill_loss is not None:
                distill_loss_accum += float(distill_loss.detach().item())
                distill_loss_count += 1
            if comp_entropy_loss is not None:
                comp_entropy_loss_accum += float(comp_entropy_loss.detach().item())
                comp_entropy_loss_count += 1
            if aux_loss_tensor is not None:
                aux_raw = float(aux_loss_tensor.detach().item())
                aux_loss_raw_accum += aux_raw
                aux_loss_scaled_accum += moe_aux_loss_weight * aux_raw
                aux_loss_count += 1

            # Backward
            total_loss = total_loss / grad_accum_steps
            if log_timing:
                sync_if_cuda(device)
                bwd_start = time.perf_counter()
            if use_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if log_timing:
                sync_if_cuda(device)
                bwd_s += time.perf_counter() - bwd_start

        if log_timing:
            sync_if_cuda(device)
            opt_start = time.perf_counter()
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if log_timing:
            sync_if_cuda(device)
            opt_s = time.perf_counter() - opt_start

        if log_timing:
            sync_if_cuda(device)
            step_s = time.perf_counter() - step_start

        lm_loss_value = lm_loss_accum / max(1, grad_accum_steps)
        distill_loss_value = (
            distill_loss_accum / distill_loss_count if distill_loss_count > 0 else None
        )
        comp_entropy_loss_value = (
            comp_entropy_loss_accum / comp_entropy_loss_count if comp_entropy_loss_count > 0 else None
        )
        aux_loss_raw_value = (
            aux_loss_raw_accum / aux_loss_count if aux_loss_count > 0 else None
        )
        aux_loss_scaled_value = (
            aux_loss_scaled_accum / aux_loss_count if aux_loss_count > 0 else None
        )
        tokens_since_log += tokens_in_step

        # Logging
        if step % train_config.log_every == 0:
            now = time.time()
            step_dt = max(1e-6, now - step_time)
            step_time = now
            tok_s = tokens_since_log / step_dt
            tokens_since_log = 0
            tok_s_effective = tok_s

            # Calculate total tokens processed
            tokens_per_step = (
                train_config.micro_batch_size * train_config.max_seq_len * grad_accum_steps
            )
            tokens_processed = (step + 1) * tokens_per_step

            # Wall clock time
            wall_clock_hours = (time.time() - training_start_time) / 3600

            log_dict = {
                "train_loss": lm_loss_value,
                "tokens_sec": tok_s,
                "tokens_sec_effective": tok_s_effective,
                "wall_clock_hours": wall_clock_hours,
                "tokens_processed": tokens_processed,
                "lr": base_lr,
            }

            # Compositional-specific metrics
            if is_compositional:
                entropy_vals = model.get_all_entropy()
                mean_entropy = sum(entropy_vals.values()) / max(1, len(entropy_vals)) if entropy_vals else 0.0

                log_dict.update({
                    "scaffold_alpha": scaffold_alpha,
                    "active_rank": float(active_rank),
                    "top_k": float(runtime_top_k if runtime_top_k is not None else router_n_experts),
                    "top_k_fc1": float(runtime_top_k_fc1 or n_primitives),
                    "top_k_fc2": float(runtime_top_k_fc2 or n_primitives),
                    "active_primitives": float(active_primitives or n_primitives),
                    "uniform_topk": float(1.0 if (phase == 1 and args.uniform_topk_phase1) else 0.0),
                    "primitive_entropy": mean_entropy,
                })

                if log_comp_stats:
                    k1 = runtime_top_k_fc1 if runtime_top_k_fc1 is not None else n_primitives
                    k2 = runtime_top_k_fc2 if runtime_top_k_fc2 is not None else n_primitives
                    comp_k = min(k1, k2)
                    comp_stats = compute_composition_stats(raw_model, comp_k)
                    if comp_stats:
                        # Rename for clarity in comparison
                        log_dict["composition_entropy"] = comp_stats.get("comp_topk_entropy", 0.0)
                        log_dict["composition_gini"] = comp_stats.get("comp_gini", 0.0)
                        log_dict["topk_utilization"] = comp_stats.get("comp_max_w", 0.0)
                        log_dict.update(comp_stats)

                if comp_entropy_loss_value is not None:
                    log_dict["comp_entropy_loss"] = comp_entropy_loss_value
                    log_dict["comp_entropy_weight"] = comp_entropy_weight

            if log_timing:
                log_dict.update({
                    "data_s": data_s,
                    "fwd_s": fwd_s,
                    "bwd_s": bwd_s,
                    "opt_s": opt_s,
                    "step_s": step_s,
                })
                if step_s > 0:
                    log_dict["tok_s_step"] = tokens_in_step / step_s
                    log_dict["tok_s_step_eff"] = tokens_in_step / step_s

            if distill_loss_value is not None:
                log_dict["distill"] = distill_loss_value
            if aux_loss_raw_value is not None:
                log_dict["aux_loss"] = aux_loss_raw_value
                log_dict["aux_loss_scaled"] = aux_loss_scaled_value
                log_dict["aux_loss_weight"] = moe_aux_loss_weight

            # Enhanced VRAM tracking
            if device.startswith("cuda"):
                log_dict["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                log_dict["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                log_dict["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                # Reset peak stats for next interval
                torch.cuda.reset_peak_memory_stats()

            log_dict["grad_norm"] = grad_norm_value

            logger.metric(step, log_dict)
            metrics.add_train_loss(step, lm_loss_value)

        # Eval
        if step % train_config.eval_every == 0:
            val_loss = 0.0
            n_batches = 0
            model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    if n_batches >= 50:
                        break
                    val_input = batch["input_ids"].to(device, non_blocking=True)
                    val_labels = batch["labels"].to(device, non_blocking=True)
                    with get_autocast_context(amp_dtype, device):
                        out = model(val_input, labels=val_labels)
                        val_loss += out["loss"].item()
                    n_batches += 1
            model.train()
            avg_val = val_loss / max(1, n_batches)
            val_ppl = math.exp(min(avg_val, 20.0))  # Clamp to avoid overflow
            metrics.add_val_loss(step, avg_val)

            # Log validation metrics
            val_log = {
                "val_loss": avg_val,
                "val_ppl": val_ppl,
                "wall_clock_hours": (time.time() - training_start_time) / 3600,
            }
            logger.metric(step, val_log)
            logger.info(f"step {step} val_loss={avg_val:.4f} val_ppl={val_ppl:.2f}")

        # Checkpoint
        if train_config.save_every > 0 and step % train_config.save_every == 0 and step > 0:
            ckpt_path = output_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, train_config, scaler)
            metrics.save(output_dir / "metrics.json")
            logger.info(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / "final_model.pt"
    save_checkpoint(final_path, model, optimizer, train_config.total_steps, train_config, scaler)
    metrics.save(output_dir / "final_metrics.json")
    logger.info(f"Training complete. Final checkpoint: {final_path}")


# ---------------------------
# CLI
# ---------------------------

def get_model_config(size: str, ffn_type: str) -> ModelConfig:
    """
    Get model configuration based on size and FFN type.

    Args:
        size: Model size ("48m" or "360m")
        ffn_type: FFN type ("compositional" or "standard")

    Returns:
        Configured ModelConfig
    """
    if size == "360m":
        return get_360m_config(ffn_type)
    elif size == "500m":
        return get_500m_config(ffn_type)
    elif size == "48m":
        # Default small model config
        config = ModelConfig()
        if ffn_type == "standard":
            config.ffn_type = "standard"
        return config
    else:
        raise ValueError(f"Unknown model size: {size}. Use '48m', '360m', or '500m'")


def main() -> None:
    parser = argparse.ArgumentParser(description="PILON-R v2 training")
    parser.add_argument("--output-dir", type=str, default="outputs/phase_a/v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro batch size override")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation override")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint save frequency (steps). 0 = disable")

    # New 360M experiment arguments
    parser.add_argument("--model-size", type=str, choices=["48m", "360m", "500m"], default="48m",
                        help="Model size (48m, 360m, or 500m)")
    parser.add_argument("--ffn-type", type=str, choices=["compositional", "standard"], default="compositional",
                        help="FFN type (compositional for PILON, standard for dense)")
    parser.add_argument("--baseline", action="store_true", help="Alias for --ffn-type standard")
    parser.add_argument("--compression-level", type=str, choices=get_all_compression_levels(), default=None,
                        help="Apply preset n_primitives/rank/top_k from compression config")
    parser.add_argument("--n-primitives", type=int, default=None, help="Override number of primitives")
    parser.add_argument("--rank", type=int, default=None, help="Override primitive rank")
    parser.add_argument("--top-k", type=int, default=None, help="Override shared top-k for compositional FFN")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to custom tokenizer (default: GPT-2)")
    parser.add_argument("--total-tokens", type=int, default=None,
                        help="Total tokens to train on (overrides --steps)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (default: 2048 for 360m, 512 for 48m)")

    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Dense teacher checkpoint path")
    parser.add_argument("--distill-temp", type=float, default=1.0, help="Distillation temperature")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path")

    parser.add_argument("--rank-start", type=int, default=None)
    parser.add_argument("--rank-mid", type=int, default=None)
    parser.add_argument("--top-k-mid", type=int, default=None)
    parser.add_argument("--top-k-fc1", type=int, default=None, help="Override top-k for fc1 (compositional)")
    parser.add_argument("--top-k-fc2", type=int, default=None, help="Override top-k for fc2 (compositional)")
    parser.add_argument("--phase1-frac", type=float, default=0.2)
    parser.add_argument("--phase2-frac", type=float, default=0.5)
    parser.add_argument("--phase1-sparse", action="store_true", help="Use top-k during phase1 (no full mix)")
    parser.add_argument("--phase1-top-k", type=int, default=None, help="Top-k to use during phase1 when sparse")
    parser.add_argument("--prims-start", type=int, default=None, help="Active primitives at start (P-schedule)")
    parser.add_argument("--prims-mid", type=int, default=None, help="Active primitives at mid (P-schedule)")
    parser.add_argument("--prims-final", type=int, default=None, help="Active primitives at end (P-schedule)")
    parser.add_argument("--uniform-topk-phase1", action="store_true", help="Use uniform weights for top-k in phase1")
    parser.add_argument("--topk-cache-steps", type=int, default=None, help="Cache top-k indices for N steps")
    parser.add_argument("--composition-temp", type=float, default=None, help="Override composition softmax temperature")
    parser.add_argument("--log-comp-stats", action="store_true", help="Log composition weight statistics")
    parser.add_argument("--freeze-primitives-phase2", dest="freeze_primitives_phase2", action="store_true",
                        help="Freeze primitives after phase1")
    parser.add_argument("--no-freeze-primitives-phase2", dest="freeze_primitives_phase2", action="store_false",
                        help="Keep primitives trainable in phase2")
    parser.set_defaults(freeze_primitives_phase2=None)
    parser.add_argument("--comp-lr-mult", type=float, default=None,
                        help="Override composition LR multiplier in phase2+")
    parser.add_argument("--comp-entropy-weight", type=float, default=0.0,
                        help="Entropy penalty weight for composition weights (phase2+)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul")

    parser.add_argument("--progressive-unfreeze", action="store_true")
    parser.add_argument("--log-timing", action="store_true", help="Log timing breakdowns per step")

    args = parser.parse_args()
    args.tf32 = not args.no_tf32
    set_seed(42)
    train_v2(args)


if __name__ == "__main__":
    main()
