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
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast

from .core.config import (
    ModelConfig,
    MoEConfig,
    TrainingConfig,
    get_compression_config,
    get_all_compression_levels,
)
from .core.model import PILONTransformer, create_model, create_baseline_model
from .core.ffn import CompositionalFFN, MoECompositionalFFN
from .core.data import load_text_dataset, get_tokenizer, create_dataloader
from .core.metrics import Logger, TrainingMetrics
from .core.early_exit import train_exit_gates
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


def compute_band_diversity_loss(model: PILONTransformer) -> Optional[torch.Tensor]:
    """
    Encourage layers in the same band to use different primitive mixes.

    Returns mean cosine similarity across layer pairs (lower is better).
    """
    if getattr(model, "primitive_banks", None) is None:
        return None
    band_to_layers: Dict[str, List[int]] = {}
    for band in model.config.primitive_config.bands:
        band_to_layers[band.name] = list(band.layers)

    pair_sims: List[torch.Tensor] = []
    for layer_indices in band_to_layers.values():
        if len(layer_indices) < 2:
            continue
        fc1_weights = []
        fc2_weights = []
        for idx in layer_indices:
            if idx < 0 or idx >= len(model.layers):
                continue
            ffn = model.layers[idx].ffn
            if isinstance(ffn, CompositionalFFN):
                fc1_weights.append(ffn.composition_weights.get_fc1_weights())
                fc2_weights.append(ffn.composition_weights.get_fc2_weights())
            elif isinstance(ffn, MoECompositionalFFN):
                fc1_expert_weights, fc2_expert_weights = ffn.expert_compositions.get_all_expert_weights()
                fc1_weights.append(fc1_expert_weights.mean(dim=0))
                fc2_weights.append(fc2_expert_weights.mean(dim=0))
        if len(fc1_weights) < 2:
            continue

        def _pairwise_cos(ws: List[torch.Tensor]) -> List[torch.Tensor]:
            sims: List[torch.Tensor] = []
            normed = [F.normalize(w, p=2, dim=0) for w in ws]
            for i in range(len(normed)):
                for j in range(i + 1, len(normed)):
                    sims.append((normed[i] * normed[j]).sum())
            return sims

        pair_sims.extend(_pairwise_cos(fc1_weights))
        pair_sims.extend(_pairwise_cos(fc2_weights))

    if not pair_sims:
        return None
    return torch.stack(pair_sims).mean()


def compute_hot_tier_bias_loss(model: PILONTransformer) -> Optional[torch.Tensor]:
    """
    Encourage composition mass to sit on hot primitives in tiered banks.
    """
    losses: List[torch.Tensor] = []
    if getattr(model, "primitive_banks", None) is None:
        return None

    for layer_idx, layer in enumerate(model.layers):
        ffn = layer.ffn
        if not isinstance(ffn, (CompositionalFFN, MoECompositionalFFN)):
            continue
        fc1_bank = model.primitive_banks.get_fc1_bank(layer_idx)
        fc2_bank = model.primitive_banks.get_fc2_bank(layer_idx)
        if not (hasattr(fc1_bank, "hot_indices") and hasattr(fc2_bank, "hot_indices")):
            continue
        fc1_hot = fc1_bank.hot_indices
        fc2_hot = fc2_bank.hot_indices
        if fc1_hot is None or fc2_hot is None:
            continue
        if isinstance(ffn, CompositionalFFN):
            fc1_w = ffn.composition_weights.get_fc1_weights()
            fc2_w = ffn.composition_weights.get_fc2_weights()
        else:
            fc1_expert_weights, fc2_expert_weights = ffn.expert_compositions.get_all_expert_weights()
            fc1_w = fc1_expert_weights.mean(dim=0)
            fc2_w = fc2_expert_weights.mean(dim=0)
        hot_mass_fc1 = fc1_w.index_select(0, fc1_hot).sum()
        hot_mass_fc2 = fc2_w.index_select(0, fc2_hot).sum()
        losses.append(1.0 - hot_mass_fc1)
        losses.append(1.0 - hot_mass_fc2)

    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_joint_exit_loss(
    exit_confidences: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Self-supervised gate target from token certainty.

    exit_confidences: (L, B, S, 1)
    logits: (B, S, V)
    """
    # Confidence target from normalized entropy: low entropy => easier token => higher skip confidence.
    probs = F.softmax(logits.detach(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)
    norm = math.log(max(2, logits.size(-1)))
    certainty = torch.clamp(1.0 - (entropy / max(norm, 1e-8)), 0.0, 1.0)

    n_layers = exit_confidences.size(0)
    layer_scale = torch.linspace(
        0.5,
        1.0,
        steps=n_layers,
        device=exit_confidences.device,
        dtype=exit_confidences.dtype,
    ).view(n_layers, 1, 1, 1)
    target = (certainty.unsqueeze(0) * layer_scale).clamp(0.0, 1.0)
    # Disable autocast — BCE is not autocast-safe
    with torch.cuda.amp.autocast(enabled=False):
        return F.binary_cross_entropy(exit_confidences.float(), target.float())


class SparseAwareAdamW(Optimizer):
    """
    AdamW variant with row-sparse updates for primitive banks.

    For parameter groups with `row_sparse=True`, updates are applied only to rows
    with non-zero gradients and maintain per-row optimizer step counters.
    """

    def __init__(
        self,
        params,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=3e-4,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            row_sparse=False,
            grad_eps=0.0,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _as_int_step(step_value: Any) -> int:
        if torch.is_tensor(step_value):
            return int(step_value.item())
        return int(step_value)

    @torch.no_grad()
    def _dense_update(
        self,
        param: nn.Parameter,
        grad: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        elif "step" not in state:
            state["step"] = 0

        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step = self._as_int_step(state["step"]) + 1
        state["step"] = step

        grad = grad.to(dtype=exp_avg.dtype)

        if weight_decay != 0.0:
            param.data.mul_(1.0 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        step_size = lr / bias_correction1
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        param.data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def _row_sparse_update(
        self,
        param: nn.Parameter,
        grad: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        if param.ndim < 2 or param.size(0) == 0:
            self._dense_update(param, grad, group, state)
            return

        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
            state["step_row"] = torch.zeros(
                param.size(0), device=param.device, dtype=torch.long
            )
        elif "step_row" not in state:
            base_step = self._as_int_step(state.get("step", 0))
            state["step_row"] = torch.full(
                (param.size(0),), base_step, device=param.device, dtype=torch.long
            )

        grad_eps = float(group.get("grad_eps", 0.0))
        row_active = grad.detach().abs().reshape(param.size(0), -1).amax(dim=1) > grad_eps
        if not bool(row_active.any()):
            return

        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])

        active_rows = torch.nonzero(row_active, as_tuple=False).squeeze(-1)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_row = state["step_row"]

        grad_sel = grad.index_select(0, active_rows).to(dtype=exp_avg.dtype)
        exp_avg_sel = exp_avg.index_select(0, active_rows)
        exp_avg_sq_sel = exp_avg_sq.index_select(0, active_rows)

        step_sel = step_row.index_select(0, active_rows) + 1
        step_row.index_copy_(0, active_rows, step_sel)

        exp_avg_sel.mul_(beta1).add_(grad_sel, alpha=1.0 - beta1)
        exp_avg_sq_sel.mul_(beta2).addcmul_(grad_sel, grad_sel, value=1.0 - beta2)

        exp_avg.index_copy_(0, active_rows, exp_avg_sel)
        exp_avg_sq.index_copy_(0, active_rows, exp_avg_sq_sel)

        param_sel = param.data.index_select(0, active_rows)
        if weight_decay != 0.0:
            param_sel.mul_(1.0 - lr * weight_decay)

        bias_correction1 = 1.0 - beta1 ** step_sel.to(dtype=exp_avg_sel.dtype)
        bias_correction2 = 1.0 - beta2 ** step_sel.to(dtype=exp_avg_sel.dtype)
        while bias_correction1.ndim < exp_avg_sel.ndim:
            bias_correction1 = bias_correction1.unsqueeze(-1)
            bias_correction2 = bias_correction2.unsqueeze(-1)

        denom = exp_avg_sq_sel.sqrt().div_(torch.sqrt(bias_correction2)).add_(eps)
        update = (exp_avg_sel / denom) * (lr / bias_correction1)
        param_sel.add_(-update)
        param.data.index_copy_(0, active_rows, param_sel)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            row_sparse = bool(group.get("row_sparse", False))
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    grad = grad.coalesce().to_dense()

                state = self.state[param]
                if row_sparse and param.ndim >= 2:
                    self._row_sparse_update(param, grad, group, state)
                else:
                    self._dense_update(param, grad, group, state)

        return loss


def build_optimizer(
    model: PILONTransformer,
    base_lr: float,
    primitive_weight_decay: float = 0.0,
) -> Optimizer:
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
        param_groups.append({
            "params": primitive_params,
            "lr": base_lr,
            "weight_decay": primitive_weight_decay,
            "name": "primitives",
            "row_sparse": True,
            "grad_eps": 0.0,
        })
    if composition_params:
        param_groups.append({
            "params": composition_params,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "composition",
            "row_sparse": False,
        })
    if base_params:
        param_groups.append({
            "params": base_params,
            "lr": base_lr,
            "weight_decay": 0.1,
            "name": "base",
            "row_sparse": False,
        })

    return SparseAwareAdamW(param_groups, betas=(0.9, 0.95))


def apply_phase_lrs(optimizer: Optimizer, base_lr: float, phase: int,
                    freeze_primitives_phase2: bool = False) -> None:
    if phase == 1:
        # Primitives train, compositions frozen via LR=0
        multipliers = {"primitives": 2.0, "composition": 0.0, "base": 1.0}
    elif phase == 2:
        # Compositions train; primitives frozen via LR=0 if requested
        prim_mult = 0.0 if freeze_primitives_phase2 else 0.5
        multipliers = {"primitives": prim_mult, "composition": 1.0, "base": 0.75}
    else:
        prim_mult = 0.0 if freeze_primitives_phase2 else 0.25
        multipliers = {"primitives": prim_mult, "composition": 0.5, "base": 0.5}

    for group in optimizer.param_groups:
        name = group.get("name", "base")
        group["lr"] = base_lr * multipliers.get(name, 1.0)


def apply_phase_lrs_with_override(
    optimizer: Optimizer,
    base_lr: float,
    phase: int,
    composition_lr_mult: Optional[float],
    freeze_primitives_phase2: bool = False,
) -> None:
    apply_phase_lrs(optimizer, base_lr, phase,
                    freeze_primitives_phase2=freeze_primitives_phase2)
    if composition_lr_mult is None or phase == 1:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "composition":
            group["lr"] = base_lr * composition_lr_mult


def compute_sparse_lr_multiplier(
    is_moe: bool,
    n_primitives: int,
    top_k_final: int,
    runtime_top_k: Optional[int],
    runtime_top_k_fc1: Optional[int],
    runtime_top_k_fc2: Optional[int],
    router_n_experts: int,
    max_mult: float,
) -> float:
    if n_primitives <= 1:
        return 1.0

    if is_moe:
        expert_k = runtime_top_k if runtime_top_k is not None else 1
        expert_k = max(1, min(router_n_experts, int(expert_k)))
        expert_frac = expert_k / float(max(1, router_n_experts))
        primitive_k = max(1, min(n_primitives, int(top_k_final)))
        primitive_frac = primitive_k / float(n_primitives)
        activity = expert_frac * primitive_frac
    else:
        k1 = runtime_top_k_fc1 if runtime_top_k_fc1 is not None else top_k_final
        k2 = runtime_top_k_fc2 if runtime_top_k_fc2 is not None else top_k_final
        k1 = max(1, min(n_primitives, int(k1)))
        k2 = max(1, min(n_primitives, int(k2)))
        activity = 0.5 * ((k1 / float(n_primitives)) + (k2 / float(n_primitives)))

    mult = 1.0 / max(1e-6, activity)
    mult = max(1.0, mult)
    return min(max(1.0, float(max_mult)), mult)


def apply_sparse_lr_compensation(optimizer: Optimizer, multiplier: float) -> None:
    if multiplier <= 1.0:
        return

    comp_mult = math.sqrt(multiplier)
    for group in optimizer.param_groups:
        name = group.get("name")
        if name == "primitives":
            group["lr"] = float(group["lr"]) * multiplier
        elif name == "composition":
            group["lr"] = float(group["lr"]) * comp_mult


def kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def save_checkpoint(
    path: Path,
    model: PILONTransformer,
    optimizer: Optimizer,
    step: int,
    config: TrainingConfig,
    scaler: Optional[GradScaler] = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": unwrap_model(model).config,  # Save ModelConfig for evaluation
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


def ensure_msvc_env() -> bool:
    """
    Ensure MSVC INCLUDE/LIB environment variables are set for torch.compile
    C++ code generation (shape guards, cpp wrappers).

    On Windows, cl.exe may be on PATH but without the include/lib paths that
    vcvars64.bat would normally set. This function auto-detects and sets them.

    Returns True if environment was configured, False if MSVC was not found.
    """
    # Skip if INCLUDE is already set (user ran vcvars64 or this was already called)
    if os.environ.get("INCLUDE"):
        return True

    import shutil
    cl_path = shutil.which("cl")
    if cl_path is None:
        return False

    # Discover MSVC version from cl.exe path
    # Expected: .../MSVC/<version>/bin/Hostx64/x64/cl.exe
    cl_path = cl_path.replace("\\", "/")
    msvc_base = None
    parts = cl_path.split("/")
    for i, part in enumerate(parts):
        if part == "MSVC" and i + 1 < len(parts):
            msvc_base = "/".join(parts[:i + 2])
            break

    if msvc_base is None:
        return False

    # Find Windows SDK
    sdk_base = "C:/Program Files (x86)/Windows Kits/10"
    sdk_ver = None
    sdk_include = os.path.join(sdk_base, "Include")
    if os.path.isdir(sdk_include):
        versions = sorted(os.listdir(sdk_include), reverse=True)
        for v in versions:
            if os.path.isdir(os.path.join(sdk_include, v, "ucrt")):
                sdk_ver = v
                break

    if sdk_ver is None:
        return False

    # Set INCLUDE
    include_paths = [
        f"{msvc_base}/include",
        f"{sdk_base}/Include/{sdk_ver}/ucrt",
        f"{sdk_base}/Include/{sdk_ver}/shared",
        f"{sdk_base}/Include/{sdk_ver}/um",
    ]
    os.environ["INCLUDE"] = ";".join(include_paths)

    # Set LIB
    lib_paths = [
        f"{msvc_base}/lib/x64",
        f"{sdk_base}/Lib/{sdk_ver}/ucrt/x64",
        f"{sdk_base}/Lib/{sdk_ver}/um/x64",
    ]
    os.environ["LIB"] = ";".join(lib_paths)

    return True


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

    if args.checkpoint_ffn is not None:
        model_config.checkpoint_ffn = args.checkpoint_ffn

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
        if args.forward_fast_mode is not None:
            model_config.primitive_config.forward_fast_mode = args.forward_fast_mode
        if args.forward_fast_min_topk is not None:
            model_config.primitive_config.forward_fast_min_topk = args.forward_fast_min_topk

        if args.moe_experts is not None:
            n_experts = max(1, int(args.moe_experts))
            top_k_experts = args.moe_top_k
            if top_k_experts is None:
                top_k_experts = min(2, n_experts)
            top_k_experts = int(max(1, min(n_experts, int(top_k_experts))))
            model_config.primitive_config.moe_config = MoEConfig(
                n_experts=n_experts,
                top_k=top_k_experts,
                router_type=args.moe_router_type,
                router_hidden_dim=args.moe_router_hidden_dim,
                aux_loss_weight=max(0.0, float(args.moe_aux_loss_weight)),
                load_balancing=not args.no_moe_load_balancing,
            )

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
        if pc.forward_fast_min_topk is not None:
            pc.forward_fast_min_topk = int(max(1, min(pc.n_primitives, int(pc.forward_fast_min_topk))))

    # Phase B.5b: Tiered primitive bank
    if model_config.ffn_type == "compositional":
        if args.n_hot is not None:
            model_config.primitive_config.n_hot = args.n_hot
        if args.swap_interval is not None:
            model_config.primitive_config.swap_interval = args.swap_interval

    # Phase B.5c: Early exit
    if args.enable_early_exit:
        model_config.enable_early_exit = True
    if args.exit_threshold is not None:
        model_config.exit_threshold = args.exit_threshold

    # Ternary quantization (BitNet b1.58)
    if model_config.ffn_type == "compositional":
        if getattr(args, 'ternary', False):
            model_config.primitive_config.ternary_primitives = True
        if getattr(args, 'use_subln', False):
            model_config.primitive_config.use_subln = True
        if getattr(args, 'use_squared_relu', False):
            model_config.primitive_config.use_squared_relu = True
        if getattr(args, 'activation_bits', None) is not None:
            model_config.primitive_config.activation_bits = args.activation_bits

    # Phase C: Attention variant
    if hasattr(args, 'attention_type') and args.attention_type != "standard_mha":
        # Normalize aliases to canonical names
        attn_type = args.attention_type
        if attn_type == "compositional_gated_recurrence":
            attn_type = "compositional_recurrence"
        elif attn_type == "hybrid_recurrent_mha":
            attn_type = "hybrid"
        model_config.attention_type = attn_type
    if hasattr(args, 'n_attn_primitives') and args.n_attn_primitives is not None:
        model_config.n_attn_primitives = args.n_attn_primitives
    if hasattr(args, 'attn_rank') and args.attn_rank is not None:
        model_config.attn_rank = args.attn_rank
    if hasattr(args, 'attn_top_k') and args.attn_top_k is not None:
        model_config.attn_top_k = args.attn_top_k

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
        # Handle --total-tokens for 48M (not handled by a dedicated config function)
        if args.total_tokens is not None:
            bs = args.batch_size or train_config.micro_batch_size
            ga = args.grad_accum or train_config.gradient_accumulation
            sl = args.seq_len or train_config.max_seq_len
            tokens_per_step = bs * ga * sl
            train_config.total_steps = max(1, args.total_tokens // tokens_per_step)
            train_config.tokens = args.total_tokens
            train_config.warmup_steps = min(500, train_config.total_steps // 10)
            train_config.save_every = max(1000, train_config.total_steps // 5)

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
    if args.num_workers is not None:
        train_config.num_workers = max(0, int(args.num_workers))
    if args.prefetch_factor is not None:
        train_config.prefetch_factor = max(1, int(args.prefetch_factor))
    if args.persistent_workers is not None:
        train_config.persistent_workers = bool(args.persistent_workers)
    if train_config.num_workers <= 0:
        train_config.persistent_workers = False

    # Store start time for wall-clock tracking
    training_start_time = time.time()

    # Allow both compositional and standard FFN for comparison experiments
    is_compositional = model_config.ffn_type == "compositional"
    if not is_compositional:
        logger.info("Training with standard FFN (dense baseline)")

    # Log ternary config
    if is_compositional and model_config.primitive_config.ternary_primitives:
        pc = model_config.primitive_config
        logger.info(
            f"Ternary quantization ENABLED: activation_bits={pc.activation_bits}, "
            f"use_subln={pc.use_subln}, use_squared_relu={pc.use_squared_relu}"
        )

    # Build model
    model = create_model(model_config)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    model.to(device)

    # Phase B.5c: Post-hoc gate training mode
    if getattr(args, "train_exit_gates", False):
        if not model_config.enable_early_exit:
            raise ValueError("--train-exit-gates requires --enable-early-exit")
        if args.resume is None:
            raise ValueError("--train-exit-gates requires --resume to load a trained checkpoint")

        # Load checkpoint
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded checkpoint from {args.resume} for gate training")

        # Create dataloader — pre-load into memory for fast multi-epoch iteration.
        # Gate training only needs 5M tokens; streaming re-init per epoch is too slow
        # for large datasets like fineweb-edu (skip + re-download each epoch).
        tokenizer = get_tokenizer(getattr(args, "tokenizer_path", None))
        logger.info("Loading gate training data (5M tokens)...")
        gate_dataset_stream = load_text_dataset(
            dataset_name=args.dataset or "Elriggs/openwebtext-100k",
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            max_tokens=5_000_000,
            split="train",
            streaming=True,
            tokenize_batch_size=args.tokenize_batch_size,
        )
        # Materialize streaming dataset into memory for multi-epoch reuse
        gate_data_list = []
        for item in gate_dataset_stream:
            gate_data_list.append(item)
        logger.info(f"Gate data loaded: {len(gate_data_list)} sequences")
        from torch.utils.data import TensorDataset
        gate_ids = torch.stack([item["input_ids"] for item in gate_data_list])
        gate_masks = torch.stack([item.get("attention_mask", torch.ones_like(item["input_ids"])) for item in gate_data_list])
        gate_dataset = TensorDataset(gate_ids, gate_masks)
        gate_loader = create_dataloader(
            gate_dataset, batch_size=args.batch_size or 4, shuffle=True,
        )

        logger.info("Starting exit gate training...")
        history = train_exit_gates(
            model, gate_loader, torch.device(device),
            epochs=3, lr=1e-3, exit_threshold=model_config.exit_threshold,
        )
        logger.info(f"Gate training complete. Losses: {history['epoch_losses']}")

        # Save model with trained gates
        gate_path = output_dir / "model_with_gates.pt"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_config,
        }, gate_path)
        logger.info(f"Saved gate-trained model to {gate_path}")
        return

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
        streaming=True,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    val_dataset = load_text_dataset(
        dataset_name=train_config.dataset,
        tokenizer=tokenizer,
        max_seq_len=train_config.max_seq_len,
        max_tokens=2_000_000,
        split="validation",
        streaming=True,
        tokenize_batch_size=args.tokenize_batch_size,
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
            # Ensure MSVC environment is set for C++ guard compilation on Windows
            if os.name == "nt":
                if ensure_msvc_env():
                    logger.info(f"MSVC environment configured: INCLUDE={os.environ.get('INCLUDE', '')[:80]}...")
                else:
                    logger.warning("Could not auto-configure MSVC environment. "
                                   "torch.compile C++ guards may fail. "
                                   "Run from a VS Developer Command Prompt or set INCLUDE/LIB manually.")
            # Allow scalar outputs (e.g. scale.item() in quantization) to be
            # captured in the graph instead of causing graph breaks.
            torch._dynamo.config.capture_scalar_outputs = True
            # Treat integer nn.Module attributes (e.g. runtime_step) as dynamic
            # so they don't trigger recompilation every step.
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            model = torch.compile(model)
            logger.info("Enabled torch.compile for model")
        except Exception as exc:
            logger.warning(f"torch.compile failed: {exc}")

    # Optimizer (created AFTER compile to ensure we capture the right parameters)
    base_lr = compute_base_lr(0, train_config)
    optimizer = build_optimizer(
        model,
        base_lr,
        primitive_weight_decay=max(0.0, float(args.primitive_weight_decay)),
    )

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
        rank_start = args.rank_start if args.rank_start is not None else rank_final
        rank_mid = args.rank_mid if args.rank_mid is not None else rank_final
        rank_start = int(max(1, min(rank_final, rank_start)))
        rank_mid = int(max(rank_start, min(rank_final, rank_mid)))
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
            unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
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
    if is_compositional and topk_cache_steps is None:
        topk_cache_steps = 10
    freeze_primitives_phase2 = train_config.freeze_primitives_in_phase2
    if args.freeze_primitives_phase2 is not None:
        freeze_primitives_phase2 = args.freeze_primitives_phase2
    comp_entropy_weight = args.comp_entropy_weight
    if comp_entropy_weight > 0 and not args.allow_entropy_regularizer:
        logger.warning(
            "Disabling composition entropy regularizer by default because it conflicts with sparse top-k usage. "
            "Pass --allow-entropy-regularizer to force-enable."
        )
        comp_entropy_weight = 0.0
    comp_lr_mult = args.comp_lr_mult
    band_diversity_weight = max(0.0, float(args.band_diversity_weight))
    hot_tier_bias_weight = max(0.0, float(args.hot_tier_bias_weight))
    joint_exit_loss_weight = max(0.0, float(args.joint_exit_loss_weight))

    step_time = time.time()
    tokens_since_log = 0
    grad_accum_steps = max(1, int(train_config.gradient_accumulation))
    tokens_per_step_effective = (
        train_config.micro_batch_size * train_config.max_seq_len * grad_accum_steps
    )
    target_tokens = args.total_tokens if args.total_tokens is not None else train_config.tokens
    planned_total_tokens = tokens_per_step_effective * train_config.total_steps
    naive_tokens_no_accum = (
        train_config.micro_batch_size * train_config.max_seq_len * train_config.total_steps
    )
    logger.info(
        "Token accounting: "
        f"steps={train_config.total_steps}, micro_batch={train_config.micro_batch_size}, "
        f"grad_accum={grad_accum_steps}, seq_len={train_config.max_seq_len}, "
        f"effective_tokens_per_step={tokens_per_step_effective}, "
        f"planned_total_tokens={planned_total_tokens}, target_tokens={target_tokens}"
    )
    if target_tokens is not None and target_tokens > 0:
        # Hard guard against silent large undertraining/overtraining.
        if planned_total_tokens < int(0.5 * target_tokens):
            raise ValueError(
                "Planned token budget is <50% of target. "
                "Check total_steps / gradient_accumulation accounting."
            )
        if planned_total_tokens > int(2.0 * target_tokens):
            raise ValueError(
                "Planned token budget is >200% of target. "
                "Check total_steps / gradient_accumulation accounting."
            )
        if grad_accum_steps > 1 and abs(naive_tokens_no_accum - target_tokens) <= tokens_per_step_effective:
            logger.info(
                "Token accounting sanity: step math already includes gradient accumulation; "
                "loop accumulation is expected and not double-counted."
            )
    moe_aux_loss_weight = 0.0
    if is_compositional and model_config.primitive_config.moe_config is not None:
        moe_aux_loss_weight = float(model_config.primitive_config.moe_config.aux_loss_weight)

    is_compiled = args.compile and hasattr(model, "_orig_mod")
    recompile_phases = args.recompile_phases and is_compiled
    prev_phase = 0  # Will trigger initial phase setup on first step

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
        if is_moe:
            # Keep MoE routing sparse and stable unless explicitly changed in config.
            runtime_top_k = router_top_k_final
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

            # Phase-specific parameter freezing
            if recompile_phases and phase != prev_phase:
                # Toggle requires_grad for proper freezing (no wasted grad compute)
                # then recompile so torch.compile builds a new optimized graph
                if phase == 1:
                    set_primitive_requires_grad(raw_model, True)
                    set_composition_requires_grad(raw_model, False)
                elif phase >= 2:
                    if freeze_primitives_phase2:
                        set_primitive_requires_grad(raw_model, False)
                    set_composition_requires_grad(raw_model, True)

                # Log frozen/trainable param counts for debugging
                n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                n_frozen = sum(p.numel() for p in raw_model.parameters() if not p.requires_grad)
                logger.info(f"Phase {prev_phase}->{phase}: trainable={n_trainable:,} frozen={n_frozen:,}")

                try:
                    torch._dynamo.reset()
                    torch.cuda.empty_cache()  # Free fragmented VRAM before recompile
                    torch._dynamo.config.capture_scalar_outputs = True
                    torch._dynamo.config.allow_unspec_int_on_nn_module = True
                    if os.name == "nt":
                        ensure_msvc_env()
                    model = torch.compile(raw_model)
                    logger.info(f"Phase {prev_phase}->{phase}: recompiled model OK")
                except Exception as exc:
                    logger.error(f"Phase {prev_phase}->{phase}: recompile FAILED: {exc}")
                    logger.info("Falling back to eager mode for remaining training")
                    model = raw_model
                    recompile_phases = False
            prev_phase = phase

            cache_frozen_primitives = phase >= 2 and freeze_primitives_phase2
            for layer in raw_model.layers:
                ffn = layer.ffn
                if hasattr(ffn, "cache_selected_primitives"):
                    ffn.cache_selected_primitives = cache_frozen_primitives

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
        sparse_lr_mult = 1.0
        if is_compositional:
            apply_phase_lrs_with_override(optimizer, base_lr, phase, comp_lr_mult,
                                         freeze_primitives_phase2=freeze_primitives_phase2)
            sparse_lr_mult = compute_sparse_lr_multiplier(
                is_moe=is_moe,
                n_primitives=n_primitives,
                top_k_final=top_k_final,
                runtime_top_k=runtime_top_k,
                runtime_top_k_fc1=runtime_top_k_fc1,
                runtime_top_k_fc2=runtime_top_k_fc2,
                router_n_experts=router_n_experts,
                max_mult=float(args.sparse_lr_max_mult),
            )
            apply_sparse_lr_compensation(optimizer, sparse_lr_mult)
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
        band_diversity_loss_accum = 0.0
        band_diversity_loss_count = 0
        hot_tier_bias_loss_accum = 0.0
        hot_tier_bias_loss_count = 0
        joint_exit_loss_accum = 0.0
        joint_exit_loss_count = 0
        aux_loss_raw_accum = 0.0
        aux_loss_scaled_accum = 0.0
        aux_loss_count = 0
        grad_norm_value = 0.0

        optimizer.zero_grad(set_to_none=True)

        # Pre-quantize ternary weights once per step (reused across micro-batches)
        if is_compositional and hasattr(raw_model, 'primitive_banks'):
            for bank in raw_model.primitive_banks.fc1_banks.values():
                bank.prepare_q_cache()
            for bank in raw_model.primitive_banks.fc2_banks.values():
                bank.prepare_q_cache()

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

                band_diversity_loss = None
                if is_compositional and band_diversity_weight > 0 and phase >= 2:
                    band_diversity_loss = compute_band_diversity_loss(raw_model)
                    if band_diversity_loss is not None:
                        total_loss = total_loss + band_diversity_weight * band_diversity_loss

                hot_tier_bias_loss = None
                if is_compositional and hot_tier_bias_weight > 0:
                    hot_tier_bias_loss = compute_hot_tier_bias_loss(raw_model)
                    if hot_tier_bias_loss is not None:
                        total_loss = total_loss + hot_tier_bias_weight * hot_tier_bias_loss

                joint_exit_loss = None
                if joint_exit_loss_weight > 0:
                    exit_confidences = outputs.get("exit_confidences")
                    if exit_confidences is not None:
                        joint_exit_loss = compute_joint_exit_loss(
                            exit_confidences=exit_confidences,
                            logits=outputs["logits"],
                        )
                        total_loss = total_loss + joint_exit_loss_weight * joint_exit_loss

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
            if band_diversity_loss is not None:
                band_diversity_loss_accum += float(band_diversity_loss.detach().item())
                band_diversity_loss_count += 1
            if hot_tier_bias_loss is not None:
                hot_tier_bias_loss_accum += float(hot_tier_bias_loss.detach().item())
                hot_tier_bias_loss_count += 1
            if joint_exit_loss is not None:
                joint_exit_loss_accum += float(joint_exit_loss.detach().item())
                joint_exit_loss_count += 1
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

        # Invalidate ternary weight cache (weights changed after optimizer step)
        if is_compositional and hasattr(raw_model, 'primitive_banks'):
            for bank in raw_model.primitive_banks.fc1_banks.values():
                bank.invalidate_q_cache()
            for bank in raw_model.primitive_banks.fc2_banks.values():
                bank.invalidate_q_cache()

        # Phase B.5b: Swap tiered primitive banks
        if args.n_hot is not None and is_compositional:
            raw_model.swap_tiers(optimizer)

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
        band_diversity_loss_value = (
            band_diversity_loss_accum / band_diversity_loss_count if band_diversity_loss_count > 0 else None
        )
        hot_tier_bias_loss_value = (
            hot_tier_bias_loss_accum / hot_tier_bias_loss_count if hot_tier_bias_loss_count > 0 else None
        )
        joint_exit_loss_value = (
            joint_exit_loss_accum / joint_exit_loss_count if joint_exit_loss_count > 0 else None
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
                primitive_group_lr = None
                composition_group_lr = None
                for group in optimizer.param_groups:
                    if group.get("name") == "primitives":
                        primitive_group_lr = float(group.get("lr", 0.0))
                    elif group.get("name") == "composition":
                        composition_group_lr = float(group.get("lr", 0.0))
                entropy_vals = raw_model.get_all_entropy()
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
                    "moe_enabled": float(1.0 if is_moe else 0.0),
                    "checkpoint_ffn": float(1.0 if model_config.checkpoint_ffn else 0.0),
                    "sparse_lr_mult": float(sparse_lr_mult),
                })
                if primitive_group_lr is not None:
                    log_dict["primitive_lr"] = primitive_group_lr
                if composition_group_lr is not None:
                    log_dict["composition_lr"] = composition_group_lr

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
                if band_diversity_loss_value is not None:
                    log_dict["band_diversity_loss"] = band_diversity_loss_value
                    log_dict["band_diversity_weight"] = band_diversity_weight
                if hot_tier_bias_loss_value is not None:
                    log_dict["hot_tier_bias_loss"] = hot_tier_bias_loss_value
                    log_dict["hot_tier_bias_weight"] = hot_tier_bias_weight
                if joint_exit_loss_value is not None:
                    log_dict["joint_exit_loss"] = joint_exit_loss_value
                    log_dict["joint_exit_weight"] = joint_exit_loss_weight

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
            # Phase B.5d: VRAM efficiency metric
            if device.startswith("cuda"):
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
                val_log["peak_vram_gb"] = peak_vram_gb
                if peak_vram_gb > 0:
                    val_log["val_loss_per_vram_gb"] = avg_val / peak_vram_gb
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
    parser.add_argument("--checkpoint-ffn", dest="checkpoint_ffn", action="store_true",
                        help="Enable FFN gradient checkpointing")
    parser.add_argument("--no-checkpoint-ffn", dest="checkpoint_ffn", action="store_false",
                        help="Disable FFN gradient checkpointing for max throughput")
    parser.set_defaults(checkpoint_ffn=None)

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
    parser.add_argument("--forward-fast-mode", type=str, choices=["auto", "on", "off"], default=None,
                        help="Compositional FFN execution path")
    parser.add_argument("--forward-fast-min-topk", type=int, default=None,
                        help="Minimum top-k for forward_fast auto mode")
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
    parser.add_argument("--allow-entropy-regularizer", action="store_true",
                        help="Allow entropy regularization with sparse top-k")
    parser.add_argument("--band-diversity-weight", type=float, default=0.0,
                        help="Penalty weight to reduce layer redundancy inside each primitive band")
    parser.add_argument("--hot-tier-bias-weight", type=float, default=0.0,
                        help="Bias composition toward hot-tier primitives when tiering is enabled")
    parser.add_argument("--joint-exit-loss-weight", type=float, default=0.0,
                        help="Jointly train exit gates during LM training with self-supervised certainty targets")
    parser.add_argument("--primitive-weight-decay", type=float, default=0.0,
                        help="Weight decay for primitive banks (default 0 to avoid dense-style drift)")
    parser.add_argument("--sparse-lr-max-mult", type=float, default=8.0,
                        help="Maximum LR multiplier for sparse primitive/composition updates")
    parser.add_argument("--moe-experts", type=int, default=None,
                        help="Enable token-level MoE routing with this many experts")
    parser.add_argument("--moe-top-k", type=int, default=None,
                        help="Top-k experts per token for MoE routing")
    parser.add_argument("--moe-router-type", type=str, choices=["linear", "mlp"], default="linear",
                        help="Router type for MoE")
    parser.add_argument("--moe-router-hidden-dim", type=int, default=None,
                        help="Hidden size for MLP MoE router")
    parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01,
                        help="Auxiliary load-balancing weight for MoE")
    parser.add_argument("--no-moe-load-balancing", action="store_true",
                        help="Disable MoE load balancing loss")
    # Phase C: Attention variants
    parser.add_argument("--attention-type", type=str, default="gated_recurrence",
                        choices=["standard_mha", "compositional_mha", "gated_recurrence",
                                 "compositional_recurrence", "compositional_gated_recurrence",
                                 "hybrid", "hybrid_recurrent_mha"],
                        help="Attention mechanism type")
    parser.add_argument("--n-attn-primitives", type=int, default=16,
                        help="Number of attention primitives (compositional attention)")
    parser.add_argument("--attn-rank", type=int, default=32,
                        help="Rank for attention primitive banks")
    parser.add_argument("--attn-top-k", type=int, default=4,
                        help="Top-k for attention composition")

    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--recompile-phases", action="store_true",
                        help="Recompile model at phase boundaries (requires --compile). "
                             "Uses requires_grad=False for frozen params + torch._dynamo.reset() "
                             "instead of LR=0, eliminating wasted gradient computation.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul")

    parser.add_argument("--progressive-unfreeze", action="store_true")
    parser.add_argument("--log-timing", action="store_true", help="Log timing breakdowns per step")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Prefetch factor per DataLoader worker")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true",
                        help="Enable persistent DataLoader workers")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false",
                        help="Disable persistent DataLoader workers")
    parser.set_defaults(persistent_workers=None)
    parser.add_argument("--tokenize-batch-size", type=int, default=64,
                        help="Number of raw texts tokenized together in streaming loader")

    # Phase B.5b: Tiered primitive bank (VRAM-efficient)
    parser.add_argument("--n-hot", type=int, default=None,
                        help="Number of hot primitives in VRAM (None = no tiering)")
    parser.add_argument("--swap-interval", type=int, default=100,
                        help="Steps between hot/warm primitive swaps")

    # Phase B.5c: Early exit
    parser.add_argument("--enable-early-exit", action="store_true",
                        help="Enable early exit gates on transformer blocks")
    parser.add_argument("--exit-threshold", type=float, default=0.5,
                        help="Skip confidence threshold for early exit")
    parser.add_argument("--train-exit-gates", action="store_true",
                        help="Post-hoc gate training mode: freeze model, train only gates")

    # Ternary quantization (BitNet b1.58)
    parser.add_argument("--ternary", action="store_true",
                        help="Enable ternary weight quantization for primitive banks")
    parser.add_argument("--activation-bits", type=int, default=8,
                        help="Activation quantization bitwidth (0 = disable)")
    parser.add_argument("--use-subln", action="store_true",
                        help="Enable SubLN normalization for ternary stability")
    parser.add_argument("--use-squared-relu", action="store_true",
                        help="Use squared ReLU activation (sparser, pairs with ternary)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.tf32 = not args.no_tf32
    set_seed(args.seed)
    train_v2(args)


if __name__ == "__main__":
    main()
