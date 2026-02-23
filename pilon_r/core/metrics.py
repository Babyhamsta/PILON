"""
PILON-R Metrics and Logging

Handles:
- Training metrics tracking
- Entropy monitoring over time
- Primitive usage heatmaps
- Gate checking
- Logging to file and console
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
import math


@dataclass
class MetricPoint:
    """Single metric measurement."""
    step: int
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: List[MetricPoint] = field(default_factory=list)
    val_loss: List[MetricPoint] = field(default_factory=list)
    val_ppl: List[MetricPoint] = field(default_factory=list)
    grad_norm: List[MetricPoint] = field(default_factory=list)
    learning_rate: List[MetricPoint] = field(default_factory=list)

    # Entropy tracking (per layer, over time)
    entropy_history: Dict[str, List[MetricPoint]] = field(default_factory=dict)

    # Primitive usage (snapshots at checkpoints)
    usage_snapshots: List[Dict] = field(default_factory=list)

    def add_train_loss(self, step: int, loss: float):
        self.train_loss.append(MetricPoint(step, loss))

    def add_val_loss(self, step: int, loss: float):
        self.val_loss.append(MetricPoint(step, loss))
        self.val_ppl.append(MetricPoint(step, math.exp(loss)))

    def add_grad_norm(self, step: int, norm: float):
        self.grad_norm.append(MetricPoint(step, norm))

    def add_lr(self, step: int, lr: float):
        self.learning_rate.append(MetricPoint(step, lr))

    def add_entropy(self, step: int, layer_name: str, entropy: float):
        if layer_name not in self.entropy_history:
            self.entropy_history[layer_name] = []
        self.entropy_history[layer_name].append(MetricPoint(step, entropy))

    def add_usage_snapshot(self, step: int, usage: Dict[str, torch.Tensor]):
        snapshot = {
            "step": step,
            "usage": {k: v.cpu().tolist() for k, v in usage.items()}
        }
        self.usage_snapshots.append(snapshot)

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        metric_list = getattr(self, metric_name, None)
        if metric_list and len(metric_list) > 0:
            return metric_list[-1].value
        return None

    def get_average(self, metric_name: str, last_n: int = 10) -> Optional[float]:
        """Get average of last n values."""
        metric_list = getattr(self, metric_name, None)
        if metric_list and len(metric_list) > 0:
            values = [p.value for p in metric_list[-last_n:]]
            return sum(values) / len(values)
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "train_loss": [(p.step, p.value) for p in self.train_loss],
            "val_loss": [(p.step, p.value) for p in self.val_loss],
            "val_ppl": [(p.step, p.value) for p in self.val_ppl],
            "grad_norm": [(p.step, p.value) for p in self.grad_norm],
            "learning_rate": [(p.step, p.value) for p in self.learning_rate],
            "entropy_history": {
                k: [(p.step, p.value) for p in v]
                for k, v in self.entropy_history.items()
            },
            "usage_snapshots": self.usage_snapshots
        }

    def save(self, path: Path):
        """Save metrics to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingMetrics":
        """Load metrics from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        metrics = cls()
        for step, value in data.get("train_loss", []):
            metrics.train_loss.append(MetricPoint(step, value))
        for step, value in data.get("val_loss", []):
            metrics.val_loss.append(MetricPoint(step, value))
        for step, value in data.get("val_ppl", []):
            metrics.val_ppl.append(MetricPoint(step, value))
        for step, value in data.get("grad_norm", []):
            metrics.grad_norm.append(MetricPoint(step, value))
        for step, value in data.get("learning_rate", []):
            metrics.learning_rate.append(MetricPoint(step, value))

        for layer_name, points in data.get("entropy_history", {}).items():
            metrics.entropy_history[layer_name] = [
                MetricPoint(step, value) for step, value in points
            ]

        metrics.usage_snapshots = data.get("usage_snapshots", [])

        return metrics


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Compute entropy of a probability distribution.

    High entropy = uniform distribution (diverse usage)
    Low entropy = peaked distribution (collapsed usage)

    For n_primitives=32 with uniform distribution:
    max_entropy = log(32) = 3.47

    Entropy > 1.0 is considered healthy for Phase A.
    """
    probs = F.softmax(logits / temperature, dim=-1)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def compute_primitive_usage_stats(
    usage_weights: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics about primitive usage.

    Returns per-layer stats:
    - entropy: diversity of usage
    - max_weight: weight of most-used primitive
    - min_weight: weight of least-used primitive
    - active_count: number of primitives with weight > threshold
    """
    stats = {}
    threshold = 0.01  # 1% threshold for "active"

    for layer_name, weights in usage_weights.items():
        weights = weights.detach().cpu()

        # Compute entropy from weights (already softmaxed)
        log_weights = torch.log(weights + 1e-8)
        entropy = -(weights * log_weights).sum().item()

        stats[layer_name] = {
            "entropy": entropy,
            "max_weight": weights.max().item(),
            "min_weight": weights.min().item(),
            "active_count": (weights > threshold).sum().item(),
            "gini": compute_gini(weights),
        }

    return stats


def compute_gini(weights: torch.Tensor) -> float:
    """
    Compute Gini coefficient of weight distribution.

    0 = perfectly equal (good)
    1 = perfectly unequal (one primitive dominates)
    """
    sorted_weights = torch.sort(weights)[0]
    n = len(sorted_weights)
    cumulative = torch.cumsum(sorted_weights, dim=0)
    gini = (n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n
    return gini.item()


# =============================================================================
# MoE-specific metrics (Phase B)
# =============================================================================

def compute_router_entropy(router_logits: torch.Tensor) -> float:
    """
    Compute entropy of router probability distribution.

    High entropy = experts used evenly (good)
    Low entropy = few experts dominate (bad, expert collapse)

    For n_experts=8 with uniform distribution:
    max_entropy = log(8) = 2.08

    Target: > 1.0 (healthy routing)

    Args:
        router_logits: Router logits (batch, seq, n_experts) or (n_experts,)
    """
    # Flatten to 2D if needed
    if router_logits.dim() > 2:
        router_logits = router_logits.view(-1, router_logits.size(-1))
    elif router_logits.dim() == 1:
        router_logits = router_logits.unsqueeze(0)

    # Per-token entropy, then mean (more sensitive to change)
    probs = F.softmax(router_logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return entropy


def compute_load_balance_metric(router_logits: torch.Tensor, top_k: int = 2) -> Dict[str, float]:
    """
    Compute load balancing metrics for MoE routing.

    Metrics:
    - cv (coefficient of variation): std/mean of expert usage
      Target: < 0.3 (even load)
    - max_load: maximum fraction of tokens assigned to any expert
      Target: < 2/n_experts (not more than 2x fair share)
    - min_load: minimum fraction
      Target: > 0.5/n_experts (at least half fair share)

    Args:
        router_logits: Router logits (batch, seq, n_experts)
        top_k: Number of experts selected per token
    """
    # Flatten to 2D
    if router_logits.dim() > 2:
        router_logits = router_logits.view(-1, router_logits.size(-1))

    n_tokens, n_experts = router_logits.shape

    # Get top-k selections
    probs = F.softmax(router_logits, dim=-1)
    _, top_indices = torch.topk(probs, top_k, dim=-1, sorted=False)  # (n_tokens, top_k)

    # Count expert usage
    expert_counts = torch.zeros(n_experts, device=router_logits.device)
    for k in range(top_k):
        expert_counts.scatter_add_(
            0,
            top_indices[:, k],
            torch.ones(n_tokens, device=router_logits.device)
        )

    # Normalize to fractions
    total_assignments = n_tokens * top_k
    expert_fractions = expert_counts / total_assignments

    # Compute metrics
    mean_frac = expert_fractions.mean().item()
    std_frac = expert_fractions.std().item()
    cv = std_frac / (mean_frac + 1e-8)

    return {
        "cv": cv,
        "max_load": expert_fractions.max().item(),
        "min_load": expert_fractions.min().item(),
        "mean_load": mean_frac,
        "std_load": std_frac,
    }


def compute_expert_specialization(
    expert_compositions: torch.Tensor
) -> Dict[str, float]:
    """
    Compute specialization metrics for expert compositions.

    Measures how different expert composition weights are from each other.
    If all experts use same primitives, MoE provides no benefit.

    Metrics:
    - mean_similarity: average pairwise cosine similarity
      Target: < 0.5 (experts are meaningfully different)
    - max_similarity: highest pairwise similarity
      Red flag if > 0.8 (experts converging)
    - min_similarity: lowest pairwise similarity

    Args:
        expert_compositions: Expert composition weights (n_experts, n_primitives)
    """
    n_experts = expert_compositions.size(0)

    # Normalize for cosine similarity
    normalized = F.normalize(expert_compositions, p=2, dim=-1)

    # Compute pairwise similarities
    similarity_matrix = torch.mm(normalized, normalized.t())

    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n_experts, n_experts, device=expert_compositions.device), diagonal=1)
    pairwise_sims = similarity_matrix[mask.bool()]

    if len(pairwise_sims) == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}

    return {
        "mean": pairwise_sims.mean().item(),
        "max": pairwise_sims.max().item(),
        "min": pairwise_sims.min().item(),
    }


@dataclass
class MoEMetrics:
    """Container for MoE-specific metrics over training."""
    router_entropy: List[MetricPoint] = field(default_factory=list)
    load_balance_cv: List[MetricPoint] = field(default_factory=list)
    expert_similarity: List[MetricPoint] = field(default_factory=list)
    aux_loss: List[MetricPoint] = field(default_factory=list)

    # Per-layer metrics
    layer_router_entropy: Dict[str, List[MetricPoint]] = field(default_factory=dict)
    layer_expert_similarity: Dict[str, List[MetricPoint]] = field(default_factory=dict)

    def add_router_entropy(self, step: int, entropy: float, layer: Optional[str] = None):
        if layer:
            if layer not in self.layer_router_entropy:
                self.layer_router_entropy[layer] = []
            self.layer_router_entropy[layer].append(MetricPoint(step, entropy))
        else:
            self.router_entropy.append(MetricPoint(step, entropy))

    def add_load_balance(self, step: int, cv: float):
        self.load_balance_cv.append(MetricPoint(step, cv))

    def add_expert_similarity(self, step: int, similarity: float, layer: Optional[str] = None):
        if layer:
            if layer not in self.layer_expert_similarity:
                self.layer_expert_similarity[layer] = []
            self.layer_expert_similarity[layer].append(MetricPoint(step, similarity))
        else:
            self.expert_similarity.append(MetricPoint(step, similarity))

    def add_aux_loss(self, step: int, loss: float):
        self.aux_loss.append(MetricPoint(step, loss))

    def get_latest(self, metric_name: str) -> Optional[float]:
        metric_list = getattr(self, metric_name, None)
        if metric_list and len(metric_list) > 0:
            return metric_list[-1].value
        return None

    def to_dict(self) -> Dict:
        return {
            "router_entropy": [(p.step, p.value) for p in self.router_entropy],
            "load_balance_cv": [(p.step, p.value) for p in self.load_balance_cv],
            "expert_similarity": [(p.step, p.value) for p in self.expert_similarity],
            "aux_loss": [(p.step, p.value) for p in self.aux_loss],
            "layer_router_entropy": {
                k: [(p.step, p.value) for p in v]
                for k, v in self.layer_router_entropy.items()
            },
            "layer_expert_similarity": {
                k: [(p.step, p.value) for p in v]
                for k, v in self.layer_expert_similarity.items()
            },
        }


class MoEGateChecker:
    """
    Checks success gates for Phase B (MoE).

    Gates:
    - B0: Training Stability (0-500 steps)
    - B1: Convergence Improvement (500-5000 steps)
    - B2: Quality at Convergence (5000-10000 steps)
    """

    def __init__(self):
        self.gate_results = {}

    def check_b0(
        self,
        step: int,
        loss_history: List[float],
        router_entropy: float,
        load_balance_cv: float,
        primitive_entropy: float,
        has_nan: bool = False
    ) -> Dict[str, Any]:
        """
        Check Gate B0: MoE Training Stability (Steps 0-500)

        PASS criteria:
        - Loss decreases (not worse than Phase A initially)
        - No NaN, no divergence
        - Router entropy > 1.0 (experts being used)
        - Load balance CV < 0.5 (no extreme expert collapse)
        - Primitive entropy still healthy (> 1.0)
        """
        results = {
            "gate": "B0",
            "step": step,
            "passed": True,
            "checks": {}
        }

        # Check loss decreasing
        if len(loss_history) >= 10:
            recent = loss_history[-10:]
            early = loss_history[:10] if len(loss_history) >= 20 else loss_history[:len(loss_history)//2]
            loss_decreasing = sum(recent) / len(recent) < sum(early) / len(early)
        else:
            loss_decreasing = True

        results["checks"]["loss_decreasing"] = loss_decreasing
        if not loss_decreasing:
            results["passed"] = False

        # Check no NaN
        results["checks"]["no_nan"] = not has_nan
        if has_nan:
            results["passed"] = False

        # Check router entropy (> 1.0 for healthy routing)
        router_ok = router_entropy > 1.0
        results["checks"]["router_entropy_ok"] = router_ok
        results["checks"]["router_entropy"] = router_entropy
        if not router_ok:
            results["passed"] = False

        # Check load balance (CV < 0.5)
        load_ok = load_balance_cv < 0.5
        results["checks"]["load_balance_ok"] = load_ok
        results["checks"]["load_balance_cv"] = load_balance_cv
        if not load_ok:
            results["passed"] = False

        # Check primitive entropy
        prim_ok = primitive_entropy > 1.0
        results["checks"]["primitive_entropy_ok"] = prim_ok
        results["checks"]["primitive_entropy"] = primitive_entropy
        if not prim_ok:
            results["passed"] = False

        self.gate_results["B0"] = results
        return results

    def check_b1(
        self,
        step: int,
        val_ppl: float,
        baseline_ppl: float,
        phase_a_ratio: float = 1.5
    ) -> Dict[str, Any]:
        """
        Check Gate B1: Convergence Improvement (Steps 500-5000)

        PASS criteria:
        - Convergence improved over Phase A
        - PPL ratio vs baseline < 1.3x at 10K steps (vs Phase A's 1.5x)
        """
        results = {
            "gate": "B1",
            "step": step,
            "passed": True,
            "checks": {}
        }

        ratio = val_ppl / baseline_ppl
        improved = ratio < phase_a_ratio  # Better than Phase A
        target_met = ratio < 1.3  # Target for Phase B

        results["checks"]["ppl_ratio"] = ratio
        results["checks"]["baseline_ppl"] = baseline_ppl
        results["checks"]["val_ppl"] = val_ppl
        results["checks"]["improved_over_phase_a"] = improved
        results["checks"]["target_met"] = target_met

        if not improved:
            results["passed"] = False

        self.gate_results["B1"] = results
        return results

    def check_b2(
        self,
        step: int,
        val_ppl: float,
        baseline_ppl: float,
        expert_similarity: float
    ) -> Dict[str, Any]:
        """
        Check Gate B2: Quality at Convergence (Steps 5000-10000)

        PASS criteria:
        - Final PPL within 1.15x of baseline
        - Experts show meaningful specialization (similarity < 0.5)
        """
        results = {
            "gate": "B2",
            "step": step,
            "passed": True,
            "checks": {}
        }

        ratio = val_ppl / baseline_ppl
        quality_ok = ratio < 1.15

        results["checks"]["ppl_ratio"] = ratio
        results["checks"]["quality_ok"] = quality_ok
        if not quality_ok:
            results["passed"] = False

        # Check expert specialization
        specialized = expert_similarity < 0.5
        results["checks"]["expert_similarity"] = expert_similarity
        results["checks"]["experts_specialized"] = specialized
        if not specialized:
            results["passed"] = False

        self.gate_results["B2"] = results
        return results

    def get_summary(self) -> str:
        """Get summary of all gate checks."""
        lines = ["=" * 50, "MOE GATE CHECK SUMMARY", "=" * 50]

        for gate_name in ["B0", "B1", "B2"]:
            if gate_name in self.gate_results:
                result = self.gate_results[gate_name]
                status = "PASS" if result["passed"] else "FAIL"
                lines.append(f"\n{gate_name} @ step {result['step']}: {status}")
                for check, value in result["checks"].items():
                    if isinstance(value, float):
                        lines.append(f"  - {check}: {value:.4f}")
                    else:
                        lines.append(f"  - {check}: {value}")

        return "\n".join(lines)


class GateChecker:
    """
    Checks success gates for Phase A.

    Gates:
    - A0: Smoke Test (0-500 steps)
    - A1: Training Stability (500-2500 steps)
    - A2: Learning Validation (2500-10000 steps)
    - A3: Functional LM (10000-25000 steps)
    """

    def __init__(self, gate_config=None):
        from .config import GateConfig
        self.config = gate_config or GateConfig()
        self.gate_results = {}

    def check_a0(
        self,
        step: int,
        loss_history: List[float],
        grad_norm: float,
        entropy_values: Dict[str, float],
        has_nan: bool = False,
        primitive_grad_stats: Optional[Dict[str, int]] = None
    ) -> Dict[str, bool]:
        """
        Check Gate A0: Smoke Test (Steps 0-500)

        PASS criteria (ALL required):
        - Loss decreases for first 500 steps
        - No NaN or Inf anywhere
        - Gradient norm < 100 and not exploding
        - Most primitives have non-zero gradient (>50% with top_k sparse selection)
        - Entropy > 1.0 (primitives being used diversely)
        """
        results = {
            "gate": "A0",
            "step": step,
            "passed": True,
            "checks": {}
        }

        # Check loss decreasing
        if len(loss_history) >= 10:
            recent = loss_history[-10:]
            early = loss_history[:10] if len(loss_history) >= 20 else loss_history[:len(loss_history)//2]
            loss_decreasing = sum(recent) / len(recent) < sum(early) / len(early)
        else:
            loss_decreasing = True  # Not enough data yet

        results["checks"]["loss_decreasing"] = loss_decreasing
        if not loss_decreasing:
            results["passed"] = False

        # Check no NaN
        results["checks"]["no_nan"] = not has_nan
        if has_nan:
            results["passed"] = False

        # Check gradient norm
        grad_ok = grad_norm < self.config.a0_max_grad_norm
        results["checks"]["grad_norm_ok"] = grad_ok
        results["checks"]["grad_norm_value"] = grad_norm
        if not grad_ok:
            results["passed"] = False

        # Check entropy
        min_entropy = min(entropy_values.values()) if entropy_values else 0
        entropy_ok = min_entropy > self.config.a0_min_entropy
        results["checks"]["entropy_ok"] = entropy_ok
        results["checks"]["min_entropy"] = min_entropy
        if not entropy_ok:
            results["passed"] = False

        # Check primitive gradients if provided
        # With top_k sparse selection, not all primitives get gradients in one batch
        # Require >50% to have gradients as evidence of gradient flow
        if primitive_grad_stats is not None:
            total = primitive_grad_stats.get("total", 0)
            with_grad = total - primitive_grad_stats.get("zero", 0)
            grad_ratio = with_grad / total if total > 0 else 0
            primitives_ok = grad_ratio > 0.5  # >50% have gradients
            results["checks"]["primitives_grad_ratio"] = grad_ratio
            results["checks"]["primitives_have_grad"] = primitives_ok
            if not primitives_ok:
                results["passed"] = False

        self.gate_results["A0"] = results
        return results

    def check_a1(
        self,
        step: int,
        loss_history: List[float],
        grad_norm_history: List[float],
        entropy_values: Dict[str, float],
        val_loss: float,
        train_loss: float
    ) -> Dict[str, bool]:
        """
        Check Gate A1: Training Stability (Steps 500-2500)

        PASS criteria:
        - Loss continues decreasing
        - Gradient norm stabilizes (not growing)
        - No sudden loss spikes (>3x moving average)
        - Entropy remains > 1.0 across all layers
        - Val loss tracks train loss (no immediate overfit)
        """
        results = {
            "gate": "A1",
            "step": step,
            "passed": True,
            "checks": {}
        }

        # Check loss still decreasing
        if len(loss_history) >= 20:
            recent = sum(loss_history[-10:]) / 10
            earlier = sum(loss_history[-20:-10]) / 10
            loss_decreasing = recent < earlier * 1.1  # Allow 10% noise
        else:
            loss_decreasing = True

        results["checks"]["loss_decreasing"] = loss_decreasing
        if not loss_decreasing:
            results["passed"] = False

        # Check for loss spikes
        if len(loss_history) >= 10:
            moving_avg = sum(loss_history[-10:-1]) / 9
            latest = loss_history[-1]
            no_spike = latest < moving_avg * self.config.a1_max_loss_spike_ratio
        else:
            no_spike = True

        results["checks"]["no_loss_spike"] = no_spike
        if not no_spike:
            results["passed"] = False

        # Check gradient norm stability
        if len(grad_norm_history) >= 10:
            recent_grads = grad_norm_history[-5:]
            earlier_grads = grad_norm_history[-10:-5]
            grad_stable = max(recent_grads) < max(earlier_grads) * 2
        else:
            grad_stable = True

        results["checks"]["grad_stable"] = grad_stable
        if not grad_stable:
            results["passed"] = False

        # Check entropy
        min_entropy = min(entropy_values.values()) if entropy_values else 0
        entropy_ok = min_entropy > self.config.a1_min_entropy
        results["checks"]["entropy_ok"] = entropy_ok
        results["checks"]["min_entropy"] = min_entropy
        if not entropy_ok:
            results["passed"] = False

        # Check val/train tracking
        val_train_ratio = val_loss / (train_loss + 1e-8)
        val_tracks_train = val_train_ratio < 1.5  # Val within 1.5x of train
        results["checks"]["val_tracks_train"] = val_tracks_train
        results["checks"]["val_train_ratio"] = val_train_ratio
        if not val_tracks_train:
            results["passed"] = False

        self.gate_results["A1"] = results
        return results

    def check_a2(
        self,
        step: int,
        val_ppl: float,
        baseline_ppl: Optional[float] = None
    ) -> Dict[str, bool]:
        """
        Check Gate A2: Learning Validation (Steps 2500-10000)

        PASS criteria:
        - Val perplexity < 200 (TinyStories is easy)
        - Val perplexity within 1.5x of baseline at same step
        """
        results = {
            "gate": "A2",
            "step": step,
            "passed": True,
            "checks": {}
        }

        # Check absolute PPL
        ppl_ok = val_ppl < self.config.a2_max_val_ppl
        results["checks"]["ppl_ok"] = ppl_ok
        results["checks"]["val_ppl"] = val_ppl
        if not ppl_ok:
            results["passed"] = False

        # Check baseline ratio (if available)
        if baseline_ppl is not None:
            ratio = val_ppl / baseline_ppl
            ratio_ok = ratio < self.config.a2_max_baseline_ratio
            results["checks"]["baseline_ratio_ok"] = ratio_ok
            results["checks"]["baseline_ratio"] = ratio
            if not ratio_ok:
                results["passed"] = False

        self.gate_results["A2"] = results
        return results

    def check_a3(
        self,
        step: int,
        val_ppl: float,
        baseline_ppl: float,
        entropy_values: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Check Gate A3: Functional LM (Steps 10000-25000)

        PASS criteria:
        - Val perplexity < 50 (TinyStories)
        - Val perplexity within 1.1x of dense baseline (±10%)
        - Primitive entropy still healthy (> 1.0)
        """
        results = {
            "gate": "A3",
            "step": step,
            "passed": True,
            "checks": {}
        }

        # Check absolute PPL
        ppl_ok = val_ppl < self.config.a3_max_val_ppl
        results["checks"]["ppl_ok"] = ppl_ok
        results["checks"]["val_ppl"] = val_ppl
        if not ppl_ok:
            results["passed"] = False

        # Check baseline ratio
        ratio = val_ppl / baseline_ppl
        ratio_ok = ratio < self.config.a3_max_baseline_ratio
        results["checks"]["baseline_ratio_ok"] = ratio_ok
        results["checks"]["baseline_ratio"] = ratio
        results["checks"]["baseline_ppl"] = baseline_ppl
        if not ratio_ok:
            results["passed"] = False

        # Check entropy
        min_entropy = min(entropy_values.values()) if entropy_values else 0
        entropy_ok = min_entropy > self.config.a3_min_entropy
        results["checks"]["entropy_ok"] = entropy_ok
        results["checks"]["min_entropy"] = min_entropy
        if not entropy_ok:
            results["passed"] = False

        self.gate_results["A3"] = results
        return results

    def get_summary(self) -> str:
        """Get summary of all gate checks."""
        lines = ["=" * 50, "GATE CHECK SUMMARY", "=" * 50]

        for gate_name in ["A0", "A1", "A2", "A3"]:
            if gate_name in self.gate_results:
                result = self.gate_results[gate_name]
                status = "PASS" if result["passed"] else "FAIL"
                lines.append(f"\n{gate_name} @ step {result['step']}: {status}")
                for check, value in result["checks"].items():
                    lines.append(f"  - {check}: {value}")

        return "\n".join(lines)


class Logger:
    """Simple logger for training."""

    def __init__(self, output_dir: Path, name: str = "training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / f"{name}.log"
        self.name = name

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")

    def info(self, message: str):
        self.log(message, "INFO")

    def warning(self, message: str):
        self.log(message, "WARN")

    def error(self, message: str):
        self.log(message, "ERROR")

    def metric(self, step: int, metrics: Dict[str, float]):
        """Log metrics in a compact format."""
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self.log(" | ".join(parts), "METRIC")


if __name__ == "__main__":
    print("Testing metrics module...")

    # Test entropy computation
    print("\n--- Entropy Tests ---")
    uniform_logits = torch.zeros(32)  # Uniform distribution
    print(f"Uniform (32) entropy: {compute_entropy(uniform_logits):.3f} (max=3.47)")

    peaked_logits = torch.zeros(32)
    peaked_logits[0] = 10.0  # One dominant
    print(f"Peaked entropy: {compute_entropy(peaked_logits):.3f}")

    # Test TrainingMetrics
    print("\n--- TrainingMetrics Tests ---")
    metrics = TrainingMetrics()
    for i in range(100):
        metrics.add_train_loss(i, 5.0 - i * 0.04)
        if i % 10 == 0:
            metrics.add_val_loss(i, 5.2 - i * 0.04)
            metrics.add_entropy(i, "layer_0_fc1", 2.5 + 0.01 * i)

    print(f"Latest train loss: {metrics.get_latest('train_loss'):.3f}")
    print(f"Average train loss (last 10): {metrics.get_average('train_loss'):.3f}")

    # Test GateChecker
    print("\n--- GateChecker Tests ---")
    checker = GateChecker()

    # Simulate A0 check
    loss_history = [5.0 - i * 0.01 for i in range(100)]
    entropy_values = {"layer_0_fc1": 2.5, "layer_0_fc2": 2.6}
    result = checker.check_a0(
        step=500,
        loss_history=loss_history,
        grad_norm=50.0,
        entropy_values=entropy_values
    )
    print(f"A0 passed: {result['passed']}")

    print(checker.get_summary())
