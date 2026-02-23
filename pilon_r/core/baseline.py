"""
PILON-R Dense Baseline Model

This module provides the dense baseline model for comparison.
The baseline is MANDATORY for valid Phase A results.

Key principle: Everything identical EXCEPT FFN implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .config import ModelConfig, TrainingConfig
from .model import PILONTransformer


def create_baseline_config(pilon_config: ModelConfig) -> ModelConfig:
    """
    Create a baseline config matching the PILON config.

    The baseline uses standard FFN instead of compositional FFN.
    Everything else is IDENTICAL:
    - Same d_model, n_layers, n_heads
    - Same vocab_size, max_seq_len
    - Same dropout, norm_type
    - Same tokenizer (implicit)

    Args:
        pilon_config: The PILON model config

    Returns:
        Baseline model config with ffn_type="standard"
    """
    return pilon_config.get_baseline_config()


def create_baseline_model(pilon_config: ModelConfig) -> PILONTransformer:
    """
    Create a dense baseline model for comparison.

    This is the MANDATORY baseline that every PILON experiment
    must compare against.

    Args:
        pilon_config: The PILON model config (will be converted to standard FFN)

    Returns:
        PILONTransformer with standard FFN
    """
    baseline_config = create_baseline_config(pilon_config)
    return PILONTransformer(baseline_config)


def compare_parameter_counts(
    pilon_model: PILONTransformer,
    baseline_model: PILONTransformer
) -> Dict[str, Dict[str, int]]:
    """
    Compare parameter counts between PILON and baseline.

    Returns breakdown by component.
    """
    pilon_counts = pilon_model.parameter_count()
    baseline_counts = baseline_model.parameter_count()

    return {
        "pilon": pilon_counts,
        "baseline": baseline_counts,
        "difference": {
            k: pilon_counts.get(k, 0) - baseline_counts.get(k, 0)
            for k in set(pilon_counts.keys()) | set(baseline_counts.keys())
        }
    }


def compute_comparison_metrics(
    pilon_loss: float,
    baseline_loss: float,
    pilon_ppl: float,
    baseline_ppl: float
) -> Dict[str, float]:
    """
    Compute comparison metrics between PILON and baseline.

    Returns ratios and differences for reporting.
    """
    return {
        "loss_ratio": pilon_loss / (baseline_loss + 1e-8),
        "loss_diff": pilon_loss - baseline_loss,
        "ppl_ratio": pilon_ppl / (baseline_ppl + 1e-8),
        "ppl_diff": pilon_ppl - baseline_ppl,
    }


def check_gate_thresholds(
    step: int,
    pilon_ppl: float,
    baseline_ppl: float,
    pilon_loss: float,
    baseline_loss: float
) -> Dict[str, bool]:
    """
    Check if current metrics pass gate thresholds.

    From Phase A plan:
    - 1K steps: loss ratio < 1.2x acceptable, < 1.5x concerning, > 1.5x fail
    - 5K steps: PPL ratio < 1.3x acceptable
    - 10K steps: PPL ratio < 1.2x acceptable
    - 25K steps: PPL ratio < 1.1x acceptable (Gate A3)
    """
    loss_ratio = pilon_loss / (baseline_loss + 1e-8)
    ppl_ratio = pilon_ppl / (baseline_ppl + 1e-8)

    checks = {}

    if step <= 1000:
        checks["loss_acceptable"] = loss_ratio < 1.2
        checks["loss_concerning"] = loss_ratio >= 1.2 and loss_ratio < 1.5
        checks["loss_fail"] = loss_ratio >= 1.5
    elif step <= 5000:
        checks["ppl_acceptable"] = ppl_ratio < 1.3
        checks["ppl_concerning"] = ppl_ratio >= 1.3 and ppl_ratio < 1.5
        checks["ppl_fail"] = ppl_ratio >= 1.5
    elif step <= 10000:
        checks["ppl_acceptable"] = ppl_ratio < 1.2
        checks["ppl_concerning"] = ppl_ratio >= 1.2 and ppl_ratio < 1.3
        checks["ppl_fail"] = ppl_ratio >= 1.3
    else:  # step > 10000 (Gate A3)
        checks["ppl_acceptable"] = ppl_ratio < 1.1
        checks["ppl_concerning"] = ppl_ratio >= 1.1 and ppl_ratio < 1.2
        checks["ppl_fail"] = ppl_ratio >= 1.2

    return checks


class BaselineTracker:
    """
    Tracks baseline metrics alongside PILON metrics.

    Used to ensure valid comparison throughout training.
    """

    def __init__(self):
        self.pilon_metrics = []
        self.baseline_metrics = []
        self.comparison_metrics = []

    def log(
        self,
        step: int,
        pilon_loss: float,
        baseline_loss: float,
        pilon_ppl: Optional[float] = None,
        baseline_ppl: Optional[float] = None
    ):
        """Log metrics at a training step."""
        pilon_ppl = pilon_ppl or torch.exp(torch.tensor(pilon_loss)).item()
        baseline_ppl = baseline_ppl or torch.exp(torch.tensor(baseline_loss)).item()

        self.pilon_metrics.append({
            "step": step,
            "loss": pilon_loss,
            "ppl": pilon_ppl
        })

        self.baseline_metrics.append({
            "step": step,
            "loss": baseline_loss,
            "ppl": baseline_ppl
        })

        comparison = compute_comparison_metrics(
            pilon_loss, baseline_loss, pilon_ppl, baseline_ppl
        )
        comparison["step"] = step

        # Add gate check
        gate_checks = check_gate_thresholds(
            step, pilon_ppl, baseline_ppl, pilon_loss, baseline_loss
        )
        comparison.update(gate_checks)

        self.comparison_metrics.append(comparison)

    def get_latest_comparison(self) -> Optional[Dict]:
        """Get most recent comparison metrics."""
        if self.comparison_metrics:
            return self.comparison_metrics[-1]
        return None

    def get_summary(self) -> Dict:
        """Get summary of all tracked metrics."""
        if not self.comparison_metrics:
            return {}

        loss_ratios = [m["loss_ratio"] for m in self.comparison_metrics]
        ppl_ratios = [m["ppl_ratio"] for m in self.comparison_metrics]

        return {
            "n_checkpoints": len(self.comparison_metrics),
            "avg_loss_ratio": sum(loss_ratios) / len(loss_ratios),
            "avg_ppl_ratio": sum(ppl_ratios) / len(ppl_ratios),
            "min_loss_ratio": min(loss_ratios),
            "max_loss_ratio": max(loss_ratios),
            "final_loss_ratio": loss_ratios[-1],
            "final_ppl_ratio": ppl_ratios[-1],
        }


def print_comparison_table(tracker: BaselineTracker):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("PILON vs Baseline Comparison")
    print("=" * 70)
    print(f"{'Step':>8} | {'PILON Loss':>12} | {'Base Loss':>12} | {'Ratio':>8} | {'Status':>12}")
    print("-" * 70)

    for pm, bm, cm in zip(tracker.pilon_metrics, tracker.baseline_metrics, tracker.comparison_metrics):
        status = "OK" if cm.get("loss_acceptable") or cm.get("ppl_acceptable") else "CONCERN" if cm.get("loss_concerning") or cm.get("ppl_concerning") else "FAIL"
        print(f"{pm['step']:>8} | {pm['loss']:>12.4f} | {bm['loss']:>12.4f} | {cm['loss_ratio']:>8.3f} | {status:>12}")

    print("=" * 70)

    summary = tracker.get_summary()
    print(f"\nSummary:")
    print(f"  Final loss ratio: {summary.get('final_loss_ratio', 'N/A'):.3f}")
    print(f"  Final PPL ratio: {summary.get('final_ppl_ratio', 'N/A'):.3f}")


if __name__ == "__main__":
    # Test baseline creation
    print("Testing baseline model creation...")

    from .config import ModelConfig

    # Create PILON config
    pilon_config = ModelConfig()
    print(f"PILON config: ffn_type={pilon_config.ffn_type}")

    # Create baseline config
    baseline_config = create_baseline_config(pilon_config)
    print(f"Baseline config: ffn_type={baseline_config.ffn_type}")

    # Create models
    pilon_model = PILONTransformer(pilon_config)
    baseline_model = create_baseline_model(pilon_config)

    # Compare parameters
    comparison = compare_parameter_counts(pilon_model, baseline_model)
    print(f"\nParameter comparison:")
    print(f"  PILON total: {comparison['pilon']['total']:,}")
    print(f"  Baseline total: {comparison['baseline']['total']:,}")
    print(f"  Difference: {comparison['difference']['total']:,}")

    # Test tracker
    print("\nTesting BaselineTracker...")
    tracker = BaselineTracker()
    tracker.log(100, 5.5, 5.4)
    tracker.log(500, 4.2, 4.1)
    tracker.log(1000, 3.5, 3.4)

    print_comparison_table(tracker)
