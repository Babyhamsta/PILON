"""
PILON-R Entropy Analysis

Visualizes entropy over training to detect:
- Early collapse (entropy drops early)
- Late collapse (entropy drops after initial training)
- Healthy training (entropy remains stable)

Healthy entropy should be > 1.0 throughout training.
For 32 primitives, max entropy = log(32) ≈ 3.47
"""

import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def plot_entropy_over_time(
    metrics: Dict,
    output_path: Optional[Path] = None,
    title: str = "Primitive Entropy Over Training"
):
    """
    Plot entropy for all layers over training steps.

    Args:
        metrics: Loaded metrics dictionary
        output_path: Path to save plot (None to display)
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    entropy_history = metrics.get("entropy_history", {})
    if not entropy_history:
        print("No entropy data found in metrics")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot fc1 entropy
    ax1 = axes[0]
    for layer_name, points in entropy_history.items():
        if "fc1" in layer_name:
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            ax1.plot(steps, values, label=layer_name, alpha=0.7)

    ax1.axhline(y=1.0, color='r', linestyle='--', label='Min threshold (1.0)')
    ax1.axhline(y=3.47, color='g', linestyle='--', label='Max (uniform, 32 prims)', alpha=0.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Entropy")
    ax1.set_title("FC1 Layer Entropy")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot fc2 entropy
    ax2 = axes[1]
    for layer_name, points in entropy_history.items():
        if "fc2" in layer_name:
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            ax2.plot(steps, values, label=layer_name, alpha=0.7)

    ax2.axhline(y=1.0, color='r', linestyle='--', label='Min threshold (1.0)')
    ax2.axhline(y=3.47, color='g', linestyle='--', label='Max (uniform, 32 prims)', alpha=0.5)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Entropy")
    ax2.set_title("FC2 Layer Entropy")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved entropy plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_entropy_summary(
    metrics: Dict,
    output_path: Optional[Path] = None
):
    """
    Plot summary statistics of entropy.

    Shows:
    - Average entropy per layer (final)
    - Min/max entropy range
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    entropy_history = metrics.get("entropy_history", {})
    if not entropy_history:
        print("No entropy data found")
        return

    # Get final entropy values
    fc1_layers = []
    fc1_values = []
    fc2_layers = []
    fc2_values = []

    for layer_name, points in entropy_history.items():
        if points:
            final_value = points[-1][1]
            if "fc1" in layer_name:
                fc1_layers.append(layer_name.replace("_fc1", "").replace("layer_", "L"))
                fc1_values.append(final_value)
            else:
                fc2_layers.append(layer_name.replace("_fc2", "").replace("layer_", "L"))
                fc2_values.append(final_value)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fc1_layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, fc1_values, width, label='FC1', color='steelblue')
    bars2 = ax.bar(x + width/2, fc2_values, width, label='FC2', color='coral')

    ax.axhline(y=1.0, color='r', linestyle='--', label='Min threshold')
    ax.axhline(y=3.47, color='g', linestyle='--', label='Max (uniform)', alpha=0.5)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Final Entropy')
    ax.set_title('Final Entropy by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(fc1_layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved entropy summary to: {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_entropy_health(metrics: Dict) -> Dict:
    """
    Analyze entropy health and return diagnostics.

    Returns warnings and status for each layer.
    """
    entropy_history = metrics.get("entropy_history", {})
    results = {
        "overall_status": "healthy",
        "layers": {},
        "warnings": []
    }

    min_threshold = 1.0

    for layer_name, points in entropy_history.items():
        if not points:
            continue

        values = [p[1] for p in points]
        steps = [p[0] for p in points]

        layer_status = {
            "final_entropy": values[-1],
            "min_entropy": min(values),
            "max_entropy": max(values),
            "status": "healthy"
        }

        # Check for collapse
        if values[-1] < min_threshold:
            layer_status["status"] = "collapsed"
            results["warnings"].append(f"{layer_name}: entropy collapsed to {values[-1]:.2f}")
            results["overall_status"] = "unhealthy"

        # Check for early collapse
        early_values = values[:len(values)//3] if len(values) > 3 else values
        if min(early_values) < min_threshold:
            layer_status["early_collapse"] = True
            results["warnings"].append(f"{layer_name}: early collapse detected")

        # Check for declining trend
        if len(values) >= 5:
            first_half = sum(values[:len(values)//2]) / (len(values)//2)
            second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            if second_half < first_half * 0.8:
                layer_status["declining_trend"] = True
                results["warnings"].append(f"{layer_name}: entropy declining over training")

        results["layers"][layer_name] = layer_status

    return results


def print_entropy_report(metrics: Dict):
    """Print a text-based entropy report."""
    analysis = analyze_entropy_health(metrics)

    print("\n" + "=" * 60)
    print("ENTROPY HEALTH REPORT")
    print("=" * 60)

    print(f"\nOverall Status: {analysis['overall_status'].upper()}")

    if analysis["warnings"]:
        print("\nWarnings:")
        for warning in analysis["warnings"]:
            print(f"  - {warning}")

    print("\nPer-Layer Summary:")
    print(f"{'Layer':<20} {'Final':>8} {'Min':>8} {'Max':>8} {'Status':>12}")
    print("-" * 60)

    for layer_name, stats in analysis["layers"].items():
        print(f"{layer_name:<20} {stats['final_entropy']:>8.2f} {stats['min_entropy']:>8.2f} {stats['max_entropy']:>8.2f} {stats['status']:>12}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze entropy from training metrics")
    parser.add_argument("metrics", type=str, help="Path to metrics JSON file")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print report
    print_entropy_report(metrics)

    # Generate plots
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_entropy_over_time(
            metrics,
            output_path=output_dir / "entropy_over_time.png"
        )
        plot_entropy_summary(
            metrics,
            output_path=output_dir / "entropy_summary.png"
        )


if __name__ == "__main__":
    main()
