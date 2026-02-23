"""
PILON-R Primitive Usage Heatmaps

Visualizes which primitives are used by each layer.
Helps identify:
- Primitive specialization (different layers use different primitives)
- Primitive collapse (all layers use same few primitives)
- Unused primitives (some primitives never selected)
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
    print("Warning: matplotlib not installed")


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def plot_usage_heatmap(
    usage_snapshot: Dict,
    output_path: Optional[Path] = None,
    title: str = "Primitive Usage Heatmap"
):
    """
    Plot heatmap of primitive usage across layers.

    Args:
        usage_snapshot: Dictionary with layer usage weights
        output_path: Path to save plot
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not installed")
        return

    usage = usage_snapshot.get("usage", {})
    if not usage:
        print("No usage data found")
        return

    # Separate fc1 and fc2
    fc1_data = {}
    fc2_data = {}

    for layer_name, weights in usage.items():
        if "fc1" in layer_name:
            layer_num = layer_name.replace("layer_", "").replace("_fc1", "")
            fc1_data[f"L{layer_num}"] = weights
        else:
            layer_num = layer_name.replace("layer_", "").replace("_fc2", "")
            fc2_data[f"L{layer_num}"] = weights

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # FC1 heatmap
    if fc1_data:
        layers = sorted(fc1_data.keys())
        matrix = np.array([fc1_data[l] for l in layers])

        ax1 = axes[0]
        im1 = ax1.imshow(matrix, aspect='auto', cmap='viridis')
        ax1.set_xlabel("Primitive Index")
        ax1.set_ylabel("Layer")
        ax1.set_yticks(range(len(layers)))
        ax1.set_yticklabels(layers)
        ax1.set_title("FC1 Primitive Usage")
        plt.colorbar(im1, ax=ax1, label="Weight")

    # FC2 heatmap
    if fc2_data:
        layers = sorted(fc2_data.keys())
        matrix = np.array([fc2_data[l] for l in layers])

        ax2 = axes[1]
        im2 = ax2.imshow(matrix, aspect='auto', cmap='viridis')
        ax2.set_xlabel("Primitive Index")
        ax2.set_ylabel("Layer")
        ax2.set_yticks(range(len(layers)))
        ax2.set_yticklabels(layers)
        ax2.set_title("FC2 Primitive Usage")
        plt.colorbar(im2, ax=ax2, label="Weight")

    plt.suptitle(f"{title} (Step {usage_snapshot.get('step', '?')})")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_usage_evolution(
    metrics: Dict,
    output_path: Optional[Path] = None
):
    """
    Plot how primitive usage evolves over training.

    Shows multiple snapshots side by side.
    """
    if not HAS_MATPLOTLIB:
        return

    snapshots = metrics.get("usage_snapshots", [])
    if len(snapshots) < 2:
        print("Not enough snapshots for evolution plot")
        return

    # Select up to 4 snapshots evenly spaced
    n_plots = min(4, len(snapshots))
    indices = [int(i * (len(snapshots) - 1) / (n_plots - 1)) for i in range(n_plots)]
    selected = [snapshots[i] for i in indices]

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, snapshot in zip(axes, selected):
        usage = snapshot.get("usage", {})

        # Use fc1 for this visualization
        fc1_data = {k: v for k, v in usage.items() if "fc1" in k}
        if not fc1_data:
            continue

        layers = sorted(fc1_data.keys())
        matrix = np.array([fc1_data[l] for l in layers])

        im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        ax.set_xlabel("Primitive")
        ax.set_ylabel("Layer")
        ax.set_title(f"Step {snapshot.get('step', '?')}")

    plt.suptitle("FC1 Primitive Usage Evolution")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved evolution plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_primitive_usage(metrics: Dict) -> Dict:
    """
    Analyze primitive usage patterns.

    Returns statistics about:
    - Most/least used primitives
    - Cross-layer correlation
    - Specialization score
    """
    snapshots = metrics.get("usage_snapshots", [])
    if not snapshots:
        return {"error": "No usage snapshots"}

    # Use final snapshot
    final = snapshots[-1]
    usage = final.get("usage", {})

    results = {
        "step": final.get("step"),
        "fc1_analysis": {},
        "fc2_analysis": {}
    }

    for prefix, analysis_key in [("fc1", "fc1_analysis"), ("fc2", "fc2_analysis")]:
        layer_weights = {k: np.array(v) for k, v in usage.items() if prefix in k}

        if not layer_weights:
            continue

        # Stack into matrix (layers x primitives)
        layers = sorted(layer_weights.keys())
        matrix = np.array([layer_weights[l] for l in layers])
        n_layers, n_primitives = matrix.shape

        # Per-primitive statistics
        primitive_avg = matrix.mean(axis=0)
        primitive_std = matrix.std(axis=0)

        # Find most/least used
        most_used = int(np.argmax(primitive_avg))
        least_used = int(np.argmin(primitive_avg))

        # Count "active" primitives (avg > 1/n_primitives)
        threshold = 1.0 / n_primitives
        n_active = int((primitive_avg > threshold * 0.5).sum())

        # Specialization: do layers use different primitives?
        # High specialization = low correlation between layers
        if n_layers > 1:
            correlations = np.corrcoef(matrix)
            avg_correlation = (correlations.sum() - n_layers) / (n_layers * (n_layers - 1))
        else:
            avg_correlation = 1.0

        # Diversity: entropy of average usage
        avg_probs = primitive_avg / primitive_avg.sum()
        diversity = -np.sum(avg_probs * np.log(avg_probs + 1e-10))

        results[analysis_key] = {
            "n_primitives": n_primitives,
            "n_layers": n_layers,
            "most_used_primitive": most_used,
            "most_used_weight": float(primitive_avg[most_used]),
            "least_used_primitive": least_used,
            "least_used_weight": float(primitive_avg[least_used]),
            "n_active_primitives": n_active,
            "avg_layer_correlation": float(avg_correlation),
            "diversity_entropy": float(diversity),
        }

    return results


def print_usage_report(metrics: Dict):
    """Print text-based usage report."""
    analysis = analyze_primitive_usage(metrics)

    print("\n" + "=" * 60)
    print("PRIMITIVE USAGE REPORT")
    print("=" * 60)

    print(f"\nSnapshot Step: {analysis.get('step', 'unknown')}")

    for key in ["fc1_analysis", "fc2_analysis"]:
        data = analysis.get(key, {})
        if not data:
            continue

        name = key.replace("_analysis", "").upper()
        print(f"\n{name} Analysis:")
        print(f"  Primitives: {data['n_primitives']}")
        print(f"  Active primitives: {data['n_active_primitives']}")
        print(f"  Most used: #{data['most_used_primitive']} (weight={data['most_used_weight']:.3f})")
        print(f"  Least used: #{data['least_used_primitive']} (weight={data['least_used_weight']:.4f})")
        print(f"  Layer correlation: {data['avg_layer_correlation']:.3f}")
        print(f"  Diversity entropy: {data['diversity_entropy']:.2f}")

        # Interpretation
        if data['n_active_primitives'] < data['n_primitives'] * 0.5:
            print(f"  WARNING: Only {data['n_active_primitives']}/{data['n_primitives']} primitives active")

        if data['avg_layer_correlation'] > 0.8:
            print(f"  WARNING: High layer correlation - layers using similar primitives")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze primitive usage heatmaps")
    parser.add_argument("metrics", type=str, help="Path to metrics JSON file")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print report
    print_usage_report(metrics)

    # Generate plots
    if not args.no_plot and HAS_MATPLOTLIB:
        # Final heatmap
        snapshots = metrics.get("usage_snapshots", [])
        if snapshots:
            plot_usage_heatmap(
                snapshots[-1],
                output_path=output_dir / "final_usage_heatmap.png",
                title="Final Primitive Usage"
            )

        # Evolution
        if len(snapshots) >= 2:
            plot_usage_evolution(
                metrics,
                output_path=output_dir / "usage_evolution.png"
            )


if __name__ == "__main__":
    main()
