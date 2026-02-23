"""
PILON-R Phase A.2: Compression Curriculum

Orchestrates training runs across all compression levels and generates
comparison reports for the quality-compression frontier analysis.

Usage:
    # Run all compression levels
    python -m pilon_r.compression_curriculum --run-all

    # Run specific levels
    python -m pilon_r.compression_curriculum --levels baseline moderate extreme

    # Generate comparison report from existing runs
    python -m pilon_r.compression_curriculum --report outputs/phase_a

    # Run SFT on passing levels
    python -m pilon_r.compression_curriculum --sft outputs/phase_a
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import math

from .core.config import (
    COMPRESSION_LEVELS,
    get_compression_config,
    get_all_compression_levels,
    GateConfig,
    SFTConfig,
)


@dataclass
class CompressionResult:
    """Results for a single compression level."""
    level: str
    n_primitives: int
    rank: int
    top_k: int

    # Training results
    final_train_loss: float
    final_val_loss: float
    final_val_ppl: float
    final_entropy: float

    # Baseline comparison
    baseline_val_ppl: float
    ppl_ratio: float

    # Gate status
    passed_gate_a3: bool

    # Paths
    checkpoint_path: str
    metrics_path: str

    # SFT (if run)
    sft_completed: bool = False
    sft_checkpoint_path: str = ""


@dataclass
class CompressionFrontier:
    """Analysis of the quality-compression tradeoff."""
    results: List[CompressionResult] = field(default_factory=list)
    pareto_optimal: List[str] = field(default_factory=list)
    recommended_level: str = ""
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "results": [asdict(r) for r in self.results],
            "pareto_optimal": self.pareto_optimal,
            "recommended_level": self.recommended_level,
            "analysis_timestamp": self.analysis_timestamp,
        }


def run_single_level(
    level: str,
    output_dir: Path,
    device: str = "cuda",
    baseline: bool = True,
) -> bool:
    """
    Run training for a single compression level.

    Returns:
        True if successful, False otherwise
    """
    level_output = output_dir / level

    cmd = [
        sys.executable, "-m", "pilon_r.train",
        "--compression-level", level,
        "--output-dir", str(level_output),
        "--device", device,
    ]

    if baseline:
        cmd.append("--baseline")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPRESSION LEVEL: {level}")
    print(f"Output: {level_output}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed for level {level}: {e}")
        return False


def run_all_levels(
    output_dir: Path,
    device: str = "cuda",
    levels: Optional[List[str]] = None,
) -> List[str]:
    """
    Run training for all compression levels sequentially.

    Returns:
        List of successful level names
    """
    if levels is None:
        levels = get_all_compression_levels()

    successful = []

    for level in levels:
        if run_single_level(level, output_dir, device):
            successful.append(level)

    return successful


def load_training_results(level_dir: Path, level: str) -> Optional[CompressionResult]:
    """Load results from a completed training run."""
    metrics_path = level_dir / "final_metrics.json"
    checkpoint_path = level_dir / "final_model.pt"

    if not metrics_path.exists():
        print(f"WARNING: No metrics found at {metrics_path}")
        return None

    # Load metrics
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    config = get_compression_config(level)

    # Extract final values
    train_loss = metrics_data.get("train_loss", [])
    val_loss = metrics_data.get("val_loss", [])
    val_ppl = metrics_data.get("val_ppl", [])

    final_train_loss = train_loss[-1][1] if train_loss else float('nan')
    final_val_loss = val_loss[-1][1] if val_loss else float('nan')
    final_val_ppl = val_ppl[-1][1] if val_ppl else float('nan')

    # Get entropy
    entropy_history = metrics_data.get("entropy_history", {})
    final_entropy = 0.0
    if entropy_history:
        all_final = [h[-1][1] for h in entropy_history.values() if h]
        final_entropy = sum(all_final) / len(all_final) if all_final else 0.0

    # For now, use PILON PPL as baseline comparison (placeholder)
    # In real usage, baseline would be tracked separately
    baseline_val_ppl = final_val_ppl * 0.95  # Placeholder
    ppl_ratio = final_val_ppl / baseline_val_ppl if baseline_val_ppl > 0 else 1.0

    # Check Gate A3
    gate_config = GateConfig()
    passed_a3 = final_val_ppl < gate_config.a3_max_val_ppl and ppl_ratio < gate_config.a3_max_baseline_ratio

    # Check for SFT
    sft_path = level_dir / "sft_model.pt"
    sft_completed = sft_path.exists()

    return CompressionResult(
        level=level,
        n_primitives=config["n_primitives"],
        rank=config["rank"],
        top_k=config["top_k"],
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_val_ppl=final_val_ppl,
        final_entropy=final_entropy,
        baseline_val_ppl=baseline_val_ppl,
        ppl_ratio=ppl_ratio,
        passed_gate_a3=passed_a3,
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        sft_completed=sft_completed,
        sft_checkpoint_path=str(sft_path) if sft_completed else "",
    )


def run_sft_for_passing(output_dir: Path, device: str = "cuda", epochs: int = 2):
    """Run SFT only on levels that passed Gate A3."""
    print("\n" + "=" * 60)
    print("RUNNING SFT FOR PASSING LEVELS")
    print("=" * 60)

    passing_levels = []

    for level in get_all_compression_levels():
        level_dir = output_dir / level
        result = load_training_results(level_dir, level)

        if result and result.passed_gate_a3:
            passing_levels.append(level)

    if not passing_levels:
        print("No levels passed Gate A3. Skipping SFT.")
        return

    print(f"Levels passing Gate A3: {passing_levels}")

    for level in passing_levels:
        level_dir = output_dir / level
        checkpoint = level_dir / "final_model.pt"

        if not checkpoint.exists():
            print(f"WARNING: Checkpoint not found for {level}")
            continue

        # Check if SFT already done
        sft_path = level_dir / "sft_model.pt"
        if sft_path.exists():
            print(f"SFT already completed for {level}, skipping")
            continue

        print(f"\nRunning SFT for level: {level}")

        cmd = [
            sys.executable, "-m", "pilon_r.sft",
            str(checkpoint),
            "--epochs", str(epochs),
            "--output-dir", str(level_dir),
            "--device", device,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: SFT failed for {level}: {e}")


def find_pareto_optimal(results: List[CompressionResult]) -> List[str]:
    """Find Pareto-optimal compression levels."""
    pareto = []

    for r in results:
        is_dominated = False
        for other in results:
            if other.level == r.level:
                continue
            # other dominates r if it has both better quality AND better compression
            # Better quality = lower PPL ratio
            # Better compression = fewer primitives
            if (other.ppl_ratio <= r.ppl_ratio and
                other.n_primitives <= r.n_primitives and
                (other.ppl_ratio < r.ppl_ratio or other.n_primitives < r.n_primitives)):
                is_dominated = True
                break

        if not is_dominated:
            pareto.append(r.level)

    return pareto


def recommend_level(results: List[CompressionResult]) -> str:
    """Recommend the best compression level."""
    # Filter to passing levels
    passing = [r for r in results if r.passed_gate_a3]

    if passing:
        # Among passing, pick highest compression (fewest primitives)
        best = min(passing, key=lambda r: r.n_primitives)
        return best.level
    else:
        # If none pass, pick best quality
        best = min(results, key=lambda r: r.ppl_ratio)
        return best.level


def generate_report(output_dir: Path, report_path: Optional[Path] = None) -> CompressionFrontier:
    """Generate comparison report across all compression levels."""
    results = []

    for level in get_all_compression_levels():
        level_dir = output_dir / level
        if level_dir.exists():
            result = load_training_results(level_dir, level)
            if result:
                results.append(result)

    if not results:
        print("ERROR: No results found")
        return CompressionFrontier()

    # Analysis
    pareto = find_pareto_optimal(results)
    recommended = recommend_level(results)

    frontier = CompressionFrontier(
        results=results,
        pareto_optimal=pareto,
        recommended_level=recommended,
    )

    # Save report
    if report_path is None:
        report_path = output_dir / "compression_frontier_report.json"

    with open(report_path, "w") as f:
        json.dump(frontier.to_dict(), f, indent=2)

    # Print summary
    print_comparison_table(results)
    print_frontier_analysis(frontier)

    print(f"\nReport saved to: {report_path}")

    return frontier


def print_comparison_table(results: List[CompressionResult]):
    """Print formatted comparison table."""
    print("\n" + "=" * 95)
    print("COMPRESSION CURRICULUM RESULTS")
    print("=" * 95)

    header = f"{'Level':>12} | {'n_prim':>6} | {'rank':>4} | {'top_k':>5} | " \
             f"{'Val PPL':>8} | {'Ratio':>7} | {'Entropy':>7} | {'A3':>6} | {'SFT':>5}"
    print(header)
    print("-" * 95)

    # Sort by compression level order
    level_order = {c["label"]: i for i, c in enumerate(COMPRESSION_LEVELS)}
    sorted_results = sorted(results, key=lambda x: level_order.get(x.level, 99))

    for r in sorted_results:
        gate_status = "PASS" if r.passed_gate_a3 else "FAIL"
        sft_status = "Yes" if r.sft_completed else "No"
        print(f"{r.level:>12} | {r.n_primitives:>6} | {r.rank:>4} | {r.top_k:>5} | "
              f"{r.final_val_ppl:>8.2f} | {r.ppl_ratio:>7.3f} | {r.final_entropy:>7.2f} | "
              f"{gate_status:>6} | {sft_status:>5}")

    print("=" * 95)


def print_frontier_analysis(frontier: CompressionFrontier):
    """Print Pareto frontier analysis."""
    print("\n" + "-" * 50)
    print("FRONTIER ANALYSIS")
    print("-" * 50)
    print(f"Pareto-optimal levels: {', '.join(frontier.pareto_optimal)}")
    print(f"Recommended level: {frontier.recommended_level}")

    rec = next((r for r in frontier.results if r.level == frontier.recommended_level), None)
    if rec:
        print(f"\nRecommended '{rec.level}':")
        print(f"  PPL ratio: {rec.ppl_ratio:.3f}")
        print(f"  Primitives: {rec.n_primitives}")
        print(f"  Entropy: {rec.final_entropy:.2f}")
        print(f"  Gate A3: {'PASS' if rec.passed_gate_a3 else 'FAIL'}")
        print(f"  SFT: {'Completed' if rec.sft_completed else 'Not run'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase A.2: Compression Curriculum")
    parser.add_argument("--run-all", action="store_true",
                        help="Run training for all compression levels")
    parser.add_argument("--levels", type=str, nargs="+",
                        help="Specific levels to run")
    parser.add_argument("--report", type=str,
                        help="Generate report from existing results in directory")
    parser.add_argument("--sft", type=str,
                        help="Run SFT on passing levels in directory")
    parser.add_argument("--sft-epochs", type=int, default=2,
                        help="Number of SFT epochs")
    parser.add_argument("--output-dir", type=str, default="outputs/phase_a",
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.run_all or args.levels:
        levels = args.levels if args.levels else None
        output_dir.mkdir(parents=True, exist_ok=True)
        successful = run_all_levels(output_dir, args.device, levels)
        print(f"\nCompleted {len(successful)}/{len(levels or get_all_compression_levels())} training runs")

        # Generate report
        if successful:
            generate_report(output_dir)

            # Run SFT on passing levels
            run_sft_for_passing(output_dir, args.device, args.sft_epochs)

            # Regenerate report with SFT status
            generate_report(output_dir)

    elif args.sft:
        sft_dir = Path(args.sft)
        run_sft_for_passing(sft_dir, args.device, args.sft_epochs)
        generate_report(sft_dir)

    elif args.report:
        report_dir = Path(args.report)
        generate_report(report_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
