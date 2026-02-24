"""
PILON-R Comprehensive Efficiency Benchmarking (Phase B.5e)

Produces reproducible benchmark results comparing PILON vs dense models:
- VRAMEfficiencyCurve: quality (val loss) vs VRAM at various scales
- ComputeEfficiency: quality vs estimated FLOPS
- InferenceProfiler: tok/s, VRAM, avg_layers_per_token, time-to-first-token
- QualityBenchmark: perplexity on held-out data

Usage:
    python -m pilon_r.benchmark_efficiency \
        --checkpoints ckpt1.pt ckpt2.pt \
        --labels "Dense-360M" "PILON-360M" \
        --output-dir benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .core.config import ModelConfig
from .core.model import PILONTransformer, create_model
from .core.data import get_tokenizer, load_text_dataset, create_dataloader
from .benchmark import load_model, benchmark_inference, benchmark_prefill, get_benchmark_autocast


# ============================================================================
# VRAM Efficiency Curve
# ============================================================================

@dataclass
class VRAMEfficiencyResult:
    """Result for a single checkpoint in the VRAM efficiency curve."""
    label: str
    model_params: int
    vram_allocated_gb: float
    vram_peak_gb: float
    val_loss: float
    val_ppl: float


def measure_vram_and_quality(
    checkpoint_path: str,
    device: str,
    val_loader,
    precision: str = "auto",
    max_val_batches: int = 50,
) -> VRAMEfficiencyResult:
    """
    Load a model, measure VRAM, and compute validation loss.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device
        val_loader: Validation dataloader
        precision: Precision for autocast
        max_val_batches: Max validation batches

    Returns:
        VRAMEfficiencyResult
    """
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, config = load_model(checkpoint_path, device)
    label = f"{config.ffn_type}"
    model_params = sum(p.numel() for p in model.parameters())

    vram_allocated = 0.0
    vram_peak = 0.0
    if device.startswith("cuda"):
        vram_allocated = torch.cuda.memory_allocated() / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9

    # Compute validation loss
    val_loss = 0.0
    n_batches = 0
    model.eval()
    with torch.inference_mode(), get_benchmark_autocast(device, precision):
        for batch in val_loader:
            if n_batches >= max_val_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(input_ids, labels=labels)
            val_loss += outputs["loss"].item()
            n_batches += 1

    if device.startswith("cuda"):
        vram_peak = max(vram_peak, torch.cuda.max_memory_allocated() / 1e9)

    avg_val = val_loss / max(n_batches, 1)
    val_ppl = math.exp(min(avg_val, 20.0))

    # Free memory
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return VRAMEfficiencyResult(
        label=label,
        model_params=model_params,
        vram_allocated_gb=vram_allocated,
        vram_peak_gb=vram_peak,
        val_loss=avg_val,
        val_ppl=val_ppl,
    )


# ============================================================================
# Compute Efficiency (FLOPS estimation)
# ============================================================================

@dataclass
class ComputeEfficiencyResult:
    """Result for compute efficiency measurement."""
    label: str
    model_params: int
    estimated_flops_per_token: float
    val_loss: float
    val_ppl: float
    flops_description: str


def estimate_flops_per_token(config: ModelConfig) -> Tuple[float, str]:
    """
    Estimate FLOPS per token for a model configuration.

    Dense: ~6 * total_params per token (standard transformer estimate)
    PILON: 6 * (attn_params + top_k/n_prims * prim_params) per token

    Returns:
        (flops_per_token, description)
    """
    d_model = config.d_model
    n_layers = config.n_layers
    d_ff = config.d_ff
    n_heads = config.n_heads
    d_head = config.d_head

    # Attention FLOPS per layer (Q, K, V projections + attention + output)
    attn_flops = 4 * d_model * n_heads * d_head  # QKV + output projections

    if config.ffn_type == "standard":
        ffn_flops = 2 * d_model * d_ff  # fc1 + fc2
        total_flops = n_layers * (attn_flops + ffn_flops)
        total_flops *= 6  # Standard 6x multiplier for forward + backward
        return total_flops, f"Dense: 6 * {n_layers} * (attn + ffn)"
    else:
        pc = config.primitive_config
        n_prims = pc.n_primitives
        top_k = pc.top_k
        rank = pc.rank

        # PILON FFN: sparse computation over top_k primitives
        # Each primitive: d_model*rank + rank*d_ff (fc1) or d_ff*rank + rank*d_model (fc2)
        fc1_flops_per_prim = d_model * rank + rank * d_ff
        fc2_flops_per_prim = d_ff * rank + rank * d_model
        ffn_flops = top_k * (fc1_flops_per_prim + fc2_flops_per_prim)

        total_flops = n_layers * (attn_flops + ffn_flops)
        total_flops *= 6
        return total_flops, f"PILON: 6 * {n_layers} * (attn + {top_k}/{n_prims} * prim)"


# ============================================================================
# Inference Profiler
# ============================================================================

@dataclass
class InferenceProfileResult:
    """Comprehensive inference profiling result."""
    label: str
    tokens_per_sec: float
    vram_peak_gb: float
    avg_layers_per_token: Optional[float]
    time_to_first_token_ms: float
    avg_latency_ms: float
    latency_per_token_ms: float
    skip_ratios: Optional[Dict[str, float]]


def profile_inference(
    checkpoint_path: str,
    device: str,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 100,
    num_warmup: int = 5,
    num_runs: int = 50,
    precision: str = "auto",
) -> InferenceProfileResult:
    """
    Profile inference including early exit metrics and time-to-first-token.
    """
    model, config = load_model(checkpoint_path, device)
    label = config.ffn_type

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)

    # Warmup
    for _ in range(num_warmup):
        with torch.inference_mode(), get_benchmark_autocast(device, precision):
            model.generate(input_tensor, max_new_tokens=10, do_sample=False)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Time to first token
    ttft_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.inference_mode(), get_benchmark_autocast(device, precision):
            model.generate(input_tensor, max_new_tokens=1, do_sample=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - start) * 1000)

    # Full generation benchmark
    latencies = []
    total_tokens = 0
    gen_start = time.perf_counter()

    for _ in range(num_runs):
        iter_start = time.perf_counter()
        with torch.inference_mode(), get_benchmark_autocast(device, precision):
            generated = model.generate(
                input_tensor, max_new_tokens=max_new_tokens, do_sample=False
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - iter_start)
        total_tokens += generated.size(1) - len(input_ids)

    total_time = time.perf_counter() - gen_start

    # Early exit metrics
    exit_metrics = (
        model.get_early_exit_metrics()
        if hasattr(model, "get_early_exit_metrics")
        else None
    )

    vram_peak = 0.0
    if device.startswith("cuda"):
        vram_peak = torch.cuda.max_memory_allocated() / 1e9

    avg_layers = None
    skip_ratios = None
    if exit_metrics is not None:
        avg_layers = exit_metrics["avg_layers_per_token"]
        skip_ratios = exit_metrics["skip_ratios"]

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return InferenceProfileResult(
        label=label,
        tokens_per_sec=total_tokens / total_time,
        vram_peak_gb=vram_peak,
        avg_layers_per_token=avg_layers,
        time_to_first_token_ms=sum(ttft_times) / len(ttft_times),
        avg_latency_ms=(sum(latencies) / len(latencies)) * 1000,
        latency_per_token_ms=(sum(latencies) / len(latencies)) / max_new_tokens * 1000,
        skip_ratios=skip_ratios,
    )


# ============================================================================
# Quality Benchmark
# ============================================================================

@dataclass
class QualityResult:
    """Quality benchmark result."""
    label: str
    val_loss: float
    val_ppl: float
    n_batches: int


def benchmark_quality(
    checkpoint_path: str,
    device: str,
    val_loader,
    precision: str = "auto",
    max_batches: int = 100,
) -> QualityResult:
    """Compute validation perplexity on held-out data."""
    model, config = load_model(checkpoint_path, device)
    label = config.ffn_type

    val_loss = 0.0
    n_batches = 0
    model.eval()

    with torch.inference_mode(), get_benchmark_autocast(device, precision):
        for batch in val_loader:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(input_ids, labels=labels)
            val_loss += outputs["loss"].item()
            n_batches += 1

    avg_loss = val_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20.0))

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return QualityResult(label=label, val_loss=avg_loss, val_ppl=ppl, n_batches=n_batches)


# ============================================================================
# Full Benchmark Orchestrator
# ============================================================================

def run_full_benchmark(
    checkpoint_paths: List[str],
    labels: List[str],
    output_dir: str,
    device: str = "cuda",
    precision: str = "auto",
    tokenizer_path: Optional[str] = None,
    max_val_batches: int = 50,
) -> Dict:
    """
    Run all benchmarks on a set of checkpoints.

    Args:
        checkpoint_paths: List of checkpoint file paths
        labels: Human-readable labels for each checkpoint
        output_dir: Directory to save results
        device: Device to run on
        precision: Precision for autocast
        tokenizer_path: Optional custom tokenizer path
        max_val_batches: Max validation batches

    Returns:
        Complete benchmark results dictionary
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(tokenizer_path)

    # Create validation dataloader
    val_dataset = load_text_dataset(
        dataset_name="Elriggs/openwebtext-100k",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_tokens=2_000_000,
        split="validation",
        streaming=True,
    )
    val_loader = create_dataloader(val_dataset, batch_size=4, shuffle=False)

    results = {"checkpoints": {}}

    for ckpt_path, label in zip(checkpoint_paths, labels):
        print(f"\n{'='*60}")
        print(f"Benchmarking: {label} ({ckpt_path})")
        print(f"{'='*60}")

        ckpt_results = {"label": label, "path": ckpt_path}

        # 1. Quality
        print("  [1/4] Quality benchmark...")
        quality = benchmark_quality(ckpt_path, device, val_loader, precision, max_val_batches)
        ckpt_results["quality"] = {
            "val_loss": quality.val_loss,
            "val_ppl": quality.val_ppl,
        }

        # 2. VRAM efficiency
        print("  [2/4] VRAM efficiency...")
        vram = measure_vram_and_quality(ckpt_path, device, val_loader, precision, max_val_batches)
        ckpt_results["vram"] = {
            "allocated_gb": vram.vram_allocated_gb,
            "peak_gb": vram.vram_peak_gb,
            "model_params": vram.model_params,
        }

        # 3. Compute efficiency
        print("  [3/4] Compute efficiency estimation...")
        model_tmp, config_tmp = load_model(ckpt_path, "cpu")
        flops, flops_desc = estimate_flops_per_token(config_tmp)
        ckpt_results["compute"] = {
            "estimated_flops_per_token": flops,
            "description": flops_desc,
        }
        del model_tmp

        # 4. Inference profiling
        print("  [4/4] Inference profiling...")
        profile = profile_inference(
            ckpt_path, device, tokenizer,
            num_warmup=5, num_runs=30, precision=precision,
        )
        ckpt_results["inference"] = {
            "tokens_per_sec": profile.tokens_per_sec,
            "vram_peak_gb": profile.vram_peak_gb,
            "avg_layers_per_token": profile.avg_layers_per_token,
            "time_to_first_token_ms": profile.time_to_first_token_ms,
            "avg_latency_ms": profile.avg_latency_ms,
            "latency_per_token_ms": profile.latency_per_token_ms,
            "skip_ratios": profile.skip_ratios,
        }

        results["checkpoints"][label] = ckpt_results

    # Compute comparisons (if multiple checkpoints)
    if len(checkpoint_paths) >= 2:
        first_label = labels[0]
        comparisons = {}
        first = results["checkpoints"][first_label]

        for label in labels[1:]:
            other = results["checkpoints"][label]
            comp = {}

            # Quality ratio
            if first["quality"]["val_loss"] > 0:
                comp["val_loss_ratio"] = other["quality"]["val_loss"] / first["quality"]["val_loss"]

            # VRAM ratio
            if first["vram"]["peak_gb"] > 0:
                comp["vram_ratio"] = other["vram"]["peak_gb"] / first["vram"]["peak_gb"]

            # Throughput ratio
            if first["inference"]["tokens_per_sec"] > 0:
                comp["throughput_ratio"] = other["inference"]["tokens_per_sec"] / first["inference"]["tokens_per_sec"]

            # FLOPS ratio
            if first["compute"]["estimated_flops_per_token"] > 0:
                comp["flops_ratio"] = other["compute"]["estimated_flops_per_token"] / first["compute"]["estimated_flops_per_token"]

            # Efficiency: quality per VRAM
            if other["vram"]["peak_gb"] > 0 and first["vram"]["peak_gb"] > 0:
                other_eff = other["quality"]["val_loss"] / other["vram"]["peak_gb"]
                first_eff = first["quality"]["val_loss"] / first["vram"]["peak_gb"]
                comp["quality_per_vram_ratio"] = other_eff / first_eff

            comparisons[f"{label}_vs_{first_label}"] = comp

        results["comparisons"] = comparisons

    # Save results
    results_path = out / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for label, ckpt in results["checkpoints"].items():
        print(f"\n{label}:")
        print(f"  Val loss: {ckpt['quality']['val_loss']:.4f}")
        print(f"  Val PPL:  {ckpt['quality']['val_ppl']:.2f}")
        print(f"  VRAM:     {ckpt['vram']['peak_gb']:.3f} GB")
        print(f"  Params:   {ckpt['vram']['model_params']:,}")
        print(f"  Tok/s:    {ckpt['inference']['tokens_per_sec']:.2f}")
        print(f"  TTFT:     {ckpt['inference']['time_to_first_token_ms']:.2f} ms")
        if ckpt["inference"]["avg_layers_per_token"] is not None:
            print(f"  Avg layers/tok: {ckpt['inference']['avg_layers_per_token']:.2f}")

    if "comparisons" in results:
        print(f"\nComparisons (vs {labels[0]}):")
        for comp_name, comp in results["comparisons"].items():
            print(f"  {comp_name}:")
            for metric, value in comp.items():
                print(f"    {metric}: {value:.3f}x")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PILON-R comprehensive efficiency benchmarking"
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to model checkpoints"
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="Labels for each checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--precision", type=str, default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
    )
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--max-val-batches", type=int, default=50)

    args = parser.parse_args()

    if len(args.checkpoints) != len(args.labels):
        raise ValueError("Number of checkpoints must match number of labels")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_full_benchmark(
        checkpoint_paths=args.checkpoints,
        labels=args.labels,
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
        tokenizer_path=args.tokenizer_path,
        max_val_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    main()
