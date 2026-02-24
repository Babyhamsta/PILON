"""
PILON-R Inference Benchmarking

Measure inference metrics for trained models:
- Tokens per second (throughput)
- Peak VRAM usage
- Latency per token

Usage:
    python -m pilon_r.benchmark outputs/360m_pilon/final_model.pt --device cuda
    python -m pilon_r.benchmark outputs/360m_dense/final_model.pt --device cuda --compare outputs/360m_pilon/final_model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from .core.config import ModelConfig
from .core.model import PILONTransformer, create_model
from .core.data import get_tokenizer


def get_benchmark_autocast(device: str, precision: str):
    """Resolve autocast context for benchmark runs."""
    if not device.startswith("cuda"):
        return nullcontext()

    prec = (precision or "auto").lower()
    if prec == "fp32":
        return nullcontext()
    if prec == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if prec == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cuda", dtype=torch.float16)

    # auto
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16)


def load_model(checkpoint_path: str, device: str = "cuda") -> tuple[PILONTransformer, ModelConfig]:
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("Checkpoint does not contain model config")

    # Create and load model
    model = create_model(config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config


def generate_tokens(
    model: PILONTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.

    Args:
        model: Model to use for generation
        input_ids: Input token IDs [batch, seq_len]
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (None = greedy)

    Returns:
        Generated token IDs [batch, seq_len + max_new_tokens]
    """
    device = next(model.parameters()).device
    generated = input_ids.to(device)

    do_sample = top_k is not None and top_k > 0
    gen_temperature = temperature if do_sample else 0.0
    return model.generate(
        generated,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
        top_k=top_k if do_sample else None,
        do_sample=do_sample,
    )


def benchmark_inference(
    model: PILONTransformer,
    config: ModelConfig,
    tokenizer,
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100,
    max_new_tokens: int = 100,
    prompt: str = "The quick brown fox",
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Benchmark inference performance.

    Args:
        model: Model to benchmark
        config: Model config
        tokenizer: Tokenizer
        device: Device
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        max_new_tokens: Tokens to generate per run
        prompt: Input prompt
        batch_size: Batch size for generation

    Returns:
        Dictionary of benchmark metrics
    """
    # Prepare input
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids] * batch_size, device=device)
    prompt_len = len(input_ids)

    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = generate_tokens(model, input_tensor, max_new_tokens=max_new_tokens)

    # Synchronize before timing
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"Running {num_runs} benchmark iterations...")
    latencies = []
    total_tokens = 0

    start_time = time.perf_counter()

    for i in range(num_runs):
        iter_start = time.perf_counter()

        generated = generate_tokens(model, input_tensor, max_new_tokens=max_new_tokens)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        iter_time = time.perf_counter() - iter_start
        latencies.append(iter_time)
        total_tokens += (generated.size(1) - prompt_len) * batch_size

    total_time = time.perf_counter() - start_time

    # Collect metrics
    results = {
        "model_type": config.ffn_type,
        "model_params": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "prompt_tokens": prompt_len,
        "generated_tokens_per_run": max_new_tokens,
        "num_runs": num_runs,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "tokens_per_sec": total_tokens / total_time,
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
        "min_latency_ms": min(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "latency_ms_per_token": (sum(latencies) / len(latencies)) / max_new_tokens * 1000,
    }

    # VRAM metrics (CUDA only)
    if device.startswith("cuda"):
        results["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        results["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        results["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

    # Early exit metrics (Phase B.5c)
    exit_metrics = model.get_early_exit_metrics() if hasattr(model, "get_early_exit_metrics") else None
    if exit_metrics is not None:
        results["skip_ratios"] = exit_metrics["skip_ratios"]
        results["avg_layers_per_token"] = exit_metrics["avg_layers_per_token"]
        results["total_skips"] = exit_metrics["total_skips"]
        results["total_tokens_exit"] = exit_metrics["total_tokens"]

    return results


def benchmark_prefill(
    model: PILONTransformer,
    config: ModelConfig,
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100,
    seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
    batch_size: int = 1,
    precision: str = "auto",
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark prefill (prompt processing) performance.

    Args:
        model: Model to benchmark
        config: Model config
        device: Device
        num_warmup: Warmup runs
        num_runs: Benchmark runs
        seq_lengths: Sequence lengths to test
        batch_size: Batch size

    Returns:
        Dictionary mapping seq_length to metrics
    """
    results = {}

    for seq_len in seq_lengths:
        if seq_len > config.max_seq_len:
            print(f"Skipping seq_len={seq_len} (exceeds max_seq_len={config.max_seq_len})")
            continue

        print(f"\nBenchmarking prefill for seq_len={seq_len}...")

        # Create random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        # Warmup
        for _ in range(num_warmup):
            with torch.inference_mode(), get_benchmark_autocast(device, precision):
                _ = model(input_ids)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        latencies = []
        start_time = time.perf_counter()

        for _ in range(num_runs):
            iter_start = time.perf_counter()

            with torch.inference_mode(), get_benchmark_autocast(device, precision):
                _ = model(input_ids)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            latencies.append(time.perf_counter() - iter_start)

        total_time = time.perf_counter() - start_time
        total_tokens = seq_len * batch_size * num_runs

        results[seq_len] = {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "precision": precision,
            "tokens_per_sec": total_tokens / total_time,
            "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
        }

        if device.startswith("cuda"):
            results[seq_len]["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

    return results


def compare_models(
    model_a_path: str,
    model_b_path: str,
    device: str = "cuda",
    tokenizer_path: Optional[str] = None,
) -> Dict:
    """
    Compare two models side by side.

    Args:
        model_a_path: Path to first model
        model_b_path: Path to second model
        device: Device
        tokenizer_path: Optional custom tokenizer path

    Returns:
        Comparison results
    """
    tokenizer = get_tokenizer(tokenizer_path)

    print(f"\n{'='*60}")
    print("Loading Model A...")
    model_a, config_a = load_model(model_a_path, device)
    print(f"  Type: {config_a.ffn_type}")
    print(f"  Params: {sum(p.numel() for p in model_a.parameters()):,}")

    print(f"\nLoading Model B...")
    model_b, config_b = load_model(model_b_path, device)
    print(f"  Type: {config_b.ffn_type}")
    print(f"  Params: {sum(p.numel() for p in model_b.parameters()):,}")

    print(f"\n{'='*60}")
    print("Benchmarking Model A...")
    results_a = benchmark_inference(model_a, config_a, tokenizer, device)

    # Clear GPU memory
    del model_a
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Benchmarking Model B...")
    results_b = benchmark_inference(model_b, config_b, tokenizer, device)

    # Compute comparison
    comparison = {
        "model_a": {
            "path": model_a_path,
            "results": results_a,
        },
        "model_b": {
            "path": model_b_path,
            "results": results_b,
        },
        "comparison": {
            "throughput_ratio": results_a["tokens_per_sec"] / results_b["tokens_per_sec"],
            "latency_ratio": results_a["latency_ms_per_token"] / results_b["latency_ms_per_token"],
            "vram_ratio": results_a.get("vram_peak_gb", 0) / max(results_b.get("vram_peak_gb", 1), 0.001),
            "param_ratio": results_a["model_params"] / results_b["model_params"],
        }
    }

    return comparison


def print_results(results: Dict[str, float], title: str = "Benchmark Results"):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print("=" * 60)

    # Model info
    print(f"\nModel Information:")
    print(f"  Type: {results.get('model_type', 'unknown')}")
    print(f"  Parameters: {results.get('model_params', 0):,}")

    # Throughput
    print(f"\nThroughput:")
    print(f"  Tokens/sec: {results.get('tokens_per_sec', 0):.2f}")
    print(f"  Total tokens: {results.get('total_tokens', 0):,}")
    print(f"  Total time: {results.get('total_time_s', 0):.2f}s")

    # Latency
    print(f"\nLatency:")
    print(f"  Avg per iteration: {results.get('avg_latency_ms', 0):.2f}ms")
    print(f"  Per token: {results.get('latency_ms_per_token', 0):.3f}ms")
    print(f"  Min: {results.get('min_latency_ms', 0):.2f}ms")
    print(f"  Max: {results.get('max_latency_ms', 0):.2f}ms")

    # VRAM
    if "vram_peak_gb" in results:
        print(f"\nVRAM Usage:")
        print(f"  Peak: {results['vram_peak_gb']:.3f} GB")
        print(f"  Allocated: {results.get('vram_allocated_gb', 0):.3f} GB")
        print(f"  Reserved: {results.get('vram_reserved_gb', 0):.3f} GB")

    # Early exit
    if "skip_ratios" in results:
        print(f"\nEarly Exit:")
        print(f"  Avg layers per token: {results['avg_layers_per_token']:.2f}")
        for layer_name, ratio in results["skip_ratios"].items():
            print(f"  {layer_name} skip ratio: {ratio:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PILON-R model inference")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Custom tokenizer path")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-runs", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--max-tokens", type=int, default=100, help="Tokens to generate per run")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    parser.add_argument("--compare", type=str, default=None, help="Compare with another checkpoint")
    parser.add_argument("--prefill", action="store_true", help="Benchmark prefill instead of generation")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="Precision for prefill benchmark (CUDA only)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")

    args = parser.parse_args()

    # Validate device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.compare:
        # Compare two models
        comparison = compare_models(
            args.checkpoint,
            args.compare,
            args.device,
            args.tokenizer_path
        )

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\nModel A ({comparison['model_a']['results']['model_type']}):")
        print(f"  Tokens/sec: {comparison['model_a']['results']['tokens_per_sec']:.2f}")
        print(f"  VRAM: {comparison['model_a']['results'].get('vram_peak_gb', 0):.3f} GB")

        print(f"\nModel B ({comparison['model_b']['results']['model_type']}):")
        print(f"  Tokens/sec: {comparison['model_b']['results']['tokens_per_sec']:.2f}")
        print(f"  VRAM: {comparison['model_b']['results'].get('vram_peak_gb', 0):.3f} GB")

        print(f"\nRatios (A/B):")
        print(f"  Throughput: {comparison['comparison']['throughput_ratio']:.3f}x")
        print(f"  Latency: {comparison['comparison']['latency_ratio']:.3f}x")
        print(f"  VRAM: {comparison['comparison']['vram_ratio']:.3f}x")
        print(f"  Parameters: {comparison['comparison']['param_ratio']:.3f}x")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        # Single model benchmark
        tokenizer = get_tokenizer(args.tokenizer_path)

        print(f"Loading model from: {args.checkpoint}")
        model, config = load_model(args.checkpoint, args.device)

        if args.prefill:
            results = benchmark_prefill(
                model, config, args.device,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                batch_size=args.batch_size,
                precision=args.precision,
            )

            print(f"\n{'='*60}")
            print("PREFILL BENCHMARK RESULTS")
            print("=" * 60)
            for seq_len, metrics in results.items():
                print(f"\nSeq Length: {seq_len}")
                print(f"  Tokens/sec: {metrics['tokens_per_sec']:.2f}")
                print(f"  Latency: {metrics['avg_latency_ms']:.2f}ms")
                if "vram_peak_gb" in metrics:
                    print(f"  VRAM: {metrics['vram_peak_gb']:.3f} GB")

        else:
            results = benchmark_inference(
                model, config, tokenizer, args.device,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                max_new_tokens=args.max_tokens,
                prompt=args.prompt,
                batch_size=args.batch_size
            )

            print_results(results, f"Inference Benchmark: {Path(args.checkpoint).name}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
