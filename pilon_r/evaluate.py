"""
PILON-R Evaluation and Generation

Handles:
- Model evaluation (perplexity, loss)
- Text generation
- Quality comparison with baseline
- Generation quality assessment
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
import argparse
import math
import time
from typing import Optional, Dict, List, Tuple
import json
from contextlib import nullcontext

from .core.config import ModelConfig
from .core.model import PILONTransformer, create_model
from .core.data import load_tinystories, get_tokenizer, create_dataloader


def device_str(device) -> str:
    """Convert device to string for comparison."""
    if isinstance(device, torch.device):
        return device.type
    return str(device)


def resolve_autocast_dtype(precision: Optional[str], device) -> Optional[torch.dtype]:
    """Resolve autocast dtype based on precision setting and device."""
    dev = device_str(device)
    if not dev.startswith("cuda"):
        return None
    if precision is None:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "fp16":
        return torch.float16
    return None


def get_autocast_context(amp_dtype: Optional[torch.dtype], device):
    """Return an autocast context manager appropriate for the device."""
    dev = device_str(device)
    if amp_dtype is None or not dev.startswith("cuda"):
        return nullcontext()
    return autocast(device_type="cuda", dtype=amp_dtype)


def sync_if_cuda(device) -> None:
    if device_str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def set_runtime_top_k(model: PILONTransformer, top_k: Optional[int]) -> None:
    if top_k is None:
        return
    for layer in model.layers:
        ffn = layer.ffn
        if hasattr(ffn, "runtime_top_k"):
            ffn.runtime_top_k = top_k


@torch.no_grad()
def compute_perplexity(
    model: PILONTransformer,
    dataloader,
    max_batches: int = 100,
    device: str = "cuda",
    precision: Optional[str] = "bf16"
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        max_batches: Maximum batches to evaluate
        device: Device

    Returns:
        Dictionary with loss and perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    amp_dtype = resolve_autocast_dtype(precision, device)

    sync_if_cuda(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in dataloader:
            if n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with get_autocast_context(amp_dtype, device):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Match shifted CE loss with ignore_index=-100 masking.
            valid = attention_mask[:, 1:].bool() & labels[:, 1:].ne(-100)
            n_tokens = valid.sum().item()
            total_loss += outputs["loss"].item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    tok_s = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "n_batches": n_batches,
        "n_tokens": total_tokens,
        "tok_s": tok_s,
        "seconds": elapsed
    }


@torch.no_grad()
def generate_samples(
    model: PILONTransformer,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
    log_speed: bool = False
) -> List[Dict[str, str]]:
    """
    Generate text samples from prompts.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device

    Returns:
        List of dictionaries with prompt and generation
    """
    model.eval()
    results = []
    total_new_tokens = 0
    sync_if_cuda(device)
    start_time = time.perf_counter()

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        # Generate
        try:
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
        except TypeError:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )

        # Decode continuation by token slice (more reliable than string slicing)
        gen_ids = output_ids[0][input_ids.shape[1]:]
        continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        total_new_tokens += int(output_ids.shape[1] - input_ids.shape[1])

        results.append({
            "prompt": prompt,
            "generation": continuation,
            "full_text": generated_text
        })

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start_time
    if log_speed:
        tok_s = total_new_tokens / elapsed if elapsed > 0 else 0.0
        print(f"Generation throughput: {tok_s:.1f} tok/s ({total_new_tokens} tokens in {elapsed:.2f}s)")

    return results


def check_generation_quality(generations: List[Dict[str, str]]) -> Dict[str, any]:
    """
    Basic quality checks on generated text.

    Checks for:
    - Repetition loops
    - Gibberish (unusual character patterns)
    - Empty or very short generations
    - Coherence (basic heuristics)
    """
    results = {
        "n_samples": len(generations),
        "issues": [],
        "quality_scores": []
    }

    for i, gen in enumerate(generations):
        text = gen["generation"]
        issues = []
        score = 1.0

        # Check for empty generation
        if len(text.strip()) < 10:
            issues.append("very_short")
            score *= 0.5

        # Check for repetition loops
        words = text.split()
        if len(words) > 5:
            # Check for repeated sequences
            for seq_len in [2, 3, 4]:
                for j in range(len(words) - seq_len * 2):
                    seq1 = " ".join(words[j:j+seq_len])
                    seq2 = " ".join(words[j+seq_len:j+seq_len*2])
                    if seq1 == seq2:
                        # Check if this repeats more
                        repeat_count = 1
                        for k in range(j + seq_len * 2, len(words) - seq_len + 1, seq_len):
                            if " ".join(words[k:k+seq_len]) == seq1:
                                repeat_count += 1
                            else:
                                break
                        if repeat_count >= 3:
                            issues.append(f"repetition_loop_{seq_len}")
                            score *= 0.3
                            break
                if "repetition" in str(issues):
                    break

        # Check for unusual character patterns (gibberish)
        if text:
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
            if alpha_ratio < 0.7:
                issues.append("unusual_characters")
                score *= 0.5

        # Check for sentence structure (has periods/question marks)
        if len(text) > 50 and not any(p in text for p in ['.', '!', '?']):
            issues.append("no_sentence_endings")
            score *= 0.8

        results["quality_scores"].append(score)
        if issues:
            results["issues"].append({"sample": i, "issues": issues})

    results["avg_quality"] = sum(results["quality_scores"]) / max(len(results["quality_scores"]), 1)
    results["n_with_issues"] = len(results["issues"])

    return results


def compare_models(
    pilon_model: PILONTransformer,
    baseline_model: PILONTransformer,
    tokenizer,
    test_prompts: List[str],
    dataloader,
    device: str = "cuda",
    precision: Optional[str] = "bf16"
) -> Dict[str, any]:
    """
    Compare PILON model with baseline.

    Returns comprehensive comparison metrics.
    """
    # Perplexity comparison
    pilon_ppl = compute_perplexity(pilon_model, dataloader, device=device, precision=precision)
    baseline_ppl = compute_perplexity(baseline_model, dataloader, device=device, precision=precision)

    # Generation comparison
    pilon_gens = generate_samples(pilon_model, tokenizer, test_prompts, device=device)
    baseline_gens = generate_samples(baseline_model, tokenizer, test_prompts, device=device)

    pilon_quality = check_generation_quality(pilon_gens)
    baseline_quality = check_generation_quality(baseline_gens)

    # Compute ratios
    ppl_ratio = pilon_ppl["perplexity"] / baseline_ppl["perplexity"]
    loss_ratio = pilon_ppl["loss"] / baseline_ppl["loss"]

    return {
        "perplexity": {
            "pilon": pilon_ppl["perplexity"],
            "baseline": baseline_ppl["perplexity"],
            "ratio": ppl_ratio
        },
        "loss": {
            "pilon": pilon_ppl["loss"],
            "baseline": baseline_ppl["loss"],
            "ratio": loss_ratio
        },
        "generation_quality": {
            "pilon": pilon_quality["avg_quality"],
            "baseline": baseline_quality["avg_quality"]
        },
        "samples": {
            "pilon": pilon_gens,
            "baseline": baseline_gens
        },
        "gate_status": {
            "ppl_within_10pct": ppl_ratio < 1.1,
            "ppl_within_20pct": ppl_ratio < 1.2,
            "quality_acceptable": pilon_quality["avg_quality"] > 0.7
        }
    }


def print_generation_samples(
    samples: List[Dict[str, str]],
    title: str = "Generation Samples",
    max_display: int = 5
):
    """Pretty print generation samples."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for i, sample in enumerate(samples[:max_display]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {sample['prompt']}")
        print(f"Generation: {sample['generation'][:200]}...")
        print()


SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_sft_prompt(instruction: str) -> str:
    """Wrap an instruction in the SFT template."""
    return SFT_TEMPLATE.format(instruction=instruction)


def ensure_sft_special_tokens(tokenizer) -> bool:
    """Ensure tokenizer has SFT special tokens (matches sft.py)."""
    added = False
    if tokenizer.eos_token == "<|endoftext|>":
        if "<|eos|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
            added = True
        else:
            tokenizer.eos_token = "<|eos|>"
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|pad|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added = True
        else:
            tokenizer.pad_token = "<|pad|>"
    return added


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    device: str = "cuda",
    precision: Optional[str] = "bf16",
    log_speed: bool = False,
    eval_top_k: Optional[int] = None,
    sft_mode: bool = False,
    custom_prompts: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate a saved checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        output_path: Optional path to save results
        device: Device

    Returns:
        Evaluation results dictionary
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", checkpoint.get("config"))

    # Handle case where checkpoint saved TrainingConfig instead of ModelConfig (legacy bug)
    if not hasattr(config, "vocab_size"):
        print("WARNING: Checkpoint config missing 'vocab_size' (likely TrainingConfig). Falling back to default ModelConfig.")
        config = ModelConfig()

    # Recreate model
    model = PILONTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    set_runtime_top_k(model, eval_top_k)

    # Setup evaluation
    tokenizer = get_tokenizer()
    # If model vocab is larger than tokenizer, add SFT special tokens
    if config.vocab_size > len(tokenizer):
        ensure_sft_special_tokens(tokenizer)
    val_dataset = load_tinystories(
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        max_tokens=1_000_000,
        split="validation",
        streaming=True
    )
    val_loader = create_dataloader(val_dataset, batch_size=4, shuffle=False)

    # Compute perplexity
    ppl_results = compute_perplexity(model, val_loader, device=device, precision=precision)

    # Generate samples
    if custom_prompts:
        test_prompts = custom_prompts
    elif sft_mode:
        test_prompts = [
            "Hi there!",
            "What is the capital of France?",
            "Write a short poem about the moon.",
            "Explain what a neural network is.",
            "Tell me a joke."
        ]
    else:
        test_prompts = [
            "Once upon a time",
            "The little dog",
            "Sarah went to the",
            "One day, a magical",
            "The big red ball"
        ]

    # Wrap in SFT template if needed
    if sft_mode:
        test_prompts = [format_sft_prompt(p) for p in test_prompts]

    generations = generate_samples(model, tokenizer, test_prompts, device=device, log_speed=log_speed)
    quality = check_generation_quality(generations)

    # Get entropy if compositional
    entropy = {}
    if config.ffn_type == "compositional":
        entropy = model.get_all_entropy()

    results = {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step", "unknown"),
        "model_type": config.ffn_type,
        "perplexity": ppl_results,
        "generation_quality": quality,
        "entropy": entropy,
        "samples": generations
    }

    # Save if requested
    if output_path:
        with open(output_path, "w") as f:
            # Convert samples for JSON serialization
            json_results = results.copy()
            json.dump(json_results, f, indent=2, default=str)

    return results


def run_full_evaluation(
    pilon_checkpoint: Path,
    baseline_checkpoint: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    precision: Optional[str] = "bf16",
    log_speed: bool = False,
    eval_top_k: Optional[int] = None,
    sft_mode: bool = False,
    custom_prompts: Optional[List[str]] = None
):
    """
    Run full evaluation comparing PILON to baseline.
    """
    output_dir = Path(output_dir) if output_dir else Path("eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PILON-R EVALUATION")
    print("=" * 60)

    # Evaluate PILON
    print("\nEvaluating PILON model...")
    pilon_results = evaluate_checkpoint(
        pilon_checkpoint,
        output_path=output_dir / "pilon_eval.json",
        device=device,
        precision=precision,
        log_speed=log_speed,
        eval_top_k=eval_top_k,
        sft_mode=sft_mode,
        custom_prompts=custom_prompts
    )

    print(f"  Perplexity: {pilon_results['perplexity']['perplexity']:.2f}")
    print(f"  Generation quality: {pilon_results['generation_quality']['avg_quality']:.2f}")
    if log_speed:
        print(f"  Eval throughput: {pilon_results['perplexity']['tok_s']:.1f} tok/s")

    if pilon_results["entropy"]:
        avg_entropy = sum(pilon_results["entropy"].values()) / len(pilon_results["entropy"])
        print(f"  Average entropy: {avg_entropy:.2f}")

    # Evaluate baseline if provided
    if baseline_checkpoint:
        print("\nEvaluating baseline model...")
        baseline_results = evaluate_checkpoint(
            baseline_checkpoint,
            output_path=output_dir / "baseline_eval.json",
            device=device,
            precision=precision,
            log_speed=log_speed,
            eval_top_k=eval_top_k,
            sft_mode=sft_mode,
            custom_prompts=custom_prompts
        )

        print(f"  Perplexity: {baseline_results['perplexity']['perplexity']:.2f}")
        print(f"  Generation quality: {baseline_results['generation_quality']['avg_quality']:.2f}")
        if log_speed:
            print(f"  Eval throughput: {baseline_results['perplexity']['tok_s']:.1f} tok/s")

        # Comparison
        ppl_ratio = pilon_results['perplexity']['perplexity'] / baseline_results['perplexity']['perplexity']
        print("\n" + "-" * 40)
        print("COMPARISON")
        print("-" * 40)
        print(f"  PPL ratio (PILON/baseline): {ppl_ratio:.3f}")
        print(f"  Gate A3 (within 10%): {'PASS' if ppl_ratio < 1.1 else 'FAIL'}")

    # Print sample generations
    print_generation_samples(pilon_results["samples"], "PILON Generations")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return pilon_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate PILON-R model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline checkpoint")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision: bf16|fp16|fp32")
    parser.add_argument("--log-speed", action="store_true", help="Log evaluation/generation throughput")
    parser.add_argument("--eval-top-k", type=int, default=None, help="Override runtime top-k for evaluation/generation")
    parser.add_argument("--sft", action="store_true", help="Use SFT instruction template for generation prompts")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for generation (can be used multiple times)", action="append")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    if args.device.startswith("cuda"):
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True

    run_full_evaluation(
        pilon_checkpoint=Path(args.checkpoint),
        baseline_checkpoint=Path(args.baseline) if args.baseline else None,
        output_dir=Path(args.output_dir),
        device=args.device,
        precision=args.precision,
        log_speed=args.log_speed,
        eval_top_k=args.eval_top_k,
        sft_mode=args.sft,
        custom_prompts=args.prompt
    )


if __name__ == "__main__":
    main()
