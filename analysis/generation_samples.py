"""
PILON-R Generation Samples Analysis

Generates and analyzes text samples from trained models.
Used for qualitative evaluation of model quality.
"""

import torch
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pilon_r.core.model import PILONTransformer
from pilon_r.core.data import get_tokenizer
from pilon_r.evaluate import generate_samples, check_generation_quality, set_runtime_top_k
from pilon_r.compress import load_compressed_model
from transformers import AutoTokenizer


# Standard prompts for TinyStories evaluation
TINYSTORIES_PROMPTS = [
    "Once upon a time, there was a little",
    "The dog went to the",
    "Sarah and her friend",
    "One sunny day,",
    "The magic fairy",
    "Mom said to",
    "In the forest, there lived",
    "The little boy was very",
    "They decided to",
    "At the end of the day,",
]

# More challenging prompts
CHALLENGE_PROMPTS = [
    "The dragon flew over the",
    "When the clock struck midnight,",
    "Deep in the ocean,",
    "The wizard cast a spell that",
    "If only she had known that",
]

# SFT instruction prompts
SFT_PROMPTS = [
    "Hi there!",
    "What is the capital of France?",
    "Write a short poem about the moon.",
    "Explain what a neural network is in simple terms.",
    "Tell me a joke.",
    "What are three tips for staying healthy?",
    "Summarize the story of Cinderella in two sentences.",
    "Why is the sky blue?",
]

SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_sft_prompt(instruction: str) -> str:
    """Wrap an instruction in the SFT template."""
    return SFT_TEMPLATE.format(instruction=instruction)


def _is_compressed_dir(path: Path) -> bool:
    return path.is_dir() and (path / "compression_config.json").exists() and (path / "primitive_banks.pt").exists()


def load_pilon_checkpoint(checkpoint_path: Path, device: str = "cuda") -> PILONTransformer:
    """Load PILON checkpoint model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = PILONTransformer(config)
    state_dict = checkpoint["model_state_dict"]
    # Drop non-parameter buffers that can change size with max_seq_len
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("attention.causal_mask")}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


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


def load_model_and_tokenizer(checkpoint_path: Path, device: str = "cuda"):
    """Load either PILON checkpoint or compressed model dir + tokenizer."""
    if _is_compressed_dir(checkpoint_path):
        model, cfg, _ = load_compressed_model(checkpoint_path, device=device, logger=None)
        tokenizer = AutoTokenizer.from_pretrained(cfg.source_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        model.eval()
        return model, tokenizer

    model = load_pilon_checkpoint(checkpoint_path, device)
    tokenizer = get_tokenizer()

    # If model vocab is larger than tokenizer, add SFT special tokens
    if model.config.vocab_size > len(tokenizer):
        ensure_sft_special_tokens(tokenizer)

    return model, tokenizer


def generate_comparison_samples(
    pilon_path: Path,
    baseline_path: Optional[Path],
    prompts: List[str],
    device: str = "cuda",
    temperature: float = 0.8,
    max_tokens: int = 256,
    log_speed: bool = False,
    eval_top_k: Optional[int] = None
) -> Dict:
    """
    Generate samples from PILON and optionally baseline.

    Returns comparison results.
    """
    print("Loading PILON model...")
    pilon_model, pilon_tokenizer = load_model_and_tokenizer(pilon_path, device)
    set_runtime_top_k(pilon_model, eval_top_k)

    pilon_samples = generate_samples(
        pilon_model, pilon_tokenizer, prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=device,
        log_speed=log_speed
    )
    pilon_quality = check_generation_quality(pilon_samples)

    results = {
        "pilon": {
            "samples": pilon_samples,
            "quality": pilon_quality
        }
    }

    if baseline_path and baseline_path.exists():
        print("Loading baseline model...")
        baseline_model, baseline_tokenizer = load_model_and_tokenizer(baseline_path, device)
        set_runtime_top_k(baseline_model, eval_top_k)

        baseline_samples = generate_samples(
            baseline_model, baseline_tokenizer, prompts,
            max_new_tokens=max_tokens,
            temperature=temperature,
            device=device,
            log_speed=log_speed
        )
        baseline_quality = check_generation_quality(baseline_samples)

        results["baseline"] = {
            "samples": baseline_samples,
            "quality": baseline_quality
        }

    return results


def print_samples_comparison(results: Dict):
    """Print side-by-side sample comparison."""
    pilon_samples = results["pilon"]["samples"]
    baseline_samples = results.get("baseline", {}).get("samples", [])

    print("\n" + "=" * 80)
    print("GENERATION SAMPLES COMPARISON")
    print("=" * 80)

    for i, pilon_sample in enumerate(pilon_samples):
        print(f"\n{'─' * 80}")
        print(f"PROMPT {i+1}: {pilon_sample['prompt']}")
        print("─" * 80)

        print(f"\n[PILON]")
        print(pilon_sample['generation'][:300])
        if len(pilon_sample['generation']) > 300:
            print("...")

        if i < len(baseline_samples):
            print(f"\n[BASELINE]")
            print(baseline_samples[i]['generation'][:300])
            if len(baseline_samples[i]['generation']) > 300:
                print("...")

    # Quality summary
    print("\n" + "=" * 80)
    print("QUALITY SUMMARY")
    print("=" * 80)

    print(f"\nPILON:")
    print(f"  Average quality score: {results['pilon']['quality']['avg_quality']:.2f}")
    print(f"  Samples with issues: {results['pilon']['quality']['n_with_issues']}/{results['pilon']['quality']['n_samples']}")
    if results['pilon']['quality']['issues']:
        print(f"  Issues found:")
        for issue in results['pilon']['quality']['issues'][:5]:
            print(f"    - Sample {issue['sample']}: {', '.join(issue['issues'])}")

    if "baseline" in results:
        print(f"\nBASELINE:")
        print(f"  Average quality score: {results['baseline']['quality']['avg_quality']:.2f}")
        print(f"  Samples with issues: {results['baseline']['quality']['n_with_issues']}/{results['baseline']['quality']['n_samples']}")


def save_samples_to_file(results: Dict, output_path: Path):
    """Save samples to a formatted text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("PILON-R GENERATION SAMPLES\n")
        f.write("=" * 80 + "\n\n")

        for i, sample in enumerate(results["pilon"]["samples"]):
            f.write(f"Sample {i+1}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Prompt: {sample['prompt']}\n\n")
            f.write(f"PILON Generation:\n{sample['generation']}\n\n")

            if "baseline" in results and i < len(results["baseline"]["samples"]):
                f.write(f"Baseline Generation:\n{results['baseline']['samples'][i]['generation']}\n\n")

            f.write("\n")

        # Quality summary
        f.write("=" * 80 + "\n")
        f.write("QUALITY SCORES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"PILON average quality: {results['pilon']['quality']['avg_quality']:.2f}\n")
        if "baseline" in results:
            f.write(f"Baseline average quality: {results['baseline']['quality']['avg_quality']:.2f}\n")

    print(f"Samples saved to: {output_path}")


def interactive_generation(checkpoint_path: Path, device: str = "cuda", max_tokens: int = 256, temperature: float = 0.8, eval_top_k: Optional[int] = None, sft_mode: bool = False):
    """Interactive generation mode."""
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
    set_runtime_top_k(model, eval_top_k)

    print("\n" + "=" * 60)
    print("INTERACTIVE GENERATION MODE" + (" (SFT)" if sft_mode else ""))
    print("Type a prompt and press Enter to generate.")
    if sft_mode:
        print("Prompts will be wrapped in instruction template automatically.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt:
            continue

        # Wrap in SFT template if needed
        formatted_prompt = format_sft_prompt(prompt) if sft_mode else prompt

        samples = generate_samples(
            model, tokenizer, [formatted_prompt],
            max_new_tokens=max_tokens,
            temperature=temperature,
            device=device,
            log_speed=True
        )

        print(f"\nGeneration:")
        print(samples[0]["generation"])


def main():
    parser = argparse.ArgumentParser(description="Generate and analyze text samples")
    parser.add_argument("checkpoint", type=str, help="Path to PILON checkpoint or compressed model dir")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline checkpoint")
    parser.add_argument("--output", type=str, default="samples.txt", help="Output file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--challenge", action="store_true", help="Use challenge prompts")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log-speed", action="store_true", help="Log generation throughput")
    parser.add_argument("--eval-top-k", type=int, default=None, help="Override runtime top-k for generation")
    parser.add_argument("--sft", action="store_true", help="Use SFT instruction template for prompts")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.interactive:
        interactive_generation(Path(args.checkpoint), args.device, args.max_tokens, args.temperature, args.eval_top_k, args.sft)
        return

    # Select prompts based on mode
    if args.sft:
        prompts = [format_sft_prompt(p) for p in SFT_PROMPTS]
    elif args.challenge:
        prompts = CHALLENGE_PROMPTS
    else:
        prompts = TINYSTORIES_PROMPTS

    results = generate_comparison_samples(
        pilon_path=Path(args.checkpoint),
        baseline_path=Path(args.baseline) if args.baseline else None,
        prompts=prompts,
        device=args.device,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_speed=args.log_speed,
        eval_top_k=args.eval_top_k
    )

    print_samples_comparison(results)
    save_samples_to_file(results, Path(args.output))


if __name__ == "__main__":
    main()
