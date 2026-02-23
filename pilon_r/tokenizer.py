"""
PILON-R Tokenizer Training

Train a BPE tokenizer on FineWeb-Edu for the 360M comparison experiment.
Uses the tokenizers library for fast training on streaming data.

Usage:
    python -m pilon_r.tokenizer --vocab-size 32000 --output tokenizer/fineweb-edu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def batch_iterator(
    dataset,
    batch_size: int = 1000,
    max_samples: Optional[int] = None,
    text_field: str = "text"
) -> Iterator[list[str]]:
    """
    Yield batches of text from a streaming dataset.

    Args:
        dataset: HuggingFace streaming dataset
        batch_size: Number of texts per batch
        max_samples: Maximum samples to process (None = unlimited)
        text_field: Name of the text field in the dataset

    Yields:
        Batches of text strings
    """
    batch = []
    count = 0

    for example in dataset:
        text = example.get(text_field, "")
        if not text:
            continue

        batch.append(text)
        count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

        if max_samples is not None and count >= max_samples:
            break

    # Yield remaining
    if batch:
        yield batch


def train_tokenizer(
    vocab_size: int = 32000,
    output_path: str = "tokenizer/fineweb-edu",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: Optional[str] = "sample-10BT",
    max_samples: int = 10_000_000,
    min_frequency: int = 2,
    show_progress: bool = True
) -> Tokenizer:
    """
    Train a BPE tokenizer on FineWeb-Edu.

    Args:
        vocab_size: Target vocabulary size
        output_path: Path to save the tokenizer (without extension)
        dataset_name: HuggingFace dataset name
        dataset_subset: Dataset subset/config name (e.g., "sample-10BT")
        max_samples: Maximum samples to train on
        min_frequency: Minimum frequency for a token to be included
        show_progress: Show training progress

    Returns:
        Trained tokenizer
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}" + (f" ({dataset_subset})" if dataset_subset else ""))

    # Load dataset in streaming mode
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Use ByteLevel pre-tokenizer (like GPT-2)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ByteLevel decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Special tokens
    special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=show_progress,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train from iterator
    print(f"Training on up to {max_samples:,} samples...")
    tokenizer.train_from_iterator(
        batch_iterator(dataset, max_samples=max_samples),
        trainer=trainer,
        length=max_samples
    )

    # Verify vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Trained tokenizer with {actual_vocab_size:,} tokens")

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer_file = str(output_path) + ".json"
    tokenizer.save(tokenizer_file)
    print(f"Saved tokenizer to: {tokenizer_file}")

    return tokenizer


def load_custom_tokenizer(path: str) -> Tokenizer:
    """
    Load a custom tokenizer from file.

    Args:
        path: Path to tokenizer file (with or without .json extension)

    Returns:
        Loaded tokenizer
    """
    if not path.endswith(".json"):
        path = path + ".json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found at: {path}")

    return Tokenizer.from_file(path)


def verify_tokenizer(tokenizer: Tokenizer, test_texts: Optional[list[str]] = None) -> dict:
    """
    Verify tokenizer works correctly.

    Args:
        tokenizer: Tokenizer to verify
        test_texts: Optional list of test texts

    Returns:
        Dictionary with verification results
    """
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can process natural language.",
            "def hello_world():\n    print('Hello, World!')",
            "1234567890",
            "Special chars: @#$%^&*()",
        ]

    results = {
        "vocab_size": tokenizer.get_vocab_size(),
        "roundtrip_success": True,
        "test_results": []
    }

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        success = decoded == text

        results["test_results"].append({
            "original": text,
            "encoded_length": len(encoded.ids),
            "decoded": decoded,
            "roundtrip_success": success
        })

        if not success:
            results["roundtrip_success"] = False

    return results


class TokenizerWrapper:
    """
    Wrapper to make tokenizers.Tokenizer compatible with HuggingFace transformers interface.

    This allows using the custom tokenizer with existing PILON-R data loading code.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

        # Get special token IDs
        vocab = tokenizer.get_vocab()
        self.pad_token_id = vocab.get("<pad>", 0)
        self.eos_token_id = vocab.get("<eos>", 1)
        self.bos_token_id = vocab.get("<bos>", 2)
        self.unk_token_id = vocab.get("<unk>", 3)

        # Set tokens for compatibility
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"

        # For compatibility with some code paths
        self.model_max_length = int(1e9)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs
    ) -> list[int]:
        """Encode text to token IDs."""
        encoded = self._tokenizer.encode(text)
        return encoded.ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        """Return vocab size."""
        return self.vocab_size

    def __call__(
        self,
        text,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        """Tokenize text (for compatibility)."""
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
        else:
            ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]

        if return_tensors == "pt":
            import torch
            if isinstance(ids[0], list):
                # Batch
                max_len = max(len(x) for x in ids)
                padded = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
                return {"input_ids": torch.tensor(padded)}
            else:
                return {"input_ids": torch.tensor([ids])}

        return {"input_ids": ids}


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on FineWeb-Edu")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer/fineweb-edu",
        help="Output path for tokenizer (default: tokenizer/fineweb-edu)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Dataset name (default: HuggingFaceFW/fineweb-edu)"
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default="sample-10BT",
        help="Dataset subset/config (default: sample-10BT)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000_000,
        help="Maximum samples to train on (default: 10M)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify tokenizer after training"
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        help="Only verify an existing tokenizer at this path"
    )

    args = parser.parse_args()

    if args.verify_only:
        # Just verify existing tokenizer
        print(f"Loading tokenizer from: {args.verify_only}")
        tokenizer = load_custom_tokenizer(args.verify_only)
        results = verify_tokenizer(tokenizer)

        print(f"\nVocabulary size: {results['vocab_size']:,}")
        print(f"All roundtrips successful: {results['roundtrip_success']}")
        print("\nTest results:")
        for r in results["test_results"]:
            status = "✓" if r["roundtrip_success"] else "✗"
            print(f"  {status} '{r['original'][:40]}...' -> {r['encoded_length']} tokens")
        return

    # Train tokenizer
    tokenizer = train_tokenizer(
        vocab_size=args.vocab_size,
        output_path=args.output,
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset if args.dataset_subset != "none" else None,
        max_samples=args.max_samples,
        min_frequency=args.min_frequency
    )

    if args.verify:
        print("\nVerifying tokenizer...")
        results = verify_tokenizer(tokenizer)
        print(f"All roundtrips successful: {results['roundtrip_success']}")
        for r in results["test_results"]:
            status = "✓" if r["roundtrip_success"] else "✗"
            print(f"  {status} '{r['original'][:40]}' -> {r['encoded_length']} tokens")


if __name__ == "__main__":
    main()
