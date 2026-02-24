"""
PILON-R Data Loading

Handles dataset loading and preprocessing for Phase A training.
Primary dataset: OpenWebText-10k (GPT-2 style, diverse content)
Alternative: TinyStories (simple, fast to train)
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from typing import Optional, Dict, List, Iterator, Tuple
import os


class TextDataset(Dataset):
    """
    Simple text dataset that returns tokenized sequences.

    Handles:
    - Loading from HuggingFace datasets
    - Tokenization with GPT-2 tokenizer
    - Chunking into fixed-length sequences
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_seq_len: int = 512,
        stride: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length
            stride: Stride for overlapping chunks
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Tokenize all texts and chunk into sequences
        self.sequences = []

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Chunk with stride
            for i in range(0, len(tokens) - max_seq_len + 1, stride):
                chunk = tokens[i:i + max_seq_len]
                if len(chunk) == max_seq_len:
                    self.sequences.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.ones_like(tokens)
        }


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large datasets.

    Processes data on-the-fly without loading everything into memory.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
        max_tokens: Optional[int] = None,
        skip_examples: int = 0,
        max_examples: Optional[int] = None,
        dataset_iterable: Optional[Iterator] = None,
        dataset_config: Optional[str] = None,
        tokenize_batch_size: int = 32,
    ):
        """
        Initialize streaming dataset.

        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length
            split: Dataset split ("train" or "validation")
            max_tokens: Maximum total tokens to yield (None = unlimited)
            skip_examples: Number of examples to skip (for splitting)
            max_examples: Maximum examples to process (for splitting)
            dataset_iterable: Pre-loaded dataset iterator (optional)
            dataset_config: Dataset config/subset name (e.g., "sample-10BT")
            tokenize_batch_size: Number of raw texts to tokenize together
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.max_tokens = max_tokens
        self.skip_examples = skip_examples
        self.max_examples = max_examples
        self.dataset_iterable = dataset_iterable
        self.dataset_config = dataset_config
        self.tokenize_batch_size = max(1, int(tokenize_batch_size))

    def _encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize a batch of texts.

        Prefers vectorized tokenizer calls when available, and falls back
        to per-item encode for compatibility with wrapper tokenizers.
        """
        try:
            encoded = self.tokenizer(texts, add_special_tokens=False, truncation=False)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                input_ids = encoded["input_ids"]
                # Some tokenizers may return a single list for batch size 1.
                if input_ids and isinstance(input_ids[0], int):
                    return [input_ids]
                return input_ids
        except Exception:
            pass
        return [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_max_tokens = self.max_tokens
        if worker_max_tokens is not None and num_workers > 1:
            base_tokens = worker_max_tokens // num_workers
            remainder = worker_max_tokens % num_workers
            worker_max_tokens = base_tokens + (1 if worker_id < remainder else 0)
            if worker_max_tokens <= 0:
                return

        dataset_sharded = False
        if self.dataset_iterable is None:
            from datasets import load_dataset

            # Load dataset in streaming mode
            try:
                if self.dataset_config:
                    dataset = load_dataset(
                        self.dataset_name,
                        self.dataset_config,
                        split=self.split,
                        streaming=True
                    )
                else:
                    dataset = load_dataset(
                        self.dataset_name,
                        split=self.split,
                        streaming=True
                    )
            except ValueError as exc:
                # Fallback only if we haven't already explicitly handled the split
                if self.split != "train" and "Bad split" in str(exc):
                    print(
                        f"WARNING: split '{self.split}' not available for {self.dataset_name}. "
                        "Falling back to 'train'."
                    )
                    dataset = load_dataset(
                        self.dataset_name,
                        split="train",
                        streaming=True
                    )
                else:
                    raise

            # Apply skipping and limiting for pseudo-splits
            if self.skip_examples > 0:
                dataset = dataset.skip(self.skip_examples)
            
            if self.max_examples is not None:
                dataset = dataset.take(self.max_examples)

            # Partition stream across dataloader workers to avoid duplicate reads.
            if num_workers > 1 and hasattr(dataset, "shard"):
                try:
                    dataset = dataset.shard(num_shards=num_workers, index=worker_id)
                    dataset_sharded = True
                except TypeError:
                    dataset = dataset.shard(num_workers, worker_id)
                    dataset_sharded = True
        else:
            dataset = self.dataset_iterable

        buffer = []
        total_tokens = 0
        seen = 0
        yielded = 0
        pending_texts: List[str] = []

        def flush_pending(texts: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
            nonlocal buffer, total_tokens
            if not texts:
                return
            token_batches = self._encode_batch(texts)
            for tokens in token_batches:
                buffer.extend(tokens)
                while len(buffer) >= self.max_seq_len:
                    # Enforce worker token budget before emitting a chunk.
                    if worker_max_tokens is not None and (total_tokens + self.max_seq_len) > worker_max_tokens:
                        return
                    chunk = buffer[:self.max_seq_len]
                    buffer = buffer[self.max_seq_len:]

                    total_tokens += len(chunk)

                    yield {
                        "input_ids": torch.tensor(chunk, dtype=torch.long),
                        "labels": torch.tensor(chunk, dtype=torch.long),
                        "attention_mask": torch.ones(len(chunk), dtype=torch.long),
                    }

        for example in dataset:
            # If remaining token budget cannot fit another full chunk,
            # stop immediately instead of scanning the stream forever.
            if worker_max_tokens is not None and (total_tokens + self.max_seq_len) > worker_max_tokens:
                return

            if self.dataset_iterable is not None:
                if seen < self.skip_examples:
                    seen += 1
                    continue
                if self.max_examples is not None and yielded >= self.max_examples:
                    break
                example_idx = yielded
                seen += 1
                yielded += 1
                if num_workers > 1 and (example_idx % num_workers) != worker_id:
                    continue
            elif num_workers > 1 and not dataset_sharded:
                # Fallback sharding for iterables that don't implement .shard().
                if (seen % num_workers) != worker_id:
                    seen += 1
                    continue
                seen += 1
            # Get text field (TinyStories uses "text")
            text = example.get("text", example.get("story", ""))
            if not text:
                continue

            pending_texts.append(text)
            if len(pending_texts) < self.tokenize_batch_size:
                continue

            for chunked in flush_pending(pending_texts):
                yield chunked
                if worker_max_tokens is not None and total_tokens >= worker_max_tokens:
                    return
            pending_texts.clear()

        # Flush trailing texts that did not fill a complete tokenization batch.
        if worker_max_tokens is not None and (total_tokens + self.max_seq_len) > worker_max_tokens:
            return

        for chunked in flush_pending(pending_texts):
            yield chunked
            if worker_max_tokens is not None and total_tokens >= worker_max_tokens:
                return


def load_tinystories(
    tokenizer,
    max_seq_len: int = 512,
    max_tokens: Optional[int] = None,
    split: str = "train",
    streaming: bool = True,
    tokenize_batch_size: int = 32,
) -> Dataset:
    """
    Load TinyStories dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
        max_tokens: Maximum tokens to load (None = all)
        split: "train" or "validation"
        streaming: Whether to use streaming mode

    Returns:
        Dataset yielding tokenized sequences
    """
    if streaming:
        return StreamingTextDataset(
            dataset_name="roneneldan/TinyStories",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split=split,
            max_tokens=max_tokens,
            tokenize_batch_size=tokenize_batch_size,
        )
    else:
        from datasets import load_dataset

        dataset = load_dataset("roneneldan/TinyStories", split=split)

        # Limit dataset size if needed
        if max_tokens:
            # Rough estimate: average ~100 tokens per story
            max_examples = max_tokens // 100
            dataset = dataset.select(range(min(len(dataset), max_examples)))

        texts = [ex["text"] for ex in dataset]

        return TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )


def load_openwebtext(
    tokenizer,
    max_seq_len: int = 512,
    max_tokens: Optional[int] = None,
    split: str = "train",
    streaming: bool = True,
    tokenize_batch_size: int = 32,
) -> Dataset:
    """
    Load OpenWebText-10k dataset (GPT-2 style web text).

    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
        max_tokens: Maximum tokens to load (None = all)
        split: Dataset split ("train" or "validation")
        streaming: Whether to use streaming mode

    Returns:
        Dataset yielding tokenized sequences
    """
    # Elriggs/openwebtext-100k only has 'train'.
    # We create a pseudo-split:
    # Validation: First 2000 examples
    # Train: Everything after 2000 examples
    VAL_EXAMPLES = 2000
    
    target_split = split
    skip_examples = 0
    max_examples = None

    # Handle pseudo-splitting
    if split == "validation":
        target_split = "train"
        max_examples = VAL_EXAMPLES
    elif split == "train":
        target_split = "train"
        skip_examples = VAL_EXAMPLES

    if streaming:
        return StreamingTextDataset(
            dataset_name="Elriggs/openwebtext-100k",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split=target_split,
            max_tokens=max_tokens,
            skip_examples=skip_examples,
            max_examples=max_examples,
            tokenize_batch_size=tokenize_batch_size,
        )
    else:
        from datasets import load_dataset

        # Note: Non-streaming load doesn't support 'skip' efficiently on load,
        # but we can slice the dataset object.
        dataset = load_dataset("Elriggs/openwebtext-100k", split="train")

        if split == "validation":
            # Select first N
            dataset = dataset.select(range(min(len(dataset), VAL_EXAMPLES)))
        else:
            # Select everything after N
            if len(dataset) > VAL_EXAMPLES:
                dataset = dataset.select(range(VAL_EXAMPLES, len(dataset)))
            else:
                # Fallback if dataset is too small
                print("WARNING: Dataset too small for validation split. Using full dataset for training.")

        # Limit dataset size if needed
        if max_tokens:
            # Rough estimate: average ~500 tokens per document
            max_docs = max_tokens // 500
            dataset = dataset.select(range(min(len(dataset), max_docs)))

        texts = [ex["text"] for ex in dataset]

        return TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )


def load_text_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_len: int = 512,
    max_tokens: Optional[int] = None,
    split: str = "train",
    streaming: bool = True,
    tokenize_batch_size: int = 32,
) -> Dataset:
    """
    Generic loader for text datasets.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
        max_tokens: Maximum tokens to load (None = all)
        split: Dataset split
        streaming: Whether to use streaming mode

    Returns:
        Dataset yielding tokenized sequences
    """
    # Local dataset saved with save_to_disk
    if os.path.isdir(dataset_name):
        info_path = os.path.join(dataset_name, "dataset_info.json")
        dict_path = os.path.join(dataset_name, "dataset_dict.json")
        if os.path.exists(info_path) or os.path.exists(dict_path):
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_name)
            if hasattr(dataset, "keys") and split in dataset:
                dataset = dataset[split]
            try:
                length = len(dataset)
                print(f"Loaded local dataset from {dataset_name} (split={split}, examples={length})")
            except Exception:
                print(f"Loaded local dataset from {dataset_name} (split={split})")
            if streaming:
                return StreamingTextDataset(
                    dataset_name=dataset_name,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    split=split,
                    max_tokens=max_tokens,
                    dataset_iterable=dataset,
                    tokenize_batch_size=tokenize_batch_size,
                )
            texts = []
            for ex in dataset:
                text = ex.get("text", ex.get("story", ex.get("content", "")))
                if text:
                    texts.append(text)
                if max_tokens and len(texts) * 200 >= max_tokens:
                    break
            return TextDataset(
                texts=texts,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len
            )

    # Use specialized loaders for known datasets
    if "tinystories" in dataset_name.lower():
        return load_tinystories(
            tokenizer,
            max_seq_len,
            max_tokens,
            split,
            streaming,
            tokenize_batch_size=tokenize_batch_size,
        )
    elif "openwebtext" in dataset_name.lower():
        return load_openwebtext(
            tokenizer,
            max_seq_len,
            max_tokens,
            split,
            streaming,
            tokenize_batch_size=tokenize_batch_size,
        )
    elif "fineweb-edu" in dataset_name.lower():
        return load_fineweb_edu(
            tokenizer,
            max_seq_len,
            max_tokens,
            split,
            streaming,
            tokenize_batch_size=tokenize_batch_size,
        )

    # Generic loader for other datasets
    if streaming:
        return StreamingTextDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split=split,
            max_tokens=max_tokens,
            tokenize_batch_size=tokenize_batch_size,
        )
    else:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)

        # Try common text field names
        texts = []
        for ex in dataset:
            text = ex.get("text", ex.get("story", ex.get("content", "")))
            if text:
                texts.append(text)
            if max_tokens and len(texts) * 200 >= max_tokens:  # rough estimate
                break

        return TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )


def load_fineweb_edu(
    tokenizer,
    max_seq_len: int = 2048,
    max_tokens: Optional[int] = None,
    split: str = "train",
    streaming: bool = True,
    dataset_subset: str = "sample-10BT",
    tokenize_batch_size: int = 32,
) -> Dataset:
    """
    Load FineWeb-Edu dataset.

    Args:
        tokenizer: HuggingFace tokenizer or TokenizerWrapper
        max_seq_len: Maximum sequence length
        max_tokens: Maximum tokens to load (None = all)
        split: Dataset split ("train" or "validation")
        streaming: Whether to use streaming mode
        dataset_subset: Dataset subset (e.g., "sample-10BT", "sample-100BT")

    Returns:
        Dataset yielding tokenized sequences
    """
    # FineWeb-Edu only has train split in the samples, we create pseudo-validation
    VAL_EXAMPLES = 10000

    target_split = "train"
    skip_examples = 0
    max_examples = None

    if split == "validation":
        max_examples = VAL_EXAMPLES
    elif split == "train":
        skip_examples = VAL_EXAMPLES

    if streaming:
        return StreamingTextDataset(
            dataset_name=f"HuggingFaceFW/fineweb-edu",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split=target_split,
            max_tokens=max_tokens,
            skip_examples=skip_examples,
            max_examples=max_examples,
            dataset_config=dataset_subset,
            tokenize_batch_size=tokenize_batch_size,
        )
    else:
        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceFW/fineweb-edu", dataset_subset, split="train")

        if split == "validation":
            dataset = dataset.select(range(min(len(dataset), VAL_EXAMPLES)))
        else:
            if len(dataset) > VAL_EXAMPLES:
                dataset = dataset.select(range(VAL_EXAMPLES, len(dataset)))

        if max_tokens:
            max_docs = max_tokens // 500
            dataset = dataset.select(range(min(len(dataset), max_docs)))

        texts = [ex["text"] for ex in dataset]

        return TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )


def load_alpaca(
    tokenizer,
    max_seq_len: int = 512
) -> Dataset:
    """
    Load Alpaca dataset for SFT.

    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length

    Returns:
        Dataset for instruction tuning
    """
    from datasets import load_dataset

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    def clean_text(text: str) -> str:
        return text.replace("<|endoftext|>", "").strip()

    # Format as instruction-response pairs
    texts = []
    for ex in dataset:
        instruction = clean_text(ex.get("instruction", ""))
        input_text = clean_text(ex.get("input", ""))
        output = clean_text(ex.get("output", ""))

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        texts.append(prompt)

    return TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=max_seq_len  # No overlap for SFT
    )


def get_tokenizer(tokenizer_path: Optional[str] = None):
    """
    Get tokenizer.

    Args:
        tokenizer_path: Path to custom tokenizer file. If None, uses GPT-2 tokenizer.

    Returns:
        Tokenizer (either custom TokenizerWrapper or GPT2Tokenizer)
    """
    if tokenizer_path is not None:
        # Load custom tokenizer
        from ..tokenizer import load_custom_tokenizer, TokenizerWrapper

        tokenizer = load_custom_tokenizer(tokenizer_path)
        return TokenizerWrapper(tokenizer)

    # Default: GPT-2 tokenizer
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    # Suppress max-length warnings since we chunk/truncate manually.
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Create a DataLoader from a dataset.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle (not applicable to streaming)
        num_workers: Number of worker processes
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader
    """
    # Streaming datasets don't support shuffle
    if isinstance(dataset, IterableDataset):
        shuffle = False

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(**loader_kwargs)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching.

    Args:
        batch: List of examples

    Returns:
        Batched tensors
    """
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
        "labels": torch.stack([ex["labels"] for ex in batch]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch])
    }


class SmokeTestDataset(Dataset):
    """
    Small dataset for smoke testing.

    Uses synthetic data to verify training works before
    committing to full dataset loading.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 512,
        n_samples: int = 1000
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_samples = n_samples

        # Generate random sequences (for quick testing only)
        # Real smoke test should use actual TinyStories
        self.sequences = [
            torch.randint(0, vocab_size, (max_seq_len,))
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.ones_like(tokens)
        }


def estimate_tokens(
    dataset_name: str,
    split: str = "train",
    sample_size: int = 1000
) -> Dict[str, float]:
    """
    Estimate total tokens in a dataset.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        sample_size: Number of examples to sample for estimation

    Returns:
        Estimation statistics
    """
    from datasets import load_dataset

    tokenizer = get_tokenizer()
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    total_tokens = 0
    n_samples = 0

    for example in dataset:
        text = example.get("text", example.get("story", ""))
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            n_samples += 1

        if n_samples >= sample_size:
            break

    avg_tokens = total_tokens / n_samples if n_samples > 0 else 0

    return {
        "sampled_examples": n_samples,
        "sampled_tokens": total_tokens,
        "avg_tokens_per_example": avg_tokens,
    }


if __name__ == "__main__":
    print("Testing data loading...")

    # Get tokenizer
    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Test smoke test dataset
    print("\n--- Smoke Test Dataset ---")
    smoke_dataset = SmokeTestDataset(n_samples=100)
    print(f"Smoke test dataset size: {len(smoke_dataset)}")
    sample = smoke_dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")

    # Test OpenWebText loading (streaming) - default dataset
    print("\n--- OpenWebText-10k (Streaming) ---")
    try:
        owt_dataset = load_openwebtext(
            tokenizer=tokenizer,
            max_seq_len=512,
            max_tokens=10000,  # Just 10K tokens for testing
            streaming=True
        )

        # Get a few samples
        dataloader = create_dataloader(owt_dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Sample decoded: {tokenizer.decode(batch['input_ids'][0][:50])}...")
    except Exception as e:
        print(f"Could not load OpenWebText (may need internet): {e}")

    # Estimate tokens
    print("\n--- Token Estimation ---")
    try:
        estimation = estimate_tokens("Elriggs/openwebtext-100k", sample_size=100)
        print(f"Avg tokens per example: {estimation['avg_tokens_per_example']:.1f}")
    except Exception as e:
        print(f"Could not estimate tokens: {e}")
