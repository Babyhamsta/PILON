"""
Evaluation Cache for Fair Model Comparison

Caches evaluation batches to ensure PILON and baseline models
are evaluated on exactly the same data for fair comparison.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from torch.utils.data import DataLoader


class EvalCache:
    """
    Cache evaluation batches for deterministic evaluation.

    Ensures different models (PILON vs baseline) are evaluated
    on exactly the same data for fair comparison.
    """

    def __init__(self, cache_path: Path, n_batches: int = 200):
        """
        Args:
            cache_path: Path to save/load cached batches
            n_batches: Number of batches to cache
        """
        self.cache_path = Path(cache_path)
        self.n_batches = n_batches
        self._batches: Optional[List[Dict[str, torch.Tensor]]] = None

    def exists(self) -> bool:
        """Check if cache file exists."""
        return self.cache_path.exists()

    def save(self, dataloader: DataLoader) -> None:
        """
        Save first n_batches from dataloader to cache.

        Args:
            dataloader: DataLoader to cache batches from
        """
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= self.n_batches:
                break

            # Store batch on CPU to avoid GPU memory issues
            cached_batch = {
                "input_ids": batch["input_ids"].cpu(),
                "labels": batch["labels"].cpu(),
            }

            # Optional attention mask
            if "attention_mask" in batch:
                cached_batch["attention_mask"] = batch["attention_mask"].cpu()

            batches.append(cached_batch)

        # Ensure parent directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata
        cache_data = {
            "batches": batches,
            "n_batches": len(batches),
            "version": "1.0"
        }
        torch.save(cache_data, self.cache_path)
        print(f"Saved {len(batches)} evaluation batches to {self.cache_path}")

    def load(self) -> List[Dict[str, torch.Tensor]]:
        """Load cached batches from file."""
        if self._batches is not None:
            return self._batches

        cache_data = torch.load(self.cache_path, weights_only=False)
        self._batches = cache_data["batches"]
        print(f"Loaded {len(self._batches)} evaluation batches from {self.cache_path}")
        return self._batches

    def get_batches(self, dataloader: Optional[DataLoader] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Get cached batches - load from cache if exists, else build and save.

        Args:
            dataloader: DataLoader to use if cache doesn't exist

        Returns:
            List of cached batches
        """
        if self.exists():
            return self.load()

        if dataloader is None:
            raise ValueError("Cache doesn't exist and no dataloader provided to build it")

        self.save(dataloader)
        return self.load()

    def iterate(self, device: str = "cuda") -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over cached batches, moving to specified device.

        Args:
            device: Device to move batches to

        Yields:
            Batches moved to device
        """
        batches = self.load()
        for batch in batches:
            yield {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def __len__(self) -> int:
        """Number of cached batches."""
        if self._batches is not None:
            return len(self._batches)
        if self.exists():
            return self.load().__len__()
        return 0


def evaluate_with_cache(
    model,
    eval_cache: EvalCache,
    device: str = "cuda",
    precision: str = "bf16"
) -> Dict[str, float]:
    """
    Evaluate model using cached batches.

    Args:
        model: Model to evaluate
        eval_cache: EvalCache containing evaluation batches
        device: Device to run evaluation on
        precision: Precision to use (bf16, fp16, fp32)

    Returns:
        Dictionary with val_loss and val_ppl
    """
    from contextlib import nullcontext

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Set up autocast
    if precision == "bf16" and device.startswith("cuda"):
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif precision == "fp16" and device.startswith("cuda"):
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    with torch.no_grad():
        for batch in eval_cache.iterate(device):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask", None)

            with autocast_ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # Count tokens (non-padding)
            if attention_mask is not None:
                n_tokens = attention_mask.sum().item()
            else:
                n_tokens = labels.numel()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)

    return {
        "val_loss": avg_loss,
        "val_ppl": perplexity,
        "n_batches": len(eval_cache),
        "n_tokens": total_tokens
    }


if __name__ == "__main__":
    # Test the eval cache
    print("Testing EvalCache...")

    # Create dummy data
    cache_path = Path("test_eval_cache.pt")

    class DummyDataset:
        def __init__(self, n_samples=100, seq_len=128):
            self.n_samples = n_samples
            self.seq_len = seq_len

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (self.seq_len,)),
                "labels": torch.randint(0, 1000, (self.seq_len,)),
                "attention_mask": torch.ones(self.seq_len)
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=8)

    # Create cache
    cache = EvalCache(cache_path, n_batches=10)

    # Save batches
    cache.save(dataloader)

    # Load and verify
    batches = cache.load()
    print(f"Loaded {len(batches)} batches")
    print(f"First batch input_ids shape: {batches[0]['input_ids'].shape}")

    # Clean up
    cache_path.unlink()
    print("Test passed!")
