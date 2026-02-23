"""
PILON-R Supervised Fine-Tuning (SFT)

Fine-tunes a pretrained PILON model on instruction-following data.
Used in Phase A.2 to verify that compression levels can follow instructions.

Usage:
    python -m pilon_r.sft checkpoint.pt --epochs 2 --output-dir outputs/sft
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import math
from typing import Optional, Dict, List
from contextlib import nullcontext
from tqdm import tqdm

from .core.config import SFTConfig, TrainingConfig
from .core.model import PILONTransformer
from .core.data import get_tokenizer
from .train import device_str, set_seed
from .core.metrics import compute_router_entropy


class InstructionDataset(Dataset):
    """Dataset for instruction-following fine-tuning.

    Optimized for large datasets using HuggingFace datasets' parallel processing.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        max_samples: Optional[int] = None,
        eos_token: Optional[str] = None,
        response_separator: str = "### Response:\n",
        max_supervised_ratio: float = 0.30,
        dataset_name: str = "teknium/OpenHermes-2.5",
        response_only: bool = True,
        num_proc: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.template = template
        self.eos_token = eos_token or ""
        self.response_separator = response_separator
        self.max_supervised_ratio = max_supervised_ratio
        self.dataset_name = dataset_name
        self.response_only = response_only
        self.eos_id = tokenizer.eos_token_id

        if self.eos_id is None:
            raise ValueError("Tokenizer eos_token_id is None.")

        # Determine number of workers for parallel processing
        if num_proc is None:
            import os
            num_proc = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid memory issues

        # Load SFT dataset
        from datasets import load_dataset
        print(f"Downloading/loading dataset from HuggingFace...")
        dataset = load_dataset(self.dataset_name, split="train")
        total_examples = len(dataset)
        print(f"Dataset has {total_examples:,} total examples")

        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
            print(f"Selected {len(dataset):,} examples")

        # Process dataset in parallel using HuggingFace's map
        print(f"Processing dataset with {num_proc} workers...")

        # First pass: extract and clean fields (batched for efficiency)
        def extract_and_clean_batch(examples):
            """Extract fields from a batch of examples."""
            results = {
                "instruction": [],
                "input_text": [],
                "output": [],
                "valid": []
            }

            for i in range(len(examples[next(iter(examples))])):
                # Build single example dict
                ex = {k: v[i] for k, v in examples.items()}
                fields = self._extract_fields(ex)

                if fields is None:
                    results["instruction"].append("")
                    results["input_text"].append("")
                    results["output"].append("")
                    results["valid"].append(False)
                    continue

                instruction = self._clean_text(fields.get("instruction", ""))
                input_text = self._clean_text(fields.get("input", ""))
                output = self._clean_text(fields.get("output", ""))

                if not output:
                    results["instruction"].append("")
                    results["input_text"].append("")
                    results["output"].append("")
                    results["valid"].append(False)
                    continue

                results["instruction"].append(instruction)
                results["input_text"].append(input_text)
                results["output"].append(output)
                results["valid"].append(True)

            return results

        # Apply extraction with progress bar
        dataset = dataset.map(
            extract_and_clean_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Extracting fields"
        )

        # Filter invalid examples
        initial_count = len(dataset)
        dataset = dataset.filter(lambda x: x["valid"], num_proc=num_proc, desc="Filtering valid")
        filtered_count = len(dataset)
        print(f"Filtered {initial_count - filtered_count:,} invalid examples")

        # Tokenize in parallel
        def tokenize_batch(examples):
            """Tokenize a batch of examples."""
            results = {
                "input_ids": [],
                "labels": [],
                "lengths": [],
                "supervised_tokens": [],
                "total_tokens": [],
                "valid": []
            }

            batch_size = len(examples["instruction"])

            for i in range(batch_size):
                instruction = examples["instruction"][i]
                input_text = examples["input_text"][i]
                output = examples["output"][i]

                if input_text:
                    full_instruction = f"{instruction}\n\nInput: {input_text}"
                else:
                    full_instruction = instruction

                prompt_text = self.template.format(instruction=full_instruction, response="")

                # Tokenize
                prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                response_tokens = self.tokenizer.encode(output, add_special_tokens=False)

                # Remove EOS and add exactly one at end
                prompt_tokens = [tid for tid in prompt_tokens if tid != self.eos_id]
                response_tokens = [tid for tid in response_tokens if tid != self.eos_id]
                response_tokens.append(self.eos_id)

                # Truncate
                if len(prompt_tokens) >= self.max_seq_len:
                    prompt_tokens = prompt_tokens[:self.max_seq_len - 1]

                max_response_len = self.max_seq_len - len(prompt_tokens)
                response_tokens = response_tokens[:max_response_len]

                if len(response_tokens) == 0:
                    results["input_ids"].append([])
                    results["labels"].append([])
                    results["lengths"].append(0)
                    results["supervised_tokens"].append(0)
                    results["total_tokens"].append(0)
                    results["valid"].append(False)
                    continue

                response_tokens[-1] = self.eos_id
                full_tokens = prompt_tokens + response_tokens
                response_start = len(prompt_tokens)

                if response_start >= len(full_tokens):
                    results["input_ids"].append([])
                    results["labels"].append([])
                    results["lengths"].append(0)
                    results["supervised_tokens"].append(0)
                    results["total_tokens"].append(0)
                    results["valid"].append(False)
                    continue

                # Build labels
                labels = list(full_tokens)
                response_len = len(full_tokens) - response_start

                mask_prefix = 0
                if not self.response_only:
                    supervised_ratio = response_len / max(len(full_tokens), 1)
                    if supervised_ratio > self.max_supervised_ratio:
                        keep = max(1, int(len(full_tokens) * self.max_supervised_ratio))
                        if keep < response_len:
                            mask_prefix = response_len - keep

                for j in range(response_start + mask_prefix):
                    labels[j] = -100

                supervised_tokens = len(full_tokens) - (response_start + mask_prefix)

                results["input_ids"].append(full_tokens)
                results["labels"].append(labels)
                results["lengths"].append(len(full_tokens))
                results["supervised_tokens"].append(supervised_tokens)
                results["total_tokens"].append(len(full_tokens))
                results["valid"].append(True)

            return results

        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=500,
            num_proc=num_proc,
            remove_columns=["instruction", "input_text", "output", "valid"],
            desc="Tokenizing"
        )

        # Filter invalid after tokenization
        pre_filter = len(dataset)
        dataset = dataset.filter(lambda x: x["valid"], num_proc=num_proc, desc="Final filtering")
        post_filter = len(dataset)
        skipped = (initial_count - filtered_count) + (pre_filter - post_filter)

        # Calculate statistics
        total_supervised = sum(dataset["supervised_tokens"])
        total_tokens = sum(dataset["total_tokens"])

        if skipped > 0:
            print(f"Skipped {skipped:,} examples total")
        if total_tokens > 0:
            avg_supervised = total_supervised / total_tokens
            print(f"Avg supervised ratio: {avg_supervised:.3f}")

        if len(dataset) == 0:
            raise ValueError(
                "SFT dataset is empty after filtering. Increase max_seq_len or reduce prompt length."
            )

        # Store the processed dataset with numpy format for faster tensor creation
        self._dataset = dataset.with_format("numpy")
        self._length = len(dataset)
        print(f"Loaded {self._length:,} examples")

    def _clean_text(self, text: str) -> str:
        """Clean text by removing special tokens."""
        text = text.replace("<|endoftext|>", "")
        if self.eos_token:
            text = text.replace(self.eos_token, "")
        return text.strip()

    def _extract_fields(self, ex: Dict) -> Optional[Dict[str, str]]:
        """Extract instruction/input/output fields from various dataset formats."""
        if "output" in ex:
            return {
                "instruction": ex.get("instruction", ""),
                "input": ex.get("input", ""),
                "output": ex.get("output", "")
            }
        if "response" in ex:
            return {
                "instruction": ex.get("instruction", ex.get("prompt", "")),
                "input": ex.get("input", ""),
                "output": ex.get("response", "")
            }
        if "messages" in ex and isinstance(ex["messages"], list):
            messages = ex["messages"]
            user_parts = []
            assistant_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in {"user", "human"}:
                    user_parts.append(content)
                elif role in {"assistant", "gpt"}:
                    assistant_parts.append(content)
            if assistant_parts:
                return {
                    "instruction": "\n".join(user_parts),
                    "input": "",
                    "output": assistant_parts[-1]
                }
        if "conversations" in ex and isinstance(ex["conversations"], list):
            user_parts = []
            assistant_parts = []
            for msg in ex["conversations"]:
                role = msg.get("from", "")
                content = msg.get("value", "")
                if role in {"human", "user"}:
                    user_parts.append(content)
                elif role in {"assistant", "gpt"}:
                    assistant_parts.append(content)
            if assistant_parts:
                return {
                    "instruction": "\n".join(user_parts),
                    "input": "",
                    "output": assistant_parts[-1]
                }
        if "text" in ex:
            return {"instruction": "", "input": "", "output": ex.get("text", "")}
        return None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example, converting to tensors on-demand."""
        item = self._dataset[idx]
        # torch.from_numpy is faster than torch.tensor for numpy arrays
        # .copy() needed because HF datasets returns non-writable arrays
        return {
            "input_ids": torch.from_numpy(item["input_ids"].copy()),
            "labels": torch.from_numpy(item["labels"].copy()),
            "attention_mask": torch.ones(item["lengths"], dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 50256) -> Dict[str, torch.Tensor]:
    """Collate and pad batch."""
    max_len = max(ex["input_ids"].size(0) for ex in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for ex in batch:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad on right
        input_ids.append(torch.cat([
            ex["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ]))
        labels.append(torch.cat([
            ex["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)  # -100 = ignore in loss
        ]))
        attention_mask.append(torch.cat([
            ex["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ]))

    attention_mask = torch.stack(attention_mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": attention_mask
    }


def resize_token_embeddings(model: PILONTransformer, new_vocab_size: int) -> None:
    """Resize token embedding and LM head to new vocab size."""
    old_vocab_size = model.config.vocab_size
    if new_vocab_size == old_vocab_size:
        return
    if new_vocab_size < old_vocab_size:
        raise ValueError("New vocab size cannot be smaller than existing vocab size.")

    device = model.token_embedding.weight.device
    dtype = model.token_embedding.weight.dtype

    new_embed = nn.Embedding(new_vocab_size, model.config.d_model).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_embed.weight[:old_vocab_size].copy_(model.token_embedding.weight)
        nn.init.normal_(new_embed.weight[old_vocab_size:], mean=0.0, std=0.02)
    model.token_embedding = new_embed

    new_lm_head = nn.Linear(model.config.d_model, new_vocab_size, bias=False).to(device=device, dtype=dtype)
    new_lm_head.weight = model.token_embedding.weight
    model.lm_head = new_lm_head
    model.config.vocab_size = new_vocab_size


def ensure_sft_special_tokens(tokenizer) -> bool:
    """
    Ensure SFT uses dedicated EOS and PAD tokens distinct from <|endoftext|>.
    Returns True if a new token was added.
    """
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


def load_pretrained_model(checkpoint_path: Path, device: str = "cuda") -> PILONTransformer:
    """Load a pretrained PILON model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not hasattr(config, "vocab_size"):
        config = checkpoint.get("model_config", config)
    if not hasattr(config, "vocab_size"):
        from .core.config import ModelConfig
        print("WARNING: Checkpoint config missing model parameters; falling back to default ModelConfig.")
        config = ModelConfig()
    model = PILONTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


class SFTCollate:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_len = max(ex["input_ids"].size(0) for ex in batch)

        # Pre-allocate output tensors (much faster than per-sample tensor creation)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        # Fill in actual values (no tensor creation in loop)
        for i, ex in enumerate(batch):
            seq_len = ex["input_ids"].size(0)
            input_ids[i, :seq_len] = ex["input_ids"]
            labels[i, :seq_len] = ex["labels"]
            attention_mask[i, :seq_len] = ex["attention_mask"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def run_sft(
    model: PILONTransformer,
    config: SFTConfig,
    output_dir: Path,
    device: str = "cuda",
    precision: str = "bf16",
    router_dense_steps: int = 500,
    router_anneal_steps: int = 500,
    max_seq_len: Optional[int] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    compile_model: bool = False,
    freeze_primitives: bool = False
) -> PILONTransformer:
    """
    Run supervised fine-tuning on instruction data.

    Args:
        model: Pretrained PILON model
        config: SFT configuration
        output_dir: Output directory
        device: Device to train on
        precision: Precision (bf16/fp16/fp32)
        batch_size: Micro batch size per GPU
        num_workers: Number of dataloader workers
        compile_model: Whether to use torch.compile
        freeze_primitives: Whether to freeze shared primitives

    Returns:
        Fine-tuned model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("SUPERVISED FINE-TUNING")
    print("=" * 50)

    # Setup
    tokenizer = get_tokenizer()
    added_tokens = ensure_sft_special_tokens(tokenizer)
    if added_tokens:
        resize_token_embeddings(model, len(tokenizer))
        print(f"Resized vocab for SFT special tokens: {model.config.vocab_size}")
    
    # Freeze primitives if requested
    if freeze_primitives:
        print("Freezing shared primitive banks (training composition weights only)...")
        if getattr(model, "primitive_banks", None) is not None:
            model.primitive_banks.requires_grad_(False)
        else:
            print("WARNING: --freeze-primitives requested but no primitive_banks found.")

    model.train()

    # Compile model if requested
    if compile_model:
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"WARNING: torch.compile failed: {e}")

    # Dataset
    print(f"Loading SFT dataset: {config.dataset}...")
    effective_max_seq_len = max_seq_len if max_seq_len is not None else min(512, model.config.max_seq_len)
    dataset = InstructionDataset(
        tokenizer=tokenizer,
        max_seq_len=effective_max_seq_len,
        template=config.template,
        eos_token=tokenizer.eos_token,
        response_separator="### Response:\n",
        dataset_name=config.dataset,
        response_only=True,
        max_samples=max_samples,
        num_proc=num_workers if num_workers > 0 else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SFTCollate(tokenizer.pad_token_id),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Tokenizer alignment checks
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer eos_token_id is None.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None.")
    print(f"Tokenizer IDs: eos_token_id={tokenizer.eos_token_id}, pad_token_id={tokenizer.pad_token_id}")
    bad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if bad_token_id is not None and bad_token_id == tokenizer.eos_token_id:
        raise ValueError("SFT requires eos_token_id != <|endoftext|> token id.")
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({len(tokenizer)}) != model vocab_size ({model.config.vocab_size})."
        )
    if bad_token_id is not None:
        # Sample check instead of iterating all examples (much faster)
        sample_size = min(1000, len(dataset))
        sample_indices = range(0, len(dataset), max(1, len(dataset) // sample_size))
        for idx in sample_indices:
            if bad_token_id in dataset._dataset[idx]["labels"]:
                raise AssertionError("JUNK TOKEN IN LABELS: <|endoftext|> found.")

    # First-example sanity print (once)
    first_batch = next(iter(dataloader))
    first_input_ids = first_batch["input_ids"][0].tolist()
    first_labels = first_batch["labels"][0].tolist()
    first_mask = first_batch["attention_mask"][0]
    if first_mask.dim() == 1:
        valid_len = int(first_mask.sum().item())
    else:
        valid_len = len(first_input_ids)
    print("\nFIRST EXAMPLE SANITY CHECK")
    print("INPUT:")
    print(tokenizer.decode(first_input_ids[:valid_len], skip_special_tokens=False))
    print("LABELS:")
    label_preview = []
    for tok_id, lab in zip(first_input_ids[:valid_len], first_labels[:valid_len]):
        if lab == -100:
            label_preview.append("-100")
        else:
            label_preview.append(tokenizer.decode([tok_id], skip_special_tokens=False))
    print(" ".join(label_preview[:200]))

    def count_params(params):
        return sum(p.numel() for p in params)

    # Optimizer with explicit param groups
    # Important: Model might be compiled, so use raw params if needed, but named_parameters works on compiled too
    router_params = []
    expert_params = []
    base_params = []
    
    # We might have frozen primitives, so filter by requires_grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = count_params(trainable_params)
    total_count = count_params(model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count:.1%})")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "router" in name:
            router_params.append(param)
        elif "experts" in name or "expert_compositions" in name:
            expert_params.append(param)
        else:
            base_params.append(param)

    print(f"Param groups: router={count_params(router_params):,} | experts={count_params(expert_params):,} | base={count_params(base_params):,}")

    optimizer = AdamW(
        [
            {"params": base_params, "lr": config.lr},
            {"params": router_params, "lr": config.lr},
            {"params": expert_params, "lr": config.lr},
        ],
        weight_decay=0.01
    )

    # Precision
    dev = device_str(device)
    amp_dtype = None
    if dev.startswith("cuda"):
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif precision == "fp16":
            amp_dtype = torch.float16

    use_scaler = amp_dtype == torch.float16
    scaler = GradScaler("cuda", enabled=use_scaler) if use_scaler else GradScaler("cuda", enabled=False)

    # MoE aux loss weight (if enabled)
    aux_loss_weight = 0.0
    if model.is_moe_model() and model.config.primitive_config.moe_config is not None:
        aux_loss_weight = 0.01
        model.config.primitive_config.moe_config.aux_loss_weight = aux_loss_weight

    # Enforce MoE routing top-k for SFT (no schedules)
    if model.is_moe_model():
        moe_cfg = model.config.primitive_config.moe_config
        if moe_cfg is None:
            raise ValueError("MoE model missing moe_config in SFT.")
        if moe_cfg.top_k is None:
            raise ValueError("moe_top_k must be set for MoE SFT (no None allowed).")
        assert moe_cfg.n_experts >= moe_cfg.top_k, "moe_top_k must be <= n_experts"
        for layer in model.layers:
            if hasattr(layer.ffn, "runtime_top_k"):
                layer.ffn.runtime_top_k = moe_cfg.top_k

    # Training loop
    steps_per_epoch = len(dataloader)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = min(config.warmup_steps, total_steps // 10)
    log_every = 100

    print(f"Training for {config.epochs} epochs ({total_steps} steps)")
    print(f"Steps per epoch: {steps_per_epoch}")
    if model.is_moe_model():
        print(f"MoE enabled: aux_loss_weight={aux_loss_weight}")

    def save_checkpoint(step: int, epoch_idx: int, is_final: bool = False) -> None:
        ckpt_name = "sft_model.pt" if is_final else f"sft_checkpoint_step_{step}.pt"
        save_path = output_dir / ckpt_name
        sft_checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "sft_config": config,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_scaler else None,
            "step": step,
            "epoch": epoch_idx,
        }
        torch.save(sft_checkpoint, save_path)
        print(f"Saved SFT checkpoint to: {save_path}")

    global_step = 0
    aux_loss_seen_nonzero = False
    last_router_entropy = None
    router_entropy_stagnation = 0
    dev = device_str(device)
    for epoch in range(config.epochs):
        # Use tensors to accumulate loss without GPU sync each step
        epoch_loss = torch.tensor(0.0, device=dev)
        epoch_total_loss = torch.tensor(0.0, device=dev)
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Learning rate warmup
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = config.lr * lr_scale

            # Forward pass
            optimizer.zero_grad(set_to_none=True)

            ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                aux_weight_value = 0.0
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                # Shift for next-token prediction (causal LM)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, model.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                aux_loss = outputs.get("aux_loss", 0.0)
                if isinstance(aux_loss, torch.Tensor):
                    aux_loss_value = aux_loss.item()
                else:
                    aux_loss_value = float(aux_loss) if aux_loss is not None else 0.0

                if model.is_moe_model():
                    aux_weight_value = min(aux_loss_weight, float(lm_loss.detach()) * 0.5)
                    loss = lm_loss + aux_weight_value * aux_loss
                else:
                    loss = lm_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"WARNING: NaN/Inf loss at step {global_step} | "
                    f"lm_loss={lm_loss.item():.6f} | aux_loss={aux_loss_value:.6f} | "
                    f"aux_weight={aux_loss_weight}"
                )

            assert loss.requires_grad, "Loss does not require grad in SFT."

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate losses without GPU sync (detach to avoid graph retention)
            epoch_loss += lm_loss.detach()
            epoch_total_loss += loss.detach()
            n_batches += 1
            if aux_loss_value > 0:
                aux_loss_seen_nonzero = True
            global_step += 1

            # Update progress bar (only sync every 10 steps to reduce overhead)
            if global_step % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{lm_loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            if global_step % log_every == 0:
                # Compute metrics only at log intervals (single GPU sync point)
                pct_supervised = (labels != -100).float().mean().item()
                avg_loss = (epoch_loss / n_batches).item()
                avg_total = (epoch_total_loss / n_batches).item()
                log_parts = [
                    f"Epoch {epoch+1}/{config.epochs}",
                    f"Step {global_step}/{total_steps}",
                    f"Loss: {avg_loss:.4f}",
                    f"Total: {avg_total:.4f}",
                    f"Aux: {aux_loss_value:.4f}",
                    f"pct_supervised={pct_supervised:.4f}",
                ]

                # MoE router metrics
                if model.is_moe_model():
                    moe_cfg = model.config.primitive_config.moe_config
                    moe_runtime_top_k = moe_cfg.top_k
                    moe_configured_top_k = moe_cfg.top_k

                    entropies = []
                    for layer in model.layers:
                        ffn = layer.ffn
                        if hasattr(ffn, "_last_router_logits") and ffn._last_router_logits is not None:
                            entropy = compute_router_entropy(ffn._last_router_logits)
                            entropies.append(entropy)

                    router_entropy = sum(entropies) / len(entropies) if entropies else None
                    router_effective_k = math.exp(router_entropy) if router_entropy is not None else None
                    if last_router_entropy is not None and router_entropy is not None:
                        if abs(router_entropy - last_router_entropy) <= 1e-4:
                            router_entropy_stagnation += 1
                        else:
                            router_entropy_stagnation = 0
                        if router_entropy_stagnation >= 5:
                            print(
                                "WARNING: Router entropy did not change for multiple log intervals; "
                                "routing may be static."
                            )
                            router_entropy_stagnation = 0
                    if router_entropy is not None:
                        last_router_entropy = router_entropy

                    # Gradient norms by component
                    def grad_norm(params):
                        grads = [p.grad for p in params if p.grad is not None]
                        if not grads:
                            return 0.0
                        return torch.stack([g.norm(2) for g in grads]).mean().item()

                    embed_params = list(model.token_embedding.parameters()) + list(model.position_embedding.parameters())
                    attn_params = [p for n, p in model.named_parameters() if "attention" in n]
                    ffn_params = [p for n, p in model.named_parameters() if "ffn" in n]
                    router_params = [p for n, p in model.named_parameters() if "router" in n]
                    expert_params = [p for n, p in model.named_parameters() if "experts" in n or "expert_compositions" in n]

                    embed_gn = grad_norm(embed_params)
                    attn_gn = grad_norm(attn_params)
                    ffn_gn = grad_norm(ffn_params)
                    router_gn = grad_norm(router_params)
                    expert_gn = grad_norm(expert_params)

                    # Hard sanity checks
                    assert any(p.grad is not None for p in router_params), "Router params have no grads in SFT."
                    assert any((p.grad is not None and p.grad.abs().sum() > 0) for p in expert_params), (
                        "Expert params have zero grads in SFT."
                    )

                    log_parts.extend([
                        f"moe_enabled={model.is_moe_model()}",
                        f"router_top_k_runtime={moe_runtime_top_k}",
                        f"router_top_k_configured={moe_configured_top_k}",
                        f"router_entropy={router_entropy:.4f}" if router_entropy is not None else "router_entropy=None",
                        f"router_effective_k={router_effective_k:.4f}" if router_effective_k is not None else "router_effective_k=None",
                        f"aux_loss_weight={aux_loss_weight}",
                        f"aux_loss_raw={aux_loss_value:.4f}",
                        f"aux_loss_scaled={(aux_weight_value * aux_loss_value):.4f}",
                        f"aux_loss_ratio={(aux_weight_value * aux_loss_value)/(avg_loss + 1e-8):.4f}",
                        f"grad_embed={embed_gn:.4f}",
                        f"grad_attn={attn_gn:.4f}",
                        f"grad_ffn={ffn_gn:.4f}",
                        f"grad_router={router_gn:.4f}",
                        f"grad_expert={expert_gn:.4f}",
                    ])

                print(" | ".join(log_parts))

            if config.save_every and global_step % config.save_every == 0:
                save_checkpoint(global_step, epoch_idx=epoch, is_final=False)

        avg_epoch_loss = (epoch_loss / n_batches).item()
        avg_epoch_total = (epoch_total_loss / n_batches).item()
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f} | Avg Total: {avg_epoch_total:.4f}")

    # Aux loss sanity check (should be non-zero occasionally if enabled)
    if model.is_moe_model() and aux_loss_weight > 0 and not aux_loss_seen_nonzero:
        raise AssertionError(
            "aux_loss_value remained zero during SFT despite aux_loss_weight > 0."
        )

    # Save model
    save_checkpoint(global_step, epoch_idx=config.epochs - 1, is_final=True)

    # Generate samples
    generate_sft_samples(model, tokenizer, output_dir, device)

    return model


def generate_sft_samples(
    model: PILONTransformer,
    tokenizer,
    output_dir: Path,
    device: str = "cuda"
):
    """Generate sample outputs from the SFT model."""
    model.eval()

    test_instructions = [
        "Write a short story about a dog.",
        "Explain what the sun is in simple terms.",
        "List three colors.",
        "What is 2 + 2?",
        "Describe a happy memory.",
    ]

    template = "### Instruction:\n{instruction}\n\n### Response:\n"

    samples = []
    for instruction in test_instructions:
        prompt = template.format(instruction=instruction)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                do_sample=True
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        samples.append({
            "instruction": instruction,
            "response": response[len(prompt):]
        })

    # Save samples
    samples_path = output_dir / "sft_samples.txt"
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write("SFT GENERATION SAMPLES\n")
        f.write("=" * 60 + "\n\n")

        for sample in samples:
            f.write(f"Instruction: {sample['instruction']}\n")
            f.write(f"Response: {sample['response']}\n")
            f.write("-" * 40 + "\n\n")

    print(f"Saved samples to: {samples_path}")

    # Print samples
    print("\n" + "=" * 60)
    print("SFT SAMPLES")
    print("=" * 60)
    for sample in samples:
        print(f"\nInstruction: {sample['instruction']}")
        print(f"Response: {sample['response'][:200]}...")

    model.train()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune PILON model on instructions")
    parser.add_argument("checkpoint", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs/sft", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps (0 = disable)")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Override max sequence length for SFT")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of SFT examples")
    parser.add_argument("--moe", action="store_true", help="Enable MoE routing (must match checkpoint)")
    parser.add_argument("--n-experts", type=int, default=None, help="Number of experts (must match checkpoint)")
    parser.add_argument("--moe-top-k", "--moe_top_k", type=int, default=None, help="Top-k experts per token for MoE")
    parser.add_argument("--aux-loss-weight", type=float, default=None, help="Auxiliary loss weight for MoE")
    parser.add_argument("--router-dense-steps", type=int, default=500,
                        help="Steps of dense routing before annealing (MoE)")
    parser.add_argument("--router-anneal-steps", type=int, default=500,
                        help="Steps to anneal router top-k to target (MoE)")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training")
    parser.add_argument("--freeze-primitives", action="store_true", help="Freeze shared primitives, train only composition weights")
    args = parser.parse_args()

    # Device fallback
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

    set_seed(42)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_pretrained_model(Path(args.checkpoint), args.device)
    print(f"Model loaded: {model.config.ffn_type} FFN")

    # Optional max_seq_len override
    if args.max_seq_len is not None:
        if args.max_seq_len <= 0:
            raise ValueError("--max-seq-len must be positive.")
        if args.max_seq_len > model.config.max_seq_len:
            # Expand position embeddings
            old_len = model.config.max_seq_len
            new_len = args.max_seq_len
            d_model = model.config.d_model
            new_pos_emb = torch.nn.Embedding(new_len, d_model).to(model.position_embedding.weight.device)
            with torch.no_grad():
                new_pos_emb.weight[:old_len].copy_(model.position_embedding.weight)
                nn.init.normal_(new_pos_emb.weight[old_len:], mean=0.0, std=0.02)
            model.position_embedding = new_pos_emb
            model.config.max_seq_len = new_len
            print(f"Expanded max_seq_len from {old_len} to {new_len}")
        else:
            print(f"Using max_seq_len={args.max_seq_len} (<= checkpoint {model.config.max_seq_len})")

    # Validate/override MoE config if requested
    if args.moe:
        if not model.is_moe_model() or model.config.primitive_config.moe_config is None:
            raise ValueError("Checkpoint model is not MoE-enabled; cannot enable --moe for SFT.")
        moe_cfg = model.config.primitive_config.moe_config
        if args.n_experts is not None and args.n_experts != moe_cfg.n_experts:
            raise ValueError(
                f"--n-experts ({args.n_experts}) must match checkpoint n_experts ({moe_cfg.n_experts})."
            )
        if args.moe_top_k is not None:
            if args.moe_top_k > moe_cfg.n_experts:
                raise ValueError(
                    f"--moe-top-k ({args.moe_top_k}) must be <= n_experts ({moe_cfg.n_experts})."
                )
            moe_cfg.top_k = args.moe_top_k
        if args.aux_loss_weight is not None:
            moe_cfg.aux_loss_weight = args.aux_loss_weight
    else:
        if args.n_experts is not None or args.moe_top_k is not None or args.aux_loss_weight is not None:
            raise ValueError("MoE args provided without --moe. Use --moe or remove MoE args.")

    # SFT config
    config = SFTConfig(epochs=args.epochs, lr=args.lr, save_every=args.save_every)

    # Run SFT
    run_sft(
        model=model,
        config=config,
        output_dir=Path(args.output_dir),
        device=args.device,
        precision=args.precision,
        router_dense_steps=args.router_dense_steps,
        router_anneal_steps=args.router_anneal_steps,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compile_model=args.compile,
        freeze_primitives=args.freeze_primitives
    )


if __name__ == "__main__":
    main()
