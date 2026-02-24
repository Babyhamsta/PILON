"""
PILON-R 360M Model Configuration

Configurations for the 360M parameter comparison experiment:
- PILON-360M: Compositional FFN with primitives
- Dense-360M: Standard dense FFN baseline

Both configurations target ~360M parameters with identical
embedding, attention, and architectural choices.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.config import ModelConfig, PrimitiveConfig, BandConfig, TrainingConfig


# ============================================================================
# Shared Base Configuration (identical for both models)
# ============================================================================

SHARED_360M_CONFIG = {
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "d_head": 64,
    "d_ff": 4096,  # 4x d_model
    "vocab_size": 50257,  # GPT-2 tokenizer
    "max_seq_len": 2048,
    "dropout": 0.1,
    "norm_type": "rmsnorm",
    "checkpoint_ffn": False,  # Throughput default; can be re-enabled when VRAM-bound
}


# ============================================================================
# PILON-360M: Compositional FFN Configuration
# ============================================================================

def _create_360m_bands() -> List[BandConfig]:
    """
    Create 3-band configuration for 24 layers.

    - Early band (layers 0-7): General pattern extraction
    - Middle band (layers 8-15): Semantic composition
    - Late band (layers 16-23): Task-specific refinement
    """
    return [
        BandConfig(name="early", layers=list(range(0, 8))),
        BandConfig(name="middle", layers=list(range(8, 16))),
        BandConfig(name="late", layers=list(range(16, 24))),
    ]


# To match ~337M params for both models, we use n_primitives=80, rank=80
# This gives PILON ~332M params vs Dense ~337M params (within 1.5%)
PILON_360M_PRIMITIVE_CONFIG = {
    "n_primitives": 80,
    "rank": 80,
    "top_k": 8,
    "share_fc1_fc2": False,  # Separate banks for fc1 and fc2
    "composition_type": "static_per_layer",
    "temperature": 1.0,
    "activation": "gelu",
    "moe_config": None,  # Static composition, not MoE
    "forward_fast_mode": "on",
    "forward_fast_min_topk": 1,
}


# Pre-built configurations
MODEL_360M_PILON = ModelConfig(
    **SHARED_360M_CONFIG,
    ffn_type="compositional",
    primitive_config=PrimitiveConfig(
        **PILON_360M_PRIMITIVE_CONFIG,
        bands=_create_360m_bands()
    )
)

MODEL_360M_DENSE = ModelConfig(
    **SHARED_360M_CONFIG,
    ffn_type="standard",
)


# ============================================================================
# Factory Functions
# ============================================================================

def get_360m_config(ffn_type: str = "compositional") -> ModelConfig:
    """
    Get 360M model configuration.

    Args:
        ffn_type: "compositional" for PILON, "standard" for dense baseline

    Returns:
        ModelConfig for 360M model
    """
    if ffn_type == "compositional":
        return get_360m_pilon_config()
    elif ffn_type == "standard":
        return get_360m_dense_config()
    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}. Use 'compositional' or 'standard'")


def get_360m_pilon_config(
    n_primitives: Optional[int] = None,
    rank: Optional[int] = None,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
) -> ModelConfig:
    """
    Get PILON-360M configuration with optional overrides.

    Args:
        n_primitives: Number of primitives (default: 80)
        rank: Rank of each primitive (default: 80)
        top_k: Sparsity - top-k primitives used (default: 8)
        temperature: Softmax temperature (default: 1.0)

    Returns:
        ModelConfig for PILON-360M
    """
    primitive_config = PrimitiveConfig(
        n_primitives=n_primitives or PILON_360M_PRIMITIVE_CONFIG["n_primitives"],
        rank=rank or PILON_360M_PRIMITIVE_CONFIG["rank"],
        top_k=top_k or PILON_360M_PRIMITIVE_CONFIG["top_k"],
        share_fc1_fc2=PILON_360M_PRIMITIVE_CONFIG["share_fc1_fc2"],
        composition_type=PILON_360M_PRIMITIVE_CONFIG["composition_type"],
        temperature=temperature or PILON_360M_PRIMITIVE_CONFIG["temperature"],
        activation=PILON_360M_PRIMITIVE_CONFIG["activation"],
        moe_config=None,
        forward_fast_mode=PILON_360M_PRIMITIVE_CONFIG["forward_fast_mode"],
        forward_fast_min_topk=PILON_360M_PRIMITIVE_CONFIG["forward_fast_min_topk"],
        bands=_create_360m_bands()
    )

    return ModelConfig(
        **SHARED_360M_CONFIG,
        ffn_type="compositional",
        primitive_config=primitive_config
    )


def get_360m_dense_config() -> ModelConfig:
    """
    Get Dense-360M baseline configuration.

    Returns:
        ModelConfig for Dense-360M
    """
    return ModelConfig(
        **SHARED_360M_CONFIG,
        ffn_type="standard",
    )


def get_360m_pilon_tiered_config(
    n_hot: int = 16,
    swap_interval: int = 100,
) -> ModelConfig:
    """
    Get PILON-360M configuration with tiered primitive bank (Phase B.5b).

    Only n_hot primitives live in VRAM; the rest are in CPU pinned memory.

    Args:
        n_hot: Number of hot primitives in VRAM
        swap_interval: Steps between hot/warm swaps

    Returns:
        ModelConfig for PILON-360M with tiered banks
    """
    primitive_config = PrimitiveConfig(
        n_primitives=PILON_360M_PRIMITIVE_CONFIG["n_primitives"],
        rank=PILON_360M_PRIMITIVE_CONFIG["rank"],
        top_k=PILON_360M_PRIMITIVE_CONFIG["top_k"],
        share_fc1_fc2=PILON_360M_PRIMITIVE_CONFIG["share_fc1_fc2"],
        composition_type=PILON_360M_PRIMITIVE_CONFIG["composition_type"],
        temperature=PILON_360M_PRIMITIVE_CONFIG["temperature"],
        activation=PILON_360M_PRIMITIVE_CONFIG["activation"],
        moe_config=None,
        forward_fast_mode=PILON_360M_PRIMITIVE_CONFIG["forward_fast_mode"],
        forward_fast_min_topk=PILON_360M_PRIMITIVE_CONFIG["forward_fast_min_topk"],
        bands=_create_360m_bands(),
        n_hot=n_hot,
        swap_interval=swap_interval,
    )

    return ModelConfig(
        **SHARED_360M_CONFIG,
        ffn_type="compositional",
        primitive_config=primitive_config,
    )


def get_360m_pilon_exit_config(
    n_hot: int = 16,
    exit_threshold: float = 0.5,
    swap_interval: int = 100,
) -> ModelConfig:
    """
    Get PILON-360M configuration with tiered bank + early exit (Phase B.5b + B.5c).

    Args:
        n_hot: Number of hot primitives in VRAM
        exit_threshold: Skip confidence threshold for early exit
        swap_interval: Steps between hot/warm swaps

    Returns:
        ModelConfig for PILON-360M with tiered banks and early exit
    """
    primitive_config = PrimitiveConfig(
        n_primitives=PILON_360M_PRIMITIVE_CONFIG["n_primitives"],
        rank=PILON_360M_PRIMITIVE_CONFIG["rank"],
        top_k=PILON_360M_PRIMITIVE_CONFIG["top_k"],
        share_fc1_fc2=PILON_360M_PRIMITIVE_CONFIG["share_fc1_fc2"],
        composition_type=PILON_360M_PRIMITIVE_CONFIG["composition_type"],
        temperature=PILON_360M_PRIMITIVE_CONFIG["temperature"],
        activation=PILON_360M_PRIMITIVE_CONFIG["activation"],
        moe_config=None,
        forward_fast_mode=PILON_360M_PRIMITIVE_CONFIG["forward_fast_mode"],
        forward_fast_min_topk=PILON_360M_PRIMITIVE_CONFIG["forward_fast_min_topk"],
        bands=_create_360m_bands(),
        n_hot=n_hot,
        swap_interval=swap_interval,
    )

    return ModelConfig(
        **SHARED_360M_CONFIG,
        ffn_type="compositional",
        primitive_config=primitive_config,
        enable_early_exit=True,
        exit_threshold=exit_threshold,
    )


# ============================================================================
# Training Configuration for 300B Token Run
# ============================================================================

def get_360m_training_config(
    total_tokens: int = 1_000_000_000,
    micro_batch_size: int = 4,
    gradient_accumulation: int = 16,
    max_seq_len: int = 2048,
) -> TrainingConfig:
    """
    Get training configuration for 360M model training.

    Args:
        total_tokens: Total tokens to train on (default: 1B)
        micro_batch_size: Micro batch size (default: 4, fits 12GB VRAM)
        gradient_accumulation: Gradient accumulation steps (default: 16)
        max_seq_len: Maximum sequence length (default: 2048)

    Returns:
        TrainingConfig for training
    """
    effective_batch_size = micro_batch_size * gradient_accumulation
    tokens_per_step = effective_batch_size * max_seq_len
    total_steps = max(1, total_tokens // tokens_per_step)

    # Scale warmup and checkpointing based on total steps
    warmup_steps = min(500, total_steps // 10)
    save_every = max(1000, total_steps // 5)  # ~5 checkpoints

    return TrainingConfig(
        dataset="HuggingFaceFW/fineweb-edu",
        tokens=total_tokens,

        # Optimizer
        optimizer="AdamW",
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,

        # Schedule
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lr_schedule="cosine",
        min_lr=3e-5,

        # Batching
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
        effective_batch_size=effective_batch_size,
        max_seq_len=max_seq_len,

        # Precision
        precision="bf16",
        grad_clip=1.0,

        # Logging and checkpointing
        log_every=50,
        eval_every=500,
        save_every=save_every,
    )


# ============================================================================
# Parameter Count Estimation
# ============================================================================

def estimate_360m_params() -> dict:
    """
    Estimate parameter counts for 360M models.

    Returns:
        Dictionary with parameter estimates for both model types
    """
    d_model = SHARED_360M_CONFIG["d_model"]
    n_layers = SHARED_360M_CONFIG["n_layers"]
    n_heads = SHARED_360M_CONFIG["n_heads"]
    d_ff = SHARED_360M_CONFIG["d_ff"]
    vocab_size = SHARED_360M_CONFIG["vocab_size"]
    max_seq_len = SHARED_360M_CONFIG["max_seq_len"]

    # Shared parameters (both models)
    token_embed = vocab_size * d_model
    pos_embed = max_seq_len * d_model

    # Attention per layer: Q, K, V projections + output projection
    qkv_proj = 3 * d_model * d_model
    out_proj = d_model * d_model
    attention_per_layer = qkv_proj + out_proj

    # Layer norms (2 per layer for pre-norm)
    norm_per_layer = 2 * d_model

    # Final layer norm
    final_norm = d_model

    # Shared components (no LM head if weight tied)
    shared = token_embed + pos_embed + final_norm
    attention_total = n_layers * (attention_per_layer + norm_per_layer)

    # Dense FFN: fc1 (d_model -> d_ff) + fc2 (d_ff -> d_model)
    dense_ffn_per_layer = d_model * d_ff * 2
    dense_total = shared + attention_total + n_layers * dense_ffn_per_layer

    # PILON FFN: primitives + composition weights
    n_primitives = PILON_360M_PRIMITIVE_CONFIG["n_primitives"]
    rank = PILON_360M_PRIMITIVE_CONFIG["rank"]
    n_bands = 3

    # Per band: fc1 bank + fc2 bank
    # fc1: n_primitives × (d_model × rank + rank × d_ff) + scale/bias
    # fc2: n_primitives × (d_ff × rank + rank × d_model) + scale/bias
    fc1_params_per_band = n_primitives * (d_model * rank + rank * d_ff) + 2 * rank
    fc2_params_per_band = n_primitives * (d_ff * rank + rank * d_model) + 2 * rank
    primitive_params = n_bands * (fc1_params_per_band + fc2_params_per_band)

    # Composition weights per layer (fc1 + fc2 logits)
    comp_per_layer = 2 * n_primitives
    comp_total = n_layers * comp_per_layer

    pilon_total = shared + attention_total + primitive_params + comp_total

    return {
        "shared_params": shared,
        "attention_params": attention_total,
        "dense_ffn_params": n_layers * dense_ffn_per_layer,
        "dense_total": dense_total,
        "pilon_primitive_params": primitive_params,
        "pilon_composition_params": comp_total,
        "pilon_total": pilon_total,
        "parameter_ratio": pilon_total / dense_total,
        "ffn_compression": (n_layers * dense_ffn_per_layer) / (primitive_params + comp_total),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PILON-R 360M Configuration Summary")
    print("=" * 60)

    print("\nShared Configuration:")
    for k, v in SHARED_360M_CONFIG.items():
        print(f"  {k}: {v}")

    print("\nPILON Primitive Configuration:")
    for k, v in PILON_360M_PRIMITIVE_CONFIG.items():
        print(f"  {k}: {v}")

    print("\nParameter Estimates:")
    estimates = estimate_360m_params()
    for k, v in estimates.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:,}")

    print("\nTraining Configuration (300B tokens):")
    train_config = get_360m_training_config()
    print(f"  total_steps: {train_config.total_steps:,}")
    print(f"  effective_batch_size: {train_config.effective_batch_size}")
    print(f"  tokens_per_step: {train_config.effective_batch_size * train_config.max_seq_len:,}")
