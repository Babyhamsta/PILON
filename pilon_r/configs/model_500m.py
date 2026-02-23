"""
PILON-R 500M Model Configuration

Configurations for a ~500M parameter comparison experiment:
- PILON-500M: Compositional FFN with primitives
- Dense-500M: Standard dense FFN baseline

Both configurations share identical embedding/attention choices.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.config import ModelConfig, PrimitiveConfig, BandConfig, TrainingConfig


# ============================================================================
# Shared Base Configuration (identical for both models)
# ============================================================================

SHARED_500M_CONFIG = {
    "d_model": 1024,
    "n_layers": 36,
    "n_heads": 16,
    "d_head": 64,
    "d_ff": 4096,  # 4x d_model
    "vocab_size": 50257,  # GPT-2 tokenizer
    "max_seq_len": 1024,
    "dropout": 0.1,
    "norm_type": "rmsnorm",
    "checkpoint_ffn": True,  # Gradient checkpointing for VRAM savings
}


# ============================================================================
# PILON-500M: Compositional FFN Configuration
# ============================================================================

def _create_500m_bands() -> List[BandConfig]:
    """
    Create 3-band configuration for 36 layers.

    - Early band (layers 0-11)
    - Middle band (layers 12-23)
    - Late band (layers 24-35)
    """
    return [
        BandConfig(name="early", layers=list(range(0, 12))),
        BandConfig(name="middle", layers=list(range(12, 24))),
        BandConfig(name="late", layers=list(range(24, 36))),
    ]


# Target ~500M params:
# Dense ~505M params with d_model=1024, n_layers=36, d_ff=4096
# PILON ~487M params with n_primitives=96, rank=96 (within ~4%)
PILON_500M_PRIMITIVE_CONFIG = {
    "n_primitives": 96,
    "rank": 96,
    "top_k": 8,
    "share_fc1_fc2": False,
    "composition_type": "static_per_layer",
    "temperature": 0.5,
    "activation": "gelu",
    "moe_config": None,
    # Keep VRAM stable by default (forward_fast only when dense)
    "forward_fast_mode": "auto",
    "forward_fast_min_topk": None,
}


# Pre-built configurations
MODEL_500M_PILON = ModelConfig(
    **SHARED_500M_CONFIG,
    ffn_type="compositional",
    primitive_config=PrimitiveConfig(
        **PILON_500M_PRIMITIVE_CONFIG,
        bands=_create_500m_bands()
    )
)

MODEL_500M_DENSE = ModelConfig(
    **SHARED_500M_CONFIG,
    ffn_type="standard",
)


# ============================================================================
# Factory Functions
# ============================================================================

def get_500m_config(ffn_type: str = "compositional") -> ModelConfig:
    """
    Get 500M model configuration.

    Args:
        ffn_type: "compositional" for PILON, "standard" for dense baseline

    Returns:
        ModelConfig for 500M model
    """
    if ffn_type == "compositional":
        return get_500m_pilon_config()
    if ffn_type == "standard":
        return get_500m_dense_config()
    raise ValueError(f"Unknown ffn_type: {ffn_type}. Use 'compositional' or 'standard'")


def get_500m_pilon_config(
    n_primitives: Optional[int] = None,
    rank: Optional[int] = None,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    forward_fast_mode: Optional[str] = None,
    forward_fast_min_topk: Optional[int] = None,
) -> ModelConfig:
    """
    Get PILON-500M configuration with optional overrides.

    Args:
        n_primitives: Number of primitives (default: 96)
        rank: Rank of each primitive (default: 96)
        top_k: Sparsity - top-k primitives used (default: 8)
        temperature: Softmax temperature (default: 0.5)
        forward_fast_mode: "auto" | "on" | "off"
        forward_fast_min_topk: Top-k threshold for auto mode

    Returns:
        ModelConfig for PILON-500M
    """
    primitive_config = PrimitiveConfig(
        n_primitives=n_primitives or PILON_500M_PRIMITIVE_CONFIG["n_primitives"],
        rank=rank or PILON_500M_PRIMITIVE_CONFIG["rank"],
        top_k=top_k or PILON_500M_PRIMITIVE_CONFIG["top_k"],
        share_fc1_fc2=PILON_500M_PRIMITIVE_CONFIG["share_fc1_fc2"],
        composition_type=PILON_500M_PRIMITIVE_CONFIG["composition_type"],
        temperature=temperature or PILON_500M_PRIMITIVE_CONFIG["temperature"],
        activation=PILON_500M_PRIMITIVE_CONFIG["activation"],
        moe_config=None,
        forward_fast_mode=forward_fast_mode or PILON_500M_PRIMITIVE_CONFIG["forward_fast_mode"],
        forward_fast_min_topk=(
            PILON_500M_PRIMITIVE_CONFIG["forward_fast_min_topk"]
            if forward_fast_min_topk is None
            else forward_fast_min_topk
        ),
        bands=_create_500m_bands()
    )

    return ModelConfig(
        **SHARED_500M_CONFIG,
        ffn_type="compositional",
        primitive_config=primitive_config
    )


def get_500m_dense_config() -> ModelConfig:
    """
    Get Dense-500M baseline configuration.

    Returns:
        ModelConfig for Dense-500M
    """
    return ModelConfig(
        **SHARED_500M_CONFIG,
        ffn_type="standard",
    )


# ============================================================================
# Training Configuration for 1B Token Run
# ============================================================================

def get_500m_training_config(
    total_tokens: int = 1_000_000_000,
    micro_batch_size: int = 2,
    gradient_accumulation: int = 32,
    max_seq_len: int = 1024,
) -> TrainingConfig:
    """
    Get training configuration for 500M model training.

    Args:
        total_tokens: Total tokens to train on (default: 1B)
        micro_batch_size: Micro batch size (default: 2, fits 12GB VRAM)
        gradient_accumulation: Gradient accumulation steps (default: 32)
        max_seq_len: Maximum sequence length (default: 1024)

    Returns:
        TrainingConfig for training
    """
    effective_batch_size = micro_batch_size * gradient_accumulation
    tokens_per_step = effective_batch_size * max_seq_len
    total_steps = max(1, total_tokens // tokens_per_step)

    warmup_steps = min(1000, max(100, total_steps // 10))
    save_every = max(1000, total_steps // 5)

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

def estimate_500m_params() -> dict:
    """
    Estimate parameter counts for 500M models.

    Returns:
        Dictionary with parameter estimates for both model types
    """
    d_model = SHARED_500M_CONFIG["d_model"]
    n_layers = SHARED_500M_CONFIG["n_layers"]
    d_ff = SHARED_500M_CONFIG["d_ff"]
    vocab_size = SHARED_500M_CONFIG["vocab_size"]
    max_seq_len = SHARED_500M_CONFIG["max_seq_len"]

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
    n_primitives = PILON_500M_PRIMITIVE_CONFIG["n_primitives"]
    rank = PILON_500M_PRIMITIVE_CONFIG["rank"]
    n_bands = 3

    fc1_params_per_band = n_primitives * (d_model * rank + rank * d_ff) + 2 * rank
    fc2_params_per_band = n_primitives * (d_ff * rank + rank * d_model) + 2 * rank
    primitive_params = n_bands * (fc1_params_per_band + fc2_params_per_band)

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
    print("PILON-R 500M Configuration Summary")
    print("=" * 60)

    print("\nShared Configuration:")
    for k, v in SHARED_500M_CONFIG.items():
        print(f"  {k}: {v}")

    print("\nPILON Primitive Configuration:")
    for k, v in PILON_500M_PRIMITIVE_CONFIG.items():
        print(f"  {k}: {v}")

    print("\nParameter Estimates:")
    estimates = estimate_500m_params()
    for k, v in estimates.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:,}")

    print("\nTraining Configuration (1B tokens):")
    train_config = get_500m_training_config()
    print(f"  total_steps: {train_config.total_steps:,}")
    print(f"  effective_batch_size: {train_config.effective_batch_size}")
    print(f"  tokens_per_step: {train_config.effective_batch_size * train_config.max_seq_len:,}")
