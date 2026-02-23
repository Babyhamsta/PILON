"""
PILON-R Phase A Configuration
All configurations for the compositional FFN experiment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# === MoE ===
# Opt-in only. When False, MoE is fully bypassed.
USE_MOE = False
MOE_NUM_EXPERTS = 16
MOE_TOP_K = None  # None = soft routing
MOE_AUX_LOSS_WEIGHT = 0.0


@dataclass
class BandConfig:
    """Configuration for a primitive band (group of layers sharing primitives)."""

    name: str
    layers: List[int]


@dataclass
class MoEConfig:
    """Configuration for Mixture-of-Experts routing (Phase B)."""

    # Expert configuration
    n_experts: int = 8  # Number of composition recipes per layer
    top_k: int = 2  # Experts selected per token

    # Router configuration
    router_type: str = "linear"  # "linear" or "mlp"
    router_hidden_dim: Optional[int] = None  # For MLP router (default: d_model // 4)

    # Load balancing
    aux_loss_weight: float = 0.01  # Weight for auxiliary loss
    load_balancing: bool = True  # Enable load balancing loss
    router_z_loss_coef: float = 0.0  # Z-loss coefficient (0 = disabled)


@dataclass
class PrimitiveConfig:
    """Configuration for compositional primitives."""

    # Generous initial config (tighten in Phase A.2, not now)
    n_primitives: int = 48
    rank: int = 48
    top_k: int = 8
    # Optional per-projection top_k overrides (None = use top_k)
    top_k_fc1: Optional[int] = None
    top_k_fc2: Optional[int] = None

    # SEPARATE banks for fc1 and fc2 (no sharing in Phase A)
    share_fc1_fc2: bool = False

    # Band strategy - layers grouped by stage
    bands: List[BandConfig] = field(
        default_factory=lambda: [
            BandConfig(name="early", layers=[0, 1, 2]),
            BandConfig(name="middle", layers=[3, 4, 5]),
            BandConfig(name="late", layers=[6, 7]),
        ]
    )

    # Static composition (no input-dependent routing)
    composition_type: str = "static_per_layer"

    # Softmax temperature for composition weights
    temperature: float = 0.5

    # Activation function
    activation: str = "gelu"

    # Forward path selection for compositional FFN
    # "auto": use forward_fast only when mixing is effectively dense
    # "on": always use forward_fast (highest throughput, more VRAM)
    # "off": never use forward_fast (lowest VRAM)
    forward_fast_mode: str = "auto"
    # In "auto" mode, require top_k >= this threshold to use forward_fast.
    # None defaults to n_primitives (dense only).
    forward_fast_min_topk: Optional[int] = None

    # MoE configuration (Phase B) - None = Phase A static composition
    moe_config: Optional[MoEConfig] = None


@dataclass
class ModelConfig:
    """Full model configuration."""

    # Core architecture
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_head: int = 64
    d_ff: int = 2048  # 4x d_model
    vocab_size: int = 50257  # GPT-2 tokenizer
    max_seq_len: int = 512

    # Attention (standard)
    attention_type: str = "standard_mha"
    pos_encoding: str = "learned"
    dropout: float = 0.1

    # Normalization
    norm_type: str = "rmsnorm"
    checkpoint_ffn: bool = True

    # FFN type: "compositional" or "standard"
    ffn_type: str = "compositional"

    # Compositional FFN config (only used if ffn_type == "compositional")
    primitive_config: PrimitiveConfig = field(default_factory=PrimitiveConfig)

    def get_baseline_config(self) -> "ModelConfig":
        """Return a copy with standard FFN for baseline comparison."""
        import copy

        baseline = copy.deepcopy(self)
        baseline.ffn_type = "standard"
        return baseline


@dataclass
class SmokeTestConfig:
    """Configuration for smoke test (fail fast)."""

    dataset: str = "Elriggs/openwebtext-100k"
    tokens: int = 2_000_000  # 2M tokens only
    steps: int = 500

    # Quick checks
    checks: List[str] = field(
        default_factory=lambda: [
            "loss_decreases",
            "no_nan",
            "gradients_flow",
            "no_dead_primitives",
            "entropy_not_collapsed",
        ]
    )


@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Dataset
    dataset: str = "Elriggs/openwebtext-100k"
    tokens: int = 50_000_000  # 50M tokens

    # Optimizer
    optimizer: str = "AdamW"
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1

    # Schedule
    warmup_steps: int = 500
    total_steps: int = 10000  # Reduced from 25k for Phase A.2
    lr_schedule: str = "cosine"
    min_lr: float = 3e-5

    # Batching
    micro_batch_size: int = 8
    gradient_accumulation: int = 8
    effective_batch_size: int = 64  # micro_batch_size * gradient_accumulation
    max_seq_len: int = 512
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False

    # Precision & stability
    precision: str = "bf16"
    grad_clip: float = 1.0

    # Logging
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 2000  # Checkpoint at 2k, 4k, 6k, 8k, 10k

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    # Random seed for reproducibility
    seed: int = 42

    # === Phase Training (PILON-specific) ===
    phase1_ratio: float = 0.05  # Phase 1 as fraction of total_steps (5%)
    disable_topk_in_phase1: bool = True  # Use all primitives in Phase 1
    freeze_primitives_in_phase2: bool = True  # Freeze primitive banks in Phase 2
    phase2_soft_steps: int = 200  # Soft-routing warm-up steps at start of Phase 2
    phase2_imbalance_weight: float = 0.01  # Encourage uneven primitive usage in Phase 2
    phase2_dominance_weight: float = 0.01  # Soft penalty for top-1 dominance in Phase 2
    prim_entropy_weight: float = (
        0.0  # Encourage higher entropy in Phase 2 (default off)
    )

    # === Learning Rate Multipliers ===
    primitives_lr_multiplier: float = 2.0  # Primitives learn faster
    composition_lr_multiplier: float = 0.5  # Composition learns slower

    # === Rank Scheduling ===
    rank_start: Optional[int] = None  # None = disabled, else starting rank
    rank_warmup_steps: int = 1000  # Steps to reach full rank
    min_rank: int = 8  # Minimum rank during Phase 2 decay


@dataclass
class SFTConfig:
    """Supervised fine-tuning configuration (after Phase A passes)."""

    dataset: str = "teknium/OpenHermes-2.5"
    epochs: int = 2
    lr: float = 1e-5
    warmup_steps: int = 100
    save_every: int = 1000  # Save checkpoint every N steps (0 = disable)

    # Chat template
    template: str = """### Instruction:
{instruction}

### Response:
"""


@dataclass
class GateConfig:
    """Success gate thresholds."""

    # Gate A0: Smoke Test (Steps 0-500)
    a0_min_entropy: float = 1.0
    a0_max_grad_norm: float = 100.0

    # Gate A1: Training Stability (Steps 500-2000)
    a1_min_entropy: float = 1.0
    a1_max_loss_spike_ratio: float = 3.0

    # Gate A2: Learning Validation (Steps 2000-5000)
    a2_max_val_ppl: float = 200.0
    a2_max_baseline_ratio: float = 1.5

    # Gate A3: Functional LM (Steps 5000-10000)
    a3_max_val_ppl: float = 50.0
    a3_max_baseline_ratio: float = 1.1  # Within 10% of baseline
    a3_min_entropy: float = 1.0


# Compression levels for Phase A.2
COMPRESSION_LEVELS = [
    {"n_primitives": 64, "rank": 64, "top_k": 12, "label": "minimal"},
    {"n_primitives": 48, "rank": 48, "top_k": 8, "label": "baseline"},
    {"n_primitives": 36, "rank": 36, "top_k": 6, "label": "mild"},
    {"n_primitives": 24, "rank": 24, "top_k": 4, "label": "moderate"},
    {"n_primitives": 16, "rank": 16, "top_k": 4, "label": "aggressive"},
    {"n_primitives": 8, "rank": 8, "top_k": 2, "label": "extreme"},
]


def get_default_configs():
    """Return all default configurations."""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "smoke_test": SmokeTestConfig(),
        "sft": SFTConfig(),
        "gates": GateConfig(),
    }


def get_baseline_model_config() -> ModelConfig:
    """Return model config for dense baseline."""
    return ModelConfig(ffn_type="standard")


def get_compression_config(level: str) -> Dict[str, int]:
    """
    Get compression parameters for a given level.

    Args:
        level: One of 'baseline', 'mild', 'moderate', 'aggressive', 'extreme'

    Returns:
        Dictionary with n_primitives, rank, top_k
    """
    for config in COMPRESSION_LEVELS:
        if config["label"] == level:
            return {
                "n_primitives": config["n_primitives"],
                "rank": config["rank"],
                "top_k": config["top_k"],
            }
    valid_levels = [c["label"] for c in COMPRESSION_LEVELS]
    raise ValueError(
        f"Unknown compression level: {level}. Valid levels: {valid_levels}"
    )


def get_model_config_for_compression(level: str) -> ModelConfig:
    """
    Create a ModelConfig with the specified compression level.

    Args:
        level: Compression level name

    Returns:
        ModelConfig configured for that compression level
    """
    compression = get_compression_config(level)

    return ModelConfig(
        primitive_config=PrimitiveConfig(
            n_primitives=compression["n_primitives"],
            rank=compression["rank"],
            top_k=compression["top_k"],
        )
    )


def get_all_compression_levels() -> List[str]:
    """Return list of all compression level names."""
    return [c["label"] for c in COMPRESSION_LEVELS]


if __name__ == "__main__":
    # Print configurations for verification
    configs = get_default_configs()
    for name, config in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        print(config)
