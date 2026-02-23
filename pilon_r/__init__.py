"""
PILON-R Phase A: From-Scratch Compositional Language Model

This module implements the Phase A experiment:
- Train a small LM with compositional FFN from scratch
- Compare to dense baseline
- Answer: "Can this architecture learn language?"

Phase A.2 extends this with compression curriculum:
- Test multiple compression levels
- Find quality-compression frontier
- Run SFT on passing levels

Key components:
- config.py: All configuration dataclasses
- primitives.py: PrimitiveBank with low-rank primitives
- ffn.py: CompositionalFFN and StandardFFN
- model.py: Full PILONTransformer
- baseline.py: Dense baseline model and comparison tools
- data.py: Dataset loading (TinyStories)
- metrics.py: Training metrics and gate checking
- train.py: Training loop
- evaluate.py: Evaluation and generation
- sft.py: Supervised fine-tuning
- compression_curriculum.py: Phase A.2 orchestration

Usage:
    # Run PILON training
    python -m pilon_r.train --model-size 360m --ffn-type compositional

    # Run training with a preset compression level
    python -m pilon_r.train --model-size 360m --compression-level moderate

    # Run dense baseline training
    python -m pilon_r.train --model-size 360m --baseline

    # Run full Phase A.2 compression curriculum
    python -m pilon_r.compression_curriculum --run-all

    # Run SFT on a checkpoint
    python -m pilon_r.sft checkpoint.pt --epochs 2

    # Evaluate checkpoint
    python -m pilon_r.evaluate checkpoint.pt --baseline baseline.pt
"""

from .core.config import (
    ModelConfig,
    TrainingConfig,
    SmokeTestConfig,
    SFTConfig,
    GateConfig,
    PrimitiveConfig,
    BandConfig,
    get_default_configs,
    COMPRESSION_LEVELS,
    get_compression_config,
    get_model_config_for_compression,
    get_all_compression_levels,
)

from .configs.model_360m import (
    get_360m_config,
    get_360m_pilon_config,
    get_360m_dense_config,
    get_360m_training_config,
    MODEL_360M_PILON,
    MODEL_360M_DENSE,
)
from .configs.model_500m import (
    get_500m_config,
    get_500m_pilon_config,
    get_500m_dense_config,
    get_500m_training_config,
    MODEL_500M_PILON,
    MODEL_500M_DENSE,
)

from .tokenizer import (
    train_tokenizer,
    load_custom_tokenizer,
    TokenizerWrapper,
)

from .core.primitives import (
    LowRankPrimitive,
    PrimitiveBank,
    BandPrimitiveBanks,
    LayerCompositionWeights,
)

from .core.ffn import (
    StandardFFN,
    CompositionalFFN,
    create_ffn,
)

from .core.model import (
    PILONTransformer,
    RMSNorm,
    MultiHeadAttention,
    TransformerBlock,
    create_model,
    create_baseline_model,
)

from .core.baseline import (
    create_baseline_config,
    compare_parameter_counts,
    BaselineTracker,
)

from .core.eval_cache import (
    EvalCache,
    evaluate_with_cache,
)

from .core.data import (
    load_tinystories,
    load_openwebtext,
    load_text_dataset,
    load_alpaca,
    get_tokenizer,
    create_dataloader,
)

from .core.metrics import (
    TrainingMetrics,
    GateChecker,
    Logger,
    compute_entropy,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "ModelConfig",
    "TrainingConfig",
    "SmokeTestConfig",
    "SFTConfig",
    "GateConfig",
    "PrimitiveConfig",
    "BandConfig",
    "get_default_configs",
    "COMPRESSION_LEVELS",
    "get_compression_config",
    "get_model_config_for_compression",
    "get_all_compression_levels",
    # 360M configs
    "get_360m_config",
    "get_360m_pilon_config",
    "get_360m_dense_config",
    "get_360m_training_config",
    "MODEL_360M_PILON",
    "MODEL_360M_DENSE",
    # 500M configs
    "get_500m_config",
    "get_500m_pilon_config",
    "get_500m_dense_config",
    "get_500m_training_config",
    "MODEL_500M_PILON",
    "MODEL_500M_DENSE",
    # Tokenizer
    "train_tokenizer",
    "load_custom_tokenizer",
    "TokenizerWrapper",
    # Primitives
    "LowRankPrimitive",
    "PrimitiveBank",
    "BandPrimitiveBanks",
    "LayerCompositionWeights",
    # FFN
    "StandardFFN",
    "CompositionalFFN",
    "create_ffn",
    # Model
    "PILONTransformer",
    "RMSNorm",
    "MultiHeadAttention",
    "TransformerBlock",
    "create_model",
    "create_baseline_model",
    # Baseline
    "create_baseline_config",
    "compare_parameter_counts",
    "BaselineTracker",
    # Eval Cache
    "EvalCache",
    "evaluate_with_cache",
    # Data
    "load_tinystories",
    "load_openwebtext",
    "load_text_dataset",
    "load_alpaca",
    "get_tokenizer",
    "create_dataloader",
    # Metrics
    "TrainingMetrics",
    "GateChecker",
    "Logger",
    "compute_entropy",
]
