"""
PILON-R Model Configurations

Pre-configured model sizes for experiments.
"""

from .model_360m import (
    get_360m_config,
    get_360m_pilon_config,
    get_360m_dense_config,
    MODEL_360M_PILON,
    MODEL_360M_DENSE,
)

__all__ = [
    "get_360m_config",
    "get_360m_pilon_config",
    "get_360m_dense_config",
    "MODEL_360M_PILON",
    "MODEL_360M_DENSE",
]
