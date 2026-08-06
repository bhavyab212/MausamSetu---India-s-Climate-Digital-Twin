"""train.config — pydantic-validated experiment configs.

Public API:
    load_config(path) -> ExperimentConfig
    ExperimentConfig — the frozen, validated root object.
"""
from .schema import (
    ExperimentConfig,
    DataConfig,
    ZonesConfig,
    ModelConfig,
    LossConfig,
    OptimConfig,
    ValidationConfig,
    EarlyStoppingConfig,
    load_config,
)

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "ZonesConfig",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "ValidationConfig",
    "EarlyStoppingConfig",
    "load_config",
]
