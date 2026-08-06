"""
training.loops — Phase 0 shim.

Preserves the ``RoundConfig`` and ``TrainingProgress`` names as inert
dataclasses so any lingering import site (e.g. ``_demo_partb.py``) does not
crash at import time. ``train_one_round`` raises when actually invoked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import TrainingRebuildInProgress


@dataclass
class RoundConfig:
    """Archived dataclass — kept only so imports don't fail. Do not populate."""
    lr: float = 0.0
    batch_size: int = 0
    epochs: int = 0
    seed: int = 0
    seq_length: int = 0
    hidden: int = 0
    dropout: float = 0.0
    residual_scale: float = 0.0
    optimizer_name: str = ""
    scheduler_name: str = ""
    mixed_precision: str = ""
    early_stopping: bool = False
    patience: int = 0
    min_delta: float = 0.0
    monitor: str = ""
    warmup_epochs: int = 0
    min_lr: float = 0.0
    grad_clip: float = 0.0
    weight_decay: float = 0.0
    gradient_accum: int = 0
    freeze_recurrent: bool = False
    physics_water_balance: float = 0.0
    physics_spatial_smooth: float = 0.0
    physics_temporal_smooth: float = 0.0
    physics_tmax_tmin: float = 0.0
    recency_weighting: bool = False
    recency_half_life_years: int = 0
    label_smoothing: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    deterministic: bool = False
    mc_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingProgress:
    """Archived dataclass — kept only so imports don't fail."""
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_rmses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    elapsed: float = 0.0
    eta: float = 0.0
    running: bool = False
    finished: bool = False
    error: str | None = None
    best_val: float = float("inf")
    patience_counter: int = 0


def train_one_round(*args, **kwargs):
    raise TrainingRebuildInProgress("loops.train_one_round")
