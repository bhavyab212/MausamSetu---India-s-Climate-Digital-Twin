"""
training.metrics — Phase 0 shim.

Every metric raises :class:`TrainingRebuildInProgress`. Phase 1 will replace
these with zone-aware metrics that carry bootstrap CIs and honour a per-zone
``min_cells`` gate.
"""
from __future__ import annotations

from . import TrainingRebuildInProgress


def masked_rmse(*args, **kwargs) -> float:
    raise TrainingRebuildInProgress("metrics.masked_rmse")


def mae(*args, **kwargs) -> float:
    raise TrainingRebuildInProgress("metrics.mae")


def bias(*args, **kwargs) -> float:
    raise TrainingRebuildInProgress("metrics.bias")


def pearson_r(*args, **kwargs) -> float:
    raise TrainingRebuildInProgress("metrics.pearson_r")


def csi(*args, **kwargs) -> float:
    raise TrainingRebuildInProgress("metrics.csi")


def ensemble_calibration(*args, **kwargs) -> float:
    raise TrainingRebuildInProgress("metrics.ensemble_calibration")


def compute_all_metrics(*args, **kwargs) -> dict:
    raise TrainingRebuildInProgress("metrics.compute_all_metrics")
