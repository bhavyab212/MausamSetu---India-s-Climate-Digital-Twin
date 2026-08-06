"""training.baselines — Phase 0 shim."""
from __future__ import annotations

from . import TrainingRebuildInProgress


def compute_baseline_metrics(*args, **kwargs) -> dict:
    raise TrainingRebuildInProgress("baselines.compute_baseline_metrics")
