"""training.state — Phase 0 shim."""
from __future__ import annotations

from . import TrainingRebuildInProgress


def __getattr__(name: str):
    def _raiser(*args, **kwargs):
        raise TrainingRebuildInProgress(f"state.{name}")
    _raiser.__name__ = name
    return _raiser
