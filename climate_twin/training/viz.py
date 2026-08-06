"""
training.viz — Phase 0 shim. Every viz helper raises during the rebuild.
"""
from __future__ import annotations

from . import TrainingRebuildInProgress


def __getattr__(name: str):
    def _raiser(*args, **kwargs):
        raise TrainingRebuildInProgress(f"viz.{name}")
    return _raiser
