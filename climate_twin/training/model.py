"""
training.model — Phase 0 shim.

``ClimateTwinModel`` raises on construction. Phase 2 will replace it with a
FiLM-conditioned zone-aware model in ``climate_twin/train/model/``.
"""
from __future__ import annotations

from . import TrainingRebuildInProgress


class ClimateTwinModel:
    """Archived model. Instantiating raises during the rebuild."""

    def __init__(self, *args, **kwargs):
        raise TrainingRebuildInProgress("model.ClimateTwinModel")
