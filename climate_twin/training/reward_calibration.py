"""
training.reward_calibration — Phase 0 shim.

CRPS-guided calibration finetune is archived. Every entry point raises.
"""
from __future__ import annotations

from . import TrainingRebuildInProgress


def calibration_finetune(*args, **kwargs):
    raise TrainingRebuildInProgress("reward_calibration.calibration_finetune")


def evaluate_calibration(*args, **kwargs):
    raise TrainingRebuildInProgress("reward_calibration.evaluate_calibration")
