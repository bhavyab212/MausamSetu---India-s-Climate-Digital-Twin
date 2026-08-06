"""
training.checkpoints — Phase 0 shim.

Passive helpers (``check_variables``, ``list_checkpoints``, ``CHECKPOINT_DIR``,
:class:`CheckpointVariableMismatch`) preserved for callers that only inspect
checkpoint metadata. Mutating operations raise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import TrainingRebuildInProgress

CheckpointVariableMismatch = TrainingRebuildInProgress

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def check_variables(expected: Sequence[str] | None,
                    saved: Sequence[str] | None,
                    *, strict: bool = True) -> tuple[bool, str]:
    """Reference-only helper kept so audit code can compare variable lists."""
    exp = [str(v).strip() for v in (expected or []) if str(v).strip()]
    got = [str(v).strip() for v in (saved or []) if str(v).strip()]
    if not exp:
        return True, "expected variables not supplied (skipping check)"
    if not got:
        return (not strict), "checkpoint has no saved variables"
    if list(exp) == list(got):
        return True, f"variables match ({got})"
    return False, f"variable-list mismatch: expected {exp}, checkpoint saved {got}"


def checkpoint_path(*args, **kwargs) -> Path:
    raise TrainingRebuildInProgress("checkpoints.checkpoint_path")


def save_checkpoint(*args, **kwargs) -> Path:
    raise TrainingRebuildInProgress("checkpoints.save_checkpoint")


def load_checkpoint(*args, **kwargs) -> dict:
    raise TrainingRebuildInProgress("checkpoints.load_checkpoint")


def list_checkpoints(region: str | None = None) -> list[dict[str, Any]]:
    """Return no checkpoints — flat-checkpoint area is empty during rebuild."""
    return []
