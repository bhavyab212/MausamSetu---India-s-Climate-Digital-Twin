"""
training.registry — Phase 0 shim.

Read-only lookups (``list_models``, ``get_model``, ``get_lineage_tree``) return
empty results so tabs that just enumerate saved models don't crash. Every
mutating call (``save_model``, ``load_into``, ``delete_model``) raises
:class:`TrainingRebuildInProgress`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import TrainingRebuildInProgress

# Re-exposed so callers that check ``except REG.ModelVariableMismatch`` still work.
ModelVariableMismatch = TrainingRebuildInProgress

# Paths kept for callers that might display them
_DIR = Path(__file__).resolve().parent
MODELS_DIR = _DIR.parent / "models"


def init_registry() -> None:
    """No-op during rebuild."""
    return None


def list_models(region: str | None = None) -> list[dict[str, Any]]:
    """Return no models — the archived registry is unreachable during rebuild."""
    return []


def get_model(name: str, region: str) -> dict | None:
    return None


def get_lineage_tree(region: str) -> list[dict[str, Any]]:
    return []


def model_dir(name: str, region: str) -> Path:
    return MODELS_DIR / region / name


def save_model(*args, **kwargs) -> Path:
    raise TrainingRebuildInProgress("registry.save_model")


def load_into(*args, **kwargs) -> tuple[bool, dict | None]:
    raise TrainingRebuildInProgress("registry.load_into")


def delete_model(*args, **kwargs) -> None:
    raise TrainingRebuildInProgress("registry.delete_model")
