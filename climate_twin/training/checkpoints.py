"""Checkpoint save/load with metadata.

Filenames include round number and validation period:
    round_012_1961-12.pt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import torch

IST = timezone(timedelta(hours=5, minutes=30))
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def checkpoint_path(round_num: int, val_period: str, region: str = "india") -> Path:
    """Build a deterministic, region-prefixed checkpoint filename.

    e.g. india_round_012_1961-12.pt / cauvery_round_003_2019.pt
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_period = val_period.replace("/", "-").replace(" ", "")
    return CHECKPOINT_DIR / f"{region}_round_{round_num:03d}_{safe_period}.pt"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    round_num: int,
    val_period: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    epoch: int,
    region: str = "india",
) -> Path:
    """Save a training checkpoint with full metadata (region-tagged)."""
    path = checkpoint_path(round_num, val_period, region)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "round_num": round_num,
        "val_period": val_period,
        "region": region,
        "config": config,
        "metrics": metrics,
        "epoch": epoch,
        "saved_at": datetime.now(IST).isoformat(),
    }, path)
    return path


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Load a checkpoint, restoring model (and optionally optimizer) state."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def list_checkpoints(region: str | None = None) -> list[dict[str, Any]]:
    """List saved checkpoints with metadata. If ``region`` is given, only that region's."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"{region}_round_*.pt" if region else "*round_*.pt"
    results = []
    for path in sorted(CHECKPOINT_DIR.glob(pattern)):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            results.append({
                "path": str(path),
                "filename": path.name,
                "region": ckpt.get("region", "india"),
                "round_num": ckpt.get("round_num"),
                "val_period": ckpt.get("val_period"),
                "metrics": ckpt.get("metrics", {}),
                "epoch": ckpt.get("epoch"),
                "saved_at": ckpt.get("saved_at"),
            })
        except Exception:
            continue
    return results
