"""Checkpoint save/load with metadata.

Filenames include round number and validation period:
    india_round_012_2024.pt / cauvery_round_003_2019.pt

Post-Phase-5 contract
---------------------
Every checkpoint stores its **variable list** (the ordered channel names the
model was trained on) and its **manifest signature** (the 12-hex fingerprint
of the processed cube that produced the training tensors). ``load_checkpoint``
refuses to restore weights into a model whose expected variable list disagrees
— this is what prevents an old rain-only checkpoint from silently loading into
a new rain+tmax+tmin+insat model.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Sequence

import torch

IST = timezone(timedelta(hours=5, minutes=30))
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class CheckpointVariableMismatch(RuntimeError):
    """Raised when a checkpoint's saved variable list disagrees with the load target."""


def checkpoint_path(round_num: int, val_period: str, region: str = "india") -> Path:
    """Build a deterministic, region-prefixed checkpoint filename.

    e.g. india_round_012_2024.pt / cauvery_round_003_2019.pt
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_period = val_period.replace("/", "-").replace(" ", "")
    return CHECKPOINT_DIR / f"{region}_round_{round_num:03d}_{safe_period}.pt"


def _norm_vars(variables: Sequence[str] | None) -> list[str]:
    """Canonical form: list[str], stripped, non-empty."""
    if variables is None:
        return []
    return [str(v).strip() for v in variables if str(v).strip()]


def check_variables(expected: Sequence[str] | None,
                    saved: Sequence[str] | None,
                    *, strict: bool = True) -> tuple[bool, str]:
    """Compare a saved checkpoint's variables against what a load target expects.

    Returns ``(ok, message)``. Order matters (channel index must line up), so
    the check is a positional equality, not a set-comparison.

    A saved-empty list is treated as "unknown" — allowed with a warning message
    only when ``strict=False`` (legacy checkpoints that pre-date this contract).
    """
    exp = _norm_vars(expected)
    got = _norm_vars(saved)
    if not exp:
        return True, "expected variables not supplied (skipping check)"
    if not got:
        if strict:
            return False, "checkpoint has no saved variables (legacy pre-Phase-5)"
        return True, "checkpoint has no saved variables — accepting under strict=False"
    if list(exp) == list(got):
        return True, f"variables match ({got})"
    return False, f"variable-list mismatch: expected {exp}, checkpoint saved {got}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    round_num: int,
    val_period: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    epoch: int,
    region: str = "india",
    variables: Sequence[str] | None = None,
    manifest_sig: str | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> Path:
    """Save a training checkpoint with full metadata (region + variables + sig)."""
    path = checkpoint_path(round_num, val_period, region)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "round_num": round_num,
        "val_period": val_period,
        "region": region,
        "variables": _norm_vars(variables),
        "manifest_sig": manifest_sig or "",
        "grid_shape": list(grid_shape) if grid_shape else [],
        "config": config,
        "metrics": metrics,
        "epoch": epoch,
        "saved_at": datetime.now(IST).isoformat(),
        "checkpoint_format": 2,  # bump when payload layout changes
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    expected_variables: Sequence[str] | None = None,
    strict: bool = True,
) -> dict:
    """Load a checkpoint, restoring model (and optionally optimizer) state.

    Refuses to load into ``model`` when ``expected_variables`` disagrees with
    the checkpoint's stored list. Set ``strict=False`` to allow loading legacy
    checkpoints that have no saved variables (they will be flagged in the
    return value under ``variable_check``).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt.get("variables", []) if isinstance(ckpt, dict) else []
    ok, msg = check_variables(expected_variables, saved, strict=strict)
    if not ok:
        raise CheckpointVariableMismatch(f"{Path(path).name}: {msg}")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ckpt["variable_check"] = msg
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
                "variables": ckpt.get("variables", []),
                "manifest_sig": ckpt.get("manifest_sig", ""),
                "grid_shape": ckpt.get("grid_shape", []),
                "checkpoint_format": ckpt.get("checkpoint_format", 1),
                "metrics": ckpt.get("metrics", {}),
                "epoch": ckpt.get("epoch"),
                "saved_at": ckpt.get("saved_at"),
            })
        except Exception:
            continue
    return results
