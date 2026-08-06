"""
runtime.run_state — pickle-serializable snapshot of the current run.

Written by the executor. Read by the dashboard and by ``ls`` on disk when
recovering from a crash. Contains the minimum surface a dashboard needs
to render a header:

    status, epoch, batch, total batches, best zone-weighted RMSE,
    device, gpu-util snapshot, start_time, elapsed seconds, config hash,
    zone signature, manifest signature.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FINISHED = "finished"
    ERRORED = "errored"


@dataclass
class RunState:
    """Minimal, pickle-safe snapshot readable by any thread.

    Consumers should treat this as read-only; the executor is the sole
    writer. Uses primitive types only.
    """
    status: str = RunStatus.IDLE.value
    model_name: str = ""
    region: str = ""
    device: str = ""
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    zone_weighted_rmse: float = float("nan")
    best_zone_weighted_rmse: float = float("inf")
    best_epoch: int = 0
    lr: float = 0.0
    grad_norm: float = 0.0
    start_time_epoch: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    config_hash: str = ""
    zone_mask_sig: str = ""
    manifest_sig: str = ""
    error_type: str = ""
    error_message: str = ""
    message: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def dump(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def hash_config(cfg_dict: dict[str, Any]) -> str:
    """12-hex hash of a serialised config dict — stable across identical inputs."""
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
