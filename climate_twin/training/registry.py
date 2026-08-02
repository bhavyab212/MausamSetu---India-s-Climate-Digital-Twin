"""
registry.py — named, resumable, validatable model registry.

A "model" here is a named training identity you can:
  - create fresh and train,
  - continue training later (resume from its saved weights),
  - select in Validation to evaluate.

Weights live at training/checkpoints/model_<region>_<name>.pt ; metadata in runs.db.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import torch

IST = timezone(timedelta(hours=5, minutes=30))
_DIR = Path(__file__).resolve().parent
CKPT_DIR = _DIR / "checkpoints"
DB_PATH = _DIR / "runs.db"


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).strip("_")[:48] or "model"


def _db() -> sqlite3.Connection:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_registry():
    conn = _db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS models (
            name TEXT, region TEXT, arch TEXT,
            created_at TEXT, updated_at TEXT,
            epochs_trained INTEGER, rounds_trained INTEGER,
            metrics TEXT, ckpt_path TEXT, notes TEXT,
            PRIMARY KEY (name, region))"""
    )
    conn.commit()
    conn.close()


def model_ckpt_path(name: str, region: str) -> Path:
    return CKPT_DIR / f"model_{region}_{_safe(name)}.pt"


def save_model(name, region, model, arch: dict, metrics: dict,
               epochs_add: int, rounds_add: int, notes: str = "") -> Path:
    """Persist weights + upsert registry metadata (accumulating epochs/rounds)."""
    path = model_ckpt_path(name, region)
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch": arch, "name": name, "region": region,
        "metrics": metrics, "saved_at": datetime.now(IST).isoformat(),
    }, path)

    conn = _db()
    now = datetime.now(IST).isoformat()
    ex = conn.execute(
        "SELECT epochs_trained, rounds_trained FROM models WHERE name=? AND region=?",
        (name, region),
    ).fetchone()
    if ex:
        conn.execute(
            "UPDATE models SET updated_at=?, epochs_trained=?, rounds_trained=?, "
            "metrics=?, arch=?, ckpt_path=? WHERE name=? AND region=?",
            (now, (ex["epochs_trained"] or 0) + epochs_add,
             (ex["rounds_trained"] or 0) + rounds_add,
             json.dumps(metrics), json.dumps(arch), str(path), name, region),
        )
    else:
        conn.execute(
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, region, json.dumps(arch), now, now,
             epochs_add, rounds_add, json.dumps(metrics), str(path), notes),
        )
    conn.commit()
    conn.close()
    return path


def _row_to_dict(r) -> dict:
    d = dict(r)
    for k in ("arch", "metrics"):
        try:
            d[k] = json.loads(d[k]) if d.get(k) else {}
        except Exception:
            d[k] = {}
    return d


def list_models(region: str | None = None) -> list[dict[str, Any]]:
    conn = _db()
    if region:
        rows = conn.execute("SELECT * FROM models WHERE region=? ORDER BY updated_at DESC", (region,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM models ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_model(name: str, region: str) -> dict | None:
    conn = _db()
    r = conn.execute("SELECT * FROM models WHERE name=? AND region=?", (name, region)).fetchone()
    conn.close()
    return _row_to_dict(r) if r else None


def load_into(name: str, region: str, model) -> tuple[bool, dict | None]:
    """Load saved weights into an existing model instance. Returns (ok, checkpoint)."""
    path = model_ckpt_path(name, region)
    if not path.exists():
        return False, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ck["model_state_dict"])
        return True, ck
    except Exception:
        return False, ck


def delete_model(name: str, region: str):
    conn = _db()
    conn.execute("DELETE FROM models WHERE name=? AND region=?", (name, region))
    conn.commit()
    conn.close()
    p = model_ckpt_path(name, region)
    if p.exists():
        p.unlink()
