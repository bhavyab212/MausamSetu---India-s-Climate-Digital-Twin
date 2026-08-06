"""Session state management + SQLite run history.

Persists walk-forward run history to climate_twin/training/runs.db
so browser refreshes don't lose progress.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

IST = timezone(timedelta(hours=5, minutes=30))
DB_PATH = Path(__file__).resolve().parent / "runs.db"


def _get_db() -> sqlite3.Connection:
    """Get or create the SQLite connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_num INTEGER NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            val_start TEXT NOT NULL,
            val_end TEXT NOT NULL,
            config_json TEXT,
            metrics_json TEXT,
            checkpoint_path TEXT,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'running',
            parent_id INTEGER DEFAULT NULL
        )
    """)
    try:
        conn.execute("ALTER TABLE rounds ADD COLUMN parent_id INTEGER DEFAULT NULL")
    except Exception:
        pass
    conn.commit()
    return conn


def insert_round(
    round_num: int,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    status: str = "running",
    parent_id: int | None = None,
) -> int:
    """Insert a new round record. Returns the row id."""
    conn = _get_db()
    cursor = conn.execute(
        """INSERT INTO rounds
           (round_num, train_start, train_end, val_start, val_end,
            config_json, metrics_json, checkpoint_path, created_at, status, parent_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            round_num, train_start, train_end, val_start, val_end,
            json.dumps(config),
            json.dumps(metrics) if metrics else None,
            checkpoint_path,
            datetime.now(IST).isoformat(),
            status,
            parent_id,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_round(row_id: int, metrics: dict[str, Any], checkpoint_path: str, status: str = "completed"):
    """Update a round with final metrics and checkpoint."""
    conn = _get_db()
    conn.execute(
        """UPDATE rounds SET metrics_json=?, checkpoint_path=?, status=? WHERE id=?""",
        (json.dumps(metrics), checkpoint_path, status, row_id),
    )
    conn.commit()


def get_all_rounds() -> list[dict[str, Any]]:
    """Return all rounds, newest first."""
    conn = _get_db()
    cursor = conn.execute(
        "SELECT id, round_num, train_start, train_end, val_start, val_end, "
        "config_json, metrics_json, checkpoint_path, created_at, status, parent_id "
        "FROM rounds ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    results = []
    for row in rows:
        entry = {
            "id": row[0],
            "round_num": row[1],
            "train_start": row[2],
            "train_end": row[3],
            "val_start": row[4],
            "val_end": row[5],
            "config": json.loads(row[6]) if row[6] else {},
            "metrics": json.loads(row[7]) if row[7] else {},
            "checkpoint_path": row[8],
            "created_at": row[9],
            "status": row[10],
            "parent_id": row[11] if len(row) > 11 else None,
        }
        results.append(entry)
    return results


def delete_round(row_id: int):
    """Delete a round from history."""
    conn = _get_db()
    conn.execute("DELETE FROM rounds WHERE id=?", (row_id,))
    conn.commit()


def clear_all_rounds():
    """Clear all run history."""
    conn = _get_db()
    conn.execute("DELETE FROM rounds")
    conn.commit()


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_training_state():
    """Initialize all training-related session state keys."""
    defaults = {
        "wf_schedule": None,
        "wf_run_mode": "auto_cascade",
        "wf_current_round": 0,
        "wf_total_rounds": 0,
        "wf_running": False,
        "wf_paused": False,
        "wf_stop_requested": False,
        "wf_stop_all_requested": False,
        "wf_degradation_threshold": 5.0,
        "wf_training_mode": "warm_start",
        "wf_reset_every_n": 10,
        "wf_last_metrics": None,
        "wf_history": [],
        "wf_live_loss": [],
        "wf_live_val_loss": [],
        "wf_live_grad_norms": [],
        "wf_live_epoch": 0,
        "wf_live_batch": 0,
        "wf_live_total_epochs": 0,
        "wf_live_total_batches": 0,
        "wf_live_elapsed": 0.0,
        "wf_live_eta": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
