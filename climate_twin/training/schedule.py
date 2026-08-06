"""training.schedule — Phase 0 shim."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from . import TrainingRebuildInProgress


MAX_ROUNDS = 0


@dataclass
class Round:
    round_num: int = 0
    train_start: date = date(1951, 1, 1)
    train_end: date = date(1951, 1, 1)
    val_start: date = date(1951, 1, 1)
    val_end: date = date(1951, 1, 1)


@dataclass
class ScheduleConfig:
    start_year: int = 1951
    end_year: int = 2025


def build_schedule(*args, **kwargs) -> list[Round]:
    raise TrainingRebuildInProgress("schedule.build_schedule")


def estimate_runtime(*args, **kwargs) -> str:
    return "n/a (rebuild in progress)"
