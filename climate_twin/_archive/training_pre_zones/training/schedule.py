"""Walk-forward schedule builder.

Builds the list of (train_start, train_end, val_start, val_end) rounds
from a user-specified configuration. Auto-caps at 500 rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta


MAX_ROUNDS = 500


@dataclass
class Round:
    """One walk-forward round definition."""
    round_num: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    @property
    def train_label(self) -> str:
        return f"{self.train_start.isoformat()} → {self.train_end.isoformat()}"

    @property
    def val_label(self) -> str:
        return f"{self.val_start.isoformat()} → {self.val_end.isoformat()}"


@dataclass
class ScheduleConfig:
    """Walk-forward schedule parameters."""
    start_year: int = 1975
    end_year: int = 2024
    initial_window_years: int = 10
    step_unit: str = "months"  # "days", "months", "years"
    step_size: int = 1
    window_mode: str = "expanding"  # "expanding" or "rolling"
    rolling_window_years: int = 10
    prediction_horizon: int = 1  # steps forward

    def step_delta(self) -> relativedelta:
        if self.step_unit == "days":
            return relativedelta(days=self.step_size)
        elif self.step_unit == "months":
            return relativedelta(months=self.step_size)
        elif self.step_unit == "years":
            return relativedelta(years=self.step_size)
        raise ValueError(f"Unknown step_unit: {self.step_unit}")

    def horizon_delta(self) -> relativedelta:
        """Duration of the prediction/validation window (same unit as step)."""
        if self.step_unit == "days":
            return relativedelta(days=self.step_size * self.prediction_horizon)
        elif self.step_unit == "months":
            return relativedelta(months=self.step_size * self.prediction_horizon)
        elif self.step_unit == "years":
            return relativedelta(years=self.step_size * self.prediction_horizon)
        raise ValueError(f"Unknown step_unit: {self.step_unit}")


def build_schedule(cfg: ScheduleConfig) -> list[Round]:
    """Generate the full walk-forward round list.

    Returns at most MAX_ROUNDS rounds. If the configuration would produce
    more, it truncates and signals via the returned list length.
    """
    data_start = date(cfg.start_year, 1, 1)
    data_end = date(cfg.end_year, 12, 31)

    # First prediction starts after the initial window
    initial_end = data_start + relativedelta(years=cfg.initial_window_years) - relativedelta(days=1)
    if initial_end >= data_end:
        return []

    step = cfg.step_delta()
    horizon = cfg.horizon_delta()

    rounds: list[Round] = []
    cursor = initial_end + relativedelta(days=1)  # first val start
    round_num = 1

    while cursor + horizon - relativedelta(days=1) <= data_end:
        if round_num > MAX_ROUNDS:
            break

        val_start = cursor
        val_end = cursor + horizon - relativedelta(days=1)

        # Training window
        train_end = val_start - relativedelta(days=1)
        if cfg.window_mode == "expanding":
            train_start = data_start
        else:  # rolling
            rolling_start = train_end - relativedelta(years=cfg.rolling_window_years) + relativedelta(days=1)
            train_start = max(data_start, rolling_start)

        rounds.append(Round(
            round_num=round_num,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))

        cursor += step
        round_num += 1

    return rounds


def estimate_runtime(n_rounds: int, seconds_per_round: float = 10.0) -> str:
    """Human-readable runtime estimate."""
    total_s = n_rounds * seconds_per_round
    if total_s < 60:
        return f"~{total_s:.0f}s"
    elif total_s < 3600:
        return f"~{total_s / 60:.0f}m"
    else:
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        return f"~{h}h {m}m"
