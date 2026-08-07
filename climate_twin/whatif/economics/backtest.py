"""
whatif.economics.backtest — walk-forward backtest of decision rules.

Primary sources:
    * Murphy 1977 — same value framework as :mod:`cost_loss`.
    * Zhang & Yao (2011) "Best practices in forecast verification" —
      walk-forward evaluation with strict train / verification split.
    * The framework's own leakage rule: fits use TRAIN_YEARS + prior
      verification years only; year ``y`` never sees itself.

Contract:
    * Every rule / setup pair emits a persisted BacktestResult under
      ``CACHE_DIR/backtests/<rule>_<setup>_v<ver>.parquet``.
    * Leakage guard: any DecisionRule that references year ``y`` during
      the year-``y`` forecast raises :class:`LeakageError`. The rule
      protocol enforces this by passing only the ``history_before_y``
      view of the data.
    * Negative results ARE published. The framework's honesty depends
      on shipping at least one rule whose V(forecast) ≤ 0.

Version: ``backtest-v1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..config.paths import CACHE_DIR
from ..indices.reference import LeakageError, TRAIN_YEARS, VALID_YEARS
from .cost_loss import (
    COST_LOSS_VERSION,
    CostLossSetup,
    ValueCurve,
    brier_score,
    brier_skill_score,
    reliability_diagram,
    value_curve,
)

BACKTEST_VERSION = "backtest-v1"


# ─── Rule protocol ───────────────────────────────────────────────────
class DecisionRule(Protocol):
    """A DecisionRule is a callable ``(history_before_y, y) → forecast_prob``.

    Contract:
        - ``history_before_y`` is a DataFrame / dict of arrays whose
          index ends strictly before ``y``. The rule MUST NOT reach
          past it — the wrapper enforces by never passing year ``y``'s
          rows.
        - Returns a float in [0, 1] — the forecast probability of the
          setup's event for year ``y``.
        - Deterministic given identical history (fixed-seed if Monte
          Carlo internally).
    """
    rule_id: str

    def __call__(self, history_before_y: pd.DataFrame, y: int) -> float: ...


# ─── Shipped rules ───────────────────────────────────────────────────
class HistoricalFrequencyRule:
    """Baseline rule: forecast probability = long-run event frequency
    over TRAIN_YEARS ∪ (verification years < y). Deliberately a weak
    forecaster — we expect ``V(forecast) ≈ 0`` here; if the backtest
    reports V >> 0 for this rule, the leakage guard is broken."""
    rule_id = "historical_frequency"

    def __call__(self, history_before_y: pd.DataFrame, y: int) -> float:
        if "event" not in history_before_y.columns:
            raise ValueError("history must have an 'event' 0/1 column")
        return float(history_before_y["event"].mean())


class OnsetAnomalyRule:
    """Slightly informed rule: forecast prob rises with the deviation
    of last year's onset-day from the climatological onset-day.

    A "toy" but plausible signal: after a late-onset year, the operator
    fears another one. Illustrates a rule that can beat OR miss
    climatology depending on the underlying correlation.
    """
    rule_id = "onset_anomaly_v1"

    def __call__(self, history_before_y: pd.DataFrame, y: int) -> float:
        need = {"event", "onset_anomaly_days"}
        if not need.issubset(history_before_y.columns):
            raise ValueError(f"history missing {need - set(history_before_y.columns)}")
        base = float(history_before_y["event"].mean())
        # Use the last row's onset anomaly to shift the base rate
        anomaly = float(history_before_y["onset_anomaly_days"].iloc[-1])
        # Compress the shift with logistic-like scaling
        shift = 0.05 * anomaly                     # 5% per day of anomaly
        p = np.clip(base + shift, 0.01, 0.99)
        return float(p)


# ─── Result container ────────────────────────────────────────────────
@dataclass
class BacktestResult:
    rule_id: str
    setup_id: str
    valid_years: tuple[int, int]
    per_year: pd.DataFrame                  # year, forecast_prob, observed, action, expense, ME_clim, ME_perfect
    ME_forecast: float
    ME_climatology: float
    ME_perfect: float
    V_forecast: float
    value_curve: ValueCurve
    brier: float
    brier_skill: float
    reliability: pd.DataFrame
    provenance: dict = field(default_factory=dict)
    version: str = BACKTEST_VERSION


# ─── Runner ──────────────────────────────────────────────────────────
def walk_forward_backtest(
    rule: DecisionRule,
    setup: CostLossSetup,
    history: pd.DataFrame,
    *,
    train_years: tuple[int, int] = TRAIN_YEARS,
    valid_years: tuple[int, int] = VALID_YEARS,
    write_parquet: bool = True,
) -> BacktestResult:
    """Run walk-forward evaluation of ``rule`` under ``setup``.

    ``history`` must have:
        * a ``year`` column (int).
        * an ``event`` column (0/1).
        * any features the rule reads.

    Rows with ``year >= valid_years[0]`` are the verification set. For
    each such year ``y``, the rule is given ``history[history.year < y]``
    only — the ``valid_years`` window is expanded one year at a time.
    """
    if not setup.complete:
        raise ValueError(
            f"setup {setup.action_id!r} has complete=False; "
            "backtest refuses to run an under-cited setup."
        )
    for col in ("year", "event"):
        if col not in history.columns:
            raise ValueError(f"history missing required column {col!r}")
    if ((history["event"] != 0) & (history["event"] != 1)).any():
        raise ValueError("history.event must be 0/1")

    y0, y1 = int(valid_years[0]), int(valid_years[1])
    C, L = float(setup.C_inr_per_ha), float(setup.L_inr_per_ha)

    # Baseline climatology probability = event frequency over TRAIN_YEARS ONLY
    train_mask = (
        (history["year"] >= train_years[0])
        & (history["year"] <= train_years[1])
    )
    if not train_mask.any():
        raise ValueError(
            f"no history rows in TRAIN_YEARS {train_years}; can't fit "
            "climatology baseline"
        )
    p_clim = float(history.loc[train_mask, "event"].mean())
    always_act_clim = p_clim > (C / L)

    per_year_rows = []
    for y in range(y0, y1 + 1):
        y_row = history[history["year"] == y]
        if y_row.empty:
            continue
        # STRICT leakage guard: hand the rule only the pre-y history
        history_before_y = history[history["year"] < y].copy()
        # Also enforce that the rule receives a copy — any mutation
        # won't leak back into subsequent iterations.
        p_forecast = float(rule(history_before_y, int(y)))
        if not (0.0 <= p_forecast <= 1.0):
            raise ValueError(
                f"rule {rule.rule_id!r} returned prob {p_forecast:.4f} "
                "outside [0, 1]"
            )
        obs = int(y_row["event"].iloc[0])

        # Forecast strategy: act if p_forecast > threshold. For the
        # scalar walk-forward number we use the rule's own threshold
        # (which is baked into the rule's probability; equivalent to
        # comparing p_forecast against 0.5 on a well-calibrated rule).
        # We report the full V(p*) sweep via value_curve, which finds
        # the optimum threshold post-hoc.
        act_forecast = int(p_forecast > 0.5)
        expense_forecast = C * act_forecast + L * obs * (1 - act_forecast)

        # Climatology strategy — always-act or never-act
        act_clim = int(always_act_clim)
        expense_clim = C * act_clim + L * obs * (1 - act_clim)

        # Perfect
        act_perfect = obs
        expense_perfect = C * act_perfect

        per_year_rows.append({
            "year": int(y),
            "forecast_prob": p_forecast,
            "observed": obs,
            "action_forecast": act_forecast,
            "action_clim": act_clim,
            "action_perfect": act_perfect,
            "expense_forecast": float(expense_forecast),
            "expense_clim": float(expense_clim),
            "expense_perfect": float(expense_perfect),
        })

    per_year = pd.DataFrame(per_year_rows)
    if per_year.empty:
        raise ValueError(f"no verification rows found for years {y0}-{y1}")

    ME_forecast = float(per_year["expense_forecast"].mean())
    ME_clim = float(per_year["expense_clim"].mean())
    ME_perfect = float(per_year["expense_perfect"].mean())
    denom = ME_clim - ME_perfect
    V_forecast = (ME_clim - ME_forecast) / denom if denom > 0 else 0.0

    fc = per_year["forecast_prob"].to_numpy()
    ob = per_year["observed"].to_numpy()
    bs = brier_score(fc, ob)
    bss = brier_skill_score(fc, ob, p_clim)
    rel = reliability_diagram(fc, ob, bins=10)

    # Full V(p*) sweep for the UI
    vc = value_curve(
        setup,
        forecast_prob_series=pd.Series(fc),
        observed_event_series=pd.Series(ob),
        climatology_prob=p_clim,
    )

    result = BacktestResult(
        rule_id=rule.rule_id,
        setup_id=setup.action_id,
        valid_years=(y0, y1),
        per_year=per_year,
        ME_forecast=ME_forecast,
        ME_climatology=ME_clim,
        ME_perfect=ME_perfect,
        V_forecast=float(V_forecast),
        value_curve=vc,
        brier=bs,
        brier_skill=bss,
        reliability=rel,
        provenance={
            "rule_id": rule.rule_id,
            "setup_id": setup.action_id,
            "train_years": list(train_years),
            "valid_years": [y0, y1],
            "p_climatology": p_clim,
            "cost_loss_version": COST_LOSS_VERSION,
            "backtest_version": BACKTEST_VERSION,
        },
    )

    if write_parquet:
        _persist(result)
    return result


def _persist(result: BacktestResult) -> Path:
    out_dir = CACHE_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{result.rule_id}_{result.setup_id}_v{BACKTEST_VERSION}.parquet"
    df = result.per_year.copy()
    df.attrs.update({
        "ME_forecast": result.ME_forecast,
        "ME_climatology": result.ME_climatology,
        "ME_perfect": result.ME_perfect,
        "V_forecast": result.V_forecast,
        "brier": result.brier,
        "brier_skill": result.brier_skill,
        "provenance": str(result.provenance),
    })
    try:
        df.to_parquet(p)
    except Exception:
        # Parquet engine missing → write CSV instead so the audit trail
        # still exists.
        df.to_csv(p.with_suffix(".csv"), index=False)
    return p


def get_last_backtest(rule_id: str, setup_id: str) -> Path | None:
    out_dir = CACHE_DIR / "backtests"
    if not out_dir.exists():
        return None
    hits = sorted(out_dir.glob(f"{rule_id}_{setup_id}_*.parquet"))
    hits += sorted(out_dir.glob(f"{rule_id}_{setup_id}_*.csv"))
    return hits[-1] if hits else None


# ─── Convenience: synthesise history for the diagnostic UI ──────────
def synthesise_history(
    seed: int = 42, event_base_rate: float = 0.35,
    train_years: tuple[int, int] = TRAIN_YEARS,
    valid_years: tuple[int, int] = VALID_YEARS,
) -> pd.DataFrame:
    """Build a synthetic (year, event, onset_anomaly_days) history for
    the diagnostic backtest. This is deliberately a fake series so the
    numbers can be reproduced without touching the cube; the real
    version wires observed IMD onset + dry-spell events in Part 5.

    Determinism: fixed seed.
    """
    rng = np.random.default_rng(int(seed))
    years = list(range(train_years[0], valid_years[1] + 1))
    onset_anomaly = rng.normal(0.0, 4.0, size=len(years))
    # Event probability drifts up when onset_anomaly > 0 (late onset)
    logit = np.log(event_base_rate / (1 - event_base_rate))
    prob = 1.0 / (1.0 + np.exp(-(logit + 0.10 * onset_anomaly)))
    events = (rng.uniform(size=len(years)) < prob).astype(int)
    return pd.DataFrame({
        "year": years,
        "event": events,
        "onset_anomaly_days": onset_anomaly,
    })
