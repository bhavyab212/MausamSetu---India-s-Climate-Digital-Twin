"""
whatif.drivers.analog_backtest — analog engine scored as a forecaster.

Closes the loop from Part 5. The analog engine produces a probabilistic
forecast over an event definition (a :class:`CostLossSetup`). Score it
using the same Murphy 1977 + Brier + reliability toolkit shipped in
Part 4.

Primary source:
    * Hamill, T.M. & Whitaker, J.S. (2006) "Probabilistic quantitative
      precipitation forecasts based on reforecast analogs: theory and
      application." Mon. Weather Rev. 134:3209-3229. Same evaluation
      protocol we use here.

Rule 1 (leakage): walk-forward strictly excludes year ``y`` and every
year ``≥ y`` from the analog pool used to forecast year y. This is the
enforcement point — the exclude-years argument is not optional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..config.paths import CACHE_DIR
from .analog_features import AnalogSpec
from .analogs import build_analog_pool, find_analogs, target_from_year
from .analog_outcomes import _compute_weights

# Lazy-imported inside the function body to avoid an import cycle
# (drivers → economics → indices → drivers). analog_backtest is only
# invoked from callers who already have economics loaded.


ANALOG_BACKTEST_VERSION = "analog-backtest-v1"


def analog_forecaster_backtest(
    setup,                                    # CostLossSetup — lazy typed
    spec: AnalogSpec,
    event_observed_by_year: dict[int, int],
    *,
    train_years: tuple[int, int] | None = None,
    valid_years: tuple[int, int] | None = None,
    k: int = 10,
    weighting: str = "inv_distance",
    softmax_tau: float = 1.0,
    write_parquet: bool = True,
):
    """Walk-forward score of the analog engine as a probabilistic forecaster.

    Parameters
    ----------
    setup :
        The event definition (:class:`CostLossSetup`).
    spec :
        The analog problem (region, window, features, metric).
    event_observed_by_year :
        Mapping ``{year: 0|1}`` for every year in
        ``train_years ∪ valid_years`` — the observed event indicator.
        The caller supplies this so the backtest doesn't touch the cube
        beyond feature building.
    train_years, valid_years, k, weighting, softmax_tau :
        Standard.

    Returns
    -------
    BacktestResult — same shape as the Part-4 economics backtest.

    Contract
    --------
    For each verification year ``y ∈ valid_years``:
      1. Build the analog pool excluding every year ``≥ y``.
      2. Compute the target features from observed data at year y.
      3. Retrieve the k closest analogs.
      4. Compute weighted probability
         ``p(y) = Σ w_i · 𝟙[event_observed_by_year[analog_year] == 1]``.
      5. Score against ``event_observed_by_year[y]``.
    """
    from ..economics.backtest import BACKTEST_VERSION, BacktestResult
    from ..economics.cost_loss import (
        COST_LOSS_VERSION,
        brier_score,
        brier_skill_score,
        reliability_diagram,
        value_curve,
    )
    from ..indices.reference import LeakageError, TRAIN_YEARS, VALID_YEARS

    train_years = train_years or TRAIN_YEARS
    valid_years = valid_years or VALID_YEARS

    if not setup.complete:
        raise ValueError(
            f"setup {setup.action_id!r} has complete=False; the analog "
            "backtest refuses to run an under-cited setup."
        )
    C, L = float(setup.C_inr_per_ha), float(setup.L_inr_per_ha)
    y0, y1 = int(valid_years[0]), int(valid_years[1])

    # Sanity: every year we'll touch must be in the observed dict
    for y in range(train_years[0], y1 + 1):
        if y not in event_observed_by_year:
            raise ValueError(
                f"event_observed_by_year missing year {y}; provide 0/1 "
                "for every year in TRAIN_YEARS ∪ VALID_YEARS"
            )

    # Climatology base rate = event frequency over TRAIN_YEARS ONLY
    train_events = np.array(
        [int(event_observed_by_year[y])
         for y in range(train_years[0], train_years[1] + 1)],
        dtype=np.int32,
    )
    p_clim = float(train_events.mean())
    always_act_clim = p_clim > (C / L)

    per_year_rows = []
    for y in range(y0, y1 + 1):
        # ── Enforce Rule 1: exclude y and every year ≥ y ──
        excl = tuple(range(int(y), int(y1) + 1))
        pool = build_analog_pool(spec, exclude_years=excl, use_cache=True)
        if not pool.years():
            # No pool available (edge case) — abstain by defaulting to
            # climatology forecast for this year.
            p_forecast = p_clim
        else:
            # Assert leakage guard held
            if any(int(yy) >= int(y) for yy in pool.years()):
                raise LeakageError(
                    f"analog pool for target year {y} contains "
                    f"forbidden year(s) {[yy for yy in pool.years() if yy >= y]}"
                )
            target = target_from_year(int(y), spec)
            matches = find_analogs(target, pool, k=int(k))
            if not matches:
                p_forecast = p_clim
            else:
                w_by_year = _compute_weights(matches, weighting, float(softmax_tau))
                p_forecast = 0.0
                for m in matches:
                    p_forecast += w_by_year[m.year] * float(
                        event_observed_by_year[m.year]
                    )
                p_forecast = float(np.clip(p_forecast, 0.0, 1.0))

        obs = int(event_observed_by_year[y])
        act_forecast = int(p_forecast > 0.5)
        act_clim = int(always_act_clim)
        act_perfect = obs

        expense_forecast = C * act_forecast + L * obs * (1 - act_forecast)
        expense_clim = C * act_clim + L * obs * (1 - act_clim)
        expense_perfect = C * act_perfect

        per_year_rows.append({
            "year": int(y),
            "forecast_prob": float(p_forecast),
            "observed": int(obs),
            "action_forecast": int(act_forecast),
            "action_clim": int(act_clim),
            "action_perfect": int(act_perfect),
            "expense_forecast": float(expense_forecast),
            "expense_clim": float(expense_clim),
            "expense_perfect": float(expense_perfect),
        })

    per_year = pd.DataFrame(per_year_rows)
    if per_year.empty:
        raise ValueError(f"no verification rows for {y0}-{y1}")

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
    vc = value_curve(
        setup, pd.Series(fc), pd.Series(ob), p_clim,
    )

    result = BacktestResult(
        rule_id=f"analog@{spec.version}",
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
            "rule": "analog_forecaster",
            "analog_spec_signature": spec.signature(),
            "analog_spec_version": spec.version,
            "features": list(spec.features),
            "metric": spec.metric,
            "window": spec.window,
            "region": f"{spec.region.kind}:{spec.region.id}",
            "k": int(k),
            "weighting": weighting,
            "softmax_tau": float(softmax_tau),
            "train_years": list(train_years),
            "valid_years": [y0, y1],
            "p_climatology": p_clim,
            "cost_loss_version": COST_LOSS_VERSION,
            "backtest_version": BACKTEST_VERSION,
            "analog_backtest_version": ANALOG_BACKTEST_VERSION,
        },
        version=BACKTEST_VERSION,
    )

    if write_parquet:
        _persist(result, spec.version)
    return result


def _persist(result: BacktestResult, spec_version: str) -> Path:
    out_dir = CACHE_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"analog_{spec_version}_{result.setup_id}.parquet"
    df = result.per_year.copy()
    df.attrs.update({
        "V_forecast": result.V_forecast,
        "brier": result.brier,
        "brier_skill": result.brier_skill,
        "provenance": str(result.provenance),
    })
    try:
        df.to_parquet(p)
    except Exception:
        df.to_csv(p.with_suffix(".csv"), index=False)
    return p
