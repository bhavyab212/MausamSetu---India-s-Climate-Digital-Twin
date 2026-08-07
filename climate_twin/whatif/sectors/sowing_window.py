"""
whatif.sectors.sowing_window — sowing-date optimiser.

Primary sources: this module composes machinery from Parts 2 and 3
whose citations appear in place — FAO Irrigation & Drainage Paper 56
(Allen et al. 1998) for the water balance, FAO-33 (Doorenbos & Kassam
1979) for the multi-stage yield-response formulation, and Hargreaves
& Samani 1985 for the reference ET0 used inside the L2 loop.

Runs the full L2+L3 chain (water balance → multi-stage FAO-33 yield)
for every candidate sowing date, across the three forecast quantiles
(q10/q50/q90) independently, and reports:

    * expected yield        (probability-weighted across quantiles)
    * probability of "good year" (Ya > threshold × Ymax)
    * worst-case yield      (q10 = 10th-percentile forcing scenario)
    * flowering heat-stress days
    * comparator baseline yield from :func:`yield_baseline`

The probability-weighting convention (p(q10)=0.25, p(q50)=0.50,
p(q90)=0.25) is a fixed default. It appears as
``QUANTILE_PROBABILITIES`` in this module and is exposed as a knob to
callers.

This is L3 output; L4 economics (Part 4) converts these into an
INR-denominated payoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from ..biophysical.water_balance import IrrigationSchedule, water_balance
from ..config.region import RegionSpec
from ..indices.et0_hargreaves import et0_hargreaves
from ..scenarios.quantiles import assert_single_quantile
from .agriculture import yield_baseline, yield_water_limited


# Fixed probability weights across the three quantile-forcing runs.
# Rationale: a symmetric 25/50/25 triangular weighting on the
# probability triple (q10, q50, q90). Exposed as a knob so downstream
# risk-sensitive callers can swap for a heavier-tailed weight.
QUANTILE_PROBABILITIES: dict[str, float] = {"q10": 0.25, "q50": 0.50, "q90": 0.25}


@dataclass(frozen=True)
class QuantileDriverBundle:
    """Three-pass driver bundle. Each key holds a Dataset with the
    variables ``rain``, ``tmax``, ``tmin`` at daily resolution and IST
    time coords, tagged with ``attrs["quantile"] = q``."""
    q10: xr.Dataset
    q50: xr.Dataset
    q90: xr.Dataset


def _spatial_mean(ds: xr.Dataset, var: str) -> float:
    """Region-mean of ``var`` (NaN-safe)."""
    if var not in ds:
        return float("nan")
    arr = ds[var].values
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


def _spatial_quantile(ds: xr.Dataset, var: str, q: float) -> float:
    arr = ds[var].values
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def optimize_sowing_window(
    crop,
    region: RegionSpec,
    driver_bundle: QuantileDriverBundle,
    candidate_sow_dates: list[date],
    irrigation: IrrigationSchedule | None = None,
    *,
    good_year_threshold: float = 0.75,     # 75 % of Ymax
    quantile_probabilities: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame ranking candidate sowing dates by expected yield.

    Columns:
        sow_date                       ISO date
        Ey                             expected Ya (t/ha)
        Ey_over_Ymax                   Ey / crop.ymax_t_per_ha
        p_good_year                    P(Ya > threshold · Ymax) across quantiles
        Ya_q10 / Ya_q50 / Ya_q90       per-quantile Ya (t/ha)
        worst_case_Ya                  = Ya_q10 (10th-percentile driver)
        heat_stress_days_flower_q50    median heat-stress days at flowering
        baseline_Ya                    climatology-only yield (t/ha)
        value_vs_baseline              Ey - baseline_Ya (t/ha)

    The DataFrame is sorted by ``Ey`` descending.
    """
    q_probs = dict(quantile_probabilities or QUANTILE_PROBABILITIES)
    if not np.isclose(sum(q_probs.values()), 1.0, atol=1e-6):
        raise ValueError(
            f"quantile_probabilities must sum to 1.0, got {sum(q_probs.values()):.6f}"
        )

    rows = []
    baseline_cache: dict[date, xr.Dataset] = {}

    for sow in candidate_sow_dates:
        per_q: dict[str, xr.Dataset] = {}
        for q_key in ("q10", "q50", "q90"):
            ds = getattr(driver_bundle, q_key)
            # Enforce quantile isolation
            _ = assert_single_quantile(ds)
            rain = ds["rain"]
            tmax = ds["tmax"]
            tmin = ds["tmin"]
            et0 = et0_hargreaves(tmax, tmin)
            wb = water_balance(crop, rain, et0, sow, region,
                                irrigation=irrigation, use_cache=True)
            y = yield_water_limited(crop, wb, tmax=tmax)
            per_q[q_key] = y

        ya_q = {k: _spatial_mean(v, "Ya") for k, v in per_q.items()}
        yof_q = {k: _spatial_mean(v, "Ya_over_Ymax") for k, v in per_q.items()}
        Ey = sum(q_probs[k] * ya_q[k] for k in ("q10", "q50", "q90"))
        p_good = sum(
            q_probs[k] * (1.0 if yof_q[k] > good_year_threshold else 0.0)
            for k in ("q10", "q50", "q90")
        )
        heat_days_med = _spatial_mean(per_q["q50"], "heat_stress_days")

        # Baseline (once per sow date, from climatology)
        if sow not in baseline_cache:
            baseline_cache[sow] = yield_baseline(crop, region, sow)
        baseline_Ya = _spatial_mean(baseline_cache[sow], "Ya")

        rows.append({
            "sow_date": sow.isoformat(),
            "Ey": Ey,
            "Ey_over_Ymax": Ey / float(crop.ymax_t_per_ha),
            "p_good_year": p_good,
            "Ya_q10": ya_q["q10"],
            "Ya_q50": ya_q["q50"],
            "Ya_q90": ya_q["q90"],
            "worst_case_Ya": ya_q["q10"],
            "heat_stress_days_flower_q50": heat_days_med,
            "baseline_Ya": baseline_Ya,
            "value_vs_baseline": Ey - baseline_Ya,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Ey", ascending=False).reset_index(drop=True)


def build_deterministic_bundle(
    rain: xr.DataArray, tmax: xr.DataArray, tmin: xr.DataArray,
) -> QuantileDriverBundle:
    """Convenience: wrap a single deterministic (historical) driver as
    a QuantileDriverBundle with the same series in all three slots.

    Used for the diagnostic panel where the observed history is played
    back as three synthetic "quantiles" so the pipeline shape can be
    exercised end-to-end.
    """
    def _tag(da: xr.DataArray, q: str) -> xr.DataArray:
        d = da.copy(deep=False)
        d.attrs = dict(da.attrs)
        d.attrs["quantile"] = q
        return d

    def _ds(q: str) -> xr.Dataset:
        return xr.Dataset({
            "rain": _tag(rain, q),
            "tmax": _tag(tmax, q),
            "tmin": _tag(tmin, q),
        }, attrs={"quantile": q, "source": "historical-as-triplet"})

    return QuantileDriverBundle(q10=_ds("q10"), q50=_ds("q50"), q90=_ds("q90"))
