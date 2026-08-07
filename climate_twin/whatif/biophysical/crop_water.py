"""
whatif.biophysical.crop_water — crop-coefficient + stage helpers.

Small, well-tested utilities shared by the water balance and the yield
model:

    kc_curve(crop, t)                      — Kc value at day t (Fig. 25 shape)
    stage_of_day(crop, t, mode)            — "ini" / "dev" / "mid" / "late"
    etc_series(crop, et0, sow_date)        — ETc = Kc × ET0 broadcast
    gdd_stage_schedule(crop, tmax, tmin,
                        sow_date)          — stage lengths in GDD (thermal time)

Primary source:
    FAO Irrigation & Drainage Paper 56 (Allen et al. 1998):
        * Ch. 6 crop coefficients.
        * Ch. 7 length of growth stages.
        * Figure 25 "Generalized crop-coefficient curve".

The Kc curve is piecewise-linear:
    ini      → constant  Kc_ini
    dev      → linear    Kc_ini → Kc_mid
    mid      → constant  Kc_mid
    late     → linear    Kc_mid → Kc_end
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..indices.gdd import gdd


Stage = Literal["ini", "dev", "mid", "late"]
Mode = Literal["fixed_days", "gdd"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def kc_curve(crop, t: int | np.ndarray) -> np.ndarray | float:
    """Kc value at ``t`` days since sowing (FAO-56 Fig. 25 shape).

    ``t`` may be a scalar int or a numpy int array. Values before day 0
    return ``Kc_ini``; values past the last day return ``Kc_end``.
    """
    d_ini = int(crop.stages_days["ini"])
    d_dev = int(crop.stages_days["dev"])
    d_mid = int(crop.stages_days["mid"])
    d_late = int(crop.stages_days["late"])
    kc_ini = float(crop.kc["ini"])
    kc_mid = float(crop.kc["mid"])
    kc_end = float(crop.kc["end"])

    t_arr = np.atleast_1d(np.asarray(t, dtype=np.int64))
    out = np.empty(t_arr.shape, dtype=np.float64)

    # 1) ini phase (constant)
    mask_ini = t_arr < d_ini
    out[mask_ini] = kc_ini

    # 2) dev phase (linear ini → mid)
    d_start_dev = d_ini
    d_end_dev = d_ini + d_dev
    mask_dev = (t_arr >= d_start_dev) & (t_arr < d_end_dev)
    if mask_dev.any():
        frac = (t_arr[mask_dev] - d_start_dev) / max(d_dev, 1)
        out[mask_dev] = kc_ini + frac * (kc_mid - kc_ini)

    # 3) mid phase (constant)
    d_start_mid = d_end_dev
    d_end_mid = d_end_dev + d_mid
    mask_mid = (t_arr >= d_start_mid) & (t_arr < d_end_mid)
    out[mask_mid] = kc_mid

    # 4) late phase (linear mid → end)
    d_start_late = d_end_mid
    d_end_late = d_end_mid + d_late
    mask_late = (t_arr >= d_start_late) & (t_arr < d_end_late)
    if mask_late.any():
        frac = (t_arr[mask_late] - d_start_late) / max(d_late, 1)
        out[mask_late] = kc_mid + frac * (kc_end - kc_mid)

    # 5) after season → Kc_end (last observed value)
    out[t_arr >= d_end_late] = kc_end
    # 6) before sowing → Kc_ini floor
    out[t_arr < 0] = kc_ini

    if np.isscalar(t):
        return float(out[0])
    return out


def stage_of_day(
    crop, t: int, *, mode: Mode = "fixed_days", gdd_accum: float | None = None,
) -> Stage:
    """Return the FAO-56 stage that day ``t`` falls in.

    * ``mode="fixed_days"``:  uses the calendar-day lengths from
      ``crop.stages_days``.
    * ``mode="gdd"``:  interprets ``t`` as accumulated GDD (in °C·day)
      and compares to per-stage GDD budgets derived from crop base
      temperature. ``gdd_accum`` overrides ``t`` if supplied.
    """
    if mode == "fixed_days":
        d_ini = int(crop.stages_days["ini"])
        d_dev = d_ini + int(crop.stages_days["dev"])
        d_mid = d_dev + int(crop.stages_days["mid"])
        d_late = d_mid + int(crop.stages_days["late"])
        if t < d_ini:  return "ini"
        if t < d_dev:  return "dev"
        if t < d_mid:  return "mid"
        if t < d_late: return "late"
        return "late"        # past end → treat as still in late

    if mode == "gdd":
        # Convert stages_days to GDD budgets using a nominal warm-day
        # rate: (t_cap + t_base)/2 - t_base = (t_cap - t_base)/2.
        # Callers who want a real GDD-based schedule pass explicit
        # ``gdd_accum`` (accumulated GDD since sowing).
        rate = 0.5 * (float(crop.t_cap_c) - float(crop.t_base_c))
        if rate <= 0:
            raise ValueError("t_cap_c must exceed t_base_c for GDD scheduling")
        budgets = {k: rate * int(v) for k, v in crop.stages_days.items()}
        thr_ini = budgets["ini"]
        thr_dev = thr_ini + budgets["dev"]
        thr_mid = thr_dev + budgets["mid"]
        thr_late = thr_mid + budgets["late"]
        g = float(gdd_accum) if gdd_accum is not None else float(t)
        if g < thr_ini:  return "ini"
        if g < thr_dev:  return "dev"
        if g < thr_mid:  return "mid"
        if g < thr_late: return "late"
        return "late"

    raise ValueError(f"unknown mode {mode!r}")


def etc_series(crop, et0_da: xr.DataArray, sow_date: date) -> xr.DataArray:
    """ETc (crop water demand) = Kc(t) × ET0(t) for every day in ``et0_da``.

    ``et0_da`` must have a ``time`` dim with tz-aware timestamps; days
    before ``sow_date`` receive Kc = Kc_ini (bare-soil approximation);
    days after the season's end receive Kc = Kc_end.
    """
    times = pd.DatetimeIndex(et0_da["time"].values)
    # tz-naive-safe delta
    days_since = np.array(
        [(pd.Timestamp(t.date()) - pd.Timestamp(sow_date)).days for t in times],
        dtype=np.int64,
    )
    kc = kc_curve(crop, days_since)                          # (T,)
    kc_da = xr.DataArray(
        kc, dims=("time",), coords={"time": et0_da["time"]}, name="kc",
    )
    etc = kc_da * et0_da
    etc.name = "etc"
    etc.attrs.update({
        "units": "mm/day",
        "source_chain": (
            f"{et0_da.attrs.get('source_chain', et0_da.attrs.get('source','ET0'))}"
            f" → biophysical.etc[crop={crop.key},sow={sow_date.isoformat()}]"
        ),
        "quantile": et0_da.attrs.get("quantile", "deterministic"),
        "crop": crop.key,
        "sow_date": sow_date.isoformat(),
    })
    return etc


def gdd_stage_schedule(
    crop, tmax: xr.DataArray, tmin: xr.DataArray, sow_date: date,
) -> xr.Dataset:
    """Return (per cell) the timestamps at which each stage boundary
    passes, based on cumulative GDD since ``sow_date``.

    Output Dataset with variables ``stage_end_ini``, ``stage_end_dev``,
    ``stage_end_mid``, ``stage_end_late`` — each a (lat, lon) DataArray
    of the day-since-sowing at which that stage ends. NaN where the
    season does not accumulate enough GDD within the input window.
    """
    if tmax.dims != tmin.dims:
        raise ValueError(f"tmax.dims {tmax.dims} != tmin.dims {tmin.dims}")

    daily_gdd = gdd(tmax, tmin,
                     t_base=float(crop.t_base_c),
                     t_cap=float(crop.t_cap_c))
    # Slice to sow_date onward
    times = pd.DatetimeIndex(daily_gdd["time"].values)
    sow_ts = pd.Timestamp(sow_date)
    days_since = np.array(
        [(pd.Timestamp(t.date()) - sow_ts).days for t in times],
        dtype=np.int64,
    )
    keep = days_since >= 0
    daily_gdd = daily_gdd.isel(time=np.flatnonzero(keep))
    cum = daily_gdd.cumsum("time", skipna=True)

    rate = 0.5 * (float(crop.t_cap_c) - float(crop.t_base_c))
    thr = {
        "ini":  rate * int(crop.stages_days["ini"]),
        "dev":  rate * (int(crop.stages_days["ini"]) + int(crop.stages_days["dev"])),
        "mid":  rate * (
            int(crop.stages_days["ini"]) + int(crop.stages_days["dev"])
            + int(crop.stages_days["mid"])
        ),
        "late": rate * int(crop.total_days),
    }

    days_since_pos = days_since[keep]
    ds = xr.Dataset()
    cum_vals = cum.values                          # (T', lat, lon)
    for stage in ("ini", "dev", "mid", "late"):
        # For each cell, find first index where cum >= threshold
        threshold = thr[stage]
        cross = (cum_vals >= threshold).astype(np.int8)
        # argmax returns 0 if all zero — mark those as NaN
        first_idx = cross.argmax(axis=0)
        any_hit = cross.any(axis=0)
        end_day = np.where(any_hit,
                            days_since_pos[np.clip(first_idx, 0, len(days_since_pos) - 1)],
                            np.nan)
        ds[f"stage_end_{stage}"] = xr.DataArray(
            end_day.astype(np.float32),
            dims=("lat", "lon"),
            coords={"lat": daily_gdd.lat, "lon": daily_gdd.lon},
        )
    ds.attrs.update({
        "crop": crop.key,
        "sow_date": sow_date.isoformat(),
        "t_base_c": float(crop.t_base_c),
        "t_cap_c": float(crop.t_cap_c),
        "gdd_thresholds": thr,
        "method": "cumulative-gdd-vs-fixed-thresholds",
    })
    return ds
