"""
whatif.biophysical.water_balance — FAO-56 single-Kc daily bucket.

Primary source:
    FAO Irrigation & Drainage Paper 56 (Allen, Pereira, Raes, Smith,
    1998), Chapter 8 "ETc under soil water stress conditions".

State (per grid cell):
    Dr(t)   root-zone depletion (mm, ≥ 0, ≤ TAW)

Daily update (equations paraphrased for readability; full form in FAO-56
eqs. 82–86):

    ETc(t)  = Kc(stage(t)) · ET0(t)                     [mm/day]
    Ks(t)   = 1                                          if Dr ≤ RAW
              (TAW − Dr) / (TAW − RAW)                   if RAW < Dr < TAW
              0                                          if Dr ≥ TAW
    ETa(t)  = Ks(t) · ETc(t)
    P_eff   = max(0, rain(t) − runoff(t))
    Irrig   = irrigation_schedule(t)                     [0 for rainfed]
    Dr(t+1) = clip(Dr(t) + ETa(t) − P_eff − Irrig, 0, TAW)
    DP(t)   = max(0, P_eff + Irrig − ETa(t) − (TAW − Dr(t)))    (deep percolation)

Notes:
    * ``runoff`` is 0 in Part 3. The plumbing is here for the Water
      sector to inject a SCS-CN routine later; when that lands, this
      module records the runoff-source version in provenance. Until
      then, every run stamps ``runoff_model = "none-v0"``.
    * Cache key includes crop registry SHA, soil source version, driver
      time-range hash, sow_date, and irrigation-schedule hash. Rerunning
      the same scenario returns byte-equal outputs.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..config.paths import CACHE_DIR
from ..config.region import RegionSpec
from ..scenarios.quantiles import assert_single_quantile
from .crop_water import etc_series, kc_curve, stage_of_day
from .soil import SoilSourceInfo, awc_mm_per_m, raw, taw

# One version per formulaic change to this module.
WATER_BALANCE_VERSION = "wb-single-kc-v1"


@dataclass(frozen=True)
class IrrigationSchedule:
    """Simple deterministic irrigation schedule.

    ``depth_mm`` mm of irrigation is applied on every calendar day
    whose day-of-year lies in ``days_of_year`` (1..366), OR every
    ``interval_days`` days after ``start_offset_days`` since sowing.

    Passed by caller. Never assumed inside the water balance.
    """
    interval_days: int | None = None       # e.g., 7 → weekly
    depth_mm: float = 0.0                  # per event
    start_offset_days: int = 0             # days after sowing to start
    days_of_year: tuple[int, ...] = ()     # explicit dates
    label: str = ""                        # for provenance

    def hash(self) -> str:
        s = f"{self.interval_days}|{self.depth_mm}|{self.start_offset_days}|{self.days_of_year}|{self.label}"
        return hashlib.sha256(s.encode()).hexdigest()[:12]


def _apply_irrigation(
    irr: IrrigationSchedule | None, days_since_sow: np.ndarray,
    doys: np.ndarray,
) -> np.ndarray:
    """Return per-day irrigation depth (mm) matching ``days_since_sow``."""
    out = np.zeros(days_since_sow.shape, dtype=np.float32)
    if irr is None or irr.depth_mm <= 0:
        return out
    if irr.interval_days:
        offset = int(irr.start_offset_days)
        step = int(irr.interval_days)
        mask = (days_since_sow >= offset) & (
            ((days_since_sow - offset) % max(step, 1)) == 0
        ) & (days_since_sow >= 0)
        out[mask] = float(irr.depth_mm)
    if irr.days_of_year:
        doy_set = set(int(d) for d in irr.days_of_year)
        for i, d in enumerate(doys):
            if int(d) in doy_set:
                out[i] = max(out[i], float(irr.depth_mm))
    return out


def _cache_key(
    crop, region: RegionSpec, sow_date: date,
    rain_hash: str, et0_hash: str,
    irr: IrrigationSchedule | None,
    soil_info: SoilSourceInfo,
) -> Path:
    parts = [
        WATER_BALANCE_VERSION,
        crop.key, crop.registry_sha256,
        region.signature(),
        sow_date.isoformat(),
        rain_hash, et0_hash,
        (irr.hash() if irr else "no-irr"),
        soil_info.version,
    ]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return CACHE_DIR / "water_balance" / f"{key}.pkl"


def _da_hash(da: xr.DataArray) -> str:
    """Cheap content hash — first + last timestamp + shape + mean."""
    times = da["time"].values
    t0 = str(times[0]) if len(times) else "empty"
    tN = str(times[-1]) if len(times) else "empty"
    shp = tuple(int(x) for x in da.shape)
    m = float(np.nanmean(da.values)) if da.size else float("nan")
    return hashlib.sha256(f"{t0}|{tN}|{shp}|{m:.6g}".encode()).hexdigest()[:12]


def water_balance(
    crop,
    rain: xr.DataArray,
    et0: xr.DataArray,
    sow_date: date,
    region: RegionSpec,
    irrigation: IrrigationSchedule | None = None,
    *,
    runoff_model: str = "none-v0",
    use_cache: bool = True,
) -> xr.Dataset:
    """Run the FAO-56 single-Kc daily bucket.

    Parameters
    ----------
    crop : ``Crop`` dataclass from :mod:`whatif.sectors.crops`.
    rain, et0 : DataArrays with a shared (time, lat, lon) grid.
    sow_date  : the transplanting / sowing calendar day.
    region    : region already applied to ``rain`` and ``et0``; used for
                cache-keying and provenance.
    irrigation: caller-supplied schedule. ``None`` ⇒ rainfed.
    runoff_model : the runoff-source version; ``"none-v0"`` means
                   runoff = 0 for every day (documented in output attrs).

    Returns
    -------
    xr.Dataset with variables (time, lat, lon):
        ETc, ETa, Ks, Dr, DP, P_eff, Irrig, stage_index
    """
    if rain.dims != et0.dims:
        raise ValueError(f"rain.dims {rain.dims} != et0.dims {et0.dims}")
    # Quantile-mixing guard: rain and et0 must share their quantile tag.
    q_tag = assert_single_quantile(rain, et0)

    # ── AWC → TAW → RAW ──
    awc_da, soil_info = awc_mm_per_m(region)
    # Align AWC to rain's (lat, lon) subset
    awc_da = awc_da.sel(lat=rain["lat"], lon=rain["lon"], method="nearest")
    taw_map = taw(crop, awc_da, stage="max")           # (lat, lon)  mm
    raw_map = raw(crop, taw_map)                        # (lat, lon)  mm

    # ── Cache ──
    cache = _cache_key(crop, region, sow_date, _da_hash(rain), _da_hash(et0),
                        irrigation, soil_info)
    if use_cache and cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    # ── ETc series ──
    etc = etc_series(crop, et0, sow_date)               # (time, lat, lon)

    # ── Day-since-sow index for stage lookup ──
    times = pd.DatetimeIndex(etc["time"].values)
    sow_ts = pd.Timestamp(sow_date)
    days_since = np.array(
        [(pd.Timestamp(t.date()) - sow_ts).days for t in times],
        dtype=np.int64,
    )
    doys = np.array([int(t.dayofyear) for t in times], dtype=np.int64)

    # Stage index for every timestep (0=ini, 1=dev, 2=mid, 3=late)
    stage_names = ["ini", "dev", "mid", "late"]
    stage_idx = np.array(
        [stage_names.index(stage_of_day(crop, int(t), mode="fixed_days"))
         for t in days_since],
        dtype=np.int8,
    )

    # ── Irrigation series ──
    irr_daily = _apply_irrigation(irrigation, days_since, doys)  # (T,)

    # ── Roll the bucket forward, vectorised over (lat, lon) ──
    T, H, W = etc.shape
    rain_v = rain.values.astype(np.float32)
    etc_v = etc.values.astype(np.float32)
    taw_v = taw_map.values.astype(np.float32)
    raw_v = raw_map.values.astype(np.float32)

    Dr = np.zeros((H, W), dtype=np.float32)             # start at field capacity
    Dr_out = np.empty_like(etc_v)
    Ks_out = np.empty_like(etc_v)
    ETa_out = np.empty_like(etc_v)
    DP_out = np.empty_like(etc_v)
    Peff_out = np.empty_like(etc_v)
    Irrig_out = np.empty_like(etc_v)

    # Precompute where TAW == RAW (avoid div-by-zero) → stress mask degenerates
    denom = np.maximum(taw_v - raw_v, 1e-6)             # (H, W)

    for t in range(T):
        etc_t = etc_v[t]                                 # (H, W)
        # Ks
        ks = np.where(
            Dr <= raw_v, 1.0,
            np.where(Dr >= taw_v, 0.0, (taw_v - Dr) / denom),
        ).astype(np.float32)
        eta = ks * etc_t
        # P_eff (runoff = 0 in Part 3)
        peff = np.maximum(rain_v[t], 0.0)
        # Irrigation is the same for every cell in the region
        irr_t = np.full_like(peff, irr_daily[t], dtype=np.float32)
        # Depletion update
        Dr_next = Dr + eta - peff - irr_t
        # Deep percolation (before clipping): any water beyond field capacity
        dp = np.maximum(-Dr_next, 0.0)                   # if Dr went negative, that's DP
        Dr_next = np.clip(Dr_next, 0.0, taw_v)

        Dr_out[t] = Dr
        Ks_out[t] = ks
        ETa_out[t] = eta
        DP_out[t] = dp
        Peff_out[t] = peff
        Irrig_out[t] = irr_t

        Dr = Dr_next

    coords = {"time": etc["time"], "lat": etc["lat"], "lon": etc["lon"]}

    def _mk(name, arr, unit):
        d = xr.DataArray(arr, dims=("time", "lat", "lon"), coords=coords, name=name)
        d.attrs["units"] = unit
        return d

    ds = xr.Dataset({
        "ETc":   _mk("ETc", etc_v, "mm/day"),
        "ETa":   _mk("ETa", ETa_out, "mm/day"),
        "Ks":    _mk("Ks", Ks_out, "1"),
        "Dr":    _mk("Dr", Dr_out, "mm"),
        "DP":    _mk("DP", DP_out, "mm"),
        "P_eff": _mk("P_eff", Peff_out, "mm"),
        "Irrig": _mk("Irrig", Irrig_out, "mm"),
        "stage_index": xr.DataArray(
            stage_idx, dims=("time",), coords={"time": etc["time"]},
            attrs={"legend": "0=ini, 1=dev, 2=mid, 3=late"},
        ),
    })
    ds.attrs.update({
        "version": WATER_BALANCE_VERSION,
        "crop": crop.key,
        "crop_registry_version": crop.registry_version,
        "crop_registry_sha256": crop.registry_sha256,
        "sow_date": sow_date.isoformat(),
        "region_kind": region.kind,
        "region_id": region.id or "",
        "soil_source": soil_info.source,
        "soil_source_version": soil_info.version,
        "soil_warning": soil_info.warning or "",
        "runoff_model": runoff_model,
        "quantile": q_tag,
        "irrigation_hash": irrigation.hash() if irrigation else "no-irr",
        "source_chain": (
            f"{et0.attrs.get('source_chain','ET0')} + rain → "
            f"biophysical.{WATER_BALANCE_VERSION}"
        ),
    })

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache, "wb") as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return ds
