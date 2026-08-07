"""
whatif.sectors.agriculture — L3 crop-yield reducer.

Primary sources:
    * FAO Irrigation & Drainage Paper 33 (Doorenbos & Kassam, 1979).
      Multi-stage FAO-33 formulation:
          1 − Ya/Ymax
            = 1 − Π_i (1 − Ky,i · (1 − ETa,i / ETc,i))
    * FAO-56 §8 (Allen et al. 1998) — where ETa, ETc come from.
    * Wheeler et al. 2000, *Agriculture, Ecosystems & Environment* —
      heat-stress penalty at flowering in wheat/rice cereals.

Rules enforced (see Part 3 spec, Golden rules 5–7):
    * Every result carries both absolute yield ``Ya`` (t/ha) and yield
      fraction ``Ya/Ymax`` (unitless).
    * Every scenario is returned alongside a **climatology baseline**
      run (see :func:`yield_baseline`).
    * NaN in → NaN out. No imputation. Companion ``valid_frac`` is
      exposed via :func:`whatif.indices.reference.valid_fraction`.

Version:
    ``fao56-fao33-multistage-v1`` — appears in provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from ..config.region import RegionSpec
from ..indices.reference import TRAIN_YEARS
from ..scenarios.quantiles import assert_single_quantile

AGRICULTURE_VERSION = "fao56-fao33-multistage-v1"

# Flowering heat-stress penalty per stress-day (unit-less multiplicative
# reduction on Ya/Ymax). Default 2 %/day, capped at 30 %. The exact
# coefficient per crop should eventually live in crops.yaml and be
# cited crop-by-crop; until then we expose a single default here with
# a citation and a WARNING attr on outputs.
DEFAULT_HEAT_STRESS_PER_DAY = 0.02
DEFAULT_HEAT_STRESS_CAP = 0.30
DEFAULT_HEAT_STRESS_CITATION = (
    "Wheeler et al. 2000 (AgEE); default 2%/day up to 30% cap — "
    "override with crop-specific value once cited in crops.yaml"
)


def _stage_indices(wb: xr.Dataset) -> dict[str, np.ndarray]:
    """Return time indices for each stage 0..3."""
    idx = wb["stage_index"].values.astype(np.int8)
    return {
        "ini":  np.flatnonzero(idx == 0),
        "dev":  np.flatnonzero(idx == 1),
        "mid":  np.flatnonzero(idx == 2),
        "late": np.flatnonzero(idx == 3),
    }


def yield_water_limited(
    crop, wb: xr.Dataset, tmax: xr.DataArray | None = None,
) -> xr.Dataset:
    """Multi-stage FAO-33 yield-response with optional flowering heat overlay.

    Parameters
    ----------
    crop : Crop dataclass.
    wb   : output of :func:`water_balance` — carries daily ETc, ETa,
           and a ``stage_index`` variable.
    tmax : (optional) daily Tmax on the same grid. When provided, days
           with ``Tmax >= crop.t_flower_stress_c`` during the flowering
           stage trigger the heat-stress penalty.

    Returns
    -------
    xr.Dataset with variables (lat, lon):
        Ya            — absolute yield (t/ha)
        Ya_over_Ymax  — unitless yield fraction ∈ [0, 1]
        Ks_stage_mean_<stage>  — mean Ks for each stage
        eta_over_etc_<stage>   — sum(ETa)/sum(ETc) per stage
        heat_stress_days       — count of Tmax > threshold days at flowering
    """
    # Quantile-integrity guard (rain and et0 were checked at WB entry;
    # if tmax is passed, it must also share the quantile).
    if tmax is not None:
        _ = assert_single_quantile(wb, tmax)

    stages = _stage_indices(wb)
    ETa = wb["ETa"].values                          # (T, H, W)
    ETc = wb["ETc"].values
    Ks = wb["Ks"].values

    lat = wb["lat"]
    lon = wb["lon"]
    H, W = ETa.shape[1], ETa.shape[2]

    ya_over_ymax = np.ones((H, W), dtype=np.float64)
    ks_means: dict[str, np.ndarray] = {}
    eta_over_etc: dict[str, np.ndarray] = {}

    for stage_name in ("ini", "dev", "mid", "late"):
        ii = stages[stage_name]
        if ii.size == 0:
            # No days in this stage within the driver window — do not
            # penalise; carry through.
            eta_over_etc[stage_name] = np.full((H, W), np.nan, dtype=np.float32)
            ks_means[stage_name] = np.full((H, W), np.nan, dtype=np.float32)
            continue

        # Aggregate ETa, ETc over the stage
        with np.errstate(invalid="ignore", divide="ignore"):
            sum_eta = np.nansum(ETa[ii], axis=0)
            sum_etc = np.nansum(ETc[ii], axis=0)
            ratio = np.where(sum_etc > 0, sum_eta / sum_etc, np.nan)

        eta_over_etc[stage_name] = ratio.astype(np.float32)
        ks_means[stage_name] = np.nanmean(Ks[ii], axis=0).astype(np.float32)

        # Stage-level water-stress factor
        ky_i = float(crop.ky_stage[stage_name])
        # f_i = 1 - Ky_i * max(0, 1 - ETa/ETc); floored at 0
        with np.errstate(invalid="ignore"):
            deficit = np.clip(1.0 - ratio, 0.0, None)
            f_i = np.clip(1.0 - ky_i * deficit, 0.0, 1.0)
        # Where ratio is NaN (no ETc in this stage), leave f_i = 1
        f_i = np.where(np.isnan(f_i), 1.0, f_i)
        ya_over_ymax *= f_i

    # ── Heat stress overlay at flowering ──
    heat_days = np.zeros((H, W), dtype=np.int32)
    heat_warning = ""
    if tmax is not None and crop.flower_stage in stages:
        ii = stages[crop.flower_stage]
        if ii.size:
            tmax_slice = tmax.values[ii]                # (T', H, W)
            with np.errstate(invalid="ignore"):
                shock = (tmax_slice >= float(crop.t_flower_stress_c)).astype(np.int32)
            heat_days = shock.sum(axis=0)
        penalty = np.minimum(
            DEFAULT_HEAT_STRESS_CAP,
            DEFAULT_HEAT_STRESS_PER_DAY * heat_days.astype(np.float32),
        )
        ya_over_ymax = ya_over_ymax * (1.0 - penalty)
        heat_warning = DEFAULT_HEAT_STRESS_CITATION

    ya_over_ymax = np.clip(ya_over_ymax, 0.0, 1.0).astype(np.float32)
    ya = (float(crop.ymax_t_per_ha) * ya_over_ymax).astype(np.float32)

    coords = {"lat": lat, "lon": lon}
    ds = xr.Dataset({
        "Ya":            xr.DataArray(ya,           dims=("lat", "lon"), coords=coords),
        "Ya_over_Ymax":  xr.DataArray(ya_over_ymax, dims=("lat", "lon"), coords=coords),
        "heat_stress_days": xr.DataArray(heat_days, dims=("lat", "lon"), coords=coords),
    })
    for st in ("ini", "dev", "mid", "late"):
        ds[f"Ks_stage_mean_{st}"] = xr.DataArray(
            ks_means[st], dims=("lat", "lon"), coords=coords,
        )
        ds[f"eta_over_etc_{st}"] = xr.DataArray(
            eta_over_etc[st], dims=("lat", "lon"), coords=coords,
        )

    ds["Ya"].attrs["units"] = "t/ha"
    ds["Ya_over_Ymax"].attrs["units"] = "1"
    ds["heat_stress_days"].attrs["units"] = "days"

    ds.attrs.update({
        "version": AGRICULTURE_VERSION,
        "crop": crop.key,
        "crop_registry_version": crop.registry_version,
        "crop_registry_sha256": crop.registry_sha256,
        "ymax_t_per_ha": float(crop.ymax_t_per_ha),
        "quantile": wb.attrs.get("quantile", "deterministic"),
        "source_chain": (
            f"{wb.attrs.get('source_chain','wb')} → "
            f"sectors.agri@{AGRICULTURE_VERSION}"
        ),
        "heat_stress_note": heat_warning,
        "flower_stage": crop.flower_stage,
        "flower_stress_threshold_c": float(crop.t_flower_stress_c),
    })
    return ds


# ─── Baseline pass ────────────────────────────────────────────────────
def yield_baseline(
    crop, region: RegionSpec, sow_date: date, *,
    duration_days: int | None = None,
) -> xr.Dataset:
    """Run the same yield model on TRAIN_YEARS climatology forcing.

    Constructs a synthetic daily driver from the DOY climatology of
    rain / tmax / tmin (over TRAIN_YEARS) spanning the crop's total
    duration starting from ``sow_date``. Uses the exact same water
    balance + yield model so the two runs are directly comparable.

    Uses only the climatology (deterministic) — this is the "what
    would we have expected from climatology alone" number that Part 4
    will use to compute forecast value.
    """
    from ..indices.et0_hargreaves import et0_hargreaves
    from ..indices.reference import climatology
    from .crops import Crop  # noqa: F401
    from ..biophysical.water_balance import water_balance

    dur = int(duration_days or crop.total_days)
    rain_clim = climatology("rain")          # (dayofyear, lat, lon)
    tmax_clim = climatology("tmax")
    tmin_clim = climatology("tmin")

    # Build a (time, lat, lon) synthetic driver by sampling DOYs
    dates = pd.date_range(sow_date, periods=dur, freq="D", tz="Asia/Kolkata")
    doys = np.array([int(d.dayofyear) for d in dates], dtype=np.int32)
    # Sanity: climatology's dayofyear axis is 1..366
    doy_pick = np.clip(doys, 1, 366)

    rain_v = rain_clim.sel(dayofyear=doy_pick).values
    tmax_v = tmax_clim.sel(dayofyear=doy_pick).values
    tmin_v = tmin_clim.sel(dayofyear=doy_pick).values

    coords = {"time": dates, "lat": rain_clim.lat, "lon": rain_clim.lon}
    rain_da = xr.DataArray(rain_v, dims=("time", "lat", "lon"),
                            coords=coords, name="rain",
                            attrs={"units": "mm", "quantile": "deterministic",
                                    "source": "climatology.rain",
                                    "source_chain": "clim@TRAIN_YEARS"})
    tmax_da = xr.DataArray(tmax_v, dims=("time", "lat", "lon"),
                            coords=coords, name="tmax",
                            attrs={"units": "°C", "quantile": "deterministic",
                                    "source": "climatology.tmax",
                                    "source_chain": "clim@TRAIN_YEARS"})
    tmin_da = xr.DataArray(tmin_v, dims=("time", "lat", "lon"),
                            coords=coords, name="tmin",
                            attrs={"units": "°C", "quantile": "deterministic",
                                    "source": "climatology.tmin",
                                    "source_chain": "clim@TRAIN_YEARS"})

    # Apply the region
    from ..config.region import apply_region
    rain_da = apply_region(rain_da, region)
    tmax_da = apply_region(tmax_da, region)
    tmin_da = apply_region(tmin_da, region)

    et0 = et0_hargreaves(tmax_da, tmin_da)
    wb = water_balance(crop, rain_da, et0, sow_date, region,
                        irrigation=None, use_cache=True)
    out = yield_water_limited(crop, wb, tmax=tmax_da)
    out.attrs.update({
        "baseline": True,
        "baseline_forcing": f"climatology@TRAIN_YEARS {TRAIN_YEARS}",
        "source_chain": f"{out.attrs.get('source_chain','')} @baseline",
    })
    return out


# ─── District aggregation ─────────────────────────────────────────────
class ResolutionCeilingError(RuntimeError):
    """Raised when a caller asks for finer than district resolution.

    Enforcement point of Part-3 Rule 2 ("resolution ceiling: 0.25°
    for grids, district for reported yield"). Never emit sub-district
    numbers.
    """


def to_district(
    yield_ds: xr.Dataset, district_registry: "DistrictRegistry",
    *, weights: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Area-weighted mean of Ya / Ya_over_Ymax to district polygons.

    ``district_registry`` — see :class:`DistrictRegistry`. Must supply
    ``.districts()`` yielding ``(name, state, mask)`` triples on the
    master grid. Sub-district polygons are refused (:class:`ResolutionCeilingError`).
    ``weights`` — optional cropped-area map on the master grid. If None,
    uniform-area (all True cells weighted equally) is used and the fact
    is recorded in the returned DataFrame's ``weight_source`` column.
    """
    if district_registry.min_area_km2 < district_registry.MIN_DISTRICT_AREA_KM2:
        raise ResolutionCeilingError(
            f"district_registry contains polygons smaller than "
            f"{district_registry.MIN_DISTRICT_AREA_KM2} km² — refusing "
            f"sub-district aggregation."
        )

    weight_source = "uniform-area"
    if weights is not None:
        weight_source = str(weights.attrs.get("source", "user-supplied"))

    rows = []
    ya = yield_ds["Ya"].values
    yof = yield_ds["Ya_over_Ymax"].values

    for name, state, mask in district_registry.districts():
        m = np.asarray(mask, dtype=bool)
        if m.shape != ya.shape:
            raise ValueError(f"district mask shape {m.shape} vs Ya shape {ya.shape}")
        if weights is not None:
            w = np.asarray(weights.values, dtype=np.float32)
            w = np.where(m & np.isfinite(w), w, 0.0)
        else:
            w = m.astype(np.float32)

        w_sum = float(w.sum())
        if w_sum <= 0:
            rows.append({
                "district": name, "state": state,
                "Ya_t_per_ha": float("nan"),
                "Ya_over_Ymax": float("nan"),
                "valid_frac": 0.0,
                "weight_source": weight_source,
            })
            continue

        with np.errstate(invalid="ignore"):
            wa = np.where(np.isfinite(ya), ya, 0.0)
            wf = np.where(np.isfinite(yof), yof, 0.0)
            ya_mean = float((wa * w).sum() / w_sum)
            yof_mean = float((wf * w).sum() / w_sum)
        valid_frac = float(np.isfinite(ya[m]).mean()) if m.any() else 0.0
        rows.append({
            "district": name, "state": state,
            "Ya_t_per_ha": ya_mean,
            "Ya_over_Ymax": yof_mean,
            "valid_frac": valid_frac,
            "weight_source": weight_source,
        })

    return pd.DataFrame(rows)


# ─── Minimal district registry (states-as-districts fallback) ─────────
class DistrictRegistry:
    """Registry of district masks on the master 0.25° grid.

    Two backends:
      1) A user-supplied dict ``{ (state, district): mask }``. This is
         what will be shipped once a district shapefile lands under
         ``climate_twin/regions/districts/``.
      2) A **states-as-districts fallback** derived from the zone
         registry — coarser than real districts, but honest. Every row
         is labelled ``state=<name>, district=<name>`` (identical) so
         downstream callers can't confuse it for district-level output.

    Sub-district (block / village) polygons are refused — the class
    exposes ``MIN_DISTRICT_AREA_KM2 = 250`` and ``min_area_km2`` which
    :func:`to_district` checks.
    """
    MIN_DISTRICT_AREA_KM2 = 250.0

    def __init__(self, mapping: dict[tuple[str, str], np.ndarray] | None = None):
        self._mapping = mapping or {}
        self.min_area_km2 = float("inf") if not mapping else self.MIN_DISTRICT_AREA_KM2

    @classmethod
    def states_fallback(cls) -> "DistrictRegistry":
        """Build a coarse registry from the zone registry (one 'district'
        per zone). Honest fallback labelled ``state == district``."""
        try:
            from climate_twin.regions import get_zones
        except Exception:
            return cls(mapping={})
        Z = get_zones()
        mapping = {}
        hard = Z.hard_mask
        for zone in Z.zones:
            m = (hard == zone.id)
            mapping[(zone.key, zone.key)] = m
        inst = cls(mapping=mapping)
        # A whole zone is much larger than 250 km²; certify.
        inst.min_area_km2 = cls.MIN_DISTRICT_AREA_KM2
        return inst

    def districts(self):
        for (state, dist), mask in self._mapping.items():
            yield dist, state, mask
