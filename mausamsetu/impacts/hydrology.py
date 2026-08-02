"""
hydrology.py
=============
Rainfall → Runoff → Reservoir Inflow (SCS-CN method).

WHY SCS-CN?
-----------
The USDA Soil Conservation Service Curve Number method is the SIMPLEST
credible way to estimate how much rainfall becomes runoff. It captures
the key non-linear behaviour: dry soil absorbs more; wet/urban land runs off more.

FORMULA
-------
  Given rainfall P (mm) and Curve Number CN (0-100):
      S  = (25400 / CN) - 254            # potential soil retention (mm)
      Ia = 0.2 · S                       # initial abstraction (soaks in first)
      if P ≤ Ia:  Q = 0                  # all rain absorbed
      else:       Q = (P - Ia)² / (P - Ia + S)   # runoff

  Higher CN → more runoff (impervious/urban surfaces).
  Lower  CN → more infiltration (forested/sandy soils).

For a whole basin, apply to each pixel with land-use-specific CN, then sum.

WHY IT MATTERS FOR MAUSAMSETU
-----------------------------
When the user runs "what-if rainfall drops 20%", we can quote NOT just
"20% less rain" but "37% less basin inflow" (non-linear response) — the
kind of actionable insight water managers actually need.
"""
from __future__ import annotations
import numpy as np
import xarray as xr

from mausamsetu import config


# ============================================================================
# SCS-CN RUNOFF
# ============================================================================
def scs_cn_runoff(rainfall_mm: np.ndarray, cn: float | np.ndarray = None) -> np.ndarray:
    """
    Compute runoff (mm) from daily rainfall (mm) using SCS-CN.

    Parameters
    ----------
    rainfall_mm : ndarray of daily rainfall (any shape)
    cn : Curve Number, either scalar or same shape as rainfall
         Default = basin mean CN from config

    Returns
    -------
    Runoff Q (same shape as rainfall, in mm)
    """
    P = np.asarray(rainfall_mm, dtype=np.float32)
    cn = np.asarray(cn if cn is not None else config.BASIN_MEAN_CN, dtype=np.float32)

    S = (25400.0 / cn) - 254.0                     # (mm)
    Ia = 0.2 * S                                    # (mm)

    Q = np.where(
        P > Ia,
        (P - Ia) ** 2 / (P - Ia + S + 1e-12),
        0.0,
    )
    return Q.astype(np.float32)


# ============================================================================
# BASIN INTEGRATION (spatial sum weighted by pixel area)
# ============================================================================
def basin_daily_inflow(
    rainfall_da: xr.DataArray,   # (time, lat, lon) in mm/day
    cn: float | np.ndarray = None,
    area_km2: float = config.BASIN_AREA_KM2,
) -> xr.DataArray:
    """
    Convert daily-per-pixel rainfall to daily basin inflow (m³).

    Steps:
      1. Compute runoff per pixel using SCS-CN
      2. Average runoff over basin (mm)
      3. Multiply by basin area to get total volume:
             volume_m3 = mean_runoff_mm * 1e-3 (→ m) * area_km2 * 1e6 (→ m²)
    """
    Q = xr.apply_ufunc(
        scs_cn_runoff, rainfall_da, kwargs={"cn": cn}, dask="parallelized"
    )
    mean_runoff_mm = Q.mean(dim=["lat", "lon"], skipna=True)     # (time,)
    volume_m3 = mean_runoff_mm * 1e-3 * area_km2 * 1e6           # (time,)
    volume_m3.name = "basin_inflow_m3"
    volume_m3.attrs = {"units": "m³/day", "long_name": "Cauvery basin inflow"}
    return volume_m3


# ============================================================================
# SCENARIO IMPACT
# ============================================================================
def compare_scenario_impact(
    baseline_rain: xr.DataArray,
    scenario_rain: xr.DataArray,
    cn: float | None = None,
) -> dict:
    """
    Compare baseline vs scenario basin inflow.

    Returns
    -------
    dict with:
      baseline_total_m3, scenario_total_m3, delta_m3, delta_pct
    """
    b = basin_daily_inflow(baseline_rain, cn=cn)
    s = basin_daily_inflow(scenario_rain, cn=cn)

    b_total = float(b.sum())
    s_total = float(s.sum())
    delta = s_total - b_total
    pct = 100 * delta / (b_total + 1e-9)
    return {
        "baseline_total_m3": b_total,
        "scenario_total_m3": s_total,
        "delta_m3": delta,
        "delta_pct": pct,
    }


# ============================================================================
# CLI DEMO
# ============================================================================
if __name__ == "__main__":
    ds = xr.open_dataset(config.CAUVERY_NC)
    # JJAS 2023 baseline rainfall
    rain_baseline = ds.rain.sel(time=slice("2023-06-01", "2023-09-30"))

    print("=" * 60)
    print("Hydrology (SCS-CN) sanity check — Cauvery JJAS 2023")
    print("=" * 60)

    inflow = basin_daily_inflow(rain_baseline)
    print(f"Basin daily inflow shape: {inflow.shape}")
    print(f"Total JJAS inflow (baseline): {float(inflow.sum()):.2e} m³")
    print(f"Mean daily inflow:            {float(inflow.mean()):.2e} m³/day")
    print(f"Peak daily inflow:            {float(inflow.max()):.2e} m³/day")

    # Scenario: −20% rainfall
    from mausamsetu.storyline.scenario import Scenario, apply_scenario
    baseline_ds = ds[["rain", "tmax", "tmin"]].sel(time=slice("2023-06-01", "2023-09-30"))
    sc = Scenario(delta_rain_pct=-20, label="drought-20")
    scenario_ds = apply_scenario(baseline_ds, sc)

    impact = compare_scenario_impact(baseline_ds.rain, scenario_ds.rain)
    print(f"\nScenario: {sc.summary()}")
    print(f"  Baseline inflow: {impact['baseline_total_m3']:.2e} m³")
    print(f"  Scenario inflow: {impact['scenario_total_m3']:.2e} m³")
    print(f"  Delta: {impact['delta_m3']:.2e} m³  ({impact['delta_pct']:+.1f}%)")
    print()
    print(f"NOTE: rain dropped 20%, but inflow dropped {abs(impact['delta_pct']):.1f}% — non-linear!")
