"""
heat.py
========
Heat-stress impact module.

WHAT WE COMPUTE
---------------
1. Heat-stress days   : days with Tmax > 40°C  (dangerous, WHO/IMD threshold)
2. Extreme heat days  : days with Tmax > 45°C  (potentially lethal)
3. Wet-bulb globe temp: physiological heat-stress index
4. Heatwave events    : ≥3 consecutive days with Tmax > threshold

WHY IT MATTERS
--------------
India's heatwaves kill hundreds each year. A digital twin that flags rising
heatwave-day counts under +2°C scenarios directly serves ADAPTATION —
early warnings, cooling-shelter planning, worker safety guidelines.

USED FOR
--------
- The dashboard's "Extremes" panel
- Scenario impact table ("+2°C → 12 more heat-stress days across the basin")
"""
from __future__ import annotations
import numpy as np
import xarray as xr

from mausamsetu import config


# ============================================================================
# DAY COUNTERS
# ============================================================================
def heat_stress_days(
    tmax_da: xr.DataArray,
    threshold: float = config.HEAT_STRESS_TEMP,
) -> xr.DataArray:
    """
    Count days per pixel where Tmax > threshold.

    Returns
    -------
    xr.DataArray with dims (lat, lon), giving counts of hot days over the
    input time range.
    """
    hot = (tmax_da > threshold).astype(np.int32)
    return hot.sum(dim="time")


def extreme_heat_days(
    tmax_da: xr.DataArray,
    threshold: float = config.EXTREME_HEAT_TEMP,
) -> xr.DataArray:
    """Count days per pixel where Tmax > 45°C (extreme)."""
    return heat_stress_days(tmax_da, threshold=threshold)


# ============================================================================
# HEATWAVE EVENTS (≥3 consecutive hot days)
# ============================================================================
def heatwave_events(
    tmax_da: xr.DataArray,
    threshold: float = config.HEAT_STRESS_TEMP,
    min_duration: int = 3,
) -> xr.DataArray:
    """
    Count heatwave EVENTS (a run of ≥ min_duration consecutive hot days).

    Returns
    -------
    xr.DataArray (lat, lon) — number of heatwave events per pixel.
    """
    hot_arr = (tmax_da > threshold).values                     # (T, lat, lon)
    T, H, W = hot_arr.shape
    events = np.zeros((H, W), dtype=np.int32)

    for i in range(H):
        for j in range(W):
            # Count runs of consecutive True values ≥ min_duration
            run_length = 0
            in_event = False
            for t in range(T):
                if hot_arr[t, i, j]:
                    run_length += 1
                    if run_length == min_duration and not in_event:
                        events[i, j] += 1
                        in_event = True
                else:
                    run_length = 0
                    in_event = False

    return xr.DataArray(
        events,
        dims=("lat", "lon"),
        coords={"lat": tmax_da.lat, "lon": tmax_da.lon},
        name="heatwave_events",
        attrs={"threshold_C": threshold, "min_duration_days": min_duration},
    )


# ============================================================================
# WET-BULB GLOBE TEMP (approx heat-stress index)
# ============================================================================
def wet_bulb_globe_temp(
    tmax_da: xr.DataArray,     # °C
    rh_frac: float = 0.6,      # relative humidity fraction (default 60%)
) -> xr.DataArray:
    """
    Simple WBGT approximation (Stull 2011).

    Parameters
    ----------
    tmax_da  : Tmax in °C
    rh_frac  : relative humidity 0-1 (rough scalar; can be a field if available)

    Returns
    -------
    Wet-bulb globe temperature (°C)

    Rough zones (WHO / OSHA):
      < 27  → safe
      27-30 → caution
      30-32 → high risk
      > 32  → extreme (heatwave, no outdoor work)
    """
    T = tmax_da
    rh_pct = rh_frac * 100
    wbt = (
        T * np.arctan(0.151977 * (rh_pct + 8.313659) ** 0.5)
        + np.arctan(T + rh_pct)
        - np.arctan(rh_pct - 1.676331)
        + 0.00391838 * (rh_pct ** 1.5) * np.arctan(0.023101 * rh_pct)
        - 4.686035
    )
    return wbt


# ============================================================================
# SCENARIO IMPACT
# ============================================================================
def compare_heat_impact(
    baseline_tmax: xr.DataArray,
    scenario_tmax: xr.DataArray,
    threshold: float = config.HEAT_STRESS_TEMP,
) -> dict:
    """
    Compare baseline vs scenario heat-stress-day counts.
    """
    b_days = heat_stress_days(baseline_tmax, threshold).sum()
    s_days = heat_stress_days(scenario_tmax, threshold).sum()

    b_ext = extreme_heat_days(baseline_tmax).sum()
    s_ext = extreme_heat_days(scenario_tmax).sum()

    return {
        "baseline_hot_pixel_days":  int(b_days),
        "scenario_hot_pixel_days":  int(s_days),
        "delta_hot_pixel_days":     int(s_days - b_days),
        "baseline_extreme_pixel_days": int(b_ext),
        "scenario_extreme_pixel_days": int(s_ext),
        "delta_extreme_pixel_days":    int(s_ext - b_ext),
    }


# ============================================================================
# CLI DEMO
# ============================================================================
if __name__ == "__main__":
    ds = xr.open_dataset(config.CAUVERY_NC)
    tmax_baseline = ds.tmax.sel(time=slice("2023-03-01", "2023-06-30"))   # pre-monsoon (hot season)
    print("=" * 60)
    print("Heat stress — Cauvery pre-monsoon (Mar-Jun) 2023")
    print("=" * 60)
    hot = heat_stress_days(tmax_baseline).sum().item()
    ext = extreme_heat_days(tmax_baseline).sum().item()
    print(f"Baseline hot-pixel-days   (Tmax > 40°C): {hot}")
    print(f"Baseline extreme-days     (Tmax > 45°C): {ext}")

    # +2°C scenario
    from mausamsetu.storyline.scenario import Scenario, apply_scenario
    baseline_ds = ds[["rain", "tmax", "tmin"]].sel(time=slice("2023-03-01", "2023-06-30"))
    sc = Scenario(delta_temp=+2.0, label="warming-2C")
    scenario_ds = apply_scenario(baseline_ds, sc)

    impact = compare_heat_impact(baseline_ds.tmax, scenario_ds.tmax)
    print(f"\nScenario: {sc.summary()}")
    for k, v in impact.items():
        print(f"  {k}: {v}")

    print(f"\n=> +2°C makes {impact['delta_hot_pixel_days']} more heat-stress pixel-days")
    print(f"=> +2°C makes {impact['delta_extreme_pixel_days']} more EXTREME heat pixel-days")
