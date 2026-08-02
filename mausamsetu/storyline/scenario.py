"""
scenario.py
============
What-If / Storyline engine — the counterfactual simulation module.

WHAT THIS IS (and why it's not a toy slider)
--------------------------------------------
Professional climate science uses "storyline analysis" (Shepherd et al.):
instead of asking "probability of X in 2050", ask "if the monsoon shifts
+2°C warmer AND rainfall drops 15%, what happens to the Cauvery basin?"

Our engine builds physically self-consistent perturbed climates:

  1. TEMPERATURE STORYLINES
     - Uniform ΔT applied to Tmax and Tmin
     - Sourced from IPCC AR6 Table SPM.1 for South Asia:
         * SSP1-2.6 (best case)   → +1.5°C
         * SSP2-4.5 (mid path)    → +2.0°C
         * SSP5-8.5 (worst case)  → +4.4°C

  2. RAINFALL STORYLINES
     - Percent multiplier on daily rainfall (± % globally, or seasonal)
     - IPCC values for the Indian summer monsoon rainfall

  3. COMBINED "PGW" (Pseudo-Global-Warming)
     - Add both deltas simultaneously to a historical period
     - Re-run forecasts + impact chain under the perturbed state

This is what the ESA/ECMWF Climate Digital Twin does. We do the same,
scaled to the Cauvery basin.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import xarray as xr

from mausamsetu import config


# ============================================================================
# SCENARIO DEFINITION
# ============================================================================
@dataclass
class Scenario:
    """
    Defines a perturbation to apply to a baseline forecast.

    delta_temp        : °C added to tmax and tmin uniformly (or seasonally)
    delta_rain_pct    : percent multiplier on rainfall (-40 = drop 40%)
    seasonal_months   : which months to apply changes to (None = all year)
    label             : human-readable name (e.g. "SSP2-4.5")
    """
    delta_temp:      float = 0.0
    delta_rain_pct:  float = 0.0
    seasonal_months: list[int] | None = None
    label:           str = "custom"

    def __post_init__(self):
        # Clip to safe ranges
        self.delta_temp = float(np.clip(self.delta_temp, -5, 8))
        self.delta_rain_pct = float(np.clip(self.delta_rain_pct, -80, 80))

    def summary(self) -> str:
        parts = [f"ΔT={self.delta_temp:+.1f}°C", f"Δrain={self.delta_rain_pct:+.0f}%"]
        if self.seasonal_months:
            parts.append(f"months={self.seasonal_months}")
        return f"[{self.label}: {' | '.join(parts)}]"


# ============================================================================
# IPCC AR6 REFERENCE SCENARIOS (from config, exposed as convenience)
# ============================================================================
IPCC_SCENARIOS: dict[str, Scenario] = {
    label: Scenario(delta_temp=v["temp"], delta_rain_pct=v["rain_pct"], label=label)
    for label, v in config.IPCC_DELTAS.items()
}


# ============================================================================
# APPLY SCENARIO TO A FORECAST
# ============================================================================
def apply_scenario(
    baseline_ds: xr.Dataset,        # variables: rain, tmax, tmin (any coordinates)
    scenario: Scenario,
) -> xr.Dataset:
    """
    Apply the storyline perturbation to a baseline forecast dataset.

    Rules:
    - Temperature deltas are ADDITIVE (°C)
    - Rainfall deltas are MULTIPLICATIVE (% shift)
    - Optional seasonal mask restricts changes to specific months
    - Physical clamps: rain ≥ 0, plausible temp range
    """
    ds = baseline_ds.copy()

    # ---- Build a time mask if seasonal ----
    if scenario.seasonal_months and "time" in ds.dims:
        months = ds.time.dt.month
        mask = xr.DataArray(np.isin(months, scenario.seasonal_months), dims=("time",))
    else:
        mask = None

    # ---- Apply ΔT to tmax and tmin ----
    if scenario.delta_temp != 0:
        if mask is not None:
            ds["tmax"] = ds["tmax"] + scenario.delta_temp * mask
            ds["tmin"] = ds["tmin"] + scenario.delta_temp * mask
        else:
            ds["tmax"] = ds["tmax"] + scenario.delta_temp
            ds["tmin"] = ds["tmin"] + scenario.delta_temp

    # ---- Apply Δrain% multiplicatively ----
    if scenario.delta_rain_pct != 0:
        factor = 1.0 + scenario.delta_rain_pct / 100.0
        if mask is not None:
            # Only multiply on masked days; leave others unchanged
            ds["rain"] = ds["rain"].where(~mask, ds["rain"] * factor)
        else:
            ds["rain"] = ds["rain"] * factor

    # ---- Physical clamps ----
    ds["rain"] = ds["rain"].clip(min=0, max=500)
    ds["tmax"] = ds["tmax"].clip(min=-5, max=55)
    ds["tmin"] = ds["tmin"].clip(min=-15, max=45)

    # Ensure Tmin ≤ Tmax (physics)
    ds["tmin"] = xr.where(ds["tmin"] > ds["tmax"], ds["tmax"] - 0.5, ds["tmin"])

    # Metadata
    ds.attrs["scenario"] = scenario.summary()
    return ds


# ============================================================================
# COMPUTE DELTAS (baseline vs scenario) — for the dashboard
# ============================================================================
def compute_delta(
    baseline: xr.Dataset,
    scenario_ds: xr.Dataset,
    variable: str = "rain",
) -> xr.DataArray:
    """
    Compute the DIFFERENCE (scenario − baseline) for a variable.
    This is what powers the "delta map" panel in the dashboard.
    """
    return scenario_ds[variable] - baseline[variable]


# ============================================================================
# CLI DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Storyline Engine — Sanity Check")
    print("=" * 60)

    # Load the processed dataset
    ds = xr.open_dataset(config.CAUVERY_NC)
    baseline = ds[["rain", "tmax", "tmin"]].sel(time=slice("2023-06-01", "2023-09-30"))
    print(f"Baseline: {len(baseline.time)} days (JJAS 2023)")
    print(f"  rain mean: {float(baseline.rain.mean()):.2f} mm/day")
    print(f"  tmax mean: {float(baseline.tmax.mean()):.2f} °C")

    print("\n--- IPCC AR6 Scenarios ---")
    for label, sc in IPCC_SCENARIOS.items():
        scenario_ds = apply_scenario(baseline, sc)
        drain = float(scenario_ds.rain.mean()) - float(baseline.rain.mean())
        dtmax = float(scenario_ds.tmax.mean()) - float(baseline.tmax.mean())
        print(f"\n{label}:")
        print(f"  applied: {sc.summary()}")
        print(f"  Δrain: {drain:+.2f} mm/day  |  ΔTmax: {dtmax:+.2f} °C")

    # Custom scenario
    print("\n--- Custom scenario: −20% rainfall in monsoon only ---")
    custom = Scenario(delta_rain_pct=-20, seasonal_months=[6, 7, 8, 9], label="monsoon-drop-20")
    scenario_ds = apply_scenario(baseline, custom)
    print(f"  JJAS baseline rain: {float(baseline.rain.mean()):.2f} mm/day")
    print(f"  JJAS scenario rain: {float(scenario_ds.rain.mean()):.2f} mm/day")
    print(f"  Change:            {(float(scenario_ds.rain.mean())/float(baseline.rain.mean())-1)*100:+.1f}%")
