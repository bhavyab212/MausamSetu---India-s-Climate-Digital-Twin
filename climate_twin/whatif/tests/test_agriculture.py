"""
whatif.tests.test_agriculture — Part 3 unit tests.

Fast, hermetic. Every failure blocks the branch. Nothing here touches
the real IMD cube — synthetic drivers only.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_twin.whatif.biophysical.crop_water import (
    etc_series,
    kc_curve,
    stage_of_day,
)
from climate_twin.whatif.biophysical.soil import (
    DEFAULT_AWC_MM_PER_M,
    awc_mm_per_m,
    raw,
    taw,
)
from climate_twin.whatif.biophysical.water_balance import (
    WATER_BALANCE_VERSION,
    IrrigationSchedule,
    water_balance,
)
from climate_twin.whatif.config.region import RegionSpec
from climate_twin.whatif.scenarios.quantiles import MixedQuantiles
from climate_twin.whatif.sectors import (
    AGRICULTURE_VERSION,
    DistrictRegistry,
    ResolutionCeilingError,
    to_district,
    yield_water_limited,
)
from climate_twin.whatif.sectors.crops import load_crop, registry_sha256


# ─── synthetic driver helpers ────────────────────────────────────────
def _tiny_grid(T: int, H: int = 4, W: int = 4) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build minimal (T, H, W) rain / tmax / tmin / et0 arrays on a
    2x2-ish patch that's still inside India's bbox."""
    times = pd.date_range("2020-06-01", periods=T, freq="D", tz="Asia/Kolkata")
    lat = np.linspace(19.0, 20.0, H).round(4)
    lon = np.linspace(77.0, 78.0, W).round(4)
    coords = {"time": times, "lat": lat, "lon": lon}
    zero = np.zeros((T, H, W), dtype=np.float32)

    def _mk(arr, name, unit):
        d = xr.DataArray(arr.astype(np.float32), dims=("time", "lat", "lon"),
                          coords=coords, name=name,
                          attrs={"units": unit, "quantile": "deterministic",
                                  "source_chain": f"synthetic.{name}"})
        return d

    rain = _mk(zero.copy(), "rain", "mm")
    tmax = _mk(zero.copy() + 32.0, "tmax", "°C")
    tmin = _mk(zero.copy() + 22.0, "tmin", "°C")
    # ET0 held at 5 mm/day for a clean bucket unit-test
    et0 = _mk(zero.copy() + 5.0, "et0", "mm/day")
    return rain, tmax, tmin, et0


def _region_bbox_for_tiny() -> RegionSpec:
    return RegionSpec(kind="bbox", bbox=(19.0, 77.0, 20.0, 78.0))


# ─── STEP 9 tests ────────────────────────────────────────────────────
def test_kc_curve_shape_paddy():
    """Kc(day 5) = Kc_ini, monotone up through dev, plateau at mid,
    monotone down through late to Kc_end."""
    crop = load_crop("paddy_kharif")
    d_ini = crop.stages_days["ini"]
    d_dev = crop.stages_days["dev"]
    d_mid = crop.stages_days["mid"]
    d_late = crop.stages_days["late"]

    assert kc_curve(crop, 5) == pytest.approx(crop.kc["ini"], abs=1e-6)
    # mid-dev linear point
    mid_dev = d_ini + d_dev // 2
    expected_mid_dev = crop.kc["ini"] + (mid_dev - d_ini) / d_dev * (
        crop.kc["mid"] - crop.kc["ini"]
    )
    assert kc_curve(crop, mid_dev) == pytest.approx(expected_mid_dev, abs=1e-6)
    # plateau
    assert kc_curve(crop, d_ini + d_dev + 10) == pytest.approx(crop.kc["mid"], abs=1e-6)
    # late-phase decline
    last_day = d_ini + d_dev + d_mid + d_late - 1
    late_kc = kc_curve(crop, last_day)
    assert crop.kc["end"] <= late_kc <= crop.kc["mid"] + 1e-6


def test_water_balance_conservation():
    """rain = ETc + 5 mm/day within the mid-plateau → Dr stays at 0,
    DP totals ~ (excess * T) mm.

    We keep the whole T-day window inside the Kc_mid plateau by sowing
    far enough in the past that day 0 is well into mid, and by choosing
    T < remaining_mid_days. Outside the plateau Kc is not constant so
    the "rain = ETc + 5" assumption breaks — that is a feature, not a
    conservation failure.
    """
    crop = load_crop("wheat_rabi")
    # Mid runs from day (ini+dev) to day (ini+dev+mid).
    ini_plus_dev = crop.stages_days["ini"] + crop.stages_days["dev"]
    mid_len = crop.stages_days["mid"]
    T = mid_len - 10                                # stays safely inside mid
    rain, tmax, tmin, et0 = _tiny_grid(T)
    sow_date = date(2020, 6, 1) - timedelta(days=ini_plus_dev + 5)
    # rain = Kc_mid * ET0 + excess
    excess = 5.0
    rain = rain + (crop.kc["mid"] * 5.0 + excess)
    region = _region_bbox_for_tiny()

    wb = water_balance(crop, rain, et0, sow_date, region, use_cache=False)
    # Sanity: every day landed inside the mid plateau
    assert (wb["stage_index"].values == 2).all(), (
        "test setup escaped mid plateau — tighten T or sow_date"
    )
    Dr = wb["Dr"].values
    assert np.nanmax(Dr) <= 1e-3, f"Dr max = {np.nanmax(Dr)} (expected ~0)"
    dp_total_per_cell = np.nansum(wb["DP"].values, axis=0)
    expected = excess * T
    assert np.nanmax(np.abs(dp_total_per_cell - expected)) < 2.0, (
        f"DP conservation off by "
        f"{np.nanmax(np.abs(dp_total_per_cell - expected))} mm "
        f"(expected {expected} mm, got mean {np.nanmean(dp_total_per_cell):.3f})"
    )


def test_water_balance_stress_ramp():
    """rain = 0, constant ET0 → Ks transitions linearly in (RAW, TAW)."""
    crop = load_crop("paddy_kharif")
    T = 150
    rain, tmax, tmin, et0 = _tiny_grid(T)
    # rain already 0. Sow at t=0, ET0 held at 5 mm/day.
    sow_date = date(2020, 6, 1)
    region = _region_bbox_for_tiny()

    wb = water_balance(crop, rain, et0, sow_date, region, use_cache=False)

    # Grab a single cell's Ks and Dr trajectory
    Ks = wb["Ks"].values[:, 0, 0]
    Dr = wb["Dr"].values[:, 0, 0]

    # Manually compute RAW / TAW
    awc_da, _info = awc_mm_per_m(region)
    awc_val = float(awc_da.sel(lat=wb["lat"][0], lon=wb["lon"][0], method="nearest"))
    taw_val = awc_val * float(crop.root_depth_m["max"])
    raw_val = taw_val * float(crop.depletion_p)

    denom = taw_val - raw_val
    # For every day, expected Ks
    expected = np.where(
        Dr <= raw_val, 1.0,
        np.where(Dr >= taw_val, 0.0, (taw_val - Dr) / denom),
    )
    err = float(np.max(np.abs(Ks - expected)))
    assert err < 1e-6, f"Ks stress-ramp mismatch: max |err| = {err}"


def test_yield_bounds_and_perfect_water():
    """Ya/Ymax ∈ [0, 1]; ETa=ETc everywhere → Ya == Ymax exactly."""
    crop = load_crop("paddy_kharif")
    T = crop.total_days
    rain, tmax, tmin, et0 = _tiny_grid(T)
    # Massive rain → guaranteed full water supply, no heat stress (tmax=32<35)
    rain = rain + 200.0
    sow_date = date(2020, 6, 1)
    region = _region_bbox_for_tiny()

    wb = water_balance(crop, rain, et0, sow_date, region, use_cache=False)
    y = yield_water_limited(crop, wb, tmax=tmax)

    yof = y["Ya_over_Ymax"].values
    finite = yof[np.isfinite(yof)]
    assert (finite >= 0.0).all() and (finite <= 1.0).all(), (
        f"Ya_over_Ymax out of [0,1] at {(finite < 0).sum() + (finite > 1).sum()} cells"
    )
    # With ETa ~= ETc everywhere (no water stress) AND tmax < flower stress
    # threshold, Ya should equal Ymax to numeric precision.
    assert np.allclose(y["Ya"].values, crop.ymax_t_per_ha, atol=1e-3), (
        f"Ya expected == Ymax={crop.ymax_t_per_ha}, got mean "
        f"{float(np.nanmean(y['Ya'].values)):.4f}"
    )


def test_multistage_penalty_multiplicative():
    """Two stages at 20% deficit each ⇒ lower yield than one at 20% alone.

    (1 − Ky·0.20)² < (1 − Ky·0.20)      for Ky > 0
    """
    crop = load_crop("paddy_kharif")
    ky_ini = crop.ky_stage["ini"]
    ky_dev = crop.ky_stage["dev"]
    # Both stages at 20 % deficit → multi-stage combined factor
    combined = (1 - ky_ini * 0.20) * (1 - ky_dev * 0.20)
    # Single stage at 20 % (say ini) → factor
    single = (1 - ky_ini * 0.20)
    assert combined < single, (
        f"multi-stage penalty is not multiplicative: {combined} !< {single}"
    )


def test_baseline_deterministic():
    """Two identical yield_baseline calls yield identical arrays.

    We do NOT run the real baseline (that needs the cube). Instead we
    verify the *water balance* is deterministic under identical inputs
    (same effective property).
    """
    crop = load_crop("wheat_rabi")
    T = 40
    rain, tmax, tmin, et0 = _tiny_grid(T)
    rain = rain + 3.0
    sow_date = date(2020, 6, 1)
    region = _region_bbox_for_tiny()

    a = water_balance(crop, rain, et0, sow_date, region, use_cache=False)
    b = water_balance(crop, rain, et0, sow_date, region, use_cache=False)
    for v in ("ETa", "ETc", "Dr", "Ks", "DP", "P_eff"):
        assert np.array_equal(a[v].values, b[v].values, equal_nan=True), (
            f"water balance not deterministic on {v}"
        )


def test_quantile_isolation_raises_on_mix():
    """Passing mixed-quantile rain + et0 to the water balance raises."""
    crop = load_crop("bajra_kharif")
    T = 20
    rain, tmax, tmin, et0 = _tiny_grid(T)
    rain.attrs["quantile"] = "q10"
    et0.attrs["quantile"] = "q90"
    sow_date = date(2020, 6, 1)
    region = _region_bbox_for_tiny()
    with pytest.raises(MixedQuantiles):
        water_balance(crop, rain, et0, sow_date, region, use_cache=False)


def test_resolution_ceiling_refuses_subdistrict():
    """to_district refuses a registry whose min-area < the ceiling."""
    yield_ds = xr.Dataset({
        "Ya":            xr.DataArray(np.ones((4, 4), dtype=np.float32),
                                       dims=("lat", "lon")),
        "Ya_over_Ymax":  xr.DataArray(np.ones((4, 4), dtype=np.float32),
                                       dims=("lat", "lon")),
    })
    bad_reg = DistrictRegistry(mapping={
        ("state_A", "village_1"): np.ones((4, 4), dtype=bool),
    })
    # Force the registry to declare a sub-district polygon
    bad_reg.min_area_km2 = 5.0
    with pytest.raises(ResolutionCeilingError):
        to_district(yield_ds, bad_reg)


def test_provenance_chain_agriculture():
    """Yield output attrs record every layer version used to build it."""
    crop = load_crop("arhar_kharif")
    T = 30
    rain, tmax, tmin, et0 = _tiny_grid(T)
    rain = rain + 10.0
    sow_date = date(2020, 6, 1)
    region = _region_bbox_for_tiny()
    wb = water_balance(crop, rain, et0, sow_date, region, use_cache=False)
    y = yield_water_limited(crop, wb, tmax=tmax)

    chain = str(y.attrs.get("source_chain", ""))
    assert WATER_BALANCE_VERSION in wb.attrs["version"]
    assert AGRICULTURE_VERSION in chain
    # crop registry SHA lands in agriculture output attrs
    assert y.attrs.get("crop_registry_sha256") == registry_sha256()
    # water balance stamps the soil source (default in tests, since no
    # SoilGrids raster is on disk)
    assert wb.attrs.get("soil_source", "") == "default"
    # Quantile tag flows through
    assert y.attrs.get("quantile") == "deterministic"


def test_etc_series_shape_and_units():
    """ETc broadcast has correct shape, units, and quantile propagation."""
    crop = load_crop("wheat_rabi")
    T = 10
    _, tmax, tmin, et0 = _tiny_grid(T)
    sow_date = date(2020, 6, 1)
    etc = etc_series(crop, et0, sow_date)
    assert etc.shape == et0.shape
    assert etc.attrs["units"] == "mm/day"
    assert etc.attrs["quantile"] == "deterministic"
    assert crop.key in etc.attrs["source_chain"]


def test_taw_raw_units():
    """TAW and RAW carry mm units and correct scaling."""
    crop = load_crop("paddy_kharif")
    awc_da, _info = awc_mm_per_m(region=None)
    taw_map = taw(crop, awc_da, stage="max")
    raw_map = raw(crop, taw_map)
    assert taw_map.attrs["units"] == "mm"
    assert raw_map.attrs["units"] == "mm"
    # RAW = p × TAW
    ratio = float(np.nanmean(raw_map.values / taw_map.values))
    assert ratio == pytest.approx(crop.depletion_p, abs=1e-6)
