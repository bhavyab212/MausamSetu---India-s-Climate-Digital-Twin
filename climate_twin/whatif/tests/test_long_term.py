"""
whatif.tests.test_long_term — Part 7 unit tests.

Fast, hermetic, deterministic. Uses small synthetic multi-model stacks
in place of the real NEX-GDDP reads (which need on-disk data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_twin.whatif.config.region import RegionSpec
from climate_twin.whatif.drivers.downscale import (
    DELTA_MEAN_VERSION,
    QDM_VERSION,
    delta_downscale_mean,
    downscale,
    qdm_downscale,
)
from climate_twin.whatif.drivers.ensemble_lt import (
    ENSEMBLE_LT_VERSION,
    decompose_uncertainty,
)
from climate_twin.whatif.drivers.nex_gddp import (
    NEX_GDDP_VERSION,
    NEXGDDPSpec,
    has_coverage,
    list_models,
    validate_spec,
    _to_engine_units,
)
from climate_twin.whatif.drivers.ssp import (
    baseline_period,
    default_display_scenarios,
    list_scenarios,
    load_scenario,
    registry_sha256,
    registry_version,
    window_for_center,
)
from climate_twin.whatif.economics.npv import (
    DEFAULT_DISCOUNT_RATES,
    adaptation_npv,
)
from climate_twin.whatif.indices.long_term import (
    LT_INDICES_VERSION,
    return_period_shift,
    time_of_emergence,
    window_climatology,
)
from climate_twin.whatif.sectors import (
    LongTermResult,
    RepresentationMismatch,
    run_long_term_scenario,
)
from climate_twin.whatif.sectors.adaptations import (
    MissingCostCitation,
    list_adaptations,
    load_adaptation,
    registry_sha256 as adapt_sha,
    registry_version as adapt_version,
)


# ─── STEP 10.1 — Catalog integrity ────────────────────────────────────
def test_catalog_lists_ten_gcm_ensemble():
    """The catalog ships with the ten-GCM ensemble from Almazroui 2020."""
    models = list_models()
    assert len(models) == 10, f"expected 10 GCMs, got {len(models)}"
    for expected in ("ACCESS-ESM1-5", "GFDL-ESM4", "MPI-ESM1-2-HR"):
        assert expected in models, f"missing {expected}"


def test_missing_gcm_ssp_combination_raises():
    """Out-of-catalog spec raises at validate_spec, not at read."""
    spec_bad_model = NEXGDDPSpec(
        variable="pr", model="NOT-A-MODEL-XYZ", ssp="ssp245",
        period="future", year_range=(2041, 2060),
    )
    with pytest.raises(ValueError):
        validate_spec(spec_bad_model)


def test_catalog_has_coverage_query():
    """has_coverage returns True for a valid (model, ssp, period, range)."""
    assert has_coverage("GFDL-ESM4", "ssp245", "future", (2041, 2060))
    assert has_coverage("GFDL-ESM4", "ssp245", "historical", (1971, 2000))
    assert not has_coverage("GFDL-ESM4", "ssp245", "future", (1971, 2000))


# ─── STEP 10.2 — Unit conversion ──────────────────────────────────────
def test_pr_kg_m2_s_to_mm_day():
    """pr in kg m-2 s-1 → mm/day is multiplication by 86400."""
    times = pd.date_range("2020-01-01", periods=3)
    da = xr.DataArray(np.ones((3, 2, 2)), dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": [10.0, 11.0],
                              "lon": [70.0, 71.0]})
    da.attrs["units"] = "kg m-2 s-1"
    out = _to_engine_units(da, "pr")
    assert np.allclose(out.values, 86400.0)
    assert out.attrs["units"] == "mm/day"


def test_tasmax_K_to_C():
    """tasmax in K → °C is subtraction of 273.15."""
    times = pd.date_range("2020-01-01", periods=3)
    da = xr.DataArray(np.full((3, 2, 2), 300.15),
                      dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": [10.0, 11.0],
                              "lon": [70.0, 71.0]})
    out = _to_engine_units(da, "tasmax")
    assert np.allclose(out.values, 27.0, atol=1e-6)
    assert out.attrs["units"] == "°C"


# ─── STEP 10.3 — QDM identity + extremes ──────────────────────────────
def _synth_daily(seed: int, T: int = 3650, H: int = 2, W: int = 2,
                  loc: float = 3.0, scale: float = 2.0) -> xr.DataArray:
    """T-day gamma-ish precipitation series on a small grid."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("1971-01-01", periods=T)
    vals = rng.gamma(shape=1.5, scale=scale, size=(T, H, W)) + loc * 0.1
    return xr.DataArray(vals, dims=("time", "lat", "lon"),
                        coords={"time": times,
                                "lat": np.linspace(19.0, 20.0, H),
                                "lon": np.linspace(77.0, 78.0, W)})


def test_qdm_identity_when_baseline_equals_future():
    """When gcm_future distribution equals gcm_baseline, QDM returns
    the observed baseline unchanged (up to numerical noise)."""
    obs = _synth_daily(seed=1, T=365)
    gcm_base = _synth_daily(seed=2, T=365)
    # gcm_future is an *independent draw* of the SAME distribution
    gcm_fut = _synth_daily(seed=2, T=365)                 # same seed → same
    out = qdm_downscale(obs, gcm_base, gcm_fut,
                          variable="pr", n_quantiles=100)
    # Since gcm_fut == gcm_base, change ratio is 1 → obs is unchanged
    # (up to the quantile-lookup rounding). Check the medians agree.
    diff = np.abs(np.nanmedian(out.values, axis=0)
                    - np.nanmedian(obs.values, axis=0))
    assert np.nanmax(diff) < 0.5, (
        f"QDM identity broke: max median diff = {np.nanmax(diff):.4f}"
    )


def test_qdm_extreme_inflation_captured():
    """A GCM future with a 50 % inflation of its top decile leaves the
    observed baseline's top decile inflated and its median roughly
    unchanged."""
    obs = _synth_daily(seed=3, T=730)
    gcm_base = _synth_daily(seed=4, T=730)
    # gcm_future = gcm_base but with top decile scaled by 1.5
    fut_vals = gcm_base.values.copy()
    for i in range(fut_vals.shape[1]):
        for j in range(fut_vals.shape[2]):
            col = fut_vals[:, i, j]
            thr = np.quantile(col, 0.90)
            col[col > thr] *= 1.5
            fut_vals[:, i, j] = col
    gcm_fut = xr.DataArray(fut_vals, dims=gcm_base.dims,
                             coords=gcm_base.coords)
    out = qdm_downscale(obs, gcm_base, gcm_fut,
                          variable="pr", n_quantiles=100)
    # Top decile of the downscaled output should exceed top decile of obs
    top_obs = np.quantile(obs.values, 0.90)
    top_out = np.quantile(out.values, 0.90)
    assert top_out > top_obs * 1.05, (
        f"QDM did not inflate the top decile: obs p90={top_obs:.3f}, "
        f"out p90={top_out:.3f}"
    )
    # Median should stay roughly the same
    med_obs = np.median(obs.values)
    med_out = np.median(out.values)
    assert abs(med_out - med_obs) / max(med_obs, 1e-3) < 0.25


def test_delta_mean_vs_qdm_on_mean_shifted_synthetic():
    """On a mean-shifted synthetic, delta_mean and QDM agree at the mean
    but QDM alone captures the tail. Delta-mean applied to precipitation
    is multiplicative — a mean-shift × k appears as a factor-k on obs."""
    obs = _synth_daily(seed=5, T=730)
    gcm_base = _synth_daily(seed=6, T=730)
    # gcm_future = gcm_base * 1.2 uniformly
    gcm_fut = xr.DataArray(gcm_base.values * 1.2, dims=gcm_base.dims,
                             coords=gcm_base.coords)
    out_delta = delta_downscale_mean(obs, gcm_base, gcm_fut, variable="pr")
    out_qdm = qdm_downscale(obs, gcm_base, gcm_fut, variable="pr")
    # Means should agree
    m_delta = float(np.nanmean(out_delta.values))
    m_qdm = float(np.nanmean(out_qdm.values))
    assert abs(m_delta - m_qdm) / max(m_delta, 1e-3) < 0.15


# ─── STEP 10.4 — Hawkins-Sutton bounds ───────────────────────────────
def _synth_multi_model_traj(seed: int, ssp_slope: float,
                              n_models: int = 3, n_years: int = 30,
                              H: int = 2, W: int = 2) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    years = np.arange(2015, 2015 + n_years)
    times = pd.date_range(f"{years[0]}-01-01", periods=n_years, freq="YS")
    x = (years - years[0]) / (years[-1] - years[0] + 1)
    per_model_slopes = ssp_slope + rng.normal(0.0, 0.2, size=n_models)
    stack = np.empty((n_models, n_years, H, W))
    for k in range(n_models):
        base = per_model_slopes[k] * x
        noise = rng.normal(0.0, 0.15, size=(n_years, H, W))
        stack[k] = base[:, None, None] + noise
    return xr.DataArray(stack, dims=("model", "time", "lat", "lon"),
                        coords={"model": [f"M{k}" for k in range(n_models)],
                                "time": times,
                                "lat": np.linspace(19.0, 20.0, H),
                                "lon": np.linspace(77.0, 78.0, W)})


def test_hawkins_sutton_fractions_sum_to_one():
    """Fractions ∈ [0, 1] and sum to 1 within 1e-6 at every timestep."""
    ens = {
        "ssp126": _synth_multi_model_traj(seed=10, ssp_slope=1.0),
        "ssp245": _synth_multi_model_traj(seed=20, ssp_slope=2.0),
        "ssp585": _synth_multi_model_traj(seed=30, ssp_slope=4.0),
    }
    dec = decompose_uncertainty(ens, smoothing_yrs=5)
    fr = dec.fractions
    total = fr["scenario"] + fr["model"] + fr["internal"]
    assert float(np.nanmin(total.values)) > 1.0 - 1e-5
    assert float(np.nanmax(total.values)) < 1.0 + 1e-5
    for k in ("scenario", "model", "internal"):
        arr = fr[k].values
        assert (arr >= -1e-9).all(), f"{k} has negative values"
        assert (arr <= 1.0 + 1e-9).all(), f"{k} exceeds 1"


# ─── STEP 10.5 — ToE monotonicity ─────────────────────────────────────
def test_time_of_emergence_monotonic_in_signal_strength():
    """Two synthetic trajectories with monotone-increasing signals but
    different slopes: the steeper one emerges earlier."""
    n_years = 86
    years = np.arange(2015, 2015 + n_years)
    times = pd.date_range(f"{years[0]}-01-01", periods=n_years, freq="YS")
    H = W = 2

    def _traj(slope: float) -> xr.DataArray:
        vals = np.zeros((1, n_years, H, W))
        for t in range(n_years):
            vals[0, t] = slope * t
        return xr.DataArray(vals, dims=("model", "time", "lat", "lon"),
                            coords={"model": ["M0"], "time": times,
                                    "lat": np.linspace(19.0, 20.0, H),
                                    "lon": np.linspace(77.0, 78.0, W)})

    obs_times = pd.date_range("1971-01-01", periods=30, freq="YS")
    obs = xr.DataArray(
        np.random.default_rng(0).normal(0, 0.5, size=(30, H, W)),
        dims=("time", "lat", "lon"),
        coords={"time": obs_times,
                "lat": np.linspace(19.0, 20.0, H),
                "lon": np.linspace(77.0, 78.0, W)},
    )
    toe_slow = time_of_emergence(_traj(slope=0.1), obs, threshold_sigma=2.0,
                                    window_yrs=5)
    toe_fast = time_of_emergence(_traj(slope=1.0), obs, threshold_sigma=2.0,
                                    window_yrs=5)
    slow_year = float(np.nanmedian(toe_slow.values))
    fast_year = float(np.nanmedian(toe_fast.values))
    assert fast_year < slow_year, (
        f"faster-signal ToE ({fast_year}) not earlier than "
        f"slower-signal ToE ({slow_year})"
    )


# ─── STEP 10.6 — Adaptation NPV determinism + cited-cost gate ─────────
def test_adaptation_npv_deterministic():
    """Two calls with the same inputs → byte-identical NPV DataFrames."""
    deltas = {"heat_tolerant_wheat_variant": 4200.0,
              "supplemental_irrigation_20mm_weekly": 6100.0}
    df1 = adaptation_npv(deltas)
    df2 = adaptation_npv(deltas)
    pd.testing.assert_frame_equal(df1, df2, check_like=True)


def test_adaptation_npv_all_three_discount_rates_reported():
    """Every row has NPV_07, NPV_08, NPV_12 columns (Rule 7)."""
    deltas = {"heat_tolerant_wheat_variant": 4200.0}
    df = adaptation_npv(deltas)
    for col in ("NPV_07", "NPV_08", "NPV_12", "BCR_08"):
        assert col in df.columns, f"missing {col}"


def test_adaptation_npv_gates_on_cited_costs():
    """If an option lacks cited costs, adaptation_npv raises
    MissingCostCitation. Today every shipped option is cited; we
    inject a synthetic uncited option to trip the gate."""
    # Monkey-patch NPV's reference to load_adaptation — npv.py imported
    # the symbol at module load, so patching sectors.adaptations here
    # would miss npv's own reference.
    from climate_twin.whatif.economics import npv as _npv
    from climate_twin.whatif.sectors import adaptations as _ad
    original = _npv.load_adaptation

    def _uncited_stub(key):
        if key == "_synthetic_uncited":
            return _ad.AdaptationOption(
                key=key, common_name="synthetic uncited",
                sector="agriculture",
                applies_to_crops=("paddy_kharif",),
                biophysical_effect="none",
                capex_inr_per_ha=1000.0, capex_inr_per_kwh=0.0,
                opex_inr_per_ha_per_yr=0.0,
                opex_pct_of_capex_per_yr=0.0,
                effective_years=5, co2_kg_per_ha_per_yr=0.0,
                side_effects="", capex_citation="",   # ← uncited
                opex_citation="",
                cost_complete=False,
                registry_version=_ad.registry_version(),
                registry_sha256=_ad.registry_sha256(),
            )
        return original(key)

    _npv.load_adaptation = _uncited_stub
    try:
        with pytest.raises(MissingCostCitation):
            adaptation_npv({"_synthetic_uncited": 5000.0})
    finally:
        _npv.load_adaptation = original


def test_adaptations_yaml_ships_four_cited_options():
    """The registry ships with the four options named in the spec."""
    for k in (
        "supplemental_irrigation_20mm_weekly",
        "heat_tolerant_wheat_variant",
        "raised_bunds_flood_protection",
        "storage_evening_solar_shift",
    ):
        opt = load_adaptation(k)
        assert opt.cost_complete, f"{k} shipped without a cited cost"


# ─── STEP 10.7 — Verb lint over long_term.py ──────────────────────────
def test_long_term_copy_no_will_or_is_going_to():
    """The Part-6 STEP-15 test already banned 'predict' and 'forecast';
    Part 7 extends the ban to 'will' and 'is going to'."""
    import inspect
    import re
    from climate_twin.whatif.ui.copy import long_term as _copy
    patterns = [r"\bwill\b", r"\bis going to\b",
                r"\bpredict", r"\bforecast"]
    for name in dir(_copy):
        if name.startswith("_"):
            continue
        obj = getattr(_copy, name)
        strings = ([obj] if isinstance(obj, str)
                    else list(obj) if isinstance(obj, tuple) else [])
        for s in strings:
            if not isinstance(s, str):
                continue
            for pat in patterns:
                assert not re.search(pat, s, flags=re.IGNORECASE), (
                    f"long_term copy string {name!r} matches banned "
                    f"pattern {pat!r}: {s!r}"
                )


# ─── STEP 10.8 — SSP registry ─────────────────────────────────────────
def test_ssp_registry_versioning():
    v = registry_version()
    s = registry_sha256()
    assert v.startswith("ssp-registry-")
    assert len(s) >= 8


def test_ssp_registry_ships_four_scenarios():
    """Every scenario in the spec is in the registry."""
    for sid in ("ssp126", "ssp245", "ssp370", "ssp585"):
        sc = load_scenario(sid)
        assert sc.narrative and sc.citation
    # Default display uses three
    assert set(default_display_scenarios()) <= set(list_scenarios())
    assert "ssp370" not in default_display_scenarios()


def test_baseline_period_is_1971_2000():
    """Rule 4: baseline is fixed at 1971-2000 unless explicitly overridden."""
    bp = baseline_period()
    assert bp == (1971, 2000)


def test_window_for_center_arithmetic():
    """2030 → 2021-2040; 2050 → 2041-2060; 2075 → 2066-2085."""
    assert window_for_center(2030) == (2021, 2040)
    assert window_for_center(2050) == (2041, 2060)
    assert window_for_center(2075) == (2066, 2085)


# ─── STEP 10.9 — Multi-model preservation + representation mismatch ──
def test_run_long_term_scenario_returns_result_with_provenance():
    """run_long_term_scenario returns a LongTermResult with the SSP
    label + window + adaptation verdicts in provenance."""
    region = RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0),
                        id="vidarbha")
    res = run_long_term_scenario(
        "ssp245", 2050, "agriculture", region,
        adaptations=["heat_tolerant_wheat_variant"],
    )
    assert isinstance(res, LongTermResult)
    p = res.provenance
    assert p["scenario_id"] == "ssp245"
    assert p["target_center_year"] == 2050
    assert p["window_20yr"] == [2041, 2060]
    assert p["baseline_period"] == [1971, 2000]
    assert p["representation"] == "multi_model_ensemble"
    verdict = p["adaptation_verdicts"]["heat_tolerant_wheat_variant"]
    assert verdict["cost_complete"] is True


def test_representation_mismatch_is_importable():
    """The exception class is exported for callers that want to raise
    it when a Short-Term three-pass array is fed into an LT codepath."""
    # We instantiate + raise/catch to prove the class works.
    with pytest.raises(RepresentationMismatch):
        raise RepresentationMismatch("test")


# ─── STEP 10.10 — window_climatology sanity ──────────────────────────
def test_window_climatology_reductions():
    da = _synth_daily(seed=99, T=365)
    m = window_climatology(da, method="mean")
    mx = window_climatology(da, method="max")
    assert m.shape == (da.sizes["lat"], da.sizes["lon"])
    assert (mx.values >= m.values).all()
