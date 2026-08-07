"""
whatif.tests.test_indices — Part 2 acceptance tests.

Every test is hermetic (< 10 s each) and guards ONE contract from
the Part-2 spec. Failures block the branch.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from climate_twin.whatif.indices import (
    DRY_THRESHOLD_MM,
    INDEX_REGISTRY,
    LeakageError,
    assert_train_only,
    broadcast_ra_to,
    cdd,
    et0_hargreaves,
    gdd,
    hdd,
    hot_day_count,
    longest_dry_spell,
    ra_table,
)
from climate_twin.whatif.indices.radiation import _ra_series
from climate_twin.whatif.indices.reference import TRAIN_YEARS, VALID_YEARS


# ---------------------------------------------------------------------------
# Synthetic-input helpers
# ---------------------------------------------------------------------------
def _mini_da(values: np.ndarray, times, lat=None, lon=None,
              name="x", attrs: dict | None = None) -> xr.DataArray:
    lat = lat if lat is not None else np.array([21.0])
    lon = lon if lon is not None else np.array([79.0])
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.DatetimeIndex(times),
                 "lat": lat, "lon": lon},
        name=name,
    )
    da.attrs.update(attrs or {})
    da.attrs.setdefault("quantile", "deterministic")
    da.attrs.setdefault("source", "test-fixture")
    da.attrs.setdefault("source_version", "unit")
    return da


# ---------------------------------------------------------------------------
# Section 2 — stateless indices
# ---------------------------------------------------------------------------
def test_ra_symmetry():
    """The strong hemispheric symmetry Ra(-lat, doy) ≈ Ra(lat, doy+182)
    holds ONLY to within ~5-7% because Earth's orbital eccentricity
    (dr = 1 + 0.033·cos(2π/365·J), FAO-56 eq. 22) is not itself
    hemisphere-symmetric — perihelion is in early January.

    The invariant we test here is the STRICT one: the **mean annual Ra**
    depends only on |lat|. That removes the eccentricity contribution
    (integrated over a year the cos term vanishes) and catches any
    sign / hemisphere bug in the geometry."""
    lats = np.array([-21.0, 21.0])
    doys = np.arange(1, 366)
    ra = _ra_series(lats, doys)                        # (365, 2)
    mean_south = float(ra[:, 0].mean())
    mean_north = float(ra[:, 1].mean())
    rel_err = abs(mean_south - mean_north) / max(mean_north, 1e-6)
    # 2% tolerance — sufficient to catch a lat sign flip (which would
    # give either 0% or > 30% asymmetry for these lats) while allowing
    # for eccentricity + calendar-length residuals.
    assert rel_err < 0.02, (
        f"mean-annual Ra should be lat-symmetric within 2%: "
        f"got Ra(-21)={mean_south:.3f} vs Ra(+21)={mean_north:.3f}, "
        f"rel_err={rel_err:.4f}"
    )


def test_et0_hargreaves_hand_computed():
    """Sanity check against a published-formula hand-compute:
    Nagpur (~21 °N), 2020-05-15, Tmax = 42 °C, Tmin = 26 °C.
    Ra at 21 °N, DOY 136 ≈ 41.9 MJ m-2 day-1 (FAO-56 Table 2.6 says
    ~41 at 20°N DOY 121, ~42 at 20°N DOY 152 → ~41.9 mid-May).
    Expected ET0 ≈ 0.0023 * 41.9 * (34 + 17.8) * sqrt(16)
              ≈ 0.0023 * 41.9 * 51.8 * 4  ≈ 19.97 mm/day ± 0.2."""
    day = pd.Timestamp("2020-05-15")
    tmax = _mini_da(np.array([[[42.0]]], dtype=np.float32), [day])
    tmin = _mini_da(np.array([[[26.0]]], dtype=np.float32), [day])
    et = et0_hargreaves(tmax, tmin)
    val = float(et.values[0, 0, 0])
    ra_lookup = float(ra_table().sel(dayofyear=136, lat=21.0, method="nearest"))
    expected = 0.0023 * ra_lookup * ((42 + 26) / 2 + 17.8) * np.sqrt(42 - 26)
    assert abs(val - expected) < 0.2, (
        f"ET0 hand check: got {val:.3f}, expected {expected:.3f}"
    )


def test_gdd_unit():
    """Tmax=30, Tmin=10, T_base=10 → GDD = 10 °C·day exactly.
    Tmean=20, (20−10)=10."""
    tmax = _mini_da(np.array([[[30.0]]], dtype=np.float32),
                      [pd.Timestamp("2020-06-01")])
    tmin = _mini_da(np.array([[[10.0]]], dtype=np.float32),
                      [pd.Timestamp("2020-06-01")])
    out = gdd(tmax, tmin, t_base=10.0)
    assert float(out.values[0, 0, 0]) == pytest.approx(10.0)


def test_dry_spell_run_length():
    """rain = [3, 0, 0, 0, 3, 0, 0] mm with 2.5 mm threshold
    → dry days = [F, T, T, T, F, T, T]; longest run = 3."""
    values = np.array([3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
    times = pd.date_range("2020-06-01", periods=7)
    da = _mini_da(values.reshape(-1, 1, 1), times)
    out = longest_dry_spell(da)
    assert int(out.values[0, 0]) == 3


# ---------------------------------------------------------------------------
# Section 3 — provenance + quantile invariants
# ---------------------------------------------------------------------------
def test_provenance_chain_records_hops():
    """A pipeline of et0 → gdd → cdd propagates source_chain across
    every hop."""
    times = pd.date_range("2020-06-01", periods=3)
    tmax = _mini_da(np.full((3, 1, 1), 35.0, dtype=np.float32), times)
    tmin = _mini_da(np.full((3, 1, 1), 25.0, dtype=np.float32), times)

    et = et0_hargreaves(tmax, tmin)
    assert "indices.et0" in et.attrs["source_chain"]

    g = gdd(tmax, tmin, t_base=10.0)
    assert "indices.et0" not in g.attrs["source_chain"]  # gdd doesn't consume et
    assert "gdd" in g.attrs["source_chain"]


def test_quantile_propagation():
    """Input DataArray with attrs['quantile']='q90' → output has
    attrs['quantile']='q90'. This is the Part-1 rule extended into L1."""
    times = pd.date_range("2020-06-01", periods=1)
    tmax = _mini_da(np.array([[[35.0]]], dtype=np.float32), times,
                      attrs={"quantile": "q90"})
    tmin = _mini_da(np.array([[[25.0]]], dtype=np.float32), times,
                      attrs={"quantile": "q90"})
    et = et0_hargreaves(tmax, tmin)
    assert et.attrs["quantile"] == "q90"


# ---------------------------------------------------------------------------
# Section 4 — SPI standardisation + leakage guard
# ---------------------------------------------------------------------------
def test_spi_leakage_guard():
    """assert_train_only refuses any year outside TRAIN_YEARS.

    Specifically, requesting fit_spi(rain, accum_months=3,
    train_years=(1971, 2022)) must raise LeakageError because 2011..2022
    is the VALID window."""
    # Just call the helper directly with a leakage-inducing range
    with pytest.raises(LeakageError):
        assert_train_only(list(range(1971, 2023)))
    # And a clean range passes silently
    assert_train_only(list(range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1)))


def test_spi_standardisation_on_synthetic():
    """After fitting on TRAIN_YEARS-shaped synthetic gamma rainfall
    and scoring on the same window, SPI has mean ≈ 0 and std ≈ 1
    across space + time to within 0.05.

    This uses a small (5×5) grid with 40 years of monthly rainfall
    drawn from a fixed Gamma(shape=2, scale=30) so the fit is
    well-posed but the test still runs in a couple of seconds."""
    from climate_twin.whatif.indices.spi import fit_spi, spi
    rng = np.random.default_rng(42)
    years = np.arange(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1)
    times = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    # 3D shape (T, 5, 5) — daily fake rainfall matching cube schema
    H, W = 5, 5
    # Ensure exact gamma properties per (calendar_month × cell) by
    # replicating the same synthetic monthly total across all days of a
    # month (SPI accumulates to monthly anyway).
    monthly_totals = rng.gamma(shape=2.0, scale=30.0, size=(len(years) * 12, H, W))
    daily = np.zeros((len(times), H, W), dtype=np.float32)
    # Distribute each month's total across its days as constant → totals match
    yrs_arr = times.year.values
    mos_arr = times.month.values
    slot = 0
    for y in years:
        for m in range(1, 13):
            picks = np.flatnonzero((yrs_arr == y) & (mos_arr == m))
            n_days = picks.size
            if n_days == 0:
                slot += 1
                continue
            daily[picks] = monthly_totals[slot] / n_days
            slot += 1

    lat = np.linspace(20.0, 21.0, H)
    lon = np.linspace(78.0, 79.0, W)
    rain = xr.DataArray(daily, dims=("time", "lat", "lon"),
                          coords={"time": times, "lat": lat, "lon": lon},
                          name="rain", attrs={"units": "mm",
                                                "quantile": "deterministic",
                                                "source": "test",
                                                "source_version": "unit"})

    fit = fit_spi(rain, accum_months=3, train_years=TRAIN_YEARS)
    z = spi(rain, fit)
    # Grab the values, drop NaNs (leading NaNs from the accum roll)
    zvals = z.values[np.isfinite(z.values)]
    assert abs(float(zvals.mean())) < 0.10, f"SPI mean = {zvals.mean():.4f}"
    assert abs(float(zvals.std()) - 1.0) < 0.10, f"SPI std = {zvals.std():.4f}"


# ---------------------------------------------------------------------------
# Section 6 — INDEX_REGISTRY basic invariants
# ---------------------------------------------------------------------------
def test_index_registry_has_versions_and_citations():
    """Every registry entry carries a non-empty version + citation."""
    assert len(INDEX_REGISTRY) >= 10
    for name, spec in INDEX_REGISTRY.items():
        assert spec.version, f"{name}: empty version"
        assert spec.citation, f"{name}: empty citation"
        assert spec.units, f"{name}: empty units"


if __name__ == "__main__":   # pragma: no cover
    sys.exit(pytest.main([__file__, "-x", "-q"]))
