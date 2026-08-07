"""
whatif.tests.test_perturbation — Part 5, Method 1 tests.

Fast, hermetic. Every failure blocks the branch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_twin.whatif.config.region import RegionSpec
from climate_twin.whatif.drivers.perturbation import (
    PERTURBATION_VERSION,
    CaveatRequiredError,
    PerturbationSpec,
    apply_perturbation,
    assert_caveat_acknowledged,
)


def _tiny_base(T: int = 5, H: int = 4, W: int = 4) -> xr.Dataset:
    times = pd.date_range("2020-06-01", periods=T, freq="D", tz="Asia/Kolkata")
    lat = np.linspace(19.0, 20.0, H).round(4)
    lon = np.linspace(77.0, 78.0, W).round(4)
    coords = {"time": times, "lat": lat, "lon": lon}
    rain = xr.DataArray(
        np.full((T, H, W), 5.0, dtype=np.float32),
        dims=("time", "lat", "lon"), coords=coords, name="rain",
        attrs={"units": "mm", "source": "synthetic"},
    )
    tmax = xr.DataArray(
        np.full((T, H, W), 32.0, dtype=np.float32),
        dims=("time", "lat", "lon"), coords=coords, name="tmax",
        attrs={"units": "°C", "source": "synthetic"},
    )
    tmin = xr.DataArray(
        np.full((T, H, W), 22.0, dtype=np.float32),
        dims=("time", "lat", "lon"), coords=coords, name="tmin",
        attrs={"units": "°C", "source": "synthetic"},
    )
    return xr.Dataset({"rain": rain, "tmax": tmax, "tmin": tmin},
                       attrs={"source_chain": "synthetic-base"})


def test_perturbation_identity_preserves_values():
    """apply_perturbation(base, PerturbationSpec()) leaves rain/tmax/tmin
    numerically identical; only attrs record the (identity) transform."""
    base = _tiny_base()
    out = apply_perturbation(base, PerturbationSpec())
    for v in ("rain", "tmax", "tmin"):
        assert np.array_equal(out[v].values, base[v].values), (
            f"{v} altered by identity perturbation"
        )
    assert "drivers.perturbation" in out.attrs["source_chain"]
    assert out.attrs["perturbation_version"] == PERTURBATION_VERSION


def test_perturbation_negative_rain_scale_rejected():
    """Rain scale < 0 is unphysical; the spec constructor raises."""
    with pytest.raises(ValueError):
        PerturbationSpec(rain_scale=-0.1)


def test_perturbation_nonfinite_shifts_rejected():
    """NaN / Inf temperature shifts are rejected at construction."""
    with pytest.raises(ValueError):
        PerturbationSpec(tmax_shift_c=float("nan"))
    with pytest.raises(ValueError):
        PerturbationSpec(tmin_shift_c=float("inf"))


def test_perturbation_applies_multiplicative_rain_and_additive_temp():
    """Verify the sign & magnitude convention: rain × 0.8, tmax + 1.5°C."""
    base = _tiny_base()
    spec = PerturbationSpec(rain_scale=0.8, tmax_shift_c=1.5)
    out = apply_perturbation(base, spec)
    assert np.allclose(out["rain"].values, base["rain"].values * 0.8)
    assert np.allclose(out["tmax"].values, base["tmax"].values + 1.5)
    # tmin unchanged
    assert np.allclose(out["tmin"].values, base["tmin"].values)


def test_caveat_gate_blocks_report_export():
    """A perturbation dataset without caveat_acknowledged raises
    CaveatRequiredError when a report exporter calls the gate."""
    base = _tiny_base()
    out = apply_perturbation(base, PerturbationSpec(rain_scale=0.9))
    with pytest.raises(CaveatRequiredError):
        assert_caveat_acknowledged(out)


def test_caveat_gate_passes_when_acknowledged():
    """The same gate returns without raising when the caveat is set."""
    base = _tiny_base()
    out = apply_perturbation(
        base, PerturbationSpec(rain_scale=0.9, caveat_acknowledged=True),
    )
    assert_caveat_acknowledged(out)         # no exception


def test_caveat_gate_ignores_non_perturbation_datasets():
    """A Dataset that isn't a perturbation output passes freely — the
    gate looks specifically at perturbation_version attrs."""
    base = _tiny_base()
    assert_caveat_acknowledged(base)        # no exception


def test_perturbation_scope_masks_outside_region():
    """A bbox scope leaves cells outside the bbox unchanged."""
    base = _tiny_base()
    # Bbox that catches only the top-left cell
    scope = RegionSpec(kind="bbox", bbox=(19.0, 77.0, 19.35, 77.35))
    spec = PerturbationSpec(rain_scale=0.0, scope=scope)
    out = apply_perturbation(base, spec)
    r = out["rain"].values
    # Cells inside the bbox → 0. Cells outside → unchanged (5.0).
    # Exact mask depends on rounding; assert both regimes exist.
    assert (r == 0.0).any(), "no cells were perturbed"
    assert (r == 5.0).any(), "all cells were perturbed (scope ignored)"
