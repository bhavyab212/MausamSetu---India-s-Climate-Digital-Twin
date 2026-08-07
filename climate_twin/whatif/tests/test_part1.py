"""
whatif.tests.test_part1 — Part 1 acceptance tests.

Each test is hermetic (mocks + tiny fixtures; no > 10 s waits) and
guards ONE contract from the Part-1 spec.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from climate_twin.whatif.config.constants import IMD_SENTINELS, N_LAT, N_LON
from climate_twin.whatif.config.region import RegionSpec
from climate_twin.whatif.drivers._common import mask_sentinels, to_ist_index
from climate_twin.whatif.drivers.driver import DriverSpec, load_driver
from climate_twin.whatif.drivers.historical import get_historical
from climate_twin.whatif.scenarios.provenance import (
    record_add_layer,
    record_close,
    record_open,
    run_from_yaml,
)
from climate_twin.whatif.scenarios.quantiles import (
    MixedQuantiles,
    assert_single_quantile,
    three_pass,
)


# ---------------------------------------------------------------------------
def test_historical_masks_sentinels():
    """Synthetic array with −999 and 99.9 → both become NaN. Idempotent."""
    src = np.array([[0.1, IMD_SENTINELS[0], 5.0],
                     [IMD_SENTINELS[1], 2.3, np.nan]], dtype=np.float32)
    out = mask_sentinels(src)
    assert np.isnan(out[0, 1])   # -999 → NaN
    assert np.isnan(out[1, 0])   # 99.9 → NaN
    # non-sentinel values preserved bit-for-bit
    assert out[0, 0] == 0.1
    assert out[1, 1] == 2.3
    # idempotent
    assert np.array_equal(mask_sentinels(out), out, equal_nan=True)


def test_master_grid_shape():
    """A 1-day historical read returns the exact (N_LAT, N_LON) shape
    from constants.py, with lat + lon ascending."""
    da = get_historical("tmax", date(2020, 6, 1), date(2020, 6, 1))
    assert da.dims == ("time", "lat", "lon"), da.dims
    assert da.sizes["lat"] == N_LAT
    assert da.sizes["lon"] == N_LON
    lat = da.lat.values
    lon = da.lon.values
    assert np.all(np.diff(lat) > 0), "lat must ascend"
    assert np.all(np.diff(lon) > 0), "lon must ascend"


def test_ist_awareness():
    """Returned times carry Asia/Kolkata tzinfo."""
    da = get_historical("tmax", date(2020, 6, 1), date(2020, 6, 3))
    t0 = da["time"].to_index()[0]
    assert t0.tzinfo is not None
    assert "Kolkata" in str(t0.tzinfo) or "IST" in str(t0.tzinfo)


def test_provenance_replay():
    """Open a record, add a layer, close it → replay yields the same
    driver spec + array shape. The bitwise-equal check is on the
    driver array."""
    spec = DriverSpec(
        mode="historical", var="tmax",
        dates=(date(2020, 6, 1), date(2020, 6, 1)),
        region=RegionSpec(kind="all_india"),
    )
    da_orig = load_driver(spec)
    rec = record_open(spec, levers={"note": "test"})
    record_add_layer(rec, "drivers.load_driver", "historical-v1",
                      {"shape": tuple(da_orig.shape)})
    path = record_close(rec, outputs_summary={"tag": "unit-test"})
    assert path.exists()

    # Replay via run_from_yaml → run_scenario → load_driver
    result = run_from_yaml(path)
    da_replay = result.driver
    assert np.array_equal(
        np.nan_to_num(da_orig.values, nan=-1e9),
        np.nan_to_num(da_replay.values, nan=-1e9),
    )


def test_mixed_quantile_raises():
    """assert_single_quantile refuses to blend inputs with different
    quantile tags (this is the fence protecting L4 economics)."""
    a = xr.DataArray(np.zeros((3, 3), dtype=np.float32),
                       dims=("lat", "lon"),
                       attrs={"quantile": "q10"})
    b = xr.DataArray(np.zeros((3, 3), dtype=np.float32),
                       dims=("lat", "lon"),
                       attrs={"quantile": "q50"})
    with pytest.raises(MixedQuantiles):
        assert_single_quantile(a, b)

    # Same quantile → OK, returns the tag
    q = assert_single_quantile(a, xr.DataArray(np.zeros((3, 3)), attrs={"quantile": "q10"}))
    assert q == "q10"


def test_three_pass_produces_three_keys():
    """three_pass(fn) executes fn once per quantile, returns {q10,q50,q90}."""
    calls = []
    def _fn(spec_):
        calls.append(spec_.quantile)
        return spec_.quantile
    spec = DriverSpec(
        mode="forecast", var="rain",
        dates=(date(2020, 6, 1), date(2020, 6, 1)),
        region=RegionSpec(kind="all_india"),
    )
    out = three_pass(spec, _fn)
    assert set(out) == {"q10", "q50", "q90"}
    assert sorted(calls) == [0.10, 0.50, 0.90]


if __name__ == "__main__":   # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))
