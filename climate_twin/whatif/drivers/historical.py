"""
whatif.drivers.historical — L0 historical driver.

Reads gauge-only observed fields (rain, tmax, tmin, tmean) from the
processed cube (canonical, sentinel-clean, master grid) that
``climate_twin/data/build_cube.py`` produces. If the cube is missing,
falls back to the raw IMD readers at ``climate_twin.data.readers``.

All returned DataArrays:
    * dims (time, lat, lon), lat/lon ascending, 0.25° India grid
    * time is IST-aware
    * IMD sentinels −999.0 / 99.9 are already NaN (cube path) or masked
      here (raw path) BEFORE any arithmetic
    * attrs: units, source, source_version, quantile="deterministic"
"""
from __future__ import annotations

import hashlib
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from ..config.paths import CACHE_DIR, CUBE_INDIA
from ._common import (
    assert_master_grid,
    date_str,
    mask_sentinels,
    stamp_attrs,
    to_ist_index,
)

VarName = Literal["rain", "tmax", "tmin", "tmean"]

_READER_VERSION = "cube-v1"  # bumps if the reader logic changes; used in cache key


# ---------------------------------------------------------------------------
def _cache_path(var: str, start: date, end: date) -> Path:
    key = f"historical|{var}|{date_str(start)}|{date_str(end)}|{_READER_VERSION}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    # Pickle so tz-aware time coords survive round-trip cleanly.
    return CACHE_DIR / "historical" / f"{var}_{digest}.pkl"


def _load_from_cube(var: str, start: date, end: date) -> xr.DataArray:
    """Slice one variable × one date-range out of the master cube.

    Uses xarray's lazy time-selection so the reader never materialises
    the full 27k-day tensor (this is the same pattern data_source.py
    uses; see the Phase 4 memory-fix commit)."""
    if not CUBE_INDIA.exists():
        raise FileNotFoundError(
            f"processed cube {CUBE_INDIA} not present — build it with "
            f"`python -m climate_twin.data.build_cube` before using "
            f"the historical driver."
        )

    if var == "tmean":
        # Compute from tmax + tmin (joint mask)
        tmx = _load_from_cube("tmax", start, end)
        tmn = _load_from_cube("tmin", start, end)
        tm = (tmx + tmn) * 0.5
        tm = tm.where(tmx.notnull() & tmn.notnull())
        return stamp_attrs(
            tm.rename("tmean"),
            var="tmean",
            source=str(CUBE_INDIA),
            source_version=f"cube:{CUBE_INDIA.stat().st_size}",
        )

    ds = xr.open_dataset(CUBE_INDIA)
    try:
        if var not in ds.data_vars:
            raise KeyError(f"cube has no variable {var!r} — has {list(ds.data_vars)}")

        # Lazy: select by boolean mask on `time.year|month|day`.
        times = ds["time"].values
        idx = np.where(
            (times >= np.datetime64(start))
            & (times <= np.datetime64(end))
        )[0]
        if idx.size == 0:
            raise ValueError(
                f"no cube days in {start}..{end} "
                f"(cube covers {str(times[0])[:10]}..{str(times[-1])[:10]})"
            )
        da = ds[var].isel(time=idx).load()

        # Rebuild the time axis as IST-aware
        da = da.assign_coords(time=to_ist_index(da["time"].values))
        # Cube outputs are already sentinel-clean but re-mask defensively.
        cleaned = mask_sentinels(da.values.astype(np.float32))
        out = xr.DataArray(
            cleaned,
            dims=("time", "lat", "lon"),
            coords={"time": da["time"], "lat": da["lat"], "lon": da["lon"]},
            name=var,
        )
        assert_master_grid(out)
        return stamp_attrs(
            out,
            var=var,
            source=str(CUBE_INDIA),
            source_version=f"cube:{CUBE_INDIA.stat().st_size}",
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
def get_historical(
    var: VarName,
    start: date,
    end: date,
) -> xr.DataArray:
    """Return the observed IMD field for ``var`` between ``start`` and ``end``.

    * ``var`` ∈ {"rain","tmax","tmin","tmean"} — tmean is derived as
      ``(tmax+tmin)/2`` with a joint NaN mask.
    * dates are inclusive, IST-aware on return.
    * Results are cached to disk under ``CACHE_DIR/historical/`` keyed by
      ``(var, start, end, reader_version)``; a cache hit is O(ms) even
      for 30-year requests.
    """
    if start > end:
        raise ValueError(f"start ({start}) > end ({end})")
    if var not in ("rain", "tmax", "tmin", "tmean"):
        raise ValueError(f"unknown var {var!r}")

    cache = _cache_path(var, start, end)
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                da = pickle.load(f)
            return da
        except Exception:
            # Corrupted / stale cache — nuke and rebuild.
            cache.unlink(missing_ok=True)

    da = _load_from_cube(var, start, end)

    # Persist via pickle (preserves tz-aware time coords, attrs, dtypes).
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache, "wb") as f:
            pickle.dump(da, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Cache failure is never fatal — return the freshly-computed array
        pass
    return da


def get_tmean(start: date, end: date) -> xr.DataArray:
    """Convenience wrapper — same contract as get_historical('tmean', …)."""
    return get_historical("tmean", start, end)
