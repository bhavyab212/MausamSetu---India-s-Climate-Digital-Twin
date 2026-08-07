"""
whatif.indices.dry_spell — dry-spell run-length indices.

Primary sources:
    * IMD's rain-day convention: a "rainy day" is defined as one with
      daily rainfall ≥ 2.5 mm. Any day with < 2.5 mm is a "dry day".
      Reference: IMD's `imd.gov.in` glossary + Guhathakurta et al. 2011.
    * ETCCDI Consecutive Dry Days (CDD): the maximum length of a run of
      consecutive dry days per period. Karl, Nicholls & Ghazi (1999);
      Zhang et al. (2011).

To avoid a clash with :mod:`whatif.indices.degree_days.cdd` (cooling
degree days, °C·day), the ETCCDI variant here is named ``cdd_wmo``.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .reference import _propagate_attrs

DRY_THRESHOLD_MM: float = 2.5  # IMD rain-day convention


def _longest_run(arr: np.ndarray) -> int:
    """Longest run of True in a 1-D boolean array. Ignores NaN by
    treating them as False (spell interrupted)."""
    if arr.size == 0:
        return 0
    best = cur = 0
    for x in arr:
        if x:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def longest_dry_spell(rain: xr.DataArray) -> xr.DataArray:
    """Longest consecutive-dry-day run over ``rain.time``.

    Returns a ``(lat, lon)`` DataArray (integer). Time axis is
    consumed; caller slices ``rain`` to the desired window first."""
    dry = (rain < DRY_THRESHOLD_MM).astype("int8")
    out = xr.apply_ufunc(
        lambda x: _longest_run(x.astype(bool)),
        dry, input_core_dims=[["time"]],
        vectorize=True, dask="parallelized",
        output_dtypes=[np.int32],
    ).rename("longest_dry_spell")
    out = _propagate_attrs(out, rain, "longest_dry_spell-v1")
    out.attrs["units"] = "days"
    out.attrs["dry_threshold_mm"] = DRY_THRESHOLD_MM
    out.attrs["method"] = "consecutive-run-length"
    return out


def cdd_wmo(rain: xr.DataArray, freq: str = "YE") -> xr.DataArray:
    """ETCCDI Consecutive Dry Days — max run length per resample bin.

    Not to be confused with :func:`whatif.indices.degree_days.cdd`."""
    dry = (rain < DRY_THRESHOLD_MM).astype("int8")

    def _per_bin(arr: np.ndarray) -> int:
        return _longest_run(arr.astype(bool))

    out = dry.resample(time=freq).map(
        lambda block: xr.apply_ufunc(
            _per_bin, block,
            input_core_dims=[["time"]],
            vectorize=True, dask="parallelized",
            output_dtypes=[np.int32],
        )
    ).rename("cdd_wmo")
    out = _propagate_attrs(out, rain, "cdd_wmo-v1")
    out.attrs["units"] = "days"
    out.attrs["dry_threshold_mm"] = DRY_THRESHOLD_MM
    out.attrs["method"] = "etccdi-cdd"
    return out
