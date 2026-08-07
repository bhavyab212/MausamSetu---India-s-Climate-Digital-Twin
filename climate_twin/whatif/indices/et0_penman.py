"""
whatif.indices.et0_penman — Penman-Monteith reference ET.

Primary source: FAO-56 Ch. 2 (Allen et al. 1998). Full PM requires
RH, wind at 2 m, and net radiation — none of which the IMD 0.25° cube
carries. This module locks the FUTURE signature so downstream code
can call it once ERA5 or Aphrodite humidity/wind arrive (Part 7).
"""
from __future__ import annotations

import xarray as xr


def et0_penman(
    tmax: xr.DataArray,
    tmin: xr.DataArray,
    rh_mean: xr.DataArray,
    wind_2m: xr.DataArray,
    rn_or_ra: xr.DataArray,
    elev_m: xr.DataArray | None = None,
) -> xr.DataArray:
    """Reserved for Part 7 — needs ERA5 humidity/wind/radiation."""
    raise NotImplementedError(
        "Penman-Monteith needs ERA5 humidity/wind/radiation. "
        "See Part 7 of the What-If build. Use "
        "whatif.indices.et0_hargreaves for the tmax/tmin-only fallback "
        "(FAO-56 §3 recommends Hargreaves-Samani in this data regime)."
    )
