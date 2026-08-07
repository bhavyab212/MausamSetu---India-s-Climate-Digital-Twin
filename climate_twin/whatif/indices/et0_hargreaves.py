"""
whatif.indices.et0_hargreaves — reference evapotranspiration, Hargreaves-Samani.

Primary source: Hargreaves & Samani (1985), "Reference crop
evapotranspiration from temperature", *Applied Engineering in Agriculture*
1(2), 96-99. FAO-56 recommends this as the fallback when full
Penman-Monteith inputs (RH, wind, radiation) are unavailable — see
FAO-56 §3, "Alternate ET0 calculation".

Formula:

    ET0 = 0.0023 · Ra · (Tmean + 17.8) · sqrt(Tmax − Tmin)     [mm/day]

    Tmean = (Tmax + Tmin) / 2                                  [°C]
    Ra    = extraterrestrial radiation                         [MJ m⁻² day⁻¹]

**Unit contract** — this is the classical Hargreaves bug source:
when the 0.0023 coefficient is used, ``Ra`` MUST be in MJ m⁻² day⁻¹
(NOT mm equivalent, NOT W m⁻²). :mod:`whatif.indices.radiation.ra_table`
guarantees this. Any change to Ra's units must be reflected in the
coefficient or the output silently drifts by a factor of ~2.45.

Bounds enforced:
    - ``tmax ≥ tmin`` cellwise; violations set the output cell to NaN
      (never silently swap the values).
    - ET0 is clipped to ``[0, ∞)`` — negative ET0 is unphysical.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .radiation import broadcast_ra_to
from .reference import _propagate_attrs


HARGREAVES_COEFF = 0.0023   # dimensionless; Hargreaves-Samani 1985 fit constant


def et0_hargreaves(tmax: xr.DataArray, tmin: xr.DataArray) -> xr.DataArray:
    """Return daily reference ET0 [mm/day] on the same (time, lat, lon) grid
    as its inputs.

    Both inputs must be °C. Non-finite inputs propagate as NaN. Cells
    where ``tmax < tmin`` (numerical noise or bad data) return NaN.
    """
    if tmax.dims != tmin.dims:
        raise ValueError(f"tmax.dims {tmax.dims} != tmin.dims {tmin.dims}")

    # Broadcast Ra onto the input grid
    ra = broadcast_ra_to(tmax)                                # MJ m-2 day-1

    tmean = (tmax + tmin) * 0.5
    diff = tmax - tmin
    # tmax < tmin → invalidate cell
    diff = diff.where(diff >= 0)
    with np.errstate(invalid="ignore"):
        et0 = HARGREAVES_COEFF * ra * (tmean + 17.8) * np.sqrt(diff)
    et0 = et0.clip(min=0.0)

    et0.name = "et0"
    et0 = _propagate_attrs(et0, tmax, "et0@hargreaves-v1")
    et0.attrs["units"] = "mm/day"
    et0.attrs["method"] = "hargreaves-samani-1985"
    et0.attrs["source_chain"] = str(et0.attrs.get("source_chain", "indices.et0@hargreaves-v1"))
    return et0
