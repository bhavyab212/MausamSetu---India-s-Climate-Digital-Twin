"""
whatif.indices.radiation — extraterrestrial radiation Ra.

Pure geometry — no data, no fits. Cached once as a ``(dayofyear, lat)``
table under ``CACHE_DIR/ra_table.nc``.

Primary source: FAO Irrigation & Drainage Paper 56 (Allen et al. 1998),
Chapter 3, "Extraterrestrial radiation for daily periods". Equations
21–24. Values checked against FAO-56 Annex 2 Table 2.6.

Formulae (per equation number in FAO-56):

    (21)  Ra = 24·60/π · Gsc · dr · [ωs·sin(φ)·sin(δ) + cos(φ)·cos(δ)·sin(ωs)]
                                                                [MJ m⁻² day⁻¹]
    (22)  dr = 1 + 0.033 · cos(2π/365 · J)             inverse Earth-Sun distance
    (23)  δ  = 0.409  · sin(2π/365 · J − 1.39)         solar declination [rad]
    (24)  ωs = arccos(-tan(φ) · tan(δ))                 sunset hour angle [rad]

Constants:
    Gsc = 0.0820  MJ m⁻² min⁻¹    solar constant  (FAO-56 §3.1.1)

φ = latitude [rad]; J = day of year (1..365 or 366).

Ra is expressed in **MJ m⁻² day⁻¹** — the unit Hargreaves-Samani expects
when the 0.0023 coefficient is used to return ET0 in mm/day.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..config.constants import LAT_MAX, LAT_MIN, N_LAT
from ..config.paths import CACHE_DIR

GSC = 0.0820  # MJ m^-2 min^-1  (FAO-56 §3.1.1)


def _ra_series(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Compute Ra for a 1-D lat array × 1-D DOY array. Returns
    ``(len(doy), len(lat))`` in MJ m⁻² day⁻¹.

    Vectorised implementation of FAO-56 eqs. 21–24."""
    phi = np.deg2rad(lat_deg)[None, :]                            # (1, LAT)
    J = doy[:, None].astype(np.float64)                           # (DOY, 1)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * J)              # (DOY, 1)
    dec = 0.409 * np.sin(2 * np.pi / 365.0 * J - 1.39)            # (DOY, 1)
    # sunset hour angle: arccos(-tan(phi)*tan(dec)); clamp to [-1,1]
    x = -np.tan(phi) * np.tan(dec)
    x = np.clip(x, -1.0, 1.0)
    omega_s = np.arccos(x)                                        # (DOY, LAT)
    ra = (
        24.0 * 60.0 / np.pi
        * GSC
        * dr
        * (omega_s * np.sin(phi) * np.sin(dec)
           + np.cos(phi) * np.cos(dec) * np.sin(omega_s))
    )
    return np.clip(ra, 0.0, None)


def ra_table(lat_deg: np.ndarray | None = None) -> xr.DataArray:
    """Return a ``(dayofyear, lat)`` Ra lookup table on the master
    grid's latitude axis (129 values from 6.5 → 38.5°N, 0.25°). DOYs 1..366.

    Cached to ``CACHE_DIR/ra_table.nc``; subsequent calls are ~O(ms)."""
    cache = CACHE_DIR / "ra_table.nc"
    if cache.exists():
        with xr.open_dataarray(cache) as _tmp:
            return _tmp.load()

    if lat_deg is None:
        lat_deg = np.round(np.linspace(LAT_MIN, LAT_MAX, N_LAT), 4)
    doys = np.arange(1, 367)
    arr = _ra_series(lat_deg, doys).astype(np.float32)
    da = xr.DataArray(
        arr,
        dims=("dayofyear", "lat"),
        coords={"dayofyear": doys, "lat": lat_deg},
        name="ra",
    )
    da.attrs.update(
        units="MJ m-2 day-1",
        method="fao56-eq21-24",
        source="whatif.indices.radiation.ra_table",
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(cache)
    return da


def broadcast_ra_to(da_ref: xr.DataArray) -> xr.DataArray:
    """Return an Ra field aligned to ``da_ref``'s (time, lat, lon) grid,
    broadcast from the (doy, lat) table. Uses the DOY of ``da_ref.time``
    and repeats along ``lon``."""
    ra_dl = ra_table()               # (dayofyear, lat)
    doy = da_ref["time.dayofyear"]
    ra_tl = ra_dl.sel(dayofyear=doy)  # (time, lat)
    # broadcast to lon
    ra_tll = ra_tl.expand_dims(lon=da_ref.lon).transpose("time", "lat", "lon")
    return ra_tll
