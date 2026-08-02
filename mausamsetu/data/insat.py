"""
insat.py
=========
Reader for INSAT-3D/3DR/3DS satellite products (HDF5) from MOSDAC.

WHAT INSAT PRODUCTS LOOK LIKE
-----------------------------
Each file is an HDF5 ".h5" archive containing multiple datasets and metadata.

Common structure:
  /LST                (2D array of Land Surface Temperature)  or
  /SST                (2D array of Sea Surface Temperature)   or
  /Precipitation      (2D array of rainfall estimate; sometimes called IMR)
  /Latitude           (2D array of lat coordinates for each pixel)
  /Longitude          (2D array of lon coordinates)
  /Times              (scan time metadata)

  Attributes on each dataset:
    scale_factor      (multiply stored int by this)
    add_offset        (then add this)
    _FillValue        (value indicating missing data)

  → real_value = stored_integer * scale_factor + add_offset

TYPICAL VALUE RANGES (after unpacking):
  LST: -50 to +60 °C  (Kelvin often; we convert to Celsius)
  SST: -2 to +35 °C
  Precipitation: 0 to ~500 mm/hr

USAGE
-----
    from mausamsetu.data.insat import read_insat
    da = read_insat("data/raw/insat/3RIMG_15JUL2020_1200_L2B_LST_V01R00.h5",
                    variable="LST")
    # returns xarray DataArray with lat/lon coords and value in °C
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import xarray as xr
import h5py

from mausamsetu import config


# ============================================================================
# CORE READER
# ============================================================================
def read_insat(
    path: Path | str,
    variable: str | None = None,
) -> xr.DataArray:
    """
    Read a single INSAT HDF5 file into an xarray DataArray.

    Parameters
    ----------
    path : path to .h5 file
    variable : the main dataset name inside the file
               Common choices: 'LST', 'SST', 'HEM' (rainfall), 'IMR', 'Precipitation'.
               If None, we try to auto-detect (pick the largest 2D dataset).

    Returns
    -------
    xr.DataArray with dims (y, x) and lat/lon 2D coordinates.
    We use (y, x) instead of (lat, lon) because satellite grids are irregular —
    lat/lon are stored per-pixel, not on a regular axis.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"INSAT HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        # --- Auto-detect variable if not specified ---
        if variable is None:
            variable = _autodetect_variable(f)
            print(f"  Auto-detected variable: '{variable}'")

        if variable not in f:
            raise KeyError(
                f"Dataset '{variable}' not found in {path.name}. "
                f"Available: {list(f.keys())}"
            )

        # --- Load the raw stored values ---
        raw = f[variable][:]

        # --- Read scale factor and offset (the packing formula) ---
        attrs = dict(f[variable].attrs)
        scale = float(attrs.get("scale_factor", 1.0))
        offset = float(attrs.get("add_offset", 0.0))
        fill = attrs.get("_FillValue", None)
        if fill is None:
            fill = attrs.get("FillValue", None)
        if fill is None:
            fill = attrs.get("missing_value", None)

        # --- Load lat/lon coordinate arrays ---
        # MOSDAC's convention varies; check common names
        lat_key = _find_key(f, ["Latitude", "latitude", "lat"])
        lon_key = _find_key(f, ["Longitude", "longitude", "lon"])

        if lat_key and lon_key:
            lat = np.asarray(f[lat_key][:])
            lon = np.asarray(f[lon_key][:])
        else:
            # Fallback: assume regular grid over India
            print("  ⚠ No lat/lon arrays found; assuming default India grid")
            ny, nx = raw.shape[-2:]
            lat = np.linspace(6.5, 38.5, ny)[:, None].repeat(nx, axis=1)
            lon = np.linspace(66.5, 99.75, nx)[None, :].repeat(ny, axis=0)

    # --- Apply scale + offset formula: real = stored × scale + offset ---
    data = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)

    # --- Mask missing values ---
    if fill is not None:
        data[raw == fill] = np.nan

    # --- If values look like Kelvin (>200 for temperature), convert to °C ---
    if variable.upper() in ("LST", "SST"):
        mean_val = np.nanmean(data)
        if mean_val > 150:   # values are in Kelvin
            data = data - 273.15
            print(f"  Converted from Kelvin to Celsius (was {mean_val:.1f}K)")

    # --- Wrap in xarray DataArray ---
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "lat": (("y", "x"), lat.astype(np.float32)),
            "lon": (("y", "x"), lon.astype(np.float32)),
        },
        name=variable.lower(),
        attrs={
            "units": _units_for_variable(variable),
            "long_name": f"INSAT {variable}",
            "source": str(path.name),
            "scale_factor": scale,
            "add_offset": offset,
        },
    )
    return da


# ============================================================================
# HELPERS
# ============================================================================
def _autodetect_variable(f: h5py.File) -> str:
    """
    Find the main 2D data variable in an HDF5 file.
    Skips coordinate/metadata datasets.
    """
    skip_names = {"latitude", "longitude", "lat", "lon", "time", "times", "x", "y"}
    best_name = None
    best_size = -1
    for name in f.keys():
        if name.lower() in skip_names:
            continue
        obj = f[name]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            size = int(np.prod(obj.shape))
            if size > best_size:
                best_size = size
                best_name = name
    if best_name is None:
        raise ValueError(f"Could not auto-detect main variable in file. Keys: {list(f.keys())}")
    return best_name


def _find_key(f: h5py.File, candidates: list[str]) -> str | None:
    """Return the first key from candidates that exists in the file."""
    for c in candidates:
        if c in f:
            return c
    return None


def _units_for_variable(variable: str) -> str:
    v = variable.upper()
    if v in ("LST", "SST"):
        return "degC"
    if v in ("HEM", "IMR", "PRECIPITATION", "RAINFALL"):
        return "mm/hr"
    return "unknown"


# ============================================================================
# CONVENIENCE: read + regrid onto our master grid
# ============================================================================
def read_insat_on_master_grid(
    path: Path | str,
    variable: str | None = None,
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Read an INSAT file and regrid its irregular satellite grid to our regular
    Cauvery master grid (0.25°). Uses scipy nearest/linear interpolation.

    Parameters
    ----------
    path : path to .h5 file
    variable : dataset name (default: auto)
    target_lat, target_lon : 1D arrays for the master grid.
        If None, use the Cauvery pilot grid from config.

    Returns
    -------
    xr.DataArray with dims (lat, lon) on the target regular grid.
    """
    from scipy.interpolate import griddata

    # --- Default target = Cauvery grid ---
    if target_lat is None:
        target_lat = np.arange(
            config.PILOT_LAT_START,
            config.PILOT_LAT_END + config.MASTER_RES / 2,
            config.MASTER_RES,
        ).astype(np.float32)
    if target_lon is None:
        target_lon = np.arange(
            config.PILOT_LON_START,
            config.PILOT_LON_END + config.MASTER_RES / 2,
            config.MASTER_RES,
        ).astype(np.float32)

    # --- Load native satellite grid ---
    da_native = read_insat(path, variable=variable)
    lat2d = da_native["lat"].values
    lon2d = da_native["lon"].values
    values = da_native.values

    # --- Filter to points near our region (for speed) ---
    mask = (
        (lat2d >= target_lat.min() - 1)
        & (lat2d <= target_lat.max() + 1)
        & (lon2d >= target_lon.min() - 1)
        & (lon2d <= target_lon.max() + 1)
        & np.isfinite(values)
    )
    pts = np.column_stack([lat2d[mask], lon2d[mask]])
    vals = values[mask]

    if len(vals) == 0:
        print(f"  ⚠ No valid points in region for {Path(path).name}")
        # Return an all-NaN array of correct shape
        grid = np.full((len(target_lat), len(target_lon)), np.nan, dtype=np.float32)
    else:
        # --- Regrid using nearest-neighbor (fast) then linear fallback ---
        LAT2, LON2 = np.meshgrid(target_lat, target_lon, indexing="ij")
        target_pts = np.column_stack([LAT2.ravel(), LON2.ravel()])
        grid = griddata(pts, vals, target_pts, method="linear", fill_value=np.nan)
        # Fill remaining NaN with nearest neighbor
        if np.isnan(grid).any():
            grid_nn = griddata(pts, vals, target_pts, method="nearest")
            nan_mask = np.isnan(grid)
            grid[nan_mask] = grid_nn[nan_mask]
        grid = grid.reshape(LAT2.shape).astype(np.float32)

    # --- Wrap into regular-grid DataArray ---
    da = xr.DataArray(
        grid,
        dims=("lat", "lon"),
        coords={"lat": target_lat, "lon": target_lon},
        name=da_native.name,
        attrs=da_native.attrs,
    )
    return da


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mausamsetu.data.insat <path_to_.h5> [variable]")
        sys.exit(1)
    path = sys.argv[1]
    var = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Inspecting INSAT file: {path}")
    with h5py.File(path, "r") as f:
        print(f"\nDatasets in file:")
        for name in f.keys():
            obj = f[name]
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        print()

    da = read_insat(path, variable=var)
    print(f"\nVariable: {da.name}")
    print(f"Shape: {da.shape}")
    print(f"Units: {da.attrs.get('units')}")
    print(f"Value stats: mean={float(np.nanmean(da.values)):.2f}, "
          f"min={float(np.nanmin(da.values)):.2f}, max={float(np.nanmax(da.values)):.2f}")
    print(f"Lat range: {float(np.nanmin(da['lat'])):.2f} to {float(np.nanmax(da['lat'])):.2f}")
    print(f"Lon range: {float(np.nanmin(da['lon'])):.2f} to {float(np.nanmax(da['lon'])):.2f}")
