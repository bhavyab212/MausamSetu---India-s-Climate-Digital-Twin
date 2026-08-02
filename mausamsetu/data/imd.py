"""
imd.py
=======
Reader for IMD Gridded Binary (.grd) files.

WHAT IMD .grd FILES CONTAIN
---------------------------
IMD Pune distributes daily gridded observations as raw binary files.
Each file = 1 YEAR of data for 1 VARIABLE.

There are TWO different grids:

  RAINFALL (0.25° resolution):
    - Grid: 135 lon × 129 lat = 17,415 pixels per day
    - Longitude: 66.5°E to 99.75°E (step 0.25°)
    - Latitude:  6.5°N to 38.5°N  (step 0.25°)
    - Values in mm/day
    - Missing data sentinel: -999.0

  TEMPERATURE (Tmax, Tmin) (1.0° resolution):
    - Grid: 31 lon × 31 lat = 961 pixels per day
    - Longitude: 67.5°E to 97.5°E (step 1.0°)
    - Latitude:  7.5°N to 37.5°N  (step 1.0°)
    - Values in °C
    - Missing data sentinel: 99.9

FILE FORMAT: RAW FLOAT32 BINARY, LATITUDE-MAJOR
-----------------------------------------------
Data layout (row-major, latitude varies slowest):
  file = concatenation of daily grids
  each daily grid = flattened [lat0_lon0, lat0_lon1, ..., lat0_lonN,
                                lat1_lon0, ..., latM_lonN]
  little-endian float32 (4 bytes per value)

USAGE
-----
    from mausamsetu.data.imd import read_imd_rainfall, read_imd_temperature
    da = read_imd_rainfall("data/raw/imd_rain/Rainfall_ind2020_rfp25.grd", year=2020)
    # da is xarray DataArray with dims (time, lat, lon)

NOTE ON ORIENTATION
-------------------
IMD stores lat SOUTH-TO-NORTH (6.5°N first, 38.5°N last).
That's what our code assumes. If a file shows mirrored maps, flip lat.
Always run the Western Ghats Test after reading to verify.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import xarray as xr

from mausamsetu import config


# ============================================================================
# CORE READER (generic — used by both rain and temp)
# ============================================================================
def _read_grd(
    path: Path | str,
    year: int,
    nlat: int,
    nlon: int,
    lat_start: float,
    lon_start: float,
    resolution: float,
    fill_value: float,
    variable_name: str,
    units: str,
    long_name: str,
) -> xr.DataArray:
    """
    Generic IMD .grd binary reader.

    Parameters
    ----------
    path : file path
    year : year the file represents (needed to build time axis)
    nlat, nlon : grid dimensions
    lat_start, lon_start : southwest corner of grid
    resolution : degrees per pixel
    fill_value : IMD's "missing data" sentinel
    variable_name, units, long_name : metadata for the output DataArray

    Returns
    -------
    xarray.DataArray with dims (time, lat, lon), coordinates set, metadata attached.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IMD .grd file not found: {path}")

    # ------------------------------------------------------------------
    # Step 1: Read raw bytes as float32
    # ------------------------------------------------------------------
    # np.fromfile reads the ENTIRE file into a 1-D array.
    # dtype='<f4' means: little-endian 32-bit float.
    raw = np.fromfile(path, dtype="<f4")

    # ------------------------------------------------------------------
    # Step 2: Verify the expected size
    # ------------------------------------------------------------------
    # Determine days in the year (365 or 366)
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    n_days_leap = 366
    n_days_normal = 365

    expected_leap = n_days_leap * nlat * nlon
    expected_normal = n_days_normal * nlat * nlon

    if raw.size == expected_leap:
        n_days = n_days_leap
    elif raw.size == expected_normal:
        n_days = n_days_normal
    else:
        raise ValueError(
            f"Unexpected file size for {path}: got {raw.size} floats, "
            f"expected {expected_normal} (365 days) or {expected_leap} (366 days). "
            f"Grid: {nlat}×{nlon}. Check year={year} matches file."
        )

    # ------------------------------------------------------------------
    # Step 3: Reshape into (time, lat, lon)
    # ------------------------------------------------------------------
    # IMD stores data lat-first (row-major, latitude varies faster than time).
    # Reshape into (n_days, nlat, nlon).
    data = raw.reshape((n_days, nlat, nlon))

    # ------------------------------------------------------------------
    # Step 4: Mask missing values
    # ------------------------------------------------------------------
    # IMD marks "no data" with a sentinel (e.g. -999 for rain, 99.9 for temp).
    # We convert to NaN so numpy/xarray ignore them in calculations.
    data = data.astype(np.float32)
    data[data == fill_value] = np.nan

    # ------------------------------------------------------------------
    # Step 5: Build coordinate arrays
    # ------------------------------------------------------------------
    # Longitudes: [lon_start, lon_start+res, ..., lon_start + (nlon-1)*res]
    lons = lon_start + resolution * np.arange(nlon, dtype=np.float32)
    lats = lat_start + resolution * np.arange(nlat, dtype=np.float32)

    # Time: one date per day of the year
    times = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")

    # ------------------------------------------------------------------
    # Step 6: Wrap into xarray DataArray with metadata
    # ------------------------------------------------------------------
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
        name=variable_name,
        attrs={
            "units": units,
            "long_name": long_name,
            "source": f"IMD Pune .grd file: {path.name}",
            "grid_resolution_deg": resolution,
        },
    )
    return da


# ============================================================================
# PUBLIC API — Rainfall
# ============================================================================
def read_imd_rainfall(path: Path | str, year: int) -> xr.DataArray:
    """
    Read one year of IMD gridded rainfall (0.25° resolution).

    Parameters
    ----------
    path : Path to the .grd file (e.g. 'Rainfall_ind2020_rfp25.grd')
    year : Year the file represents (e.g. 2020)

    Returns
    -------
    xr.DataArray with dims (time, lat, lon), values in mm/day.
    Shape: (365 or 366, 129, 135). NaN over ocean / outside India.
    """
    return _read_grd(
        path=path,
        year=year,
        nlat=config.IMD_RAIN_NLAT,
        nlon=config.IMD_RAIN_NLON,
        lat_start=config.IMD_RAIN_LAT_START,
        lon_start=config.IMD_RAIN_LON_START,
        resolution=config.IMD_RAIN_RES,
        fill_value=config.IMD_RAIN_FILL,
        variable_name="rain",
        units="mm/day",
        long_name="Daily gridded rainfall (IMD 0.25°)",
    )


# ============================================================================
# PUBLIC API — Temperature (Tmax or Tmin)
# ============================================================================
def read_imd_temperature(
    path: Path | str,
    year: int,
    kind: Literal["tmax", "tmin"] = "tmax",
) -> xr.DataArray:
    """
    Read one year of IMD gridded temperature (1.0° resolution).

    Parameters
    ----------
    path : Path to the .GRD file
    year : Year the file represents
    kind : 'tmax' or 'tmin'  (only for metadata labeling)

    Returns
    -------
    xr.DataArray with dims (time, lat, lon), values in °C.
    Shape: (365 or 366, 31, 31). NaN over ocean / outside India.
    """
    long_name = {
        "tmax": "Daily maximum air temperature (IMD 1.0°)",
        "tmin": "Daily minimum air temperature (IMD 1.0°)",
    }[kind]
    return _read_grd(
        path=path,
        year=year,
        nlat=config.IMD_TEMP_NLAT,
        nlon=config.IMD_TEMP_NLON,
        lat_start=config.IMD_TEMP_LAT_START,
        lon_start=config.IMD_TEMP_LON_START,
        resolution=config.IMD_TEMP_RES,
        fill_value=config.IMD_TEMP_FILL,
        variable_name=kind,
        units="degC",
        long_name=long_name,
    )


# ============================================================================
# CONVENIENCE — read multiple years and concatenate
# ============================================================================
def read_imd_rainfall_years(directory: Path | str, years: list[int]) -> xr.DataArray:
    """
    Read multiple years of IMD rainfall from a directory and concatenate along time.

    Expected filename pattern: contains the year (e.g., 'Rainfall_ind2020_rfp25.grd').
    We scan the directory and match files to years.
    """
    directory = Path(directory)
    arrays = []
    for year in years:
        # Try to find a file whose name contains str(year)
        matches = list(directory.glob(f"*{year}*.grd"))
        if not matches:
            matches = list(directory.glob(f"*{year}*.GRD"))
        if not matches:
            print(f"⚠ No IMD rainfall file found for {year} in {directory}. Skipping.")
            continue
        path = matches[0]
        print(f"Reading {path.name} ({year})...")
        arrays.append(read_imd_rainfall(path, year))
    if not arrays:
        raise FileNotFoundError(f"No IMD rainfall files found in {directory}")
    return xr.concat(arrays, dim="time")


def read_imd_temperature_years(
    directory: Path | str,
    years: list[int],
    kind: Literal["tmax", "tmin"] = "tmax",
) -> xr.DataArray:
    """Read multiple years of IMD temperature and concatenate."""
    directory = Path(directory)
    arrays = []
    for year in years:
        matches = list(directory.glob(f"*{year}*.grd")) + list(directory.glob(f"*{year}*.GRD"))
        if not matches:
            print(f"⚠ No IMD {kind} file found for {year} in {directory}. Skipping.")
            continue
        print(f"Reading {matches[0].name} ({year})...")
        arrays.append(read_imd_temperature(matches[0], year, kind=kind))
    if not arrays:
        raise FileNotFoundError(f"No IMD {kind} files found in {directory}")
    return xr.concat(arrays, dim="time")


# ============================================================================
# SANITY CHECK
# ============================================================================
def western_ghats_test(rainfall: xr.DataArray, month: int = 7) -> bool:
    """
    Verify the reading orientation is correct.

    In the given monsoon month (default July), the WEST coast of India should
    have MUCH higher average rainfall than the EAST coast, thanks to the
    Western Ghats orographic effect.

    Returns True if the test passes, False otherwise (and prints a warning).
    """
    da = rainfall.sel(time=rainfall.time.dt.month == month).mean("time")
    # West strip: first 8 columns (~lon 66.5-68.5°E covers the Ghats)
    west_mean = float(da.isel(lon=slice(0, 8)).mean(skipna=True))
    # East strip: last 8 columns
    east_mean = float(da.isel(lon=slice(-8, None)).mean(skipna=True))

    print(f"\nWestern Ghats test (month = {month}):")
    print(f"  West edge mean rainfall: {west_mean:.2f} mm/day")
    print(f"  East edge mean rainfall: {east_mean:.2f} mm/day")
    passed = west_mean > east_mean * 1.5   # west should be MUCH wetter
    print(f"  {'✓ PASS' if passed else '✗ FAIL'} — West should be significantly wetter than East")
    return passed


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mausamsetu.data.imd <path_to_rainfall.grd> <year>")
        print("Example: python -m mausamsetu.data.imd data/raw/imd_rain/Rainfall_ind2020_rfp25.grd 2020")
        sys.exit(1)

    path = sys.argv[1]
    year = int(sys.argv[2])
    print(f"Reading IMD rainfall: {path} (year {year})")
    da = read_imd_rainfall(path, year)
    print("\nShape:", da.shape)
    print("Coord ranges:")
    print(f"  time: {da.time.values[0]} → {da.time.values[-1]}")
    print(f"  lat:  {float(da.lat.min())} → {float(da.lat.max())}")
    print(f"  lon:  {float(da.lon.min())} → {float(da.lon.max())}")
    print(f"Value stats: mean={float(da.mean(skipna=True)):.2f} mm, "
          f"max={float(da.max(skipna=True)):.1f} mm")

    western_ghats_test(da)
