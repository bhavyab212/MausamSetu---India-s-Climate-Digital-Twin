"""
data_source.py — SINGLE SOURCE OF TRUTH for all climate data in the cloned app.

All data access routes through this module. It reads ONLY the user's local raw
IMD files and derives every region from them:

  - India   : the raw 0.25° rainfall grid (129 x 135) + 1° temperature regridded.
  - Cauvery : the same raw grids, clipped to the Cauvery basin polygon taken from
              Subbasin/Subbasin.shp (reprojected from Lambert Conformal Conic to
              WGS84) and subset to the basin bounding box (19 x 17).

No HuggingFace / cloud downloads. No pre-existing processed cubes are read; the
processed .nc files written here are just a rebuild cache derived from raw.

Raw layout (verified on disk):
  data/IMD rainfall data/Rainfall_indYYYY_rfp25.grd  float32 (days,129,135) fill -999.0
  data/IMD max temp data/Maxtemp_MaxT_YYYY.GRD       float32 (days,31,31)   fill 99.9
  data/IMD min temp data/Mintemp_MinT_YYYY.GRD       float32 (days,31,31)   fill 99.9
"""
from __future__ import annotations
from pathlib import Path
import glob
import re
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# --- optional Streamlit caches (no-op when imported outside a Streamlit run) ---
try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:  # pragma: no cover
    def _noop(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco
    cache_resource = _noop
    cache_data = _noop


# ============================================================================
# PATHS
# ============================================================================
REPO = Path(__file__).resolve().parent          # climate_twin/
PROJECT_ROOT = REPO.parent                        # L:\MausamSetu

RAW_RAIN_DIR = PROJECT_ROOT / "data" / "IMD rainfall data"
RAW_TMAX_DIR = PROJECT_ROOT / "data" / "IMD max temp data"
RAW_TMIN_DIR = PROJECT_ROOT / "data" / "IMD min temp data"
SUBBASIN_SHP = PROJECT_ROOT / "Subbasin" / "Subbasin.shp"
SUBBASIN_PRJ = PROJECT_ROOT / "Subbasin" / "Subbasin.prj"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================================
# RAW IMD GRID GEOMETRY (from IMD documentation, verified against files)
# ============================================================================
RAIN_NLAT, RAIN_NLON = 129, 135
RAIN_LAT0, RAIN_LAT1 = 6.5, 38.5     # °N, index 0 = south
RAIN_LON0, RAIN_LON1 = 66.5, 100.0   # °E, index 0 = west
RAIN_FILL = -999.0

TEMP_NLAT, TEMP_NLON = 31, 31
TEMP_LAT0, TEMP_LAT1 = 7.5, 37.5
TEMP_LON0, TEMP_LON1 = 67.5, 97.5
TEMP_FILL_MIN = 90.0                  # values > 90 are IMD fill (99.9)

DEFAULT_START_YEAR = 1975
DEFAULT_END_YEAR = 2025

# master (India) coordinate axes at 0.25°
RAIN_LAT = np.round(np.linspace(RAIN_LAT0, RAIN_LAT1, RAIN_NLAT), 4)
RAIN_LON = np.round(np.linspace(RAIN_LON0, RAIN_LON1, RAIN_NLON), 4)


# ============================================================================
# REGION REGISTRY  —  India vs Cauvery differ only in these entries.
# ============================================================================
REGIONS = {
    "india": {
        "label": "India (Full)",
        "raw_dir": RAW_RAIN_DIR,
        "extent": {"lat": (6.5, 38.5), "lon": (66.5, 100.0)},
        "rain_res_deg": 0.25,
        "temp_res_deg": 1.0,
        "expected_shape": (129, 135),
        "processed_nc": PROCESSED_DIR / "india.nc",
        "ckpt_prefix": "india",
    },
    "cauvery": {
        "label": "Cauvery Basin",
        "raw_dir": PROJECT_ROOT / "Subbasin",
        "extent": {"lat": (10.0, 14.5), "lon": (75.5, 79.5)},
        "rain_res_deg": 0.25,
        "temp_res_deg": 1.0,
        "expected_shape": (19, 17),
        "processed_nc": PROCESSED_DIR / "cauvery.nc",
        "ckpt_prefix": "cauvery",
    },
}


def region_grid(region: str):
    """Return (lat_axis, lon_axis) 1-D arrays for a region, subset from the master grid."""
    ext = REGIONS[region]["extent"]
    lat0, lat1 = ext["lat"]
    lon0, lon1 = ext["lon"]
    lat_idx = np.where((RAIN_LAT >= lat0 - 1e-6) & (RAIN_LAT <= lat1 + 1e-6))[0]
    lon_idx = np.where((RAIN_LON >= lon0 - 1e-6) & (RAIN_LON <= lon1 + 1e-6))[0]
    return RAIN_LAT[lat_idx], RAIN_LON[lon_idx], lat_idx, lon_idx


# ============================================================================
# RAW FILE DISCOVERY
# ============================================================================
def _year_file_map(directory: Path, prefix: str, ext: str) -> dict[int, Path]:
    """Map {year: path} for files like <prefix>YYYY<...>.<ext>, skipping ' (1)' dupes."""
    out: dict[int, Path] = {}
    for p in sorted(glob.glob(str(directory / f"*.{ext}"))):
        name = Path(p).name
        m = re.search(r"(19|20)\d{2}", name)
        if not m:
            continue
        year = int(m.group(0))
        # prefer the canonical file over a "(1)" duplicate
        if year in out and "(1)" in name:
            continue
        if year in out and "(1)" not in Path(out[year]).name:
            continue
        out[year] = Path(p)
    return out


def rain_year_files() -> dict[int, Path]:
    return _year_file_map(RAW_RAIN_DIR, "Rainfall_ind", "grd")


def tmax_year_files() -> dict[int, Path]:
    return _year_file_map(RAW_TMAX_DIR, "Maxtemp_MaxT_", "GRD")


def tmin_year_files() -> dict[int, Path]:
    return _year_file_map(RAW_TMIN_DIR, "Mintemp_MinT_", "GRD")


# ============================================================================
# RAW READERS  (fill values -> NaN)
# ============================================================================
def read_rain_grd(path: Path) -> np.ndarray:
    """(days, 129, 135) float64, IMD fill -999 -> NaN."""
    raw = np.fromfile(str(path), dtype=np.float32)
    days = raw.size // (RAIN_NLAT * RAIN_NLON)
    g = raw[: days * RAIN_NLAT * RAIN_NLON].reshape(days, RAIN_NLAT, RAIN_NLON).astype(np.float64)
    g[g <= RAIN_FILL + 1e-3] = np.nan
    return g


def read_temp_grd(path: Path) -> np.ndarray:
    """(days, 31, 31) float64, IMD fill 99.9 -> NaN."""
    raw = np.fromfile(str(path), dtype=np.float32)
    days = raw.size // (TEMP_NLAT * TEMP_NLON)
    g = raw[: days * TEMP_NLAT * TEMP_NLON].reshape(days, TEMP_NLAT, TEMP_NLON).astype(np.float64)
    g[g > TEMP_FILL_MIN] = np.nan
    return g


def _regrid_temp(day_31: np.ndarray, tgt_lat: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    """Bilinear regrid a single 31x31 (1°) temperature field onto a target 0.25° grid."""
    lat_src = np.linspace(TEMP_LAT0, TEMP_LAT1, TEMP_NLAT)
    lon_src = np.linspace(TEMP_LON0, TEMP_LON1, TEMP_NLON)
    interp = RegularGridInterpolator(
        (lat_src, lon_src), day_31, bounds_error=False, fill_value=np.nan
    )
    la, lo = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
    pts = np.stack([la.ravel(), lo.ravel()], axis=1)
    return interp(pts).reshape(len(tgt_lat), len(tgt_lon))


# ============================================================================
# BASIN MASK  (Cauvery polygon from the Subbasin shapefile)
# ============================================================================
@cache_resource(show_spinner=False)
def cauvery_polygon():
    """Reproject the 3 Cauvery sub-basin polygons to WGS84 and dissolve into one shape."""
    import shapefile
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import unary_union, transform as shp_transform

    r = shapefile.Reader(str(SUBBASIN_SHP))
    recs = r.records()
    shapes = r.shapes()
    idx = [i for i, rec in enumerate(recs) if str(rec["ba_name"]) == "Cauvery Basin"]
    wkt = SUBBASIN_PRJ.read_text()
    tr = Transformer.from_crs(wkt, "EPSG:4326", always_xy=True)
    polys = []
    for i in idx:
        g = shape(shapes[i].__geo_interface__)
        g = shp_transform(lambda x, y, z=None: tr.transform(x, y), g)
        polys.append(g)
    return unary_union(polys)


def _basin_mask(region: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean (len(lat), len(lon)) — True inside the region polygon. India = all True."""
    if region != "cauvery":
        return np.ones((len(lat), len(lon)), dtype=bool)
    from shapely.geometry import Point
    poly = cauvery_polygon()
    prep = poly.buffer(0)  # heal any invalid geometry
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            mask[i, j] = prep.contains(Point(float(lo), float(la)))
    # if the coarse 0.25° grid misses thin basins, fall back to bbox so we never
    # return an all-empty mask
    if not mask.any():
        mask[:] = True
    return mask


# ============================================================================
# CUBE BUILDER  —  annual aggregates (rain sum, temp max) per year + land mask
# ============================================================================
def build_cube(region: str, start_year: int = DEFAULT_START_YEAR,
               end_year: int = DEFAULT_END_YEAR, verbose: bool = True) -> xr.Dataset:
    """Build the annual aggregate cube for a region straight from raw IMD grids."""
    lat, lon, lat_idx, lon_idx = region_grid(region)
    rain_files = rain_year_files()
    tmax_files = tmax_year_files()
    tmin_files = tmin_year_files()

    years = [y for y in range(start_year, end_year + 1)
             if y in rain_files and y in tmax_files]
    if not years:
        raise RuntimeError(f"No overlapping rain/temp years for {region} in {start_year}-{end_year}")

    n = len(years)
    H, W = len(lat), len(lon)
    rain = np.zeros((n, H, W), dtype=np.float32)
    tmax = np.zeros((n, H, W), dtype=np.float32)
    tmin = np.full((n, H, W), np.nan, dtype=np.float32)
    land = None

    for k, y in enumerate(years):
        if verbose:
            print(f"[{region}] {y} ({k + 1}/{n})")
        # ---- rainfall: annual cumulative ----
        rg = read_rain_grd(rain_files[y])                       # (days,129,135)
        rg = rg[:, lat_idx][:, :, lon_idx]                      # subset to region
        if land is None:
            land = ~np.isnan(rg[0])                             # land where day0 is valid
        rain[k] = np.nansum(rg, axis=0)

        # ---- tmax: annual maximum, regridded 1°->0.25° ----
        tg = read_temp_grd(tmax_files[y])                       # (days,31,31)
        tg_annual = np.nanmax(tg, axis=0)                       # (31,31)
        tmax[k] = _regrid_temp(tg_annual, lat, lon)

        # ---- tmin: annual minimum (optional; not all years present) ----
        if y in tmin_files:
            tng = read_temp_grd(tmin_files[y])
            tng_annual = np.nanmin(tng, axis=0)
            tmin[k] = _regrid_temp(tng_annual, lat, lon)

    basin = _basin_mask(region, lat, lon)
    mask = (land & basin).astype(np.float32)

    # oceans/outside-basin -> 0 for rain/temp (matches the app's convention)
    rain = np.where(mask[None] == 1, np.nan_to_num(rain, nan=0.0), 0.0).astype(np.float32)
    tmax = np.where(mask[None] == 1, tmax, 0.0)
    # fill residual land NaNs with per-year land mean
    for k in range(n):
        lm = mask == 1
        if lm.any():
            tv = tmax[k][lm]
            fill = np.nanmean(tv) if np.isfinite(np.nanmean(tv)) else 0.0
            tmax[k] = np.where(np.isnan(tmax[k]), fill, tmax[k])
    tmax = tmax.astype(np.float32)

    ds = xr.Dataset(
        {
            "rain": (("year", "lat", "lon"), rain),
            "tmax": (("year", "lat", "lon"), tmax),
            "tmin": (("year", "lat", "lon"), tmin),
            "mask": (("lat", "lon"), mask),
        },
        coords={"year": np.array(years), "lat": lat, "lon": lon},
        attrs={
            "region": region,
            "label": REGIONS[region]["label"],
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
            "source": "IMD raw .grd (rebuilt by data_source.build_cube)",
        },
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(REGIONS[region]["processed_nc"])
    if verbose:
        print(f"[{region}] wrote {REGIONS[region]['processed_nc']}  dims={dict(ds.sizes)}")
    return ds


# ============================================================================
# PUBLIC API
# ============================================================================
@cache_resource(show_spinner="Building climate cube from raw IMD data...")
def load_region(region: str) -> xr.Dataset:
    """Load the processed cube for a region, building it from raw grids if absent."""
    if region not in REGIONS:
        raise KeyError(f"Unknown region '{region}'. Known: {list(REGIONS)}")
    nc = REGIONS[region]["processed_nc"]
    exp = REGIONS[region]["expected_shape"]
    if nc.exists():
        ds = xr.open_dataset(nc)
        if (ds.sizes.get("lat"), ds.sizes.get("lon")) == exp:
            return ds
        ds.close()  # shape mismatch -> rebuild
    return build_cube(region)


@cache_data(show_spinner=False)
def load_aggregates(region: str):
    """Return (rain, temp, mask, years) — the arrays the app's load_artifacts needs.

    rain : (n_years, H, W) annual cumulative rainfall
    temp : (n_years, H, W) annual maximum temperature
    mask : (H, W) 1=valid land inside region, 0 otherwise
    years: (n_years,) int
    """
    ds = load_region(region)
    return (
        np.asarray(ds["rain"].values, dtype=np.float32),
        np.asarray(ds["tmax"].values, dtype=np.float64),
        np.asarray(ds["mask"].values, dtype=np.float32),
        np.asarray(ds["year"].values, dtype=int),
    )


@cache_data(show_spinner=False)
def daily_rain(region: str, year: int) -> np.ndarray | None:
    """(days, H, W) daily rainfall for a region/year from raw .grd, region-clipped."""
    files = rain_year_files()
    if year not in files:
        return None
    _, _, lat_idx, lon_idx = region_grid(region)
    rg = read_rain_grd(files[year])[:, lat_idx][:, :, lon_idx]
    lat, lon, _, _ = region_grid(region)
    basin = _basin_mask(region, lat, lon)
    rg = np.where(basin[None], np.nan_to_num(rg, nan=0.0), np.nan)
    return rg


@cache_data(show_spinner=False)
def daily_tmax(region: str, year: int) -> np.ndarray | None:
    """(days, H, W) daily max temperature for a region/year, regridded + region-clipped."""
    files = tmax_year_files()
    if year not in files:
        return None
    lat, lon, _, _ = region_grid(region)
    tg = read_temp_grd(files[year])                    # (days,31,31)
    basin = _basin_mask(region, lat, lon)
    out = np.empty((tg.shape[0], len(lat), len(lon)), dtype=np.float32)
    for d in range(tg.shape[0]):
        out[d] = _regrid_temp(tg[d], lat, lon)
    out = np.where(basin[None], out, np.nan)
    return out


@cache_data(show_spinner=False)
def region_info(region: str) -> dict:
    """Summary for the UI: dims, extent, year coverage, missing %, checkpoint prefix."""
    ds = load_region(region)
    mask = ds["mask"].values
    rain = ds["rain"].values
    valid = int((mask == 1).sum())
    total = int(mask.size)
    ext = REGIONS[region]["extent"]
    return {
        "region": region,
        "label": REGIONS[region]["label"],
        "shape": (int(ds.sizes["lat"]), int(ds.sizes["lon"])),
        "extent": ext,
        "years": (int(ds["year"].values.min()), int(ds["year"].values.max())),
        "n_years": int(ds.sizes["year"]),
        "valid_cells": valid,
        "total_cells": total,
        "land_pct": round(100.0 * valid / total, 1) if total else 0.0,
        "processed_nc": str(REGIONS[region]["processed_nc"]),
        "nc_exists": REGIONS[region]["processed_nc"].exists(),
        "ckpt_prefix": REGIONS[region]["ckpt_prefix"],
    }


def region_matches_shape(region: str, shape: tuple[int, int]) -> bool:
    """True if a (lat, lon) shape matches the region's expected grid (for checkpoint guards)."""
    return tuple(shape) == tuple(REGIONS[region]["expected_shape"])


if __name__ == "__main__":
    import sys
    regs = sys.argv[1:] or ["india", "cauvery"]
    for rg in regs:
        ds = build_cube(rg)
        print(region_info(rg))

