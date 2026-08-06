"""
build_cube.py — Phase 4.

One region-parameterized builder that turns the readers + harmonizer into
a validated NetCDF cube per region, alongside `<region>_norm_stats.json` and
a shared `manifest.yaml`. India and Cauvery share a single code path; Cauvery
is a subset of the India grid clipped by the Subbasin polygon (guarantees the
two are consistent by construction).

Layout produced:
    L:/MausamSetu/data/processed/
        india.nc                (dims: time, lat, lon; vars: rain, tmax, tmin, [insat_lst])
        cauvery.nc              (same vars; subset of india)
        india_norm_stats.json   (per-variable min/max/mean/std fit on TRAIN YEARS ONLY)
        cauvery_norm_stats.json
        manifest.yaml           (source hashes, reader params, regrid methods,
                                 missing policy, variable list, split, git hash)

Chunks (NetCDF encoding): time=100 so date-slicing does not pull the whole cube.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, os, subprocess, sys
from typing import Sequence
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.readers.imd_rainfall_nc import read_rainfall_range
from data.readers.imd_temperature import read_temp_range
from data.readers.insat_lst import read_insat_range
from data import harmonize as HM

IST = timezone(timedelta(hours=5, minutes=30))

REPO       = Path(__file__).resolve().parents[2]        # L:/MausamSetu
DATA_ROOT  = REPO / "data"
PROCESSED  = DATA_ROOT / "processed"
SUBBASIN   = REPO / "Subbasin" / "Subbasin.shp"
PRJ_FILE   = REPO / "Subbasin" / "Subbasin.prj"
RAIN_DIR   = DATA_ROOT / "Rainfall"
TMAX_DIR   = DATA_ROOT / "max_temp"
TMIN_DIR   = DATA_ROOT / "min_temp"
INSAT_DIR  = DATA_ROOT / "INSAT"

# Region registry — same shape as data_source.REGIONS but built off the
# Phase-3 master grid. Cauvery is derived from India via the shapefile.
REGIONS = {
    "india":   {"label": "India (Full)",   "extent": {"lat": (6.5, 38.5), "lon": (66.5, 100.0)}},
    "cauvery": {"label": "Cauvery Basin",  "extent": {"lat": (10.0, 14.5), "lon": (75.5, 79.5)}},
}

# Which files a variable needs (used for the coverage matrix).
_VAR_FILES = {
    "rain":       lambda y: RAIN_DIR / f"RF25_ind{y}_rfp25.nc",
    "tmax":       lambda y: TMAX_DIR / f"Maxtemp_MaxT_{y}.GRD",
    "tmin":       lambda y: TMIN_DIR / f"Mintemp_MinT_{y}.GRD",
    # insat handled separately (per-day files, not per-year)
}


# ─────────────────────────── helpers ───────────────────────────

def _md5(path: Path, cap_bytes: int = 32 * 1024 * 1024) -> str:
    """MD5 of the first `cap_bytes` — fast fingerprint for reproducibility."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(cap_bytes))
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _cauvery_polygon():
    """Reuse the same polygon logic as data_source (LCC→WGS84, 3 Cauvery polys)."""
    import shapefile
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import unary_union, transform as shp_transform
    r = shapefile.Reader(str(SUBBASIN))
    recs = r.records(); shapes = r.shapes()
    idx = [i for i, rec in enumerate(recs) if str(rec["ba_name"]) == "Cauvery Basin"]
    wkt = PRJ_FILE.read_text()
    tr = Transformer.from_crs(wkt, "EPSG:4326", always_xy=True)
    polys = []
    for i in idx:
        g = shape(shapes[i].__geo_interface__)
        g = shp_transform(lambda x, y, z=None: tr.transform(x, y), g)
        polys.append(g)
    return unary_union(polys)


def _basin_mask_for(region: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean (lat,lon) mask: True inside the region polygon. India = all True."""
    if region == "india":
        return np.ones((len(lat), len(lon)), dtype=bool)
    from shapely.geometry import Point
    poly = _cauvery_polygon().buffer(0)
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            mask[i, j] = poly.contains(Point(float(lo), float(la)))
    if not mask.any():
        mask[:] = True
    return mask


# ─────────────────────────── main builder ──────────────────────

def build_cube(region: str, years: tuple[int, int], variables: Sequence[str],
               train_years: tuple[int, int] | None = None,
               apply_missingness: bool = True) -> Path:
    """Build one region's cube + norm-stats. Returns the .nc path.

    Args:
        region      : "india" | "cauvery"
        years       : (start, end) inclusive year range for the cube's TIME axis
        variables   : subset of ["rain","tmax","tmin","insat_lst"] to include
        train_years : (start, end) inclusive — norm stats fit ONLY on these years
                      (no leakage). Defaults to the first 80% of `years`.
    """
    assert region in REGIONS, f"unknown region {region!r}"
    y0, y1 = years
    if train_years is None:
        span = y1 - y0 + 1
        train_years = (y0, y0 + max(1, int(0.8 * span)) - 1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    lat, lon = HM.canonical_grid()
    basin = _basin_mask_for(region, lat, lon)                                          # India = True

    # ── read + harmonize each requested variable on the India master grid ──
    das: dict[str, xr.DataArray] = {}
    if "rain" in variables:
        r = read_rainfall_range(RAIN_DIR, years)
        das["rain"] = HM.rainfall_to_master(r)
    if "tmax" in variables:
        das["tmax"] = HM.regrid_temp_to_master(read_temp_range(TMAX_DIR, years, "tmax"))
    if "tmin" in variables:
        das["tmin"] = HM.regrid_temp_to_master(read_temp_range(TMIN_DIR, years, "tmin"))
    if "insat_lst" in variables:
        from datetime import date
        try:
            insat = read_insat_range(INSAT_DIR, date(y0, 1, 1), date(y1, 12, 31))
            das["insat_lst"] = HM.regrid_insat_conservative(insat)
        except FileNotFoundError:
            print(f"[{region}] INSAT unavailable for {years} — skipping this channel.")

    if not das:
        raise ValueError(f"No variables produced for {region} {years}")

    # ── align on the common time axis (INSAT is a sparse subset) ──
    all_times = sorted(set().union(*(set(d.time.values) for d in das.values())))
    time_axis = np.array(all_times, dtype="datetime64[D]")
    canvas: dict[str, xr.DataArray] = {}
    for name, da in das.items():
        canvas[name] = da.reindex(time=time_axis).transpose("time", "lat", "lon")

    # ── clip to the region polygon (Cauvery = subset of India) ──
    if region != "india":
        for name, da in list(canvas.items()):
            v = da.values.copy()
            v[:, ~basin] = np.nan
            canvas[name] = xr.DataArray(v, dims=da.dims, coords=da.coords,
                                        name=da.name, attrs=da.attrs)

    # ── physical consistency check ──
    physcheck = HM.check_tmax_ge_tmin(canvas["tmax"], canvas["tmin"]) if {"tmax", "tmin"}.issubset(canvas) else {"violations": 0}

    # ── missingness policy (interp isolated ≤3-day gaps, keep sidecar mask) ──
    interp_masks: dict[str, xr.DataArray] = {}
    if apply_missingness:
        for name in ("tmax", "tmin"):
            if name in canvas:
                filled, mask = HM.apply_missingness_policy(canvas[name], max_gap_days=3)
                canvas[name] = filled
                interp_masks[name] = mask

    # ── land mask (True where ANY of the core vars is ever non-NaN inside region) ──
    core = [canvas[k] for k in ("rain", "tmax", "tmin") if k in canvas]
    if core:
        land = np.zeros(basin.shape, dtype=bool)
        for da in core:
            land |= ~np.isnan(da.values).all(axis=0)
        land &= basin
    else:
        land = basin.copy()
    land_da = xr.DataArray(land.astype(np.uint8), dims=("lat", "lon"),
                           coords={"lat": lat, "lon": lon}, name="mask",
                           attrs={"long_name": "1 = valid land inside region",
                                  "region": region})

    # ── norm stats on TRAIN YEARS ONLY (no leakage) ──
    tstart = np.datetime64(f"{train_years[0]}-01-01")
    tend   = np.datetime64(f"{train_years[1]}-12-31")
    norm_stats: dict[str, dict[str, float]] = {}
    for name, da in canvas.items():
        sel = da.sel(time=slice(tstart, tend))
        v = sel.values
        v = v[np.isfinite(v)]
        if v.size == 0:
            norm_stats[name] = {"min": None, "max": None, "mean": None, "std": None, "n": 0}
            continue
        norm_stats[name] = {
            "min":  float(np.nanmin(v)),
            "max":  float(np.nanmax(v)),
            "mean": float(np.nanmean(v)),
            "std":  float(np.nanstd(v)),
            "n":    int(v.size),
        }

    # ── assemble the xr.Dataset ──
    dvars = {name: (("time", "lat", "lon"), da.values, dict(da.attrs))
             for name, da in canvas.items()}
    for name, m in interp_masks.items():
        dvars[m.name] = (("time", "lat", "lon"), m.values.astype(np.uint8), dict(m.attrs))
    dvars["mask"] = (("lat", "lon"), land_da.values, dict(land_da.attrs))

    ds = xr.Dataset(
        dvars,
        coords={"time": time_axis, "lat": lat, "lon": lon},
        attrs={
            "title": f"ClimateTwin Lab cube — {REGIONS[region]['label']}",
            "region": region,
            "region_label": REGIONS[region]["label"],
            "master_grid": "0.25° India (129×135, lat 6.5..38.5, lon 66.5..100)",
            "years": [int(y0), int(y1)],
            "train_years": [int(train_years[0]), int(train_years[1])],
            "variables": list(canvas.keys()),
            "physcheck_tmax_ge_tmin_violations": int(physcheck["violations"]),
            "build_time_ist": datetime.now(IST).isoformat(),
            "git_hash": _git_hash(),
        },
    )

    # ── write NetCDF (chunked so date-slicing is cheap) ──
    out = PROCESSED / f"{region}.nc"
    encoding = {}
    for v in ds.data_vars:
        if ds[v].ndim == 3:
            encoding[v] = {"zlib": True, "complevel": 4,
                           "chunksizes": (min(100, ds.sizes["time"]),
                                          ds.sizes["lat"], ds.sizes["lon"])}
        else:
            encoding[v] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(out, engine="netcdf4", encoding=encoding)

    # ── norm-stats sidecar (readable at inference without opening the cube) ──
    (PROCESSED / f"{region}_norm_stats.json").write_text(json.dumps({
        "region": region,
        "train_years": list(train_years),
        "stats": norm_stats,
    }, indent=2))

    ds.close()
    return out


def write_manifest(regions_built: list[str],
                   years: tuple[int, int],
                   train_years: tuple[int, int],
                   variables: Sequence[str],
                   extra_notes: str = "") -> Path:
    """Write manifest.yaml capturing everything needed to reproduce the cubes."""
    import yaml

    files = {}
    for label, pattern in [("rainfall", "Rainfall/*.nc"),
                           ("tmax",     "max_temp/*.GRD"),
                           ("tmin",     "min_temp/*.GRD"),
                           ("insat",    "INSAT/**/*.h5")]:
        matches = sorted(DATA_ROOT.glob(pattern))
        files[label] = {
            "count":   len(matches),
            "first":   matches[0].name if matches else None,
            "last":    matches[-1].name if matches else None,
            "md5_first": _md5(matches[0]) if matches else None,
            "md5_last":  _md5(matches[-1]) if matches else None,
        }

    manifest = {
        "generator": "climate_twin/data/build_cube.py",
        "generated_at_ist": datetime.now(IST).isoformat(),
        "git_hash": _git_hash(),
        "master_grid": {
            "extent_lat": [HM.MASTER_LAT0, HM.MASTER_LAT1],
            "extent_lon": [HM.MASTER_LON0, HM.MASTER_LON1],
            "n_lat": HM.MASTER_NLAT, "n_lon": HM.MASTER_NLON,
            "resolution_deg": 0.25,
        },
        "regions_built": regions_built,
        "years": list(years),
        "train_years": list(train_years),
        "variables": list(variables),
        "readers": {
            "rainfall": {"module": "data.readers.imd_rainfall_nc",
                         "format": "NetCDF-4", "dtype": "float32", "fill": "NaN (built-in)"},
            "tmax":     {"module": "data.readers.imd_temperature",
                         "format": "IMD .GRD", "dtype": "float32", "shape": "days×31×31",
                         "sentinel": 99.9, "unit": "degC"},
            "tmin":     {"module": "data.readers.imd_temperature",
                         "format": "IMD .GRD", "dtype": "float32", "shape": "days×31×31",
                         "sentinel": 99.9, "unit": "degC"},
            "insat_lst": {"module": "data.readers.insat_lst",
                          "format": "HDF5 curvilinear 974×1067", "dtype": "float32",
                          "sentinel": -999.0, "unit_native": "Kelvin",
                          "cadence_native": "daily (already averaged per file description)"},
        },
        "regrid": {
            "rain": "native (0.25°, no regrid)",
            "tmax": "bilinear 1°→0.25°",
            "tmin": "bilinear 1°→0.25°",
            "insat_lst": "conservative area-average curvilinear→0.25°",
        },
        "missing_policy": {
            "interp_gap_days_max": 3,
            "gap_gt_max": "left NaN",
            "cell_missing_gt_20pct_of_window": "reported (not zero-filled)",
            "recorded_as_sidecar": ["tmax_is_interpolated", "tmin_is_interpolated"],
        },
        "sources": files,
        "outputs": {
            "cubes": [str((PROCESSED / f"{r}.nc").as_posix()) for r in regions_built],
            "norm_stats": [str((PROCESSED / f"{r}_norm_stats.json").as_posix()) for r in regions_built],
        },
        "notes": extra_notes,
    }

    out = PROCESSED / "manifest.yaml"
    out.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    # short signature line that data_source can cache-bust on:
    sig = hashlib.md5(out.read_bytes()).hexdigest()[:12]
    (PROCESSED / "manifest.sig").write_text(sig)
    return out


# ─────────────────────────── validation ────────────────────────

def validate_cube(region: str) -> dict:
    """Load the written cube and re-assert every rule from the spec."""
    path = PROCESSED / f"{region}.nc"
    ds = xr.open_dataset(path)
    report = {"path": str(path), "size_mb": round(path.stat().st_size / 1e6, 2)}
    # dims
    report["dims"] = dict(ds.sizes)
    report["variables"] = list(ds.data_vars)
    # no sentinels
    for v in ("rain", "tmax", "tmin"):
        if v not in ds.data_vars:
            continue
        a = ds[v].values
        survivors = a[np.isfinite(a)]
        assert not np.any(survivors <= -998.0), f"{v}: -999 survived"
        if v in ("tmax", "tmin"):
            assert not np.any(survivors >= 90.0), f"{v}: 99.9 survived"
    # tmax >= tmin
    if {"tmax", "tmin"}.issubset(ds.data_vars):
        a, b = ds["tmax"].values, ds["tmin"].values
        both = np.isfinite(a) & np.isfinite(b)
        report["tmax_ge_tmin_violations"] = int(((a < b) & both).sum())
    # time monotonic + no dupes
    t = ds.time.values
    report["time_monotonic"] = bool(np.all(np.diff(t) > np.timedelta64(0)))
    report["time_duplicates"] = int(len(t) - len(np.unique(t)))
    # per-variable attrs present
    missing_attrs = {}
    for v in ds.data_vars:
        need = {"units", "long_name", "source"} - set(ds[v].attrs)
        if need:
            missing_attrs[v] = sorted(need)
    report["missing_attrs"] = missing_attrs
    # region tag / provenance
    report["attrs"] = {k: ds.attrs[k] for k in ds.attrs if k in
                       ("region", "region_label", "years", "train_years",
                        "variables", "physcheck_tmax_ge_tmin_violations",
                        "build_time_ist", "git_hash")}
    ds.close()
    return report


def validate_cauvery_subset_of_india(sample_time: str | None = None) -> dict:
    """Spot-check: same day + same cell values in Cauvery cube match india cube."""
    ip = PROCESSED / "india.nc"; cp = PROCESSED / "cauvery.nc"
    if not (ip.exists() and cp.exists()):
        return {"skipped": True, "reason": "one of the cubes not built"}
    di = xr.open_dataset(ip); dc = xr.open_dataset(cp)
    t = sample_time or str(di.time.values[len(di.time) // 2])[:10]
    v = next(iter(v for v in di.data_vars if v in dc.data_vars and v != "mask"))
    a = di[v].sel(time=t).values
    b = dc[v].sel(time=t).values
    # Cauvery cells (finite in b) should match India at the same (lat,lon).
    common = np.isfinite(b)
    diff = np.abs(a[common] - b[common])
    result = {
        "sample_time": t, "sample_var": v,
        "cauvery_cells_checked": int(common.sum()),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "identical": bool(np.allclose(a[common], b[common], equal_nan=True)),
    }
    di.close(); dc.close()
    return result


# ─────────────────────────── entry point ───────────────────────

def build_all(years: tuple[int, int] = (2018, 2025),
              train_years: tuple[int, int] = (2018, 2023),
              variables: Sequence[str] = ("rain", "tmax", "tmin", "insat_lst")) -> dict:
    print(f"[build_all] years={years}  train_years={train_years}  vars={list(variables)}")
    for region in ("india", "cauvery"):
        print(f"\n=== building {region} ===")
        p = build_cube(region, years, variables, train_years=train_years)
        rep = validate_cube(region)
        print(f"[{region}] wrote {p}  {rep['size_mb']} MB  dims={rep['dims']}  vars={rep['variables']}")
        if "tmax_ge_tmin_violations" in rep:
            print(f"[{region}] tmax≥tmin violations: {rep['tmax_ge_tmin_violations']}")
        print(f"[{region}] time monotonic: {rep['time_monotonic']}  duplicates: {rep['time_duplicates']}")

    mf = write_manifest(regions_built=["india", "cauvery"], years=years,
                        train_years=train_years, variables=variables)
    print(f"\n[manifest] {mf}")

    subset = validate_cauvery_subset_of_india()
    print(f"[cauvery⊂india spot-check] {subset}")

    return {"cubes_built": ["india", "cauvery"], "manifest": str(mf), "subset_check": subset}


if __name__ == "__main__":
    build_all()
