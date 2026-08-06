"""
data_source.py — single access seam for the ClimateTwin Lab app.

Post-Phase-4 rewrite: every consumer reads through this module. It loads the
processed cubes built by ``climate_twin.data.build_cube`` and NEVER rebuilds
from raw silently. If the cubes are missing or their manifest signature
disagrees with the last-seen signature, all Streamlit caches are cleared so
stale data can never leak across a rebuild.

Cubes (produced by ``python -m climate_twin.data.build_cube``):

    data/processed/india.nc      dims (time, lat, lon)  vars rain,tmax,tmin,insat_lst,*_is_interpolated,mask
    data/processed/cauvery.nc    same grid, NaN outside the Cauvery polygon
    data/processed/<region>_norm_stats.json   train-years-only normalisation
    data/processed/manifest.yaml              readers, sources, md5s, git hash
    data/processed/manifest.sig               12-hex signature for cache busting

Contract:
  - India and Cauvery share ONE 0.25° master grid (129 × 135). Cauvery = India
    NaN-masked outside the basin. This is a hard subset guarantee.
  - Normalisation stats were fit on TRAIN YEARS ONLY (no leakage).
  - Missing values are NaN. No zero-fill. Callers must respect the mask.
  - If ``manifest.sig`` differs from what we last saw, caches are wiped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Optional Streamlit caches (no-op when imported outside a Streamlit run)
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _HAS_ST = True
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:  # pragma: no cover
    _HAS_ST = False
    def _noop(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco
    cache_resource = _noop
    cache_data = _noop


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent            # climate_twin/
PROJECT_ROOT = REPO.parent                        # L:\MausamSetu

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_YAML = PROCESSED_DIR / "manifest.yaml"
MANIFEST_SIG = PROCESSED_DIR / "manifest.sig"

# raw dirs are still exposed so tests / diagnostics can point at them, but
# the app itself never reads raw grids through data_source anymore.
RAW_RAIN_DIR = PROJECT_ROOT / "data" / "Rainfall"
RAW_TMAX_DIR = PROJECT_ROOT / "data" / "max_temp"
RAW_TMIN_DIR = PROJECT_ROOT / "data" / "min_temp"
RAW_INSAT_DIR = PROJECT_ROOT / "data" / "INSAT"
SUBBASIN_SHP = PROJECT_ROOT / "Subbasin" / "Subbasin.shp"


# ---------------------------------------------------------------------------
# Master grid (canonical 0.25° India — verified against cubes at load time)
# ---------------------------------------------------------------------------
MASTER_NLAT = 129
MASTER_NLON = 135
MASTER_LAT0, MASTER_LAT1 = 6.5, 38.5
MASTER_LON0, MASTER_LON1 = 66.5, 100.0
MASTER_LAT = np.round(np.linspace(MASTER_LAT0, MASTER_LAT1, MASTER_NLAT), 4)
MASTER_LON = np.round(np.linspace(MASTER_LON0, MASTER_LON1, MASTER_NLON), 4)

# legacy aliases — old code imports DS.RAIN_LAT / DS.RAIN_LON
RAIN_LAT = MASTER_LAT
RAIN_LON = MASTER_LON

DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2025


# ---------------------------------------------------------------------------
# Region registry
# ---------------------------------------------------------------------------
REGIONS: dict[str, dict[str, Any]] = {
    "india": {
        "label": "India (Full)",
        "extent": {"lat": (MASTER_LAT0, MASTER_LAT1), "lon": (MASTER_LON0, MASTER_LON1)},
        "expected_shape": (MASTER_NLAT, MASTER_NLON),
        "processed_nc": PROCESSED_DIR / "india.nc",
        "norm_stats_json": PROCESSED_DIR / "india_norm_stats.json",
        "ckpt_prefix": "india",
    },
    "cauvery": {
        "label": "Cauvery Basin",
        # Cauvery cube is on the FULL India grid with NaN outside the basin.
        # The old (19, 17) shape belonged to a pre-Phase-4 pipeline — archived.
        "extent": {"lat": (MASTER_LAT0, MASTER_LAT1), "lon": (MASTER_LON0, MASTER_LON1)},
        "expected_shape": (MASTER_NLAT, MASTER_NLON),
        "processed_nc": PROCESSED_DIR / "cauvery.nc",
        "norm_stats_json": PROCESSED_DIR / "cauvery_norm_stats.json",
        "ckpt_prefix": "cauvery",
    },
}


# ---------------------------------------------------------------------------
# Manifest handling + cache-bust on signature change
# ---------------------------------------------------------------------------
class ProcessedDataMissing(FileNotFoundError):
    """Raised when the processed cubes required by the app are not on disk."""


def _read_sig_from_disk() -> str | None:
    if not MANIFEST_SIG.exists():
        return None
    try:
        return MANIFEST_SIG.read_text().strip() or None
    except Exception:
        return None


def manifest_sig() -> str | None:
    """Return the 12-hex signature written by build_cube.build_all()."""
    return _read_sig_from_disk()


def manifest_dict() -> dict:
    """Return the parsed manifest.yaml (empty dict if missing/unparseable)."""
    if not MANIFEST_YAML.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(MANIFEST_YAML.read_text()) or {}
    except Exception:
        return {}


def _bust_streamlit_caches_if_sig_changed() -> str | None:
    """If the on-disk manifest.sig differs from the last-seen value, wipe
    Streamlit's data/resource caches so no stale cube can be served.

    Called once per session at import time and again at the top of every
    public loader (cheap — just a stat + text read)."""
    sig = _read_sig_from_disk()
    if not _HAS_ST:
        return sig
    try:
        prev = st.session_state.get("_climate_twin_manifest_sig")
    except Exception:
        prev = None
    if prev != sig:
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        try:
            st.session_state["_climate_twin_manifest_sig"] = sig
        except Exception:
            pass
    return sig


# fire once at import so a stale-cache scenario is caught before the first read
_bust_streamlit_caches_if_sig_changed()


# ---------------------------------------------------------------------------
# Region grid helpers (both regions share the master grid)
# ---------------------------------------------------------------------------
def region_grid(region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lat_axis, lon_axis, lat_idx, lon_idx)`` for a region.

    Post-Phase-4 both regions live on the same master grid, so the index
    arrays are always ``arange(NLAT/NLON)``. Kept as a tuple for API
    compatibility with the callers that pattern-match the old signature.
    """
    if region not in REGIONS:
        raise KeyError(f"Unknown region '{region}'. Known: {list(REGIONS)}")
    lat_idx = np.arange(MASTER_NLAT)
    lon_idx = np.arange(MASTER_NLON)
    return MASTER_LAT, MASTER_LON, lat_idx, lon_idx


def region_matches_shape(region: str, shape: tuple[int, int]) -> bool:
    """True iff the ``(lat, lon)`` shape matches this region's expected grid.

    This is the load-time gate: a checkpoint saved on a different-shape grid
    (e.g. the archived (19, 17) Cauvery models) is rejected here.
    """
    return tuple(shape) == tuple(REGIONS[region]["expected_shape"])


# ---------------------------------------------------------------------------
# Cube loaders  (all cache-keyed by the manifest signature)
# ---------------------------------------------------------------------------
def _ensure_processed_present() -> None:
    """Fail fast with actionable advice when the pipeline hasn't been run."""
    missing = [str(REGIONS[r]["processed_nc"])
               for r in REGIONS if not REGIONS[r]["processed_nc"].exists()]
    if missing or not MANIFEST_SIG.exists():
        raise ProcessedDataMissing(
            "Processed cubes are missing. Build them from raw with:\n"
            "    python -m climate_twin.data.build_cube\n"
            f"Missing files: {missing or [str(MANIFEST_SIG)]}"
        )


def _validate_cube_geometry(ds: xr.Dataset, region: str) -> None:
    """Assert the cube matches the master grid before serving it."""
    exp = REGIONS[region]["expected_shape"]
    got = (int(ds.sizes.get("lat", 0)), int(ds.sizes.get("lon", 0)))
    if got != exp:
        raise RuntimeError(
            f"Cube shape mismatch for {region}: expected {exp}, got {got}. "
            "Rebuild with `python -m climate_twin.data.build_cube`."
        )
    lat0, lat1 = float(ds.lat.values[0]), float(ds.lat.values[-1])
    lon0, lon1 = float(ds.lon.values[0]), float(ds.lon.values[-1])
    if not (abs(lat0 - MASTER_LAT0) < 1e-3 and abs(lat1 - MASTER_LAT1) < 1e-3
            and abs(lon0 - MASTER_LON0) < 1e-3 and abs(lon1 - MASTER_LON1) < 1e-3):
        raise RuntimeError(
            f"Cube for {region} is not on the master grid "
            f"(lat {lat0}..{lat1}, lon {lon0}..{lon1}). Rebuild required."
        )


@cache_resource(show_spinner="Loading processed cube…")
def _load_region_cached(region: str, sig: str | None) -> xr.Dataset:
    """The actual cube open — signature is part of the cache key so a rebuild
    forces a re-open on the next call even if Streamlit didn't clear."""
    _ensure_processed_present()
    if region not in REGIONS:
        raise KeyError(f"Unknown region '{region}'. Known: {list(REGIONS)}")
    ds = xr.open_dataset(REGIONS[region]["processed_nc"])
    _validate_cube_geometry(ds, region)
    return ds


def load_region(region: str) -> xr.Dataset:
    """Load the daily processed cube for ``region``.

    Raises ``ProcessedDataMissing`` when the pipeline hasn't been run.
    """
    sig = _bust_streamlit_caches_if_sig_changed()
    return _load_region_cached(region, sig)


def load_daily(region: str, variables: list[str] | None = None) -> xr.Dataset:
    """Return the daily cube, optionally sliced to a variable list.

    Convenient entry point for the new 4-channel training path: pass
    ``variables=["rain", "tmax", "tmin", "insat_lst"]`` and get a Dataset
    exposing only those.
    """
    ds = load_region(region)
    if variables:
        keep = [v for v in variables if v in ds.data_vars]
        return ds[keep + [v for v in ("mask",) if v in ds.data_vars]]
    return ds


@cache_data(show_spinner=False)
def load_norm_stats(region: str, sig: str | None = None) -> dict:
    """Return the train-years-only normalisation stats for a region."""
    _bust_streamlit_caches_if_sig_changed()
    p = REGIONS[region]["norm_stats_json"]
    if not p.exists():
        raise ProcessedDataMissing(f"Missing norm stats: {p}")
    return json.loads(p.read_text())


@cache_data(show_spinner=False)
def list_available_variables(region: str, sig: str | None = None) -> list[str]:
    """Return the variable names that are present in the cube (excluding mask
    and *_is_interpolated sidecars)."""
    ds = load_region(region)
    core = [v for v in ds.data_vars
            if v != "mask" and not v.endswith("_is_interpolated")]
    return core


@cache_data(show_spinner=False)
def coverage_matrix(region: str, sig: str | None = None) -> dict[str, dict[int, float]]:
    """For each core variable, return ``{year: fraction_of_days_with_any_valid_cell}``.

    Used by the coverage-aware ``rounds.yaml`` generator: a variable is only
    included in a round's variable list if its coverage in that round's year
    range exceeds a threshold.
    """
    ds = load_region(region)
    core = list_available_variables(region, sig=sig)
    years = np.asarray(ds["time.year"].values)
    out: dict[str, dict[int, float]] = {}
    for v in core:
        arr = ds[v].values  # (T, H, W)
        # per-day: is there ANY non-NaN cell?
        per_day = np.isfinite(arr).any(axis=(1, 2))
        y_out: dict[int, float] = {}
        for y in np.unique(years):
            m = years == y
            y_out[int(y)] = float(per_day[m].mean()) if m.any() else 0.0
        out[v] = y_out
    return out


# ---------------------------------------------------------------------------
# Legacy annual-aggregate API (kept so display code doesn't break)
# ---------------------------------------------------------------------------
@cache_data(show_spinner=False)
def _annual_aggregates(region: str, sig: str | None = None):
    """Aggregate the daily cube to (n_years, H, W) rain-sum + tmax-max.

    This is the shape the *old* callers of load_aggregates() expect. New code
    should use ``load_region()`` / ``load_daily()`` directly.
    """
    ds = load_region(region)
    years = np.asarray(ds["time.year"].values)
    uniq = np.unique(years)
    H, W = ds.sizes["lat"], ds.sizes["lon"]

    rain = np.zeros((len(uniq), H, W), dtype=np.float32)
    tmax = np.full((len(uniq), H, W), np.nan, dtype=np.float64)
    for k, y in enumerate(uniq):
        sel = years == y
        r = ds["rain"].values[sel]
        t = ds["tmax"].values[sel]
        # rain: annual sum, treating NaN outside basin/land as 0 for the sum
        rain[k] = np.nansum(r, axis=0).astype(np.float32)
        # tmax: annual max ignoring NaNs; keep NaN if the whole column is NaN
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                tmax[k] = np.where(np.all(np.isnan(t), axis=0), np.nan, np.nanmax(t, axis=0))
    mask = ds["mask"].values.astype(np.float32)
    return rain, tmax, mask, uniq.astype(int)


def load_aggregates(region: str):
    """Return ``(rain, temp, mask, years)`` for legacy annual-training callers.

    - ``rain``  : ``(n_years, H, W)`` annual cumulative rainfall (mm)
    - ``temp``  : ``(n_years, H, W)`` annual maximum temperature (°C)
    - ``mask``  : ``(H, W)`` 1 = valid land inside region, 0 outside
    - ``years`` : ``(n_years,)`` int

    Aggregated from the daily cube. Prefer ``load_daily()`` for new code.
    """
    sig = _bust_streamlit_caches_if_sig_changed()
    return _annual_aggregates(region, sig=sig)


@cache_data(show_spinner=False)
def _daily_slice(region: str, year: int, var: str, sig: str | None = None) -> np.ndarray | None:
    ds = load_region(region)
    if var not in ds.data_vars:
        return None
    years = np.asarray(ds["time.year"].values)
    m = years == int(year)
    if not m.any():
        return None
    return np.asarray(ds[var].values[m])  # (days, H, W)


def daily_rain(region: str, year: int) -> np.ndarray | None:
    """``(days, H, W)`` daily rainfall for a region/year, or None if missing."""
    _bust_streamlit_caches_if_sig_changed()
    return _daily_slice(region, int(year), "rain", sig=manifest_sig())


def daily_tmax(region: str, year: int) -> np.ndarray | None:
    """``(days, H, W)`` daily max-temperature for a region/year, or None if missing."""
    _bust_streamlit_caches_if_sig_changed()
    return _daily_slice(region, int(year), "tmax", sig=manifest_sig())


def daily_tmin(region: str, year: int) -> np.ndarray | None:
    """``(days, H, W)`` daily min-temperature for a region/year, or None if missing."""
    _bust_streamlit_caches_if_sig_changed()
    return _daily_slice(region, int(year), "tmin", sig=manifest_sig())


def daily_insat_lst(region: str, year: int) -> np.ndarray | None:
    """``(days, H, W)`` daily INSAT LST for a region/year, or None if missing."""
    _bust_streamlit_caches_if_sig_changed()
    return _daily_slice(region, int(year), "insat_lst", sig=manifest_sig())


# ---------------------------------------------------------------------------
# Region info panel (real coverage from manifest + cube)
# ---------------------------------------------------------------------------
@cache_data(show_spinner=False)
def region_info(region: str, sig: str | None = None) -> dict:
    """Summary the UI uses in the sidebar + data-provenance panels.

    Blends cube stats (shape, valid cells) with manifest facts (readers,
    sources, git hash, train years, build time).
    """
    ds = load_region(region)
    mask = ds["mask"].values
    valid = int((mask == 1).sum())
    total = int(mask.size)
    years = np.unique(np.asarray(ds["time.year"].values)).astype(int).tolist()

    variables = list_available_variables(region, sig=sig)
    cov = coverage_matrix(region, sig=sig)

    mani = manifest_dict()
    train_years = mani.get("train_years") or [None, None]

    return {
        "region": region,
        "label": REGIONS[region]["label"],
        "shape": (int(ds.sizes["lat"]), int(ds.sizes["lon"])),
        "extent": REGIONS[region]["extent"],
        "years": (min(years), max(years)) if years else (None, None),
        "all_years": years,
        "train_years": tuple(train_years) if len(train_years) == 2 else None,
        "n_time": int(ds.sizes.get("time", 0)),
        "valid_cells": valid,
        "total_cells": total,
        "land_pct": round(100.0 * valid / total, 1) if total else 0.0,
        "variables": variables,
        "coverage": cov,
        "processed_nc": str(REGIONS[region]["processed_nc"]),
        "nc_exists": REGIONS[region]["processed_nc"].exists(),
        "ckpt_prefix": REGIONS[region]["ckpt_prefix"],
        "manifest_sig": manifest_sig(),
        "manifest_generated_at_ist": mani.get("generated_at_ist", ""),
        "manifest_git_hash": mani.get("git_hash", ""),
        "manifest_readers": mani.get("readers", {}),
        "manifest_regrid": mani.get("regrid", {}),
        "manifest_sources": mani.get("sources", {}),
    }


# ---------------------------------------------------------------------------
# Diagnostic entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    regs = sys.argv[1:] or list(REGIONS.keys())
    print(f"manifest_sig = {manifest_sig()}")
    for rg in regs:
        try:
            info = region_info(rg, sig=manifest_sig())
        except ProcessedDataMissing as e:
            print(f"[{rg}] MISSING: {e}")
            continue
        print(f"[{rg}] shape={info['shape']}  years={info['years']}  "
              f"train_years={info['train_years']}  land={info['land_pct']}%  "
              f"vars={info['variables']}  time={info['n_time']}")
        for v in info["variables"]:
            avg_cov = float(np.mean(list(info["coverage"][v].values())))
            print(f"    {v:12s} avg-daily-coverage={avg_cov*100:5.1f}%")
