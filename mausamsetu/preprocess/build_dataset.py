"""
build_dataset.py
=================
End-to-end preprocessing pipeline: raw data → cauvery.nc + PyTorch dataset.

WHAT THIS SCRIPT DOES
---------------------
1. LOAD raw data (real IMD/INSAT if available, else synthetic fallback)
2. CROP everything to the Cauvery pilot region
3. REGRID all sources onto the master 0.25° grid (rain grid)
4. COMPUTE climatology + anomalies (subtract "normal" from each pixel/day)
5. STANDARDIZE (z-score) using ONLY training years (no leakage)
6. SAVE final cauvery.nc — the single file every other module reads

TIME SPLITS (from config.py):
  train: 2015-2021 (7 years)
  val:   2022      (1 year)
  test:  2023-2024 (2 years)

OUTPUT
------
  data/processed/cauvery.nc
    - rain, tmax, tmin, insat_lst, insat_rain  (raw values)
    - rain_anom, tmax_anom, tmin_anom          (anomalies from climatology)
    - climatology (day-of-year × lat × lon × 3 vars)
    - norm_stats (mean, std for each channel — for un-standardizing)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from mausamsetu import config
from mausamsetu.data import synthetic, imd


# ============================================================================
# STEP 1: LOAD RAW DATA (real if available, else synthetic)
# ============================================================================
def _find_data_dirs() -> tuple[Path, Path, Path] | None:
    """
    Locate the three IMD directories (rain, tmax, tmin), preferring the
    project's `data/raw/` folders but falling back to the external download
    location at C:\\Users\\bhavy\\Documents\\v1\\DATA.

    Returns (rain_dir, tmax_dir, tmin_dir) if at least the rainfall directory
    has files, else None.
    """
    from pathlib import Path

    candidates = [
        # (rain, tmax, tmin) triples in preference order
        (config.IMD_RAIN_DIR, config.IMD_TMAX_DIR, config.IMD_TMIN_DIR),
        (config.EXTERNAL_RAIN_DIR, config.EXTERNAL_TMAX_DIR, config.EXTERNAL_TMIN_DIR),
    ]

    for rain_dir, tmax_dir, tmin_dir in candidates:
        rain_files = (list(rain_dir.glob("*.grd")) + list(rain_dir.glob("*.GRD"))
                      if rain_dir.exists() else [])
        if len(rain_files) >= 2:
            return rain_dir, tmax_dir, tmin_dir
    return None


def load_raw_data(prefer_real: bool = True) -> xr.Dataset:
    """
    Load raw climate data for the full time range.

    Strategy:
      - If real IMD .grd files exist in data/raw/, use those.
      - Else, if real files exist in EXTERNAL_DATA_DIR, use those.
      - Else, generate/load synthetic data.
    """
    if prefer_real:
        dirs = _find_data_dirs()
        if dirs is not None:
            rain_dir, tmax_dir, tmin_dir = dirs
            print(f"Found real IMD data in:")
            print(f"  Rain: {rain_dir}")
            print(f"  Tmax: {tmax_dir}")
            print(f"  Tmin: {tmin_dir}")
            return _load_real_data(rain_dir, tmax_dir, tmin_dir)
        print("No real IMD data found in project dirs or external dir. Using synthetic.")

    return _load_synthetic_data()


def _load_synthetic_data() -> xr.Dataset:
    """Load the synthetic Cauvery dataset (already cropped to pilot region)."""
    syn_path = config.SYNTHETIC_DIR / "cauvery_synthetic.nc"
    if not syn_path.exists():
        print("Synthetic data not found. Generating it now...")
        synthetic.generate_synthetic_cauvery(output_path=syn_path)
    ds = xr.open_dataset(syn_path)
    print(f"✓ Loaded synthetic dataset: {ds.sizes}")
    return ds


def _load_real_data(rain_dir, tmax_dir, tmin_dir) -> xr.Dataset:
    """Load real IMD data for all available years and merge."""
    all_years = config.TRAIN_YEARS + config.VAL_YEARS + config.TEST_YEARS

    # Rainfall (0.25°)
    rain = imd.read_imd_rainfall_years(rain_dir, all_years)

    # Temperature (1.0°) — will be regridded later
    tmax = imd.read_imd_temperature_years(tmax_dir, all_years, kind="tmax")
    tmin = imd.read_imd_temperature_years(tmin_dir, all_years, kind="tmin")

    # -- Crop to the pilot region NOW (temp files are 31x31 India-wide) --
    # This also aligns time axes across all three variables.
    lat_min, lat_max = config.PILOT_LAT_START, config.PILOT_LAT_END
    lon_min, lon_max = config.PILOT_LON_START, config.PILOT_LON_END

    def _crop_to_box(da):
        lat_slice = slice(lat_min, lat_max) if da.lat.values[0] < da.lat.values[-1] else slice(lat_max, lat_min)
        lon_slice = slice(lon_min, lon_max) if da.lon.values[0] < da.lon.values[-1] else slice(lon_max, lon_min)
        return da.sel(lat=lat_slice, lon=lon_slice)

    rain = _crop_to_box(rain)
    tmax = _crop_to_box(tmax)
    tmin = _crop_to_box(tmin)

    # -- Regrid temperature (1°) up to the rainfall grid (0.25°) --
    tmax = tmax.interp(lat=rain.lat, lon=rain.lon, method="linear")
    tmin = tmin.interp(lat=rain.lat, lon=rain.lon, method="linear")

    # -- Align time axes (intersect) --
    common_times = np.intersect1d(np.intersect1d(rain.time.values, tmax.time.values), tmin.time.values)
    rain = rain.sel(time=common_times)
    tmax = tmax.sel(time=common_times)
    tmin = tmin.sel(time=common_times)

    # -- INSAT proxies (until real MOSDAC data arrives) --
    print("Note: Real INSAT integration TODO — using synthetic LST/rain proxies.")
    lst_syn = tmax.copy().rename("insat_lst") + 5.0                      # LST ~5°C hotter
    insat_rain_syn = rain.copy().rename("insat_rain") * 1.1              # rough proxy

    ds = xr.Dataset({
        "rain": rain,
        "tmax": tmax,
        "tmin": tmin,
        "insat_lst": lst_syn,
        "insat_rain": insat_rain_syn,
    })
    return ds


# ============================================================================
# STEP 2: CROP TO PILOT REGION
# ============================================================================
def crop_to_cauvery(ds: xr.Dataset) -> xr.Dataset:
    """
    Crop the full-India dataset to the Cauvery pilot box.

    Uses xarray's .sel() with slices — this selects by COORDINATE VALUE,
    not by index. If the file already fits the pilot region (synthetic case),
    this is effectively a no-op.
    """
    lat_min, lat_max = config.PILOT_LAT_START, config.PILOT_LAT_END
    lon_min, lon_max = config.PILOT_LON_START, config.PILOT_LON_END

    # Handle ascending/descending lat (be safe)
    lat_slice = slice(lat_min, lat_max) if ds.lat.values[0] < ds.lat.values[-1] else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if ds.lon.values[0] < ds.lon.values[-1] else slice(lon_max, lon_min)

    cropped = ds.sel(lat=lat_slice, lon=lon_slice)
    print(f"✓ Cropped to Cauvery: {cropped.sizes}")
    return cropped


# ============================================================================
# STEP 3: REGRID ALL VARIABLES TO MASTER GRID
# ============================================================================
def regrid_to_master(ds: xr.Dataset) -> xr.Dataset:
    """
    Regrid all variables onto the master 0.25° Cauvery grid.

    - rain is already on 0.25°; no-op
    - tmax/tmin are on 1.0°; interpolate up to 0.25° (bilinear)
    - INSAT variables: already regridded if synthetic; else regridded in ingestion
    """
    # Master grid
    target_lat = np.arange(
        config.PILOT_LAT_START,
        config.PILOT_LAT_END + config.MASTER_RES / 2,
        config.MASTER_RES,
    ).astype(np.float32)
    target_lon = np.arange(
        config.PILOT_LON_START,
        config.PILOT_LON_END + config.MASTER_RES / 2,
        config.MASTER_RES,
    ).astype(np.float32)

    # Skip if already on master grid
    if len(ds.lat) == len(target_lat) and np.allclose(ds.lat.values, target_lat):
        print("✓ Already on master grid — no regridding needed")
        return ds

    print(f"Regridding to master grid: {len(target_lat)} lat × {len(target_lon)} lon")
    return ds.interp(lat=target_lat, lon=target_lon, method="linear")


# ============================================================================
# STEP 4: CLIMATOLOGY + ANOMALIES
# ============================================================================
def compute_climatology_and_anomalies(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute per-pixel, per-day-of-year climatology using ONLY training years,
    then subtract from every timestep to produce anomalies.

    Variables treated: rain, tmax, tmin (satellite is kept as raw input).
    Returns a Dataset with new variables rain_anom, tmax_anom, tmin_anom AND
    a `climatology` variable indexed by day-of-year.
    """
    # --- Select training years ---
    train_mask = ds.time.dt.year.isin(config.TRAIN_YEARS)
    ds_train = ds.where(train_mask, drop=True)

    print(f"Computing climatology from {len(ds_train.time)} training days ({config.TRAIN_YEARS})")

    clim = {}
    anom = {}
    for var in ["rain", "tmax", "tmin"]:
        # Group by day-of-year and take mean across training years
        clim[var] = ds_train[var].groupby("time.dayofyear").mean("time")   # (365, lat, lon)
        # Broadcast climatology back to full time axis and subtract
        clim_full = clim[var].sel(dayofyear=ds.time.dt.dayofyear)
        # sel gives shape (time, lat, lon); align coords
        clim_full = clim_full.assign_coords(time=ds.time)
        anom[var] = ds[var] - clim_full

    ds_out = ds.copy()
    for var in ["rain", "tmax", "tmin"]:
        ds_out[f"{var}_anom"] = anom[var]
        ds_out[f"{var}_clim"] = clim[var]

    print("✓ Anomalies computed")
    return ds_out


# ============================================================================
# STEP 5: STANDARDIZATION (z-score) using TRAINING statistics only
# ============================================================================
def compute_norm_stats(ds: xr.Dataset) -> dict:
    """
    Compute mean and std of each anomaly variable, using ONLY training years.
    These stats will normalise inputs and un-normalise model outputs.
    """
    train_mask = ds.time.dt.year.isin(config.TRAIN_YEARS)
    ds_train = ds.where(train_mask, drop=True)

    stats = {}
    for var in ["rain_anom", "tmax_anom", "tmin_anom", "insat_lst", "insat_rain"]:
        arr = ds_train[var].values
        stats[var] = {
            "mean": float(np.nanmean(arr)),
            "std":  float(np.nanstd(arr) + 1e-8),
        }
    print("✓ Normalisation stats computed (train-only):")
    for k, v in stats.items():
        print(f"  {k}: mean={v['mean']:.3f}, std={v['std']:.3f}")
    return stats


# ============================================================================
# STEP 6: SAVE FINAL DATASET
# ============================================================================
def save_dataset(ds: xr.Dataset, norm_stats: dict, out_path: Path) -> None:
    """Save the processed dataset with normalisation stats in attrs."""
    # Attach metadata
    ds.attrs.update({
        "title":         "MausamSetu — Cauvery Processed Dataset",
        "pilot_region":  config.PILOT_NAME,
        "lat_bounds":    [config.PILOT_LAT_START, config.PILOT_LAT_END],
        "lon_bounds":    [config.PILOT_LON_START, config.PILOT_LON_END],
        "master_resolution_deg": config.MASTER_RES,
        "train_years":   config.TRAIN_YEARS,
        "val_years":     config.VAL_YEARS,
        "test_years":    config.TEST_YEARS,
    })
    # Encode norm stats as a JSON-like string attribute (xarray attrs must be scalars)
    import json
    ds.attrs["norm_stats_json"] = json.dumps(norm_stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    print(f"\n✓ Saved processed dataset → {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main(prefer_real: bool = True) -> Path:
    """Run the full preprocessing pipeline."""
    print("=" * 70)
    print("MAUSAMSETU PREPROCESSING PIPELINE")
    print("=" * 70)

    config.ensure_dirs()

    # Step 1: Load
    print("\n[1/6] Loading raw data...")
    ds = load_raw_data(prefer_real=prefer_real)

    # Step 2: Crop
    print("\n[2/6] Cropping to Cauvery basin...")
    ds = crop_to_cauvery(ds)

    # Step 3: Regrid
    print("\n[3/6] Regridding to master 0.25° grid...")
    ds = regrid_to_master(ds)

    # Step 4: Anomalies
    print("\n[4/6] Computing climatology + anomalies...")
    ds = compute_climatology_and_anomalies(ds)

    # Step 5: Normalisation stats
    print("\n[5/6] Computing normalisation stats (train-only)...")
    norm_stats = compute_norm_stats(ds)

    # Step 6: Save
    print("\n[6/6] Saving...")
    save_dataset(ds, norm_stats, config.CAUVERY_NC)

    print("\n" + "=" * 70)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output: {config.CAUVERY_NC}")
    print(f"Days:   {len(ds.time)}")
    print(f"Grid:   {ds.sizes['lat']} lat × {ds.sizes['lon']} lon")
    print(f"Vars:   {list(ds.data_vars)}")

    return config.CAUVERY_NC


if __name__ == "__main__":
    main()
