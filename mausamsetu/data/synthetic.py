"""
synthetic.py
=============
Generates realistic FAKE climate data for the Cauvery pilot region.

WHY WE NEED THIS
----------------
Real IMD/INSAT downloads take 1–2 days (approvals) + hours (large files).
To develop the pipeline in parallel, we generate synthetic data that has:
  - The same shape (time × lat × lon)
  - The same coordinate system
  - The same NetCDF file format
  - The same VALUE RANGES (mm rainfall, °C temperature)
  - Realistic patterns (monsoon seasonality, spatial gradients, noise)

When real data arrives, the pipeline works identically — no code changes needed.
This is called "programming against an interface" — a professional engineering pattern.

WHAT IT GENERATES
-----------------
A single NetCDF file `data/synthetic/cauvery_synthetic.nc` containing:
  - rain      (mm/day)  — seasonal monsoon signal + spatial gradient + noise
  - tmax      (°C)      — hot summer, cool winter, latitude gradient
  - tmin      (°C)      — always < tmax, similar seasonal cycle
  - insat_lst (°C)      — LST is slightly hotter than air temp during day
  - insat_rain (mm)     — noisier satellite estimate of rain (biased vs truth)

Time span: 2015-01-01 to 2024-12-31 (10 years, ~3652 days)
Grid: Cauvery box at 0.25° → 19 lat × 17 lon = 323 pixels
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from mausamsetu import config


# ============================================================================
# HELPER: build the exact Cauvery grid coordinates
# ============================================================================
def cauvery_coords() -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lat, lon) arrays for the Cauvery pilot region at master 0.25° resolution.

    Latitude:  10.0°N to 14.5°N in 0.25° steps → 19 points [10.0, 10.25, ..., 14.5]
    Longitude: 75.5°E to 79.5°E in 0.25° steps → 17 points [75.5, 75.75, ..., 79.5]
    """
    lat = np.arange(config.PILOT_LAT_START,
                    config.PILOT_LAT_END + config.MASTER_RES / 2,
                    config.MASTER_RES).astype(np.float32)
    lon = np.arange(config.PILOT_LON_START,
                    config.PILOT_LON_END + config.MASTER_RES / 2,
                    config.MASTER_RES).astype(np.float32)
    return lat, lon


# ============================================================================
# HELPER: monsoon-shaped seasonal function
# ============================================================================
def monsoon_seasonal(day_of_year: np.ndarray) -> np.ndarray:
    """
    Returns a factor 0.0 to 1.0 that follows the Indian monsoon cycle.

    - Jan-May (days 1-150):   DRY (0.05-0.2)
    - Jun-Sep (days 152-273): WET peak (0.5-1.0), max around Jul 15 (day 196)
    - Oct-Dec (days 274-365): DRY tail (0.1-0.3)

    Approximated with a Gaussian centred on July 15.
    """
    peak_day = 196  # July 15
    width = 60      # standard deviation in days
    # Gaussian: exp(-(x-mu)^2 / (2*sigma^2))
    seasonal = np.exp(-((day_of_year - peak_day) ** 2) / (2 * width ** 2))
    # Add a small baseline (some rain in other months)
    seasonal = 0.05 + 0.95 * seasonal
    return seasonal


def temperature_seasonal(day_of_year: np.ndarray) -> np.ndarray:
    """
    Returns a factor 0.0 to 1.0 for the temperature cycle.

    - Jan-Feb: COLD (0.2-0.3) — winter minimum
    - Apr-May: HOT (0.9-1.0) — summer peak before monsoon cooling
    - Jun-Sep: WARM but moderated by monsoon clouds (0.6-0.7)
    - Oct-Dec: COOLING (0.5 → 0.3)

    Approximated with a sinusoid peaking in early May.
    """
    peak_day = 130  # May 10 (before monsoon arrives)
    # Cosine peaking at peak_day
    return 0.3 + 0.7 * (0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - peak_day) / 365))


# ============================================================================
# HELPER: spatial gradient (Western Ghats effect)
# ============================================================================
def western_ghats_pattern(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Returns a 2D grid (nlat × nlon) representing the west-to-east rainfall pattern.

    The Western Ghats are on the WEST side of Cauvery (~75.5°E).
    Rainfall should be HIGH on the west, LOW on the east (rain shadow).

    Output values ~ 0.3 (dry east) to 1.5 (wet west), centred around 1.0.
    """
    nlat = len(lat)
    nlon = len(lon)
    # 2D grid via broadcasting
    _, LON = np.meshgrid(lat, lon, indexing="ij")   # LON has shape (nlat, nlon)

    # Linear west→east ramp, normalised to 0..1 (west=1, east=0)
    west_to_east = 1.0 - (LON - config.PILOT_LON_START) / (config.PILOT_LON_END - config.PILOT_LON_START)

    # Sharpen so the west is much wetter than the east
    ghats = 0.3 + 1.5 * west_to_east ** 2   # west end: ~1.8, east end: ~0.3
    return ghats.astype(np.float32)


def latitude_gradient(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Returns a temperature gradient — south is warmer, north is cooler.
    Effect is small over Cauvery (only 4.5° span), but adds realism.
    """
    nlat = len(lat)
    nlon = len(lon)
    LAT, _ = np.meshgrid(lat, lon, indexing="ij")
    # 1.0 at south (10°N), 0.7 at north (14.5°N)
    grad = 1.0 - 0.3 * (LAT - config.PILOT_LAT_START) / (config.PILOT_LAT_END - config.PILOT_LAT_START)
    return grad.astype(np.float32)


# ============================================================================
# MAIN GENERATOR
# ============================================================================
def generate_synthetic_cauvery(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    output_path: Path | None = None,
) -> xr.Dataset:
    """
    Generate a synthetic Cauvery basin dataset.

    Parameters
    ----------
    start_date, end_date : str
        Date range (YYYY-MM-DD). Default = 10 years (2015–2024).
    seed : int
        Random seed for reproducibility.
    output_path : Path | None
        If given, save the xarray Dataset as NetCDF here.

    Returns
    -------
    xr.Dataset with variables: rain, tmax, tmin, insat_lst, insat_rain
    """
    rng = np.random.default_rng(seed)

    # --- Time & space coordinates ---
    time = pd.date_range(start_date, end_date, freq="D")
    n_days = len(time)
    lat, lon = cauvery_coords()
    nlat, nlon = len(lat), len(lon)

    print(f"Generating synthetic data for {config.PILOT_NAME}")
    print(f"  Days: {n_days} ({start_date} → {end_date})")
    print(f"  Grid: {nlat} lat × {nlon} lon = {nlat*nlon} pixels")

    # --- Day of year for each day (for seasonal cycles) ---
    doy = time.dayofyear.values.astype(np.float32)   # shape (n_days,)

    # --- Spatial patterns (same every day) ---
    ghats = western_ghats_pattern(lat, lon)          # (nlat, nlon)
    lat_grad = latitude_gradient(lat, lon)           # (nlat, nlon)

    # --- Time-varying seasonal factors ---
    rain_season = monsoon_seasonal(doy)              # (n_days,)
    temp_season = temperature_seasonal(doy)          # (n_days,)

    # ------------------------------------------------------------------
    # 1. RAINFALL (mm/day)
    # ------------------------------------------------------------------
    # Base = monsoon seasonal × Western Ghats spatial pattern
    # + Random daily variability (some days heavy rain, others dry)
    # + Zero inflation (most non-monsoon days have 0 rain)
    print("Building rainfall...")
    rain = np.zeros((n_days, nlat, nlon), dtype=np.float32)
    for t in tqdm(range(n_days), desc="  rain days", ncols=70):
        # Base intensity for this day (monsoon season × spatial pattern)
        base = 30.0 * rain_season[t] * ghats           # peak ~ 45 mm/day on west coast in July
        # Multiplicative noise (log-normal): occasional heavy days
        noise = rng.lognormal(mean=0.0, sigma=0.6, size=(nlat, nlon)).astype(np.float32)
        day_rain = base * noise
        # Zero inflation: on ~30% of days, force to 0 (dry day)
        if rain_season[t] < 0.15 and rng.random() < 0.6:
            day_rain *= 0.0
        rain[t] = np.clip(day_rain, 0.0, 500.0)        # cap at 500 mm

    # ------------------------------------------------------------------
    # 2. MAX TEMPERATURE (°C)
    # ------------------------------------------------------------------
    # Base = temperature seasonal × latitude gradient, in typical range 20-40°C
    print("Building tmax...")
    tmax_base = 22.0 + 18.0 * temp_season[:, None, None] * lat_grad[None, :, :]
    # Add small daily noise
    tmax_noise = rng.normal(0.0, 1.5, size=(n_days, nlat, nlon)).astype(np.float32)
    tmax = tmax_base + tmax_noise
    # Cool the wet pixels (rainy days are cooler)
    tmax -= 0.05 * rain
    tmax = np.clip(tmax, 12.0, 46.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. MIN TEMPERATURE (°C) — always < tmax
    # ------------------------------------------------------------------
    print("Building tmin...")
    # Diurnal range shrinks in monsoon (cloudy), widens in summer (clear skies)
    diurnal_range = 12.0 - 6.0 * rain_season[:, None, None]
    tmin = tmax - diurnal_range - rng.normal(0.0, 0.8, size=(n_days, nlat, nlon)).astype(np.float32)
    tmin = np.clip(tmin, 5.0, 32.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. INSAT LST (°C) — hotter than air Tmax during day, correlated
    # ------------------------------------------------------------------
    print("Building insat_lst...")
    # LST typically 3-8°C hotter than air temperature over land (daytime satellite pass)
    lst_offset = 3.0 + 5.0 * temp_season[:, None, None]
    insat_lst = tmax + lst_offset + rng.normal(0.0, 2.0, size=(n_days, nlat, nlon)).astype(np.float32)
    insat_lst = np.clip(insat_lst, 15.0, 60.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. INSAT rainfall estimate — noisy version of ground truth rain
    # ------------------------------------------------------------------
    # Real INSAT rainfall is biased AND noisy vs IMD ground truth.
    # We simulate: multiplicative bias (0.7-1.3) + additive noise + false alarms
    print("Building insat_rain...")
    bias = rng.uniform(0.7, 1.3, size=(1, nlat, nlon)).astype(np.float32)      # per-pixel bias
    noise = rng.normal(0.0, 3.0, size=(n_days, nlat, nlon)).astype(np.float32)
    insat_rain = rain * bias + noise
    # A few false alarms (cirrus clouds fake rain)
    false_alarm_mask = rng.random((n_days, nlat, nlon)) < 0.02
    insat_rain[false_alarm_mask] += rng.uniform(2, 10, size=false_alarm_mask.sum()).astype(np.float32)
    insat_rain = np.clip(insat_rain, 0.0, 500.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Build the xarray Dataset (proper coordinates + metadata)
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        data_vars={
            "rain":       (("time", "lat", "lon"), rain,
                           {"units": "mm/day", "long_name": "Daily rainfall", "source": "synthetic (IMD proxy)"}),
            "tmax":       (("time", "lat", "lon"), tmax,
                           {"units": "degC", "long_name": "Daily maximum air temperature", "source": "synthetic (IMD proxy)"}),
            "tmin":       (("time", "lat", "lon"), tmin,
                           {"units": "degC", "long_name": "Daily minimum air temperature", "source": "synthetic (IMD proxy)"}),
            "insat_lst":  (("time", "lat", "lon"), insat_lst,
                           {"units": "degC", "long_name": "INSAT Land Surface Temperature", "source": "synthetic (INSAT proxy)"}),
            "insat_rain": (("time", "lat", "lon"), insat_rain,
                           {"units": "mm/day", "long_name": "INSAT rainfall estimate", "source": "synthetic (INSAT proxy)"}),
        },
        coords={
            "time": time,
            "lat":  ("lat", lat, {"units": "degrees_north", "long_name": "Latitude"}),
            "lon":  ("lon", lon, {"units": "degrees_east",  "long_name": "Longitude"}),
        },
        attrs={
            "title":        "MausamSetu Synthetic Cauvery Basin Data",
            "description":  "Physically-plausible synthetic climate data for development. NOT REAL DATA.",
            "pilot_region": config.PILOT_NAME,
            "resolution":   f"{config.MASTER_RES} degrees",
            "created_by":   "mausamsetu.data.synthetic",
            "seed":         seed,
        },
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = config.SYNTHETIC_DIR / "cauvery_synthetic.nc"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # NetCDF write with compression (much smaller file)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"\n✓ Saved synthetic dataset → {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Sanity summary
    print("\n=== SANITY SUMMARY ===")
    print(f"Rain:       mean = {float(ds.rain.mean()):.2f} mm/day, "
          f"max = {float(ds.rain.max()):.1f} mm/day")
    print(f"Tmax:       mean = {float(ds.tmax.mean()):.2f} °C, "
          f"range = [{float(ds.tmax.min()):.1f}, {float(ds.tmax.max()):.1f}]")
    print(f"Tmin:       mean = {float(ds.tmin.mean()):.2f} °C")
    print(f"INSAT LST:  mean = {float(ds.insat_lst.mean()):.2f} °C")
    print(f"INSAT rain: mean = {float(ds.insat_rain.mean()):.2f} mm/day")
    # Verify Western Ghats pattern in July average
    jul = ds.rain.sel(time=ds.time.dt.month == 7).mean("time")
    west_mean = float(jul.isel(lon=slice(0, 3)).mean())
    east_mean = float(jul.isel(lon=slice(-3, None)).mean())
    print(f"\nWestern Ghats test (Jul mean):")
    print(f"  West edge rain: {west_mean:.1f} mm/day")
    print(f"  East edge rain: {east_mean:.1f} mm/day")
    print(f"  {'✓' if west_mean > east_mean else '✗'} West is wetter than East (expected)")

    return ds


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    ds = generate_synthetic_cauvery()
    print("\nDone. Dataset:")
    print(ds)
