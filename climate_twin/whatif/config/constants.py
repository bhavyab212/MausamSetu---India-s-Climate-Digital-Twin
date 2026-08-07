"""
whatif.config.constants — physical + protocol constants.

Anything that describes the IMD data or the master grid lives here.
The zone registry itself lives at ``climate_twin/regions/india_zones.yaml``;
this file only mirrors the master grid geometry so downstream modules
don't have to open the yaml just to know N_LAT.
"""
from __future__ import annotations

# ── IMD raw-file sentinels — MASK BEFORE ANY MATH ─────────────────
# rain (.GRD) uses -999.0, temp (.GRD) uses 99.9.  Any downstream code
# that reads raw grids must call:
#     arr[np.isclose(arr, IMD_SENTINELS[0])] = np.nan
#     arr[np.isclose(arr, IMD_SENTINELS[1])] = np.nan
# BEFORE aggregating.  The processed cubes at data/processed/*.nc are
# already sentinel-clean.
IMD_SENTINELS = (-999.0, 99.9)

# ── Master grid (0.25° India) ─────────────────────────────────────
GRID_DEG = 0.25
N_LAT = 129
N_LON = 135
LAT_MIN, LAT_MAX = 6.5, 38.5
LON_MIN, LON_MAX = 66.5, 100.0

# ── Time zone ─────────────────────────────────────────────────────
TZ = "Asia/Kolkata"

# ── Unit labels — every displayed number must include one ─────────
UNITS = {
    # gauge / cube
    "rain":  "mm",
    "tmax":  "°C",
    "tmin":  "°C",
    "tmean": "°C",
    # indices
    "et0":  "mm/day",
    "spi":  "σ",
    "spei": "σ",
    "gdd":  "°C·day",
    "cdd":  "°C·day",
    "hdd":  "°C·day",
    "wbgt": "°C",
    "hi":   "°C",         # heat index
    "vpd":  "kPa",
    # sector outputs
    "inr":  "₹",
    "mwh":  "MWh",
    "kwh":  "kWh",
    "ha":   "ha",
    "yield_t_ha": "t/ha",
    # water
    "runoff":    "mm",
    "storage_m3": "m³",
    # health
    "mortality_per_100k": "per 100 000",
}

# ── IMD official daily-rainfall categories (mm/day) ──────────────
# Comparable nationally, published by IMD's Numerical Weather Products
# unit.  Downstream indices + validation use these AND per-zone
# percentile thresholds; do not tune either based on a metric.
IMD_RAINFALL_CATEGORIES_MM_DAY = {
    "no_rain":         (0.0,   0.1),
    "very_light":      (0.1,   2.4),
    "light":           (2.5,  15.5),
    "moderate":        (15.6, 64.4),
    "heavy":           (64.5, 115.5),
    "very_heavy":      (115.6, 204.4),
    "extremely_heavy": (204.5, 9999.0),
}
