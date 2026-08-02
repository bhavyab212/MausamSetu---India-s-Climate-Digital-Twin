"""
config.py
==========
CENTRAL CONFIGURATION FILE for MausamSetu.

Every hardcoded constant lives here. Every other module imports from here.
This way, if we change (say) the pilot region, we change it ONCE.

Grid constants come directly from IMD documentation:
  - Rainfall: 0.25° × 0.25°, 135 lon × 129 lat, lon 66.5→99.75°E, lat 6.5→38.5°N
  - Temperature: 1.0° × 1.0°, 31 lon × 31 lat, lon 67.5→97.5°E, lat 7.5→37.5°N
"""
from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
# ROOT = the top-level MausamSetu/ folder (auto-detected from this file's location)
ROOT = Path(__file__).resolve().parent.parent

# Data locations
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
SYNTHETIC_DIR  = DATA_DIR / "synthetic"

IMD_RAIN_DIR   = RAW_DIR / "imd_rain"
IMD_TMAX_DIR   = RAW_DIR / "imd_tmax"
IMD_TMIN_DIR   = RAW_DIR / "imd_tmin"
INSAT_DIR      = RAW_DIR / "insat"
IMDAA_DIR      = RAW_DIR / "imdaa"

# --- EXTERNAL DATA FALLBACK ---
# When data isn't in data/raw/, check this external folder next.
# This is where the team's real IMD downloads live.
EXTERNAL_DATA_DIR = Path(r"C:\Users\bhavy\Documents\v1\DATA")
EXTERNAL_RAIN_DIR = EXTERNAL_DATA_DIR / "IMD rainfall data"
EXTERNAL_TMAX_DIR = EXTERNAL_DATA_DIR / "IMD max temp data"
EXTERNAL_TMIN_DIR = EXTERNAL_DATA_DIR / "IMD min temp data"

# Where trained models are saved
CHECKPOINT_DIR = ROOT / "checkpoints"

# Where forecasts / plots / reports are saved
OUTPUT_DIR     = ROOT / "outputs"

# The main processed file (created by preprocessing pipeline)
CAUVERY_NC     = PROCESSED_DIR / "cauvery.nc"


# ==============================================================================
# FULL INDIA GRID CONSTANTS (from IMD official documentation)
# ==============================================================================
# These are FACTS about the data files; do not change.

# --- IMD Rainfall Grid (0.25° resolution) ---
IMD_RAIN_LAT_START = 6.5     # southernmost latitude (°N)
IMD_RAIN_LAT_END   = 38.5    # northernmost latitude (°N)
IMD_RAIN_LON_START = 66.5    # westernmost longitude (°E)
IMD_RAIN_LON_END   = 99.75   # easternmost longitude (°E)
IMD_RAIN_RES       = 0.25    # degrees per pixel
IMD_RAIN_NLAT      = 129     # number of latitude points
IMD_RAIN_NLON      = 135     # number of longitude points
IMD_RAIN_FILL      = -999.0  # "missing data" sentinel value in .grd files

# --- IMD Temperature Grid (1.0° resolution) — SAME for Tmax and Tmin ---
IMD_TEMP_LAT_START = 7.5     # southernmost latitude (°N)
IMD_TEMP_LAT_END   = 37.5    # northernmost latitude (°N)
IMD_TEMP_LON_START = 67.5    # westernmost longitude (°E)
IMD_TEMP_LON_END   = 97.5    # easternmost longitude (°E)
IMD_TEMP_RES       = 1.0     # degrees per pixel
IMD_TEMP_NLAT      = 31
IMD_TEMP_NLON      = 31
IMD_TEMP_FILL      = 99.9    # "missing data" sentinel value (varies; some files use 99.9)


# ==============================================================================
# PILOT REGION — CAUVERY BASIN
# ==============================================================================
# The box we crop everything to. All downstream code operates on THIS area.

PILOT_NAME       = "Cauvery Basin"
PILOT_LAT_START  = 10.0      # °N (southern edge of Cauvery)
PILOT_LAT_END    = 14.5      # °N (northern edge)
PILOT_LON_START  = 75.5      # °E (western edge, includes Western Ghats)
PILOT_LON_END    = 79.5      # °E (eastern edge, includes Mettur reservoir)

# The master grid we regrid everything to (matches IMD rainfall resolution)
MASTER_RES       = 0.25      # degrees

# Computed pilot grid dimensions:
# lat: 10.0, 10.25, 10.5, ..., 14.25, 14.5  → 19 points
# lon: 75.5, 75.75, 76.0, ..., 79.25, 79.5  → 17 points
# (These are computed at runtime by numpy; documented here for clarity)


# ==============================================================================
# TIME SPLITS (crucial for honest validation — split by year, not random!)
# ==============================================================================
TRAIN_YEARS = [2020, 2021]      # real IMD data span
VAL_YEARS   = [2022]
TEST_YEARS  = [2023]

# Climatology baseline period (for computing "normal")
CLIM_YEARS  = [2020, 2021]


# ==============================================================================
# MODEL HYPERPARAMETERS (ConvLSTM Tier-C model)
# ==============================================================================
INPUT_DAYS     = 6          # lookback window (past days shown to model)
FORECAST_DAYS  = 7          # forecast horizon (days to predict)

# Channels the model sees per day (see preprocess/build_dataset.py)
INPUT_CHANNELS = 5    # rain, tmax, tmin, insat_lst, insat_rain
OUTPUT_VARS    = 3    # rain, tmax, tmin (INSAT not predicted, only used as input)

# ConvLSTM architecture
HIDDEN_CHANNELS = [64, 64, 64]   # 3 stacked ConvLSTM layers
KERNEL_SIZE     = 3               # 3x3 convolution filter
DROPOUT         = 0.2             # for MC-Dropout uncertainty

# Training
BATCH_SIZE      = 8
EPOCHS          = 50
LR              = 1e-3            # learning rate
EARLY_STOP      = 10              # stop if val loss doesn't improve for N epochs

# Loss weights
LOSS_MSE_W      = 1.0
LOSS_OCC_W      = 0.5             # rainfall occurrence (yes/no) head
LOSS_PHYSICS_W  = 0.1             # physics penalty (no negative rain, Tmin<Tmax)

# MC-Dropout uncertainty
MC_SAMPLES      = 20              # forward passes at inference for uncertainty


# ==============================================================================
# ASSIMILATION (EnKF) SETTINGS
# ==============================================================================
ENKF_ENSEMBLE_SIZE = 20           # number of ensemble members
ENKF_INFLATION     = 1.05         # covariance inflation factor
OBS_ERROR_LST      = 2.0          # INSAT LST std (°C)
OBS_ERROR_IMC      = 5.0          # INSAT rainfall std (mm)


# ==============================================================================
# STORYLINE / WHAT-IF (from IPCC AR6 published deltas)
# ==============================================================================
# Ranges for user sliders in the dashboard
SCENARIO_TEMP_RANGE     = (-2.0, 4.0)   # °C delta
SCENARIO_RAIN_RANGE     = (-40, 40)     # % delta
SCENARIO_DEFAULT_TEMP   = 0.0
SCENARIO_DEFAULT_RAIN   = 0

# IPCC AR6 delta references (annual mean, South Asia)
IPCC_DELTAS = {
    "SSP1-2.6 (1.5°C)": {"temp": +1.5, "rain_pct": +5.0},
    "SSP2-4.5 (2.0°C)": {"temp": +2.0, "rain_pct": +8.0},
    "SSP5-8.5 (4.4°C)": {"temp": +4.4, "rain_pct": +15.0},
}


# ==============================================================================
# IMPACT MODEL SETTINGS
# ==============================================================================
# Heat stress thresholds (°C, Tmax)
HEAT_STRESS_TEMP     = 40.0       # dangerous heat day
EXTREME_HEAT_TEMP    = 45.0       # extreme (lethal) heat day

# Cauvery basin approx characteristics (for hydrology)
BASIN_AREA_KM2       = 81155      # km² (Cauvery basin total)
BASIN_MEAN_CN        = 72         # SCS Curve Number (mixed land use)

# Simple ₹ risk factors (per district per event, rough calibration)
RUPEE_PER_HEAT_DAY_CRORE   = 5    # ₹ crore per district per extreme heat day
RUPEE_PER_DROUGHT_MM_CRORE = 0.1  # ₹ crore per mm rainfall deficit


# ==============================================================================
# DASHBOARD SETTINGS
# ==============================================================================
DASHBOARD_TITLE   = "MausamSetu — Digital Twin of India's Climate"
DASHBOARD_ICON    = "🇮🇳"
DASHBOARD_LAYOUT  = "wide"


# ==============================================================================
# HELPER: ensure all directories exist
# ==============================================================================
def ensure_dirs():
    """Create all project directories if missing. Called by main entrypoints."""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR,
              IMD_RAIN_DIR, IMD_TMAX_DIR, IMD_TMIN_DIR, INSAT_DIR, IMDAA_DIR,
              CHECKPOINT_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Run `python -m mausamsetu.config` to sanity-check paths
    ensure_dirs()
    print("=" * 60)
    print(f"MausamSetu config check — ROOT = {ROOT}")
    print("=" * 60)
    print(f"Pilot region: {PILOT_NAME}")
    print(f"  Lat: {PILOT_LAT_START}°N – {PILOT_LAT_END}°N")
    print(f"  Lon: {PILOT_LON_START}°E – {PILOT_LON_END}°E")
    print(f"  Master resolution: {MASTER_RES}°")
    print(f"Data dir: {DATA_DIR}")
    print(f"  Raw: {RAW_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("✓ All directories exist.")
