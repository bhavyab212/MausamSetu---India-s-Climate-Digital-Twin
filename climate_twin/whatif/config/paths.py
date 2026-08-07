"""
whatif.config.paths — absolute paths, resolved from the project root.

Everything downstream imports these constants; nobody re-derives a path
from ``__file__`` or hardcodes a string.  Every value below was verified
against the on-disk layout during the Part-0 recon
(``docs/whatif_recon.md``).
"""
from __future__ import annotations

from pathlib import Path

# ── Streamlit app root (this package lives inside it) ───────────────────
STREAMLIT_ROOT: Path = Path(__file__).resolve().parents[2]      # …/climate_twin

# ── Raw data root (mounted volume, holds the official IMD + INSAT feeds) ─
DATA_ROOT: Path = Path(r"L:\MausamSetu\data")

# Yearly IMD .GRD / .nc feeds
RAINFALL_DIR: Path = DATA_ROOT / "Rainfall"
TMAX_DIR: Path = DATA_ROOT / "max_temp"
TMIN_DIR: Path = DATA_ROOT / "min_temp"
INSAT_DIR: Path = DATA_ROOT / "INSAT"

# CWC sub-basin shapefile bundle (WGS-1984 Lambert Conformal Conic)
SUBBASIN_DIR: Path = Path(r"L:\MausamSetu\Subbasin")

# ── Processed cube outputs (built by climate_twin/data/build_cube.py) ─
PROCESSED_DIR: Path = DATA_ROOT / "processed"
CUBE_INDIA: Path = PROCESSED_DIR / "india.nc"
CUBE_CAUVERY: Path = PROCESSED_DIR / "cauvery.nc"
MANIFEST_YAML: Path = PROCESSED_DIR / "manifest.yaml"
MANIFEST_SIG: Path = PROCESSED_DIR / "manifest.sig"

# ── Zone registry (built by climate_twin/regions/build_mask.py) ─────
STATES_GEOJSON: Path = STREAMLIT_ROOT / "India_States_2024.geojson"
ZONES_DIR: Path = STREAMLIT_ROOT / "regions"
ZONE_MASK_NPY: Path = ZONES_DIR / "zone_mask.npy"
ZONE_MEMBERSHIP_NPY: Path = ZONES_DIR / "zone_membership.npy"
ZONE_MASK_SHA: Path = ZONES_DIR / "zone_mask.sha256"
ZONE_STATS_JSON: Path = ZONES_DIR / "zone_stats.json"

# ── Model checkpoints (project-level checkpoints dir, one level up) ──
CHECKPOINTS: Path = STREAMLIT_ROOT.parent / "checkpoints"

# ── What-If session cache ────────────────────────────────────────────
CACHE_DIR: Path = STREAMLIT_ROOT / ".whatif_cache"
