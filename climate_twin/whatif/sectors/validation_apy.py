"""
whatif.sectors.validation_apy — APY validation for the agriculture sector.

Primary source:
    India APY dataset (Area / Production / Yield) — Directorate of
    Economics & Statistics, Ministry of Agriculture, published on
    data.gov.in. When ingested to disk, the parquet lands at
    ``CACHE_DIR/apy/apy_district_<version>.parquet`` with columns
    ``state, district, year, crop, area_ha, production_t, yield_t_ha``.

Skepticism, not decoration.
    * We run the yield model on each year of ``VALID_YEARS`` forced
      with observed IMD data, aggregate to district, join to APY.
    * We report Pearson r, Spearman ρ, RMSE, MAE, MPE.
    * We publish the numbers whether they're good or bad.
    * If the model doesn't beat district climatology, we say so.

If the APY parquet isn't on disk yet, this module is deliberately a
stub that gates activation on file presence — it does NOT scrape or
guess numbers. :func:`get_last_validation` returns a status dict that
the UI renders as a "not-yet-ingested" banner.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config.paths import CACHE_DIR
from ..indices.reference import VALID_YEARS

APY_PARQUET_GLOB = "apy_district_*.parquet"


def _apy_parquet_path() -> Path | None:
    """Find the most-recent APY parquet on disk, if any."""
    apy_dir = CACHE_DIR / "apy"
    if not apy_dir.exists():
        return None
    candidates = sorted(apy_dir.glob(APY_PARQUET_GLOB))
    return candidates[-1] if candidates else None


def apy_available() -> bool:
    return _apy_parquet_path() is not None


def _validation_path(crop_key: str, version: str) -> Path:
    p = CACHE_DIR / "validation" / "agri" / f"{crop_key}_{version}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_last_validation(crop_key: str) -> dict[str, Any]:
    """Return a status dict the UI can render.

    Keys:
        ok            — bool
        reason        — str (empty when ok)
        scores        — dict of metrics (empty when not ok)
        n_districts   — int
        parquet_path  — str or empty
    """
    apy_path = _apy_parquet_path()
    if apy_path is None:
        return {
            "ok": False,
            "reason": "APY snapshot not yet ingested — model unvalidated.",
            "scores": {},
            "n_districts": 0,
            "parquet_path": "",
        }

    # Locate most-recent validation for this crop
    val_dir = CACHE_DIR / "validation" / "agri"
    if not val_dir.exists():
        return {
            "ok": False,
            "reason": "APY on disk, but no validation run recorded yet.",
            "scores": {},
            "n_districts": 0,
            "parquet_path": "",
        }
    candidates = sorted(val_dir.glob(f"{crop_key}_*.parquet"))
    if not candidates:
        return {
            "ok": False,
            "reason": f"APY on disk, but no {crop_key} validation run recorded yet.",
            "scores": {},
            "n_districts": 0,
            "parquet_path": "",
        }
    path = candidates[-1]
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"ok": False, "reason": f"couldn't read {path}: {e}",
                "scores": {}, "n_districts": 0, "parquet_path": str(path)}

    scores = _score_frame(df)
    return {
        "ok": True,
        "reason": "",
        "scores": scores,
        "n_districts": int(df["district"].nunique()),
        "parquet_path": str(path),
    }


def _score_frame(df: pd.DataFrame) -> dict[str, float]:
    """Compute Pearson r, Spearman ρ, RMSE, MAE, MPE on a
    (observed, modelled) DataFrame."""
    if not {"observed_t_ha", "modelled_t_ha"} <= set(df.columns):
        return {}

    obs = df["observed_t_ha"].to_numpy(dtype=np.float64)
    mod = df["modelled_t_ha"].to_numpy(dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[m], mod[m]
    if obs.size < 3:
        return {"n": int(obs.size)}

    r = float(np.corrcoef(obs, mod)[0, 1])
    # Spearman via ranks
    from scipy.stats import spearmanr
    rho = float(spearmanr(obs, mod).correlation)
    err = mod - obs
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        mpe = float(np.nanmean(np.where(obs > 0, (mod - obs) / obs, np.nan))) * 100.0
    return {"n": int(obs.size), "pearson_r": r, "spearman_rho": rho,
            "rmse_t_ha": rmse, "mae_t_ha": mae, "mpe_pct": mpe}


def run_validation(
    crop, region_kind: str = "all_india",
    *, write_parquet: bool = True,
) -> dict[str, Any]:
    """Full APY validation pipeline. Requires APY parquet on disk.

    Iterates ``VALID_YEARS``, runs the model on observed IMD forcing,
    aggregates to (proxy) districts, joins to APY, writes the results
    parquet, and returns a status dict identical in shape to
    :func:`get_last_validation`.

    Left as a documented no-op when APY isn't on disk; the recon
    (Part 3 STEP 7) confirms the file isn't present today, so this
    function raises :class:`FileNotFoundError` — Part 4 will land it.
    """
    apy_path = _apy_parquet_path()
    if apy_path is None:
        raise FileNotFoundError(
            "APY snapshot not on disk (looked in "
            f"{CACHE_DIR / 'apy'}). Ingest APY as a parquet named "
            f"apy_district_<version>.parquet before running validation."
        )
    raise NotImplementedError(
        "run_validation lands in Part 4 (Economics / Payoff). Today we "
        "only report whether APY is on disk; the joining + scoring "
        "pipeline is documented above."
    )
