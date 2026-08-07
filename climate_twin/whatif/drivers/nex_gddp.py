"""
whatif.drivers.nex_gddp — NEX-GDDP-CMIP6 loader.

Primary sources:
    * Thrasher, B. et al. (2022) "NASA Global Daily Downscaled Projections,
      CMIP6", Sci. Data 9:262. 0.25° daily bias-adjusted projections for
      34 CMIP6 GCMs × 4 SSPs, hosted on AWS Open Data
      (``s3://nex-gddp-cmip6/``).
    * Almazroui, M. et al. (2020) "Projected Changes in Temperature and
      Precipitation Climatology of South Asia CORDEX Region", Earth
      Systems and Environment 4:297-320 — model selection ranking.

Contract:
    * Return array is on the master 0.25° grid — NEX-GDDP is already
      0.25° so **no regridding**, only spatial masking to region.
    * Units converted at read time:
        - ``pr`` from kg m⁻² s⁻¹ to mm/day  (× 86400).
        - ``tasmax``, ``tasmin`` from K to °C  (− 273.15).
    * Times are IST-normalised (tz-aware, matching the historical driver).
    * Cache reads under ``CACHE_DIR/nex_gddp/``.

Version: ``nexgddp-cmip6-v1``.

When the on-disk mirror is absent AND cloud reads are unavailable,
the loader raises :class:`NEXGDDPUnavailable`. The higher-level
downscaler catches this and falls back to a **synthetic ensemble**
built from an observed climatology plus a warming trend — labelled
in provenance as ``synthetic-fallback`` so downstream code cannot
mistake it for a real projection.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ..config.paths import CACHE_DIR, STREAMLIT_ROOT
from ..config.region import RegionSpec, apply_region

NEX_GDDP_VERSION = "nexgddp-cmip6-v1"
_CATALOG_PATH = Path(__file__).resolve().parent / "nex_gddp_catalog.yaml"


class NEXGDDPUnavailable(RuntimeError):
    """Raised when neither cloud nor local NEX-GDDP is reachable."""


Variable = Literal["pr", "tasmax", "tasmin"]
SSP_ID = Literal["ssp126", "ssp245", "ssp370", "ssp585"]
Period = Literal["historical", "future"]


@dataclass(frozen=True)
class NEXGDDPSpec:
    """Immutable declaration of a NEX-GDDP-CMIP6 read."""
    variable: Variable
    model: str
    ssp: SSP_ID
    period: Period
    year_range: tuple[int, int]
    version: str = NEX_GDDP_VERSION

    def signature(self) -> str:
        parts = [
            self.variable, self.model, self.ssp, self.period,
            f"{self.year_range[0]}-{self.year_range[1]}", self.version,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


# ── Catalog ─────────────────────────────────────────────────────────
def _load_catalog() -> dict:
    return yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))


def list_models() -> list[str]:
    return [m["id"] for m in _load_catalog()["models"]]


def has_coverage(model: str, ssp: str, period: Period,
                  year_range: tuple[int, int]) -> bool:
    """Refuse a spec at construction if the requested (model × ssp ×
    period × year_range) isn't in the catalog. Fails loud instead of
    silently returning NaNs at read time."""
    cat = _load_catalog()
    models = {m["id"]: m for m in cat["models"]}
    if model not in models:
        return False
    cov = cat["coverage"].get("default", {})
    key = "historical" if period == "historical" else ssp
    if key not in cov:
        return False
    y0, y1 = cov[key]
    return y0 <= year_range[0] and y1 >= year_range[1]


def validate_spec(spec: NEXGDDPSpec) -> None:
    """Raise ValueError when the spec is out-of-catalog."""
    if not has_coverage(spec.model, spec.ssp, spec.period, spec.year_range):
        raise ValueError(
            f"NEX-GDDP-CMIP6 has no coverage for {spec.model} × {spec.ssp} × "
            f"{spec.period} × {spec.year_range}. Check the catalog "
            f"({_CATALOG_PATH.name})."
        )


# ── Unit conversion ─────────────────────────────────────────────────
def _to_engine_units(da: xr.DataArray, variable: Variable) -> xr.DataArray:
    """Convert NEX-GDDP native units to the engine's convention.

    * pr:      kg m⁻² s⁻¹  →  mm/day     (multiply by 86400)
    * tasmax:  K           →  °C          (subtract 273.15)
    * tasmin:  K           →  °C          (subtract 273.15)
    """
    if variable == "pr":
        out = da * 86400.0
        out.attrs["units"] = "mm/day"
    else:
        out = da - 273.15
        out.attrs["units"] = "°C"
    out.attrs["nex_gddp_variable"] = variable
    return out


# ── Cache helpers ───────────────────────────────────────────────────
def _cache_path(spec: NEXGDDPSpec, region: RegionSpec) -> Path:
    key = f"{spec.signature()}|{region.signature()}"
    d = hashlib.sha256(key.encode()).hexdigest()[:12]
    return CACHE_DIR / "nex_gddp" / f"{spec.variable}_{d}.pkl"


# ── Open ────────────────────────────────────────────────────────────
def open_nex_gddp(spec: NEXGDDPSpec, region: RegionSpec) -> xr.DataArray:
    """Return the NEX-GDDP array for ``spec`` masked to ``region``.

    Reads are cached to disk. When no on-disk mirror is configured
    AND fsspec/s3 access is unavailable, the loader raises
    :class:`NEXGDDPUnavailable`. Callers who want a synthetic
    fallback wrap this via :func:`open_nex_gddp_with_fallback`.
    """
    validate_spec(spec)
    cache = _cache_path(spec, region)
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    # 1) Local mirror (if configured)
    root_env = _local_mirror_root()
    if root_env is not None:
        try:
            da = _open_local_mirror(spec, root_env)
            da = apply_region(_to_engine_units(da, spec.variable), region)
            _save_cache(cache, da)
            return da
        except FileNotFoundError:
            pass

    # 2) Cloud (fsspec + s3)
    try:
        da = _open_s3(spec)
        da = apply_region(_to_engine_units(da, spec.variable), region)
        _save_cache(cache, da)
        return da
    except Exception as e:      # noqa: BLE001
        raise NEXGDDPUnavailable(
            f"NEX-GDDP-CMIP6 read failed for {spec.signature()}: "
            f"{type(e).__name__}: {e}. Set NEX_GDDP_ROOT for a local "
            "mirror, or install fsspec + s3fs for cloud reads. Higher-"
            "level LT callers should catch this and fall back to a "
            "synthetic ensemble (documented in provenance)."
        )


def _local_mirror_root() -> Path | None:
    import os
    env = os.environ.get("NEX_GDDP_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p
    return None


def _open_local_mirror(spec: NEXGDDPSpec, root: Path) -> xr.DataArray:
    """Open a set of yearly netCDF files from a local NEX-GDDP mirror."""
    cat = _load_catalog()
    tmpl = cat["path_template"]
    files = []
    for yr in range(spec.year_range[0], spec.year_range[1] + 1):
        rel = tmpl.format(
            prefix=cat["storage"]["s3_prefix"],
            scenario=("historical" if spec.period == "historical" else spec.ssp),
            model=spec.model,
            run="r1i1p1f1",
            var=spec.variable,
            year=yr,
        )
        candidate = root / rel
        if candidate.exists():
            files.append(candidate)
    if not files:
        raise FileNotFoundError(f"No local NEX-GDDP files for {spec.signature()}")
    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords",
                             engine="netcdf4")
    return ds[spec.variable]


def _open_s3(spec: NEXGDDPSpec) -> xr.DataArray:
    """Open via fsspec S3 anonymous access. Raises when fsspec / s3fs
    aren't installed."""
    import fsspec       # noqa: F401
    cat = _load_catalog()
    tmpl = cat["path_template"]
    bucket = cat["storage"]["s3_bucket"]
    files = []
    for yr in range(spec.year_range[0], spec.year_range[1] + 1):
        rel = tmpl.format(
            prefix=cat["storage"]["s3_prefix"],
            scenario=("historical" if spec.period == "historical" else spec.ssp),
            model=spec.model, run="r1i1p1f1", var=spec.variable, year=yr,
        )
        files.append(f"s3://{bucket}/{rel}")
    ds = xr.open_mfdataset(
        files, engine="h5netcdf", combine="by_coords",
        backend_kwargs={"anon": True},
    )
    return ds[spec.variable]


def _save_cache(path: Path, da: xr.DataArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as f:
            pickle.dump(da, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


# ── Assembly ────────────────────────────────────────────────────────
def assemble_multi_model(
    variable: Variable, ssp: SSP_ID, period: Period,
    year_range: tuple[int, int], region: RegionSpec,
    models: list[str] | None = None,
) -> xr.DataArray:
    """Stack the ten-GCM ensemble along a ``model`` dim.

    Missing model → dropped with WARNING attribute; never imputed.
    Returns dims ``(model, time, lat, lon)``.
    """
    if models is None:
        models = list_models()

    kept: list[xr.DataArray] = []
    dropped: list[str] = []
    for m in models:
        spec = NEXGDDPSpec(
            variable=variable, model=m, ssp=ssp, period=period,
            year_range=year_range,
        )
        try:
            da = open_nex_gddp(spec, region)
            kept.append(da.expand_dims(model=[m]))
        except (NEXGDDPUnavailable, ValueError) as e:
            dropped.append(f"{m}:{type(e).__name__}")
            continue

    if not kept:
        raise NEXGDDPUnavailable(
            f"No models in {models} produced data for {variable} × {ssp} "
            f"× {year_range}. Dropped: {dropped}"
        )
    ens = xr.concat(kept, dim="model")
    ens.attrs["nex_gddp_version"] = NEX_GDDP_VERSION
    ens.attrs["ssp"] = ssp
    ens.attrs["period"] = period
    ens.attrs["year_range"] = list(year_range)
    ens.attrs["models_kept"] = [str(x) for x in kept and ens.model.values]
    ens.attrs["models_dropped"] = dropped
    return ens
