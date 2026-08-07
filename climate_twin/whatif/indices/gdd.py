"""
whatif.indices.gdd — Growing degree days.

Primary source: McMaster & Wilhelm (1997), "Growing degree-days: one
equation, two interpretations", *Agricultural and Forest Meteorology*
87, 291-300. This module implements **Method 1**:

    GDD_day = max(0, Tmean − Tbase)                            [°C·day]

with ``Tmean = (Tmax + Tmin) / 2``. An optional upper cap ``T_cap`` is
applied to ``Tmean`` BEFORE subtracting ``Tbase`` (this is the standard
"cap before subtract" convention; McMaster & Wilhelm §2.2).

Named-crop base/cap constants live in ``crops.yaml`` alongside a source
citation per crop. Do not add a crop to the yaml without one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from .reference import _propagate_attrs

_CROPS_YAML = Path(__file__).resolve().parent / "crops.yaml"


def gdd(tmax: xr.DataArray, tmin: xr.DataArray,
         t_base: float, t_cap: float | None = None) -> xr.DataArray:
    """Return daily GDD in °C·day. NaN cells propagate."""
    tmean = (tmax + tmin) * 0.5
    if t_cap is not None:
        tmean = tmean.clip(max=float(t_cap))
    out = (tmean - float(t_base)).clip(min=0.0)
    out.name = "gdd"
    out = _propagate_attrs(out, tmax, f"gdd@Tb{t_base}Tc{t_cap or 'inf'}-v1")
    out.attrs["units"] = "°C·day"
    out.attrs["t_base_c"] = float(t_base)
    if t_cap is not None:
        out.attrs["t_cap_c"] = float(t_cap)
    out.attrs["method"] = "mcmaster-wilhelm-1997-method-1"
    return out


def _load_crops() -> dict:
    if not _CROPS_YAML.exists():
        return {}
    return yaml.safe_load(_CROPS_YAML.read_text(encoding="utf-8")) or {}


def gdd_for_crop(crop: str, tmax: xr.DataArray, tmin: xr.DataArray) -> xr.DataArray:
    """Convenience wrapper — look up ``(t_base, t_cap)`` from crops.yaml.

    Raises ``KeyError`` when the crop is not defined (rather than
    silently falling through to a global default; per Part-2 rule 1)."""
    crops = _load_crops()
    if crop not in crops:
        raise KeyError(
            f"crop {crop!r} not in {_CROPS_YAML.name}. Known: "
            f"{sorted(crops.keys())}"
        )
    entry = crops[crop]
    return gdd(tmax, tmin,
                t_base=float(entry["t_base_c"]),
                t_cap=float(entry["t_cap_c"]) if entry.get("t_cap_c") else None)
