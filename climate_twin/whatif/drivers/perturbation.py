"""
whatif.drivers.perturbation — L0, Method 1: delta-change perturbation.

Primary sources:
    * Räisänen, J. & Räty, O. (2013) "Analysis of changes in temperature
      and precipitation using two additional bias-correction methods."
      Clim. Dyn. 41:1553-1568. §4 documents the physical-inconsistency
      pitfall of the delta method.
    * IPCC AR6 WGI, Interactive Atlas — Technical Annex, delta
      downscaling protocol.

Physical-inconsistency note (mandatory, Part-5 Golden rule 4):
    Delta perturbation modifies mean state without preserving inter-
    variable consistency. A 20% rainfall reduction with unchanged
    humidity, wind, and radiation is not a physically realisable
    world. Use Method 2 (historical analogs) whenever possible; use
    this only for exploratory sensitivity work.

Convention:
    * Rainfall is scaled MULTIPLICATIVELY (rain * rain_scale). Rain is
      strictly non-negative and scale-consistent to first order.
    * Temperature is shifted ADDITIVELY (tmax + tmax_shift_c). Multi-
      plying a Celsius temperature is meaningless.
    * Applied only inside ``scope``; a 2-cell taper is exposed as a
      flag (default off).

Version: ``perturbation-v1``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import xarray as xr

from ..config.region import RegionSpec, region_mask_array

PERTURBATION_VERSION = "perturbation-v1"


class CaveatRequiredError(RuntimeError):
    """Raised by report exporters when a perturbation scenario is being
    exported without ``caveat_acknowledged=True``. Prevents accidental
    report generation without the physical-inconsistency disclaimer."""


@dataclass(frozen=True)
class PerturbationSpec:
    """Immutable delta-change specification.

    ``caveat_acknowledged`` gates report export (never blocks
    exploratory runs); the UI must set it True before the "export card"
    button becomes active.
    """
    rain_scale: float = 1.0
    tmax_shift_c: float = 0.0
    tmin_shift_c: float = 0.0
    scope: RegionSpec = field(default_factory=lambda: RegionSpec("all_india"))
    taper: bool = False
    caveat_acknowledged: bool = False
    version: str = PERTURBATION_VERSION

    def __post_init__(self) -> None:
        if not np.isfinite(self.rain_scale) or self.rain_scale < 0.0:
            raise ValueError(
                f"rain_scale must be ≥ 0 (rainfall is non-negative); "
                f"got {self.rain_scale}"
            )
        for k in ("tmax_shift_c", "tmin_shift_c"):
            v = getattr(self, k)
            if not np.isfinite(v):
                raise ValueError(f"{k} must be finite, got {v}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rain_scale": float(self.rain_scale),
            "tmax_shift_c": float(self.tmax_shift_c),
            "tmin_shift_c": float(self.tmin_shift_c),
            "scope": {
                "kind": self.scope.kind, "id": self.scope.id,
                "bbox": list(self.scope.bbox) if self.scope.bbox else None,
                "point": list(self.scope.point) if self.scope.point else None,
            },
            "taper": bool(self.taper),
            "caveat_acknowledged": bool(self.caveat_acknowledged),
            "version": self.version,
        }

    def is_identity(self) -> bool:
        return (self.rain_scale == 1.0
                 and self.tmax_shift_c == 0.0
                 and self.tmin_shift_c == 0.0)


def _mask_for_scope(scope: RegionSpec, lat, lon) -> np.ndarray:
    """Return a boolean mask (H, W) True inside ``scope``.

    For all_india / bbox / point (no explicit mask), we synthesise one
    from the coordinate axes so the multiplication is uniform inside
    the requested area.
    """
    H, W = len(lat), len(lon)
    if scope.kind == "all_india":
        return np.ones((H, W), dtype=bool)
    if scope.kind == "bbox":
        lat_v = np.asarray(lat)
        lon_v = np.asarray(lon)
        lo_la, lo_lo, hi_la, hi_lo = scope.bbox or (0, 0, 0, 0)
        mask = (
            (lat_v[:, None] >= min(lo_la, hi_la))
            & (lat_v[:, None] <= max(lo_la, hi_la))
            & (lon_v[None, :] >= min(lo_lo, hi_lo))
            & (lon_v[None, :] <= max(lo_lo, hi_lo))
        )
        return mask
    m = region_mask_array(scope)
    if m is None:
        return np.ones((H, W), dtype=bool)
    return np.asarray(m, dtype=bool)


def _apply_var(
    da: xr.DataArray, mask: np.ndarray, *, op: str, arg: float,
) -> xr.DataArray:
    """Apply ``op(arg)`` inside ``mask`` on a (time, lat, lon) DataArray.

    op ∈ {"scale", "shift"}. Preserves NaNs and attrs.
    """
    if da.ndim != 3:
        return da
    vals = da.values.astype(np.float32, copy=True)
    m3 = np.broadcast_to(mask[None, :, :], vals.shape)
    if op == "scale":
        vals = np.where(m3, vals * float(arg), vals)
    elif op == "shift":
        vals = np.where(m3, vals + float(arg), vals)
    else:
        raise ValueError(f"unknown op {op!r}")
    out = da.copy(data=vals)
    return out


def apply_perturbation(
    base: xr.Dataset, spec: PerturbationSpec,
) -> xr.Dataset:
    """Apply a delta-change perturbation to a driver Dataset.

    ``base`` must have variables ``rain``, ``tmax``, ``tmin`` on a
    (time, lat, lon) grid. Returns a new Dataset with the same shape
    and dims, with ``attrs["perturbation"]``, ``attrs["source_chain"]``
    updated to record what was applied.
    """
    required = {"rain", "tmax", "tmin"}
    missing = required - set(base.data_vars)
    if missing:
        raise ValueError(f"apply_perturbation: base missing {sorted(missing)}")

    lat = base["lat"].values
    lon = base["lon"].values
    mask = _mask_for_scope(spec.scope, lat, lon)

    out = xr.Dataset(coords=base.coords, attrs=dict(base.attrs))
    out["rain"] = _apply_var(base["rain"], mask, op="scale", arg=spec.rain_scale)
    out["tmax"] = _apply_var(base["tmax"], mask, op="shift", arg=spec.tmax_shift_c)
    out["tmin"] = _apply_var(base["tmin"], mask, op="shift", arg=spec.tmin_shift_c)

    # Preserve any other variables verbatim
    for v in base.data_vars:
        if v not in ("rain", "tmax", "tmin"):
            out[v] = base[v]

    out.attrs["perturbation"] = spec.to_dict()
    out.attrs["perturbation_version"] = PERTURBATION_VERSION
    chain = str(base.attrs.get("source_chain", base.attrs.get("source", "?")))
    out.attrs["source_chain"] = f"{chain} → drivers.perturbation@{PERTURBATION_VERSION}"
    # Physical-inconsistency banner — always attached, gates ONLY report export
    out.attrs["caveat"] = (
        "Method-1 delta perturbation modifies mean state without "
        "preserving inter-variable consistency (Räisänen & Räty 2013). "
        "Use Method 2 (analogs) for reported scenarios."
    )
    out.attrs["caveat_acknowledged"] = bool(spec.caveat_acknowledged)
    return out


def assert_caveat_acknowledged(ds: xr.Dataset) -> None:
    """Report exporters call this on a perturbation Dataset before
    committing to disk. Raises :class:`CaveatRequiredError` if the
    caveat wasn't explicitly acknowledged."""
    if ds.attrs.get("perturbation_version") != PERTURBATION_VERSION:
        return       # not a perturbation dataset — nothing to gate
    if not bool(ds.attrs.get("caveat_acknowledged", False)):
        raise CaveatRequiredError(
            "Cannot export a perturbation scenario without "
            "caveat_acknowledged=True. Set PerturbationSpec("
            "caveat_acknowledged=True) explicitly to opt in — this "
            "gate prevents accidental report generation without the "
            "physical-inconsistency disclaimer."
        )
