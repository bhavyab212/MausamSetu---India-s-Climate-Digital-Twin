"""
whatif.config.region — RegionSpec + apply_region helper.

A RegionSpec is an immutable declarative slice of India. The engine
uses it to mask / subset any (time, lat, lon) DataArray to the actual
domain the user cares about.

Supported kinds today (Part 1):
    all_india   — no masking; full 0.25° grid
    bbox        — inclusive latitude / longitude window
    subbasin    — polygon from L:/MausamSetu/Subbasin/ (Cauvery ships)
    zone        — from the frozen climate_twin.regions registry
                  (if present); otherwise raises RegionNotAvailable
    point       — single (lat, lon) — nearest-cell lookup

`district` is reserved for a Part-2 add-on when the district
shapefile is bundled.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from .constants import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON
from .paths import CACHE_DIR, SUBBASIN_DIR


class RegionNotAvailable(RuntimeError):
    """The requested region kind/id is not resolvable in the current build."""


@dataclass(frozen=True)
class RegionSpec:
    """Immutable region declaration. Hashable → cache-key friendly."""
    kind: Literal["all_india", "zone", "district", "subbasin", "bbox", "point"]
    id: str | None = None
    bbox: tuple[float, float, float, float] | None = None   # lat_min, lon_min, lat_max, lon_max
    point: tuple[float, float] | None = None                # (lat, lon)

    def signature(self) -> str:
        """12-hex fingerprint for cache keys + provenance."""
        parts = [self.kind, str(self.id), str(self.bbox), str(self.point)]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Master-grid axes (kept aligned with whatif.config.constants)
# ---------------------------------------------------------------------------
def master_axes() -> tuple[np.ndarray, np.ndarray]:
    lat = np.round(np.linspace(LAT_MIN, LAT_MAX, N_LAT), 4)
    lon = np.round(np.linspace(LON_MIN, LON_MAX, N_LON), 4)
    return lat, lon


# ---------------------------------------------------------------------------
# Subbasin mask (Cauvery ships; other basins use the same code path)
# ---------------------------------------------------------------------------
def _load_subbasin_mask(basin_id: str) -> np.ndarray:
    """Rasterise a Subbasin.shp polygon onto the 0.25° master grid.

    Reprojects the (LCC) shapefile to WGS84 first. Cached under
    CACHE_DIR/masks/subbasin_<basin_id>.npy so subsequent calls are ~free.
    """
    cache_dir = CACHE_DIR / "masks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"subbasin_{basin_id}.npy"
    if cache.exists():
        return np.load(cache)

    import shapefile
    from pyproj import Transformer
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union, transform as shp_transform

    shp_path = SUBBASIN_DIR / "Subbasin.shp"
    prj_path = SUBBASIN_DIR / "Subbasin.prj"
    if not shp_path.exists():
        raise RegionNotAvailable(f"missing {shp_path}")

    r = shapefile.Reader(str(shp_path))
    recs = r.records()
    shapes = r.shapes()

    # Simple key match: 'cauvery' → any record with ba_name containing 'Cauvery'
    want = basin_id.lower()
    idx = [
        i for i, rec in enumerate(recs)
        if want in str(rec["ba_name"]).lower()
        or want in str(rec.get("sub_basin", "")).lower()
    ]
    if not idx:
        raise RegionNotAvailable(
            f"subbasin id {basin_id!r} not found in {shp_path.name}"
        )

    tr = Transformer.from_crs(prj_path.read_text(), "EPSG:4326", always_xy=True)
    polys = []
    for i in idx:
        g = shape(shapes[i].__geo_interface__)
        g = shp_transform(lambda x, y, z=None: tr.transform(x, y), g)
        polys.append(g)
    poly = unary_union(polys).buffer(0)

    lat, lon = master_axes()
    mask = np.zeros((N_LAT, N_LON), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if poly.contains(Point(float(lo), float(la))):
                mask[i, j] = True
    if not mask.any():
        # fall back to bounding box so we never return an empty mask
        b = poly.bounds  # (minx, miny, maxx, maxy)
        mask = (
            (lat[:, None] >= b[1]) & (lat[:, None] <= b[3])
            & (lon[None, :] >= b[0]) & (lon[None, :] <= b[2])
        )
    np.save(cache, mask)
    return mask


# ---------------------------------------------------------------------------
# Region application
# ---------------------------------------------------------------------------
def apply_region(da: xr.DataArray, region: RegionSpec) -> xr.DataArray:
    """Subset / mask a (time, lat, lon) DataArray to the requested region.

    Behaviour by kind:
      all_india   — return unchanged
      bbox        — slice lat/lon to the requested window
      subbasin    — apply Cauvery mask (or other listed basins);
                    cells outside → NaN
      zone        — apply frozen zone-registry mask if available
      point       — return a (time,) DataArray for the nearest cell
      district    — raises RegionNotAvailable (deferred)
    """
    if region.kind == "all_india":
        return da

    if region.kind == "bbox":
        if region.bbox is None:
            raise ValueError("RegionSpec(kind='bbox') requires bbox=…")
        lat_min, lon_min, lat_max, lon_max = region.bbox
        return da.sel(
            lat=slice(min(lat_min, lat_max), max(lat_min, lat_max)),
            lon=slice(min(lon_min, lon_max), max(lon_min, lon_max)),
        )

    if region.kind == "point":
        if region.point is None:
            raise ValueError("RegionSpec(kind='point') requires point=…")
        pla, plo = region.point
        return da.sel(lat=pla, lon=plo, method="nearest")

    if region.kind == "subbasin":
        if not region.id:
            raise ValueError("RegionSpec(kind='subbasin') requires id=…")
        mask = _load_subbasin_mask(region.id)
        return da.where(mask)

    if region.kind == "zone":
        # Defer to the frozen zone registry if it's importable.
        try:
            from climate_twin.regions import get_zones
        except Exception as e:  # noqa: BLE001
            raise RegionNotAvailable(
                f"zone registry not importable: {e}. "
                "Run `python -m climate_twin.regions.build_mask` first."
            ) from e
        Z = get_zones()
        if not region.id:
            raise ValueError("RegionSpec(kind='zone') requires id=…")
        try:
            zone = Z.by_key(region.id)
        except KeyError as e:
            raise RegionNotAvailable(
                f"zone id {region.id!r} not in registry (known: "
                f"{[z.key for z in Z.zones]})"
            ) from e
        mask = Z.hard_mask == zone.id
        return da.where(mask)

    if region.kind == "district":
        raise RegionNotAvailable(
            "district-level RegionSpec is not implemented yet. "
            "It needs an India-districts polygon layer under Subbasin/."
        )

    raise ValueError(f"unknown RegionSpec.kind={region.kind!r}")


def region_mask_array(region: RegionSpec) -> np.ndarray | None:
    """Return the raw boolean mask used by ``apply_region`` (for provenance
    hashing). Returns None for all_india / bbox / point (no mask needed)."""
    if region.kind in ("all_india", "bbox", "point"):
        return None
    if region.kind == "subbasin":
        return _load_subbasin_mask(region.id or "cauvery")
    if region.kind == "zone":
        from climate_twin.regions import get_zones
        Z = get_zones()
        zone = Z.by_key(region.id or "")
        return Z.hard_mask == zone.id
    return None
