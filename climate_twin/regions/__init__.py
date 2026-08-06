"""
climate_twin.regions — the SINGLE authoritative zone registry.

Every module that needs to know "which zone is this cell in?" imports from
here. The registry is frozen: on first load it computes a SHA-256 over
`india_zones.yaml` + `zone_mask.npy` + `zone_membership.npy` and refuses to
serve state if that signature drifts from `zone_mask.sha256`.

Public API
----------
``get_zones()``     -> :class:`ZoneRegistry`   (cached singleton)
``ZoneRegistry``    exposes:
    - `.zones`           : list of `Zone` dataclasses in id-order
    - `.by_key(key)`     / `.by_id(zid)`
    - `.hard_mask`       : ``int8[nlat, nlon]``, 0 = unassigned
    - `.membership`      : ``float32[nlat, nlon, K]`` (soft blend)
    - `.mask_signature`  : 12-hex fingerprint (from zone_mask.sha256)
    - `.imd_categories`  : IMD rainfall category dict
    - `.physics_bounds()` -> per-zone physics bounds
"""
from .india_zones import (
    Zone,
    ZoneRegistry,
    ZoneMaskDrift,
    get_zones,
)

__all__ = ["Zone", "ZoneRegistry", "ZoneMaskDrift", "get_zones"]
