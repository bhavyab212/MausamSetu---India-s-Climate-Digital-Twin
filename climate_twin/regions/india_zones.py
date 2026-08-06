"""
india_zones.py — accessor for the frozen zone registry.

The mask and membership are BUILT ONCE by `build_mask.py` and then never
re-derived at runtime. This module reads them and validates the SHA-256
signature. Any drift raises ``ZoneMaskDrift`` — because a changed mask
makes all prior metrics incomparable.

Callers should almost always use::

    from climate_twin.regions import get_zones
    Z = get_zones()
    Z.hard_mask          # (129, 135) int8
    Z.membership         # (129, 135, 9) float32 rows sum to 1
    Z.zones              # list[Zone]  ordered by id
    Z.by_key("thar_arid")

The dataclass ``Zone`` carries every parameter frozen at build time.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
YAML_PATH = HERE / "india_zones.yaml"
MASK_PATH = HERE / "zone_mask.npy"
MEMBERSHIP_PATH = HERE / "zone_membership.npy"
HASH_PATH = HERE / "zone_mask.sha256"


class ZoneMaskDrift(RuntimeError):
    """Raised when the on-disk mask, membership, or yaml disagree with the
    frozen ``zone_mask.sha256`` signature. This means somebody changed the
    definition after training; refuse to proceed until they intended to."""


@dataclass(frozen=True)
class Zone:
    """One zone's frozen parameters."""
    key: str
    id: int
    label: str
    priority: int
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    states: tuple[str, ...]
    rationale: str
    seasonality_wet_months: tuple[int, ...]
    elev_min_m: float | None = None
    # populated at load time from `physics_bounds:` block
    rain_max_mm_day: float | None = None
    tmax_c: tuple[float, float] | None = None
    tmin_c: tuple[float, float] | None = None
    # populated at load time from `validation_tier:` block
    validation_tier: str = "full"


@dataclass(frozen=True)
class ZoneRegistry:
    """Singleton view of the frozen zone system."""
    yaml_path: Path
    mask_path: Path
    membership_path: Path
    mask_signature: str
    zones: tuple[Zone, ...]
    hard_mask: np.ndarray       # (nlat, nlon) int8
    membership: np.ndarray      # (nlat, nlon, K) float32
    zone_ids: tuple[int, ...]   # ordered same as membership last axis
    master_grid: dict[str, Any]
    imd_categories: dict[str, tuple[float, float]]
    soft_transition_cells: int
    min_cells_per_split: int
    loss_weighting_mode: str
    loss_weighting_clip_ratio: tuple[float, float]

    # ── lookups ─────────────────────────────────────────────────
    def by_key(self, key: str) -> Zone:
        for z in self.zones:
            if z.key == key:
                return z
        raise KeyError(f"unknown zone key {key!r}")

    def by_id(self, zid: int) -> Zone:
        for z in self.zones:
            if z.id == zid:
                return z
        raise KeyError(f"unknown zone id {zid}")

    def keys(self) -> list[str]:
        return [z.key for z in self.zones]

    def ids(self) -> list[int]:
        return list(self.zone_ids)

    def n_zones(self) -> int:
        return len(self.zones)


# ---------------------------------------------------------------------------
# freeze verification
# ---------------------------------------------------------------------------
def _compute_signature(hard: np.ndarray, memb: np.ndarray, yaml_text: str) -> str:
    h = hashlib.sha256()
    h.update(hard.tobytes(order="C"))
    h.update(memb.tobytes(order="C"))
    h.update(yaml_text.encode("utf-8"))
    return h.hexdigest()[:12]


def _load_and_verify() -> ZoneRegistry:
    for p in (YAML_PATH, MASK_PATH, MEMBERSHIP_PATH, HASH_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"zone registry incomplete — {p.name} missing. "
                f"Run `python -m climate_twin.regions.build_mask` first."
            )

    yaml_text = YAML_PATH.read_text(encoding="utf-8")
    cfg = yaml.safe_load(yaml_text)
    hard = np.load(MASK_PATH)
    memb = np.load(MEMBERSHIP_PATH)
    stored_sig = HASH_PATH.read_text().strip()
    computed_sig = _compute_signature(hard, memb, yaml_text)

    if stored_sig != computed_sig:
        raise ZoneMaskDrift(
            f"zone signature drift: stored={stored_sig!r}  computed={computed_sig!r}. "
            f"Yaml, mask, or membership was modified after freeze. "
            f"Either revert changes or rebuild with `python -m climate_twin.regions.build_mask` "
            f"and be aware every prior metric is now incomparable."
        )

    # Bind Zone dataclasses
    phys = cfg.get("physics_bounds", {})
    tiers = cfg.get("validation_tier", {})
    zones: list[Zone] = []
    for zc in sorted(cfg["zones"], key=lambda z: z["id"]):
        pb = phys.get(zc["key"], {})
        zones.append(Zone(
            key=zc["key"],
            id=int(zc["id"]),
            label=zc["label"],
            priority=int(zc["priority"]),
            lat_range=tuple(zc["lat_range"]),
            lon_range=tuple(zc["lon_range"]),
            states=tuple(zc["states"]),
            rationale=zc.get("rationale", "").strip(),
            seasonality_wet_months=tuple(zc.get("seasonality_wet_months", [])),
            elev_min_m=zc.get("elev_min_m"),
            rain_max_mm_day=pb.get("rain_max_mm_day"),
            tmax_c=tuple(pb["tmax_c"]) if pb.get("tmax_c") else None,
            tmin_c=tuple(pb["tmin_c"]) if pb.get("tmin_c") else None,
            validation_tier=str(tiers.get(zc["key"], "full")),
        ))

    zone_ids = tuple(z.id for z in zones)
    imd_cat = {k: tuple(v) for k, v in cfg.get("imd_rainfall_categories_mm_day", {}).items()}
    lw = cfg.get("loss_weighting", {}) or {}

    return ZoneRegistry(
        yaml_path=YAML_PATH,
        mask_path=MASK_PATH,
        membership_path=MEMBERSHIP_PATH,
        mask_signature=stored_sig,
        zones=tuple(zones),
        hard_mask=hard,
        membership=memb,
        zone_ids=zone_ids,
        master_grid=dict(cfg["master_grid"]),
        imd_categories=imd_cat,
        soft_transition_cells=int(cfg.get("soft_transition_cells", 2)),
        min_cells_per_split=int(cfg.get("min_cells_per_split", 5)),
        loss_weighting_mode=str(lw.get("mode", "inverse_variance")),
        loss_weighting_clip_ratio=tuple(lw.get("clip_ratio", [0.2, 5.0])),
    )


@lru_cache(maxsize=1)
def get_zones() -> ZoneRegistry:
    """Cached-singleton entry point. Every call after the first is O(1)."""
    return _load_and_verify()


if __name__ == "__main__":
    Z = get_zones()
    print(f"signature = {Z.mask_signature}")
    print(f"n_zones   = {Z.n_zones()}")
    print(f"hard_mask = {Z.hard_mask.shape}  {Z.hard_mask.dtype}")
    print(f"soft memb = {Z.membership.shape} {Z.membership.dtype}")
    print()
    print("Zones (by id):")
    for z in Z.zones:
        cnt = int((Z.hard_mask == z.id).sum())
        print(f"  {z.id}  {z.key:22s}  cells={cnt:5d}  "
              f"tier={z.validation_tier}  rain_max={z.rain_max_mm_day} mm/day  "
              f"tmax={z.tmax_c}  tmin={z.tmin_c}")
