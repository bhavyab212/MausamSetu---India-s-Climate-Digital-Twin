"""
regions/build_mask.py — rasterise `india_zones.yaml` onto the 0.25° master grid.

Runs once (from the CLI) to produce three artefacts, all committed as the
frozen definition of the zoning:

    zone_mask.npy         int8   (129, 135)      hard argmax mask (0 = unassigned)
    zone_membership.npy   float32 (129, 135, 9)  soft membership, rows sum to 1
    zone_mask.sha256      12-hex signature over yaml + mask + membership

Priority rule (per `india_zones.yaml`):
  For each cell (lat_i, lon_j):
    - Determine which state (from India_States_2024.geojson) contains the cell.
    - For each zone (walked in priority order 1..9), the zone "matches" iff
        (lat_i, lon_j) ∈ lat_range × lon_range AND state ∈ zone.states.
    - The winning zone is the lowest-priority-number match.
    - Cells matching no zone → id 0 (unassigned).

Soft membership:
  Start with a one-hot vector for the hard zone id.  In a 2-cell
  (`soft_transition_cells`) band around every zone boundary, linearly blend
  each cell's own zone weight with the weight of the neighbouring zone.
  The blend weight at distance k cells from the boundary is
      w_self  = 0.5 + 0.5 * k / T          (T = soft_transition_cells)
      w_other = 1 - w_self
  Interior cells stay one-hot; boundary cells become mixtures.  A cell can
  neighbour multiple zones — the resulting vector is renormalised to sum
  to 1 at the very end.

Elevation: not used (no DEM in project).  Zone 8 (Himalayan) falls back on
`lat > 30° AND state ∈ Himalayan states`, documented in `india_zones.yaml`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import shape, Point
from shapely.prepared import prep

IST = timezone(timedelta(hours=5, minutes=30))
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
YAML_PATH = HERE / "india_zones.yaml"
STATES_GEOJSON = REPO / "India_States_2024.geojson"
MASK_PATH = HERE / "zone_mask.npy"
MEMBERSHIP_PATH = HERE / "zone_membership.npy"
HASH_PATH = HERE / "zone_mask.sha256"
BUILD_LOG_PATH = HERE / "build_mask.log.json"


# ---------------------------------------------------------------------------
# yaml + geometry loaders
# ---------------------------------------------------------------------------
def load_yaml() -> dict[str, Any]:
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def load_states() -> dict[str, Any]:
    """Return {state_name: prepared_polygon}."""
    gj = json.loads(STATES_GEOJSON.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for feat in gj["features"]:
        name = feat["properties"]["ST_NM"]
        geom = shape(feat["geometry"]).buffer(0)      # heal invalid polys
        out[name] = prep(geom)                        # prepared → fast .contains
    return out


# ---------------------------------------------------------------------------
# mask construction
# ---------------------------------------------------------------------------
def build_hard_mask(cfg: dict, states: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (hard_mask_int8, state_index_int8, stats)."""
    mg = cfg["master_grid"]
    nlat, nlon = mg["n_lat"], mg["n_lon"]
    lat0, lat1 = mg["extent_lat"]
    lon0, lon1 = mg["extent_lon"]
    lats = np.linspace(lat0, lat1, nlat)
    lons = np.linspace(lon0, lon1, nlon)

    # 1. State-of-cell raster (int8; -1 = ocean/outside India)
    print(f"[build_mask] rasterising {len(states)} states to {nlat}x{nlon} grid…")
    state_names = list(states.keys())
    state_idx = np.full((nlat, nlon), -1, dtype=np.int8)
    for si, name in enumerate(state_names):
        poly = states[name]
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                if state_idx[i, j] != -1:
                    continue
                if poly.contains(Point(float(lo), float(la))):
                    state_idx[i, j] = si

    n_land = int((state_idx >= 0).sum())
    print(f"[build_mask] {n_land}/{nlat*nlon} cells fall inside some state polygon")

    # 2. Priority-ordered zone assignment
    zones_by_priority = sorted(cfg["zones"], key=lambda z: z["priority"])
    hard = np.zeros((nlat, nlon), dtype=np.int8)   # 0 = unassigned
    per_zone_counts: dict[str, int] = {z["key"]: 0 for z in zones_by_priority}

    for zone in zones_by_priority:
        zid = int(zone["id"])
        lat_range = zone["lat_range"]
        lon_range = zone["lon_range"]
        allowed_states = set(zone["states"])
        allowed_idx = {state_names.index(n) for n in allowed_states if n in state_names}

        for i, la in enumerate(lats):
            if not (lat_range[0] <= la <= lat_range[1]):
                continue
            for j, lo in enumerate(lons):
                if hard[i, j] != 0:
                    continue                               # earlier priority won
                if not (lon_range[0] <= lo <= lon_range[1]):
                    continue
                if state_idx[i, j] not in allowed_idx:
                    continue
                hard[i, j] = zid
                per_zone_counts[zone["key"]] += 1

    unassigned = int((hard == 0).sum())
    unassigned_land = int(((hard == 0) & (state_idx >= 0)).sum())
    stats = {
        "n_cells_total": int(nlat * nlon),
        "n_land_cells": n_land,
        "n_assigned_cells": int((hard > 0).sum()),
        "n_unassigned_ocean": int(((hard == 0) & (state_idx == -1)).sum()),
        "n_unassigned_land": unassigned_land,
        "per_zone_counts": per_zone_counts,
    }
    return hard, state_idx, stats


def build_soft_membership(hard: np.ndarray, zone_ids: list[int],
                          transition_cells: int) -> np.ndarray:
    """Linear soft blending in a T-cell band around every zone boundary.

    Returns (nlat, nlon, K) float32, rows summing to 1.  Unassigned cells
    (hard == 0) get an all-zero row.
    """
    T = int(transition_cells)
    nlat, nlon = hard.shape
    K = len(zone_ids)
    zid_to_k = {zid: k for k, zid in enumerate(zone_ids)}

    memb = np.zeros((nlat, nlon, K), dtype=np.float32)
    # seed with one-hot
    for i in range(nlat):
        for j in range(nlon):
            zid = int(hard[i, j])
            if zid == 0:
                continue
            memb[i, j, zid_to_k[zid]] = 1.0

    if T <= 0:
        return memb

    # For every cell, look up neighbours within T cells (Chebyshev distance).
    # If a neighbour belongs to a DIFFERENT zone, mix in that zone's weight
    # proportionally to (T - d + 1) / T (nearer = higher influence).
    for i in range(nlat):
        for j in range(nlon):
            zid = int(hard[i, j])
            if zid == 0:
                continue
            k_self = zid_to_k[zid]
            neighbour_influence = {}     # neighbour zone k → summed weight
            for di in range(-T, T + 1):
                for dj in range(-T, T + 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if not (0 <= ii < nlat and 0 <= jj < nlon):
                        continue
                    other = int(hard[ii, jj])
                    if other == 0 or other == zid:
                        continue
                    d = max(abs(di), abs(dj))
                    w = (T - d + 1) / (T * (2 * T + 1))    # normalised inside band
                    neighbour_influence[zid_to_k[other]] = (
                        neighbour_influence.get(zid_to_k[other], 0.0) + w
                    )
            total_influence = sum(neighbour_influence.values())
            if total_influence > 0:
                # Cap the "outside" influence at 0.5 — interior cells keep
                # majority self-weight, deep boundary cells drop to ~0.5.
                total_influence = min(total_influence, 0.5)
                memb[i, j, k_self] = 1.0 - total_influence
                for k_other, w in neighbour_influence.items():
                    memb[i, j, k_other] = total_influence * (
                        w / sum(neighbour_influence.values())
                    )

    # Renormalise defensively
    row_sums = memb.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        memb = np.where(row_sums > 0, memb / np.maximum(row_sums, 1e-12), 0.0)
    return memb.astype(np.float32)


# ---------------------------------------------------------------------------
# freeze + hash
# ---------------------------------------------------------------------------
def compute_signature(hard: np.ndarray, memb: np.ndarray, yaml_text: str) -> str:
    h = hashlib.sha256()
    h.update(hard.tobytes(order="C"))
    h.update(memb.tobytes(order="C"))
    h.update(yaml_text.encode("utf-8"))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    cfg = load_yaml()
    yaml_text = YAML_PATH.read_text(encoding="utf-8")
    states = load_states()

    hard, state_idx, stats = build_hard_mask(cfg, states)

    zone_ids = [z["id"] for z in sorted(cfg["zones"], key=lambda z: z["id"])]
    memb = build_soft_membership(
        hard, zone_ids,
        transition_cells=int(cfg["soft_transition_cells"]),
    )

    sig = compute_signature(hard, memb, yaml_text)

    # ── write
    np.save(MASK_PATH, hard)
    np.save(MEMBERSHIP_PATH, memb)
    HASH_PATH.write_text(sig)

    log = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "signature": sig,
        "yaml_bytes": len(yaml_text.encode("utf-8")),
        "mask_shape": list(hard.shape),
        "membership_shape": list(memb.shape),
        "stats": stats,
        "per_zone_counts": {
            z["key"]: int((hard == z["id"]).sum()) for z in cfg["zones"]
        },
    }
    BUILD_LOG_PATH.write_text(json.dumps(log, indent=2, default=str))

    print()
    print("── build_mask summary ─────────────────────────────────────")
    print(f"  mask_signature     = {sig}")
    print(f"  total cells        = {stats['n_cells_total']}")
    print(f"  inside-any-state   = {stats['n_land_cells']}")
    print(f"  assigned           = {stats['n_assigned_cells']}")
    print(f"  unassigned (ocean) = {stats['n_unassigned_ocean']}")
    print(f"  unassigned (land)  = {stats['n_unassigned_land']}"
          + ("  ⚠ some land cells outside all zone rectangles" if stats["n_unassigned_land"] else ""))
    print()
    print("  per-zone counts:")
    for z in sorted(cfg["zones"], key=lambda z: z["id"]):
        c = int((hard == z["id"]).sum())
        print(f"    {z['id']}  {z['key']:22s}  {c:5d} cells  ({z['label']})")
    print()
    print(f"  wrote {MASK_PATH.name}, {MEMBERSHIP_PATH.name}, {HASH_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
