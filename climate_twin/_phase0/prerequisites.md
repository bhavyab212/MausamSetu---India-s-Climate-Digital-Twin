# Phase 0b — Prerequisite Verification

Read-only verification of every asset the Phase 1 zone registry will need. Findings recorded verbatim from filesystem inspection; no downloads performed.

## 1. Subdivision / boundary geometry — what actually exists

### `L:\MausamSetu\Subbasin\` — inspected

Not IMD-36. This is the CWC (Central Water Commission) India sub-basin catalogue.

| File | Size | Notes |
|------|-----:|-------|
| Subbasin.shp | 32.8 MB | 101 polygons |
| Subbasin.shx | 908 B | shape index |
| Subbasin.dbf | 31.6 KB | attribute table |
| Subbasin.prj | 480 B | **WGS_1984_Lambert_Conformal_Conic**, false easting 4 000 000, false northing 4 000 000, central meridian 80°E, standard parallels 12.47/35.17°N, latitude of origin 24°N, units metres |
| Subbasin.cpg | 5 B | UTF-8 |
| Subbasin.sbn / .sbx | 1.1 KB / 212 B | spatial index |
| Subbasin.shp.xml | 114 KB | ISO-19139 metadata |

Attribute fields: `objectid` (N), `sbconc` (C6), `sbcode` (C3), `bacode` (C2), `ba_name` (C150), `sub_basin` (C100), `shape_Leng` (F), `shape_Area` (F). Sample basin names: "Ganga Basin (Above Ramganga Confluence)", "Banas", "Bhagirathi", "Brahmani-Baitarni", "Indus (Barmer, Beas)", "Krishna (Bhima Upper/Lower)", "Kutch-Saurashtra", "Barak", "West-flowing rivers South of Tapi", "Cauvery Basin" (3 polys).

**Verdict:** hydrological, not meteorological. Cannot be used as an IMD subdivision proxy.

### Other polygon assets

- `L:\MausamSetu\climate_twin\India_States_2024.geojson` — 1024198 B — 36 Indian states/UTs. WGS84 lat/lon.
- `L:\MausamSetu\climate_twin\data\India_States_2024.geojson` — 1024198 B — duplicate (`app_v2.py:234` reads this copy).
- No other India polygon files anywhere in the tree.

### Decision (per user approval)

Zone boundaries in Phase 1 will be defined by the **intersection of a lat/lon rectangle** (from `india_zones.yaml`) **and a set of state names** (from `India_States_2024.geojson`). This is the "1 + 2 best possible things" answer the user selected.

Per-zone specification will be, e.g.:

```yaml
tamilnadu_ne:
  lat_range: [8.0, 13.5]
  lon_range: [77.0, 80.5]
  states: ["Tamil Nadu", "Puducherry"]
  mask_rule: "intersection"   # cell ∈ zone iff inside BOTH rectangle AND state polygon
```

`build_mask.py` will rasterise state polygons onto the 0.25° master grid once, then AND with the rectangle.

## 2. DEM / elevation raster

Recursive search across `L:\MausamSetu\` for `*.tif`, `*.tiff`, `*.hgt`, `*.dem`, `*.asc`, `SRTM*`, `ETOPO*`, `elevation*`, `dem*`:

All hits are third-party library artefacts inside `L:\MausamSetu\venv\Lib\site-packages\` (cartopy's `srtm.py` module, pygame test tif, absl `demangle.h`, dask `demo.py`). **No project-owned elevation raster exists.**

### Decision (per user approval)

Zone 8 Himalayan uses **latitude-only fallback**: cell ∈ zone 8 iff `lat > 30.0°N` AND (state ∈ {"Jammu & Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand", "Sikkim", "Arunachal Pradesh"}). Documented limitation: this admits some low-elevation valley cells that a proper `elev > 2500 m` filter would exclude. If a DEM is later supplied, `build_mask.py` will accept it and refine the mask; the zone hash will change and Phase 2's checkpoint contract will refuse old models — which is the correct behaviour.

## 3. Prior benchmark — what to beat

### `L:\MausamSetu\climate_twin\models\india\` — empty (0 model folders)
### `L:\MausamSetu\climate_twin\training\checkpoints\` — empty (0 files)
### `L:\MausamSetu\climate_twin\_archive_pre_official_data\` — 2.5 GB archive

Contents:

| Path | Contents |
|------|----------|
| `_archive_pre_official_data/checkpoints/` | 2,450 `.pt` files (cauvery/india walk-forward rounds) |
| `_archive_pre_official_data/models_registry/cauvery/cauvery_v1/` | rain-only rmse=0.0529 on old 19×17 grid |
| `_archive_pre_official_data/models_registry/cauvery/india_v1/` | rmse=0.0490 same grid |
| `_archive_pre_official_data/models_registry/india/india_v1/` | rmse=0.0619, 129×135 |
| `_archive_pre_official_data/models_registry/india/india_v112/` | rmse=0.0356, 129×135, 500 rounds, 3497 epochs |
| `_archive_pre_official_data/processed/cauvery_v2.nc` | old 12 MB annual cube |
| `_archive_pre_official_data/README.md` | "Do not restore. Old annual 2–3 channels vs new daily 4 channels — incompatible." |

**All four are on the OLD annual 2-channel normalised aggregate.** RMSE=0.0356 is on `[0,1]` normalised annual rain, not physical mm/day. Cannot be compared to the new daily 4-channel cube.

### `*report*.json`, `*metrics*.json`, `benchmark*.md` searches

`find L:/MausamSetu/climate_twin/ -iname "*report*.json" -o -iname "*metrics*.json" -o -iname "benchmark*.md" -o -iname "*benchmark*.json"` — **zero matches**.

### Decision (per user approval)

**Benchmark recording deferred to end of Phase 1.** Rationale: an honest number-to-beat requires zone-aware per-zone persistence + climatology skill on the 2024-2025 holdout. Zones don't exist yet, so this computation can't happen in Phase 0.

Phase 0 records `benchmark_status = "deferred_to_phase_1_close"` in `PHASE0_REPORT.md`. Phase 1's exit criterion becomes:

> Emit `climate_twin/_phase0/benchmark_persistence_climatology.json` with per-zone RMSE, MAE, CSI (both IMD-absolute and zone-percentile), and CRPS for persistence and climatology on 2024-2025 holdout.

Any model trained in Phase 2 must beat these numbers per zone (or state explicitly which zones regressed and why).

## 4. Processed cubes + manifest

### PRE-rebuild (baseline for the diff in the Phase 0 report)

| File | Size | mtime | Notes |
|------|-----:|-------|-------|
| `data/processed/india.nc` | 115.44 MB | 2026-08-06 11:35 IST | dims (2922, 129, 135), years 2018-2025 |
| `data/processed/cauvery.nc` | 4.94 MB | 2026-08-06 11:35 IST | same dims, NaN outside basin |
| `data/processed/manifest.yaml` | 2502 B | 2026-08-06 11:36 IST | 8-year cube, INSAT 2020-2021 partial |
| `data/processed/manifest.sig` | 12 B | 2026-08-06 11:36 IST | `737d81c2f0c9` |
| `data/processed/india_norm_stats.json` | 792 B | train_years 2018-2023 |
| `data/processed/cauvery_norm_stats.json` | 790 B | train_years 2018-2023 |

### POST-rebuild (target — after Phase 0c completes)

- 1951-2025 (75 years), same 0.25° 129×135 grid
- INSAT stays 2020-2021 (raw data-limited)
- New manifest.sig (12-hex)
- New norm_stats fit on train_years 1951-2022 (72 yrs)
- Expected india.nc ~1.1 GB, cauvery.nc ~50 MB
- **The diff between the old and new manifest.sig will be recorded** — this alone is the "cache-bust invalidator" the rest of the system trusts (`data_source.py:_bust_streamlit_caches_if_sig_changed`).

## 5. Contradictions with the plan text

### Contradiction 5.1 — subdivisions

**Plan text (Phase 1b):** _"Build the mask by RASTERISING IMD subdivision shapefiles onto the 0.25 degree master grid."_

**Reality:** No IMD subdivision shapefile exists in the project. Only assets are `Subbasin.shp` (101 hydrological basins) and `India_States_2024.geojson` (36 states).

**Resolution (approved):** Phase 1 uses the lat/lon-rectangle + state-list rule. This is documented as an approximation of IMD-36, not IMD-36 itself.

### Contradiction 5.2 — DEM

**Plan text (Phase 1a, zone 8):** _"HIMALAYAN >30N, elev>2500m snow physics not rain."_

**Reality:** No DEM. `elev > 2500 m` cannot be evaluated.

**Resolution (approved):** Latitude+state fallback documented above. Phase 1 report will render `qc/zone_08_himalayan_fallback.png` with an explicit "elevation not used — see prerequisites.md §2".

### Contradiction 5.3 — 73-year data richness

**Plan text (Phase 2d):** _"ConvLSTM backbone sees all of India (learns general monsoon dynamics from the full 73 years — this is where the data richness lives)."_

**Reality of the pre-rebuild cube:** covers 2018-2025 (8 years) only. Raw sources go back to 1951, but the cube was truncated.

**Resolution (approved):** Phase 0c rebuilds the cube for 1951-2025 (75 years, 72 train + 1 val + 2 test). "Full 73 years" is a good approximation of the 72 training years. After Phase 0c the plan text is factually correct.

### Contradiction 5.4 — no compatible benchmark

**Plan text (Phase 0b):** _"Current global RMSE/CSI per variable on the test year (this is the number to beat)."_

**Reality:** All prior checkpoints are on incompatible cubes; no comparable RMSE exists.

**Resolution (approved):** Deferred to Phase 1 close per §3. Phase 0 does not emit a placeholder number.

### Contradiction 5.5 — INSAT partial coverage

**Not in the plan text but relevant:** INSAT LST files cover only 2020-04-01 → 2021-09-30 (362 daily files, ~1.5 yr). The plan's Phase 2e satellite-fusion path assumes INSAT is a channel; that channel will be all-NaN outside 2020-2021 in every historical training year.

**Resolution:** The Phase 5d checkpoint contract + Phase 5c coverage-aware rounds already handle this correctly. INSAT is dropped from any round whose years touch pre-2020 or post-2021. Documented — no plan change needed.

## Approvals checklist for Phase 1 entry

- [x] Zone-boundary source decided (lat/lon + state polygon intersection)
- [x] DEM fallback decided (latitude + state, documented limitation)
- [x] Benchmark strategy decided (per-zone persistence+climatology at Phase 1 close)
- [x] Cube-rebuild scope decided (1951-2025, train 1951-2022, val 2023, test 2024-2025)
- [ ] Cube rebuild completed (Phase 0c) — awaiting completion; report will state new manifest.sig
- [ ] `tests/test_phase5.py` re-run against new cube (67/69 physical-consistency checks) — awaiting rebuild
- [x] Archive plan decided (atomic rename at end of Phase 0, with import shim so app still boots)
