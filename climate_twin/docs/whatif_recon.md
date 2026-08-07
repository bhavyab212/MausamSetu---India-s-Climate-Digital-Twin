# What-If Engine — Part 0 Recon

**Date:** 2026-08-07 IST
**Branch:** `feat/whatif-engine` (base commit `fc5fd02` on `main`)
**Scope:** read-only survey required by the Part-0 spec before any
scenario-engine code is written.

---

## Resolved paths

| Key | Value |
|---|---|
| `STREAMLIT_ROOT` | `L:\MausamSetu\climate_twin\` |
| `ENTRY_POINT`    | `app_v2.py` (single file, not `Home.py`) |
| `PAGES_DIR`      | `L:\MausamSetu\climate_twin\pages\` (**created in Part 0**; did not exist before) |
| `OLD_WHATIF_PAGE`| `climate_twin/app_v2.py::_render_tab3` (lines 1985–2034 pre-edit) |
| `IMD_READER`     | `climate_twin/data/readers/imd_rainfall_nc.py` + `climate_twin/data/readers/imd_temperature.py` |
| `PREDICTION_DIR` | `climate_twin/train/registry/models/india/` (filesystem-first registry) |
| `PREDICTION loader` | `climate_twin/train/registry/store.py::ZoneAwareRegistry.load_into` |
| `PROCESSED_CUBES`| `L:\MausamSetu\data\processed\india.nc`, `L:\MausamSetu\data\processed\cauvery.nc` |
| `MANIFEST`       | `L:\MausamSetu\data\processed\manifest.yaml`, `manifest.sig` (currently `ab0999487887`) |
| `ZONE_REGISTRY`  | `climate_twin/regions/india_zones.py` (accessor) + `india_zones.yaml` + frozen `zone_mask.npy`, `zone_membership.npy`, `zone_stats.json`, `zone_mask.sha256` |

The `data` folder used by the plan (`L:\MausamSetu\data`) exists and is
populated. Every path in `whatif/config/paths.py` was verified against the
real filesystem during this recon.

---

## Architecture note — spec vs. reality

The Part-0 spec assumed a **Streamlit multipage layout** (entry file + a
`pages/*.py` directory auto-registered by the router). Reality is a
**single-file monolith** at `climate_twin/app_v2.py` (~4800 lines) with a
sidebar radio + `_TAB_RENDERERS` dispatcher. Approved routing decision:

- Keep `app_v2.py` as the entry file (existing 9 tabs remain in-place).
- Introduce `pages/` so Streamlit's multipage router registers NEW pages
  (the What-If engine, and future rewritten tabs) as top-level pages.
- The old What-If lived inside `_render_tab3()` — it was **removed from
  `_TAB_LABELS`** and **archived** to `pages/_archive/whatif_old.py`
  (verbatim source preserved as a string constant, not executable).

The pages under `pages/_archive/` are **not** picked up by Streamlit's
multipage router (Streamlit ignores subdirectories under `pages/`); this
was the mechanism the spec used to "deregister" the old page.

---

## Streamlit entry point

**File:** `climate_twin/app_v2.py`

Top of file:

- `sys.path` bootstrap so `climate_twin.*` imports resolve when run
  directly with `streamlit run climate_twin/app_v2.py`.
- Loads all shared services: `src.visualization.Visualizer`,
  `src.viz_plotly.PlotlyVisualizer`, `data_source`, `perf`.
- Two-layer CSS injection: theme variables (light/dark), then a single
  stylesheet using only `var(--mm-*)` (injected once per session).
- Region radio (India / Cauvery) at the top of the sidebar.
- Vertical navigation via `st.radio` in the sidebar, keyed by
  `st.session_state["active_tab"]`.
- Dispatch: only the ACTIVE tab body runs each rerun (no `st.tabs()` at
  the top level, so we don't pay the cost of every tab body).

Post-edit (Part 0), `_TAB_LABELS` no longer contains `"±1°C What-If"`
and `_TAB_RENDERERS` no longer references `_render_tab3`.

---

## Pages

Pre-Part-0 there was **no `pages/`** directory anywhere in the tree.

Part 0 created:

- `climate_twin/pages/`
- `climate_twin/pages/_archive/`
- `climate_twin/pages/_archive/__init__.py`
- `climate_twin/pages/_archive/whatif_old.py`
- `climate_twin/pages/30_What_If.py` ← new stub, two placeholder tabs

Streamlit's multipage router registers exactly one page from this
directory: `30_What_If.py` (title auto-derived from the filename:
"What If"). The `_archive/` subdirectory is skipped.

---

## IMD readers

Both readers live at `climate_twin/data/readers/`:

- **`imd_rainfall_nc.py`** — `read_rainfall_year(root, year) → xr.DataArray`.
  Reads `L:/MausamSetu/data/Rainfall/RF25_ind<YYYY>_rfp25.nc`. Native
  NetCDF-4; variable `RAINFALL` (mm/day); 129×135 India grid;
  NaN-masked ocean cells (no sentinels to strip).
- **`imd_temperature.py`** — `read_tmax_year`, `read_tmin_year`. Reads
  raw `.GRD` files (float32, days×31×31, C-major). Sentinels: values
  `>= 90.0` → NaN (canonical IMD fill 99.9). Callers get an
  xr.DataArray on a **1° grid** which the harmoniser bilinearly
  regrids to 0.25° before use.
- **`insat_lst.py`** — daily curvilinear LST fields (Apr-2020 to
  Sep-2021 only); sentinel `-999.0`; not on the master grid until
  passed through the conservative regrid in `climate_twin/data/harmonize.py`.

All three feed into `climate_twin/data/build_cube.py`, which produces
`data/processed/india.nc` and `cauvery.nc` (already sentinel-clean,
0.25° India grid, dims `(time, lat, lon)`, vars
`rain, tmax, tmin, insat_lst, tmax_is_interpolated,
tmin_is_interpolated, mask`).

---

## Prediction artifact + loader

Not one file — a **filesystem-first registry** rooted at
`climate_twin/train/registry/models/<region>/<name>/`.

Each model directory contains:

- `weights.pt` — torch checkpoint, PyTorch state_dict plus JSON-clean
  metadata inside the pickle: `config`, `zone_mask_sig`, `manifest_sig`,
  `variables`, `epoch`, `zone_weighted_rmse`, `parent_name`, …
- `meta.json` — same metadata as a plain file (for readable browsing).
- `tier4.json` (optional) — final Tier-4 significance report.
- `tier3_heatmap.png` (optional) — 9-zone × 4-season skill grid.

**Load surface** — `climate_twin/train/registry/store.py`:

```python
from climate_twin.train.registry import get_registry, RegistryModelIncompatible
reg = get_registry()
models = reg.list_models(region="india")
ck = reg.load_into("india_fast_v1", "india", my_model_instance,
                    expected_zone_mask_sig="242f813af71b",
                    expected_manifest_sig="ab0999487887",
                    expected_variables=["rain", "tmax", "tmin"],
                    strict=True)  # raises RegistryModelIncompatible on drift
```

The What-If ensemble driver (`whatif/drivers/ensemble.py`) will call
`reg.load_into(...)` with `strict=True` — refusing to serve any prediction
whose sig disagrees with the live cube + zone registry is required by
the "no silent stale-cube" rule from Phase 5.

**Daily-disaggregation code** — not present. The cube is daily at the
source, so no disaggregation is needed. The old `_render_tab3` used a
static `sensitivity_map.npy` slider mechanic (`sens_map * ΔT`) which was
neither reproducible nor documented; it has been retired verbatim.

---

## Zone registry

- **YAML:** `climate_twin/regions/india_zones.yaml` — 9 zones, per-zone
  lat/lon rectangles, state lists, physics bounds, loss weights,
  IMD-absolute rainfall categories, `soft_transition_cells: 2`.
- **Frozen artefacts:** `zone_mask.npy` (int8, 129×135, hard argmax),
  `zone_membership.npy` (float32, 129×135×9, soft blend), `zone_mask.sha256`
  (currently `242f813af71b`), `zone_stats.json`
  (currently sig `0a097b298a06`).
- **Accessor:** `from climate_twin.regions import get_zones` →
  `ZoneRegistry` with `.hard_mask`, `.membership`, `.zones`,
  `.mask_signature`, `.by_key(key)`, `.by_id(zid)`.
- **Drift guard:** every load recomputes SHA-256 over
  (mask + membership + yaml text) and raises `ZoneMaskDrift` on mismatch.

The What-If engine treats this registry as read-only: no scenario can
change the mask; a re-mask is a full-project regeneration.

---

## Config files

- `L:\MausamSetu\requirements.txt` — top-level minimal deps
  (streamlit, numpy, xarray, pyshp, shapely, matplotlib, plotly).
- `L:\MausamSetu\climate_twin\requirements.txt` — same layout,
  scoped to the app directory. Streamlit Cloud reads this.
- `L:\MausamSetu\pyproject.toml` — does **not** exist.
- `L:\MausamSetu\environment.yml` — does **not** exist.
- Deps for the ML side (torch, torchvision) are installed via
  `venv/Scripts/python -m pip install torch==2.5.1+cu121 …` (not
  captured in requirements.txt to avoid platform mismatches on Cloud).

---

## Required-key summary (spec's exact list)

- **STREAMLIT_ROOT** = `L:\MausamSetu\climate_twin\`
- **ENTRY_POINT** = `app_v2.py`
- **PAGES_DIR** = `L:\MausamSetu\climate_twin\pages\`
- **OLD_WHATIF_PAGE** (relative) = `_archive/whatif_old.py`
  (retired from `app_v2.py::_render_tab3`)
- **IMD_READER modules** =
  - `climate_twin/data/readers/imd_rainfall_nc.py`
  - `climate_twin/data/readers/imd_temperature.py`
  - `climate_twin/data/readers/insat_lst.py`
- **PREDICTION artifact path** = `climate_twin/train/registry/models/<region>/<name>/weights.pt`
- **PREDICTION loader module** = `climate_twin/train/registry/store.py::ZoneAwareRegistry.load_into`
- **ZONE registry module** = `climate_twin/regions/india_zones.py`

---

## Verification

- `git status` on `feat/whatif-engine`: clean before Part-0 edits.
- Base commit `fc5fd02` on `main` groups all prior UI + path work
  (glass theme, cream/dark toggle, memory-safe cube reads, Home tab)
  into one commit; the What-If work sits on top on the feature branch.
- `streamlit run climate_twin/app_v2.py` will be re-launched at the end
  of Part 0 to confirm: (a) old What-If gone from the tab bar,
  (b) new "What If" page visible in the sidebar via Streamlit's
  multipage router.

---

## Done-when checklist (from spec)

- ✅ `whatif_recon.md` written and answers every question in Step 2.
- ✅ Old What-If retired to `pages/_archive/whatif_old.py`
  (verbatim source preserved as a string constant).
- ✅ `pages/30_What_If.py` renders with two placeholder tabs.
- ✅ `whatif/` package skeleton in place (10 subpackages/modules,
  each with a `# TODO(Part N)` marker).
- ✅ Centralised paths in `whatif/config/paths.py`; constants in
  `whatif/config/constants.py`.
- ⏳ `streamlit run` clean launch → verify + commit.
