# MausamSetu · ClimateTwin Lab — Project Memory

> **Codename:** **ClimateTwin Lab** (folder: `climate_twin/`)
> Part of the **MausamSetu · मौसम सेतु** project (India climate digital twin).
> Last updated: 2026-08-02

---

## 0. What this is & where it came from

**ClimateTwin Lab** is an enhanced fork of a cloned Hugging Face Space:

- **Origin clone:** `https://huggingface.co/spaces/Sujith2005/ISRO-Climate-Digital-Twin`
- Cloned into the top-level folder **`climate_twin/`** inside the `L:\MausamSetu` repo.
- The original was a Streamlit app (TensorFlow/Keras ConvLSTM + U-Net ensemble) that
  predicted annual rainfall/temperature for India and pulled its data from a HF dataset.

We rerouted it to the user's **local IMD data**, added a **region toggle (India / Cauvery)**,
a full **walk-forward Training + Validation system** (PyTorch, GPU), a **model registry**,
rich **W&B-style dashboards**, **matplotlib animations**, a **Cauvery spotlight map**, and did a
full **performance pass**. The original cloud/data logic is preserved but bypassed.

**Entry point:** `climate_twin/app_v2.py` (Streamlit). Runs on port **8502**.

---

## 1. How to run

```bash
# from the repo, start the app (must cd into climate_twin so relative paths resolve)
cd /l/MausamSetu/climate_twin
MSYS_NO_PATHCONV=1 ../venv/Scripts/streamlit run app_v2.py --server.port 8502
# open http://localhost:8502

# stop
netstat -ano | grep :8502           # find PID
taskkill //PID <PID> //F
```

- Python venv: `L:\MausamSetu\venv` (Windows, Git Bash shell).
- GPU: PyTorch 2.5.1+cu121 (CUDA) for training; TensorFlow is CPU-only on native Windows (inference).
- Config: `climate_twin/.streamlit/config.toml` (headless, runOnSave off, fileWatcher none).

---

## 2. Data pipeline (the reroute)

**Single source of truth: `climate_twin/data_source.py`.** All data access routes through it.
No cloud/HF downloads — everything is built from the user's **local raw IMD files**.

### Raw inputs (never modified)
| Folder | Files | Grid | Fill |
|---|---|---|---|
| `L:\MausamSetu\data\IMD rainfall data\` | `Rainfall_indYYYY_rfp25.grd` (1901–2025) | 129×135 @ 0.25° | −999.0 |
| `L:\MausamSetu\data\IMD max temp data\` | `Maxtemp_MaxT_YYYY.GRD` (1951–2025) | 31×31 @ 1° | 99.9 |
| `L:\MausamSetu\data\IMD min temp data\` | `Mintemp_MinT_YYYY.GRD` (1951–2025) | 31×31 @ 1° | 99.9 |
| `L:\MausamSetu\Subbasin\` | `Subbasin.shp` (+.dbf/.shx/.prj…) | ESRI shapefile, **101 all-India sub-basins**, Lambert-Conformal (metres) | — |

**Key correction discovered:** `Subbasin/` is a **boundary shapefile, not climate data**. The
Cauvery region's values come from **clipping the raw India grids** with the Cauvery polygons
(records 14/15/16 = "Cauvery Basin", reprojected LCC→WGS84 via pyproj).

### Region registry (`data_source.REGIONS`)
| region | label | extent | grid | ckpt prefix | processed cache |
|---|---|---|---|---|---|
| `india` | India (Full) | lat 6.5–38.5°N, lon 66.5–100.0°E | **129×135** | `india_round_*` | `data/processed/india.nc` |
| `cauvery` | Cauvery Basin | lat 10.0–14.5°N, lon 75.5–79.5°E | **19×17** | `cauvery_round_*` | `data/processed/cauvery.nc` |

### What data_source.py builds
- Reads `.grd` (rainfall 129×135, fill→NaN), `.GRD` temp (31×31 → bilinear regrid to 0.25°).
- `cauvery_polygon()` — reprojects the 3 Cauvery polygons to WGS84, dissolves them.
- `_basin_mask()` — rasterizes the polygon onto the region grid (India = all land).
- `build_cube(region)` — **annual** aggregates: rain = yearly cumulative, tmax = yearly max,
  tmin = yearly min; masked; written to `data/processed/<region>.nc` (rebuild cache).
- Public API: `load_region()`, `load_aggregates()` → `(rain, temp, mask, years)`,
  `daily_rain()`, `daily_tmax()`, `region_info()`, `region_matches_shape()`, `region_grid()`.
- **India cube reproduces the original `aggregates.npz` bit-for-bit** (verified corr=1.0) — proof
  the readers/orientation are correct. Cauvery = 51 years (1975–2025), 109 in-basin cells of 323.

### The rerouted seam (in `app_v2.py` / `src/`)
- `src/utils.py::ensure_data_file` — cloud download **removed** (returns local path only).
- `src/daily_temp_loader.py` — `hf_hub_download` block **removed**.
- `app_v2.py::load_artifacts(region)` — new; calls `DS.load_aggregates(region)` (original kept as
  `_legacy_load_artifacts`).
- `app_v2.py::get_nc_year_data(year, region)` — reads local `.grd` via `DS.daily_rain` (no NetCDF).
- Daily map temp via `DS.daily_tmax`.
- **Zero external data calls** anywhere now.

---

## 3. Region toggle (India ↔ Cauvery)

- Global **🌍 Region** radio at the top of the sidebar (`key="active_region"`), plus a context panel
  (grid, land %, extent, cube status, checkpoint prefix). Devanagari branding preserved.
- `active_region()` helper; everything region-keyed.
- On region switch, `st.cache_data.clear()` runs so no cross-region array leaks (129×135 vs 19×17).
- `plotly_vis` (PlotlyVisualizer) rebuilt per region with that region's extent.
- Checkpoints & registry models are **region-prefixed** and region-filtered in every dropdown.

---

## 4. Walk-Forward Training + Validation system (🔥 Training tab, `_render_tab9`)

GPU-only (refuses CPU). Two workflows via a top toggle: **🏋️ Train** and **🔬 Validate**.

### training/ modules
| file | role |
|---|---|
| `training/schedule.py` | `ScheduleConfig`, `build_schedule`, `MAX_ROUNDS=500`, `Round` |
| `training/loops.py` | `RoundConfig`, `TrainingProgress`, `train_one_round`, masked-Huber loss |
| `training/model.py` | `ClimateTwinModel` (PyTorch residual ConvLSTM, ~113K params, `predict_ensemble` MC-dropout) |
| `training/metrics.py` | masked rmse/mae/bias/pearson, pod/far/csi, ensemble_calibration, compute_all_metrics |
| `training/baselines.py` | persistence + climatology baselines |
| `training/checkpoints.py` | region-prefixed round checkpoints `\<region>_round_NNN_\<period>.pt` |
| `training/state.py` | SQLite `runs.db` — per-round history (config, metrics, curves, baselines) |
| `training/registry.py` | **named model registry** (NEW) — see §5 |
| `training/viz.py` | figures: reliability, metric bars, sample maps, **walkforward_progress_figure**, **training_dashboard_figure** |

### 🏋️ Train
- **Model identity:** "Create new" (name it) or "Continue existing" (resume a saved model's weights).
- Full schedule + hyperparameter controls, **each with an ⓘ help tooltip** explaining what it does & its impact.
- **Live dashboard** (single, in-place — not a growing list): top progress bar (round/epoch/samples-s/GPU/ETA),
  6 metric tiles (Epoch, Train loss, Val loss, Val RMSE, GPU mem, LR), a **multi-panel diagnostics chart**
  (Loss / Grad norm / LR), and a collapsible **scientific console** narrating every phase + per-epoch line.
- On finish → **saved to the registry** under its name (accumulates epochs/rounds), with a success summary vs baselines.

### 🔬 Validate
- Select a saved model + a year → 20-pass MC-dropout inference → full report:
  - **Plain-language verdict card** (Strong/Decent/Weak) with physical-unit misses (mm, °C) and % skill vs baselines — laymen-friendly.
  - Scorecard tiles, detailed comparison table (Model vs Persistence vs Climatology, with Winner column).
  - **9-map gallery** (physical units): rain Observed/Predicted/Error/Uncertainty, the two baseline maps, temp Observed/Predicted/Error.
  - Metric-comparison bars + reliability diagram + "how to read this".

### Run History (redesigned, click-to-inspect)
- **📊 Training Dashboard** (DataRobot-style): multi-model overlay of Loss (train solid/val dotted) / Skill / LR / Grad
  on a 2×2 grid, x=iterations, with a **hyperparameter comparison table** (rows highlighted where models differ),
  and a "Models to show" multiselect.
- **Walk-Forward Progression** (big graph): 4 stacked panels vs round — RMSE (+persistence/climatology), CSI, train/val loss, bias.
- **🔍 Inspect a run**: dropdown → single-run detail (config chips, metrics-vs-baselines table, its own loss/grad/LR curves).
- Overview strip (rounds, completed, best RMSE, latest) + full table in an expander + Clear/Reset actions.

---

## 5. Model registry (`training/registry.py`, NEW)

- Named, resumable, validatable models. Table `models` in `runs.db`; weights at
  `training/checkpoints/model_<region>_<name>.pt`.
- `save_model` (upsert, accumulates epochs/rounds), `list_models(region)`, `get_model`, `load_into`, `delete_model`.
- **Region-scoped** (India 129×135 and Cauvery 19×17 can't share a model). If the active region has no
  models but others do, the UI tells you which region to switch to (this resolved a "model missing" confusion).
- **📦 Model Library** panel (in Train & Validate): list + delete + **📥 Import a .pt/.pth file** (our checkpoints
  or a custom PyTorch state_dict), shape-checked against the region grid, then registered.

---

## 6. Visualization changes

- **Maps smoothed + hi-res** (`src/viz_plotly.py`): `zsmooth="best"` on all heatmaps; small regions
  (Cauvery) upsampled 6× (India stays native to protect payload); light state-boundary outline for geographic context.
- **Cauvery spotlight-on-India** (`app_v2.focus_india_figure`): in Cauvery region, Daily Explorer maps render
  on the **full India canvas** with the basin bright and the rest a dark silhouette with an
  `exp(−distance)` darkening gradient (go.Image RGBA). India region keeps the normal full map. `_emit_map()` switches.
- **Matplotlib animation** ("Animated Indian Twin", replaced the CesiumJS globe):
  `build_matplotlib_animation_gif` renders the chrome once + PIL-composites frames (~17× faster than FuncAnimation).
  Smooth (gaussian+bicubic upsample), region-aware, Rainfall/Temperature, speed slider.
  **Full-year option**: `🗓️ Animate full year` → all 365 days (Jan 1→Dec 31), `month_idx=None`, date labels.

---

## 7. Performance pass (app was unusably laggy → smooth)

Measured via a toggleable **🐢 Perf panel** (`climate_twin/perf.py`, `@timed`).
- **Animation:** FuncAnimation redraw (~17 s) → chrome-once + PIL compose (~1 s). Biggest single win.
- **Fragments:** each of the 9 tabs wrapped in `@st.fragment`; a widget interaction reruns only its tab.
- **Active-tab-only render:** replaced `st.tabs()` (which executes all 9 bodies every rerun) with a
  **segmented control** that dispatches to only the selected tab's fragment → warm rerun 1752 ms → ~516 ms.
- **Caching:** `EnsemblePredictor` is a `@st.cache_resource` singleton (loaded once); data cubes cached to `.nc`.
- **MC-dropout** batched (chunks of 8) with `torch.inference_mode()`.
- **Config:** `runOnSave=false`, `fileWatcherType="none"`, `gatherUsageStats=false`, `fastReruns=true`.
- GPU is **not** touched on non-training pages (TF is CPU on Windows; PyTorch/CUDA only in Training).

---

## 8. Key bug fixes

- **Training "did nothing / GPU idle":** `DataLoader(drop_last=True, batch_size=8)` with ~5 annual samples =
  0 batches. Fixed: `batch=min(bs, n)`, `drop_last=False`, `num_workers=0`.
- **`_masked_loss`** indexed a `(H,W)` mask as `(B,H,W)` → crash on first real forward. Fixed to broadcast `(1,1,H,W)`.
- **`ensemble_calibration`** fed batched arrays → fixed to average per-sample 2-D fields.
- **All rounds skipped "not enough data":** `seq_len = min(30, window)` equalled the window → 0 samples.
  Fixed: `seq_len = max(2, min(5, train_win-2))` (annual data needs context < window).
- **Cross-region shape crashes** (`(129,135) vs (19,17)`): future-prediction/climatology caches weren't region-keyed →
  cleared `st.cache_data` on region switch; daily helpers now pass `active_region()`.
- **GIF `BytesIO` save** → PillowWriter needs a path → save via `BytesIO` GIF writer / temp handling.

---

## 9. File map (climate_twin/)

```
app_v2.py            Streamlit entry — all tabs, training UI, focus map, animation
data_source.py       SINGLE data source of truth (regions, raw IMD, shapefile clip) [NEW]
perf.py              perf harness + 🐢 Perf panel [NEW]
utils.py / src/utils.py   ensure_data_file (cloud removed), scalers, masked metrics
src/model.py, unet_model.py, ensemble.py   Keras inference ensemble (India-trained weights)
src/viz_plotly.py    PlotlyVisualizer (smoothed, hi-res, outline)
src/daily_temp_loader.py, data_preprocessing.py, zones.py, visualization.py
training/            walk-forward training package (see §4) + registry.py [NEW]
config/rounds.yaml   default schedule/hyperparameter doc [NEW]
data/                cloned artifacts (aggregates.npz, *.weights.h5, geojson, scalers) — legacy, mostly bypassed
.streamlit/config.toml   perf-tuned server config
PROJECT_MEMORY.md    this file
```
Shared with parent repo: `L:\MausamSetu\data\processed\{india,cauvery}.nc` (rebuild cache),
`L:\MausamSetu\data\IMD *` (raw), `L:\MausamSetu\Subbasin\` (basin shapefile).

---

## 10. Notes / limitations

- Data is **annual** (yearly cumulative rain, yearly max temp) → validation "misses" are in mm/year; a
  daily training cube would be needed for sub-annual skill.
- Cauvery inference with the **Keras** ensemble uses India-trained conv weights (kernels transfer;
  quality improves once a Cauvery model is trained in the Training tab).
- The Cauvery spotlight silhouette shows *where* Cauvery sits in India (not real dimmed India data).
- Don't touch: `web/`, `mausamsetu/dashboard/api/`. Only `climate_twin/` + shared `data/processed` + checkpoints.
- Rule kept throughout: **all work stays in the cloned `climate_twin/` repo**; the old `command_center.py` app is untouched.


