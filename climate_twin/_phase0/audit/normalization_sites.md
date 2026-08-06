# Normalization Sites

Phase 0a audit. Every place normalization/denormalization is applied. Phase 1 must consolidate all of these into a single per-zone transform driven by `regions/india_zones.yaml`. Right now normalization stats come from *at least three* independent sources with different formulas, several of which recompute stats on the fly inside the UI — a leakage hazard.

## Canonical source of truth (post-Phase-4)

| File | Detail |
|------|--------|
| `data/processed/india_norm_stats.json` | Per-variable min/max/mean/std/n. Computed on TRAIN YEARS ONLY by `data/build_cube.py`. |
| `data/processed/cauvery_norm_stats.json` | Same. |
| `data_source.py:274` | `load_norm_stats(region, sig=None)` — returns the JSON dict. Streamlit-cached, keyed on `manifest.sig` so a rebuild invalidates it. |

This is the ONLY source that respects the no-leakage rule. Every other path violates it in some way (see below).

## In-repo violations of the canonical path

### A. Streamlit "shared normalized cube" (major offender)

`climate_twin/app_v2.py` **recomputes** min/max on the fly, from the already-loaded aggregate `_rain_agg` and `_temp_agg` arrays, and does so inside `_render_tab9` (Training tab). These are ANNUAL aggregates over the full year span (train + val + test), so the min/max include validation and test years — **direct temporal leakage**:

| File:Line | Code |
|-----------|------|
| `app_v2.py:3151` | `_scalers = {"rain_min": float(np.nanmin(_rl)), "rain_max": float(np.nanmax(_rl)), ...}` |
| `app_v2.py:3152` | `..., "temp_min": float(np.nanmin(_tl)), "temp_max": float(np.nanmax(_tl))}` |
| `app_v2.py:3153-3154` | `_rn = np.clip((_rain_agg - _scalers["rain_min"]) / (_scalers["rain_max"] - _scalers["rain_min"] + 1e-8), 0, 1)` |
| `app_v2.py:3545-3547` | Same pattern repeated inside walk-forward training path (`scalers = {"rain_min": ..., "rain_max": ..., "temp_min": ..., "temp_max": ...}` recomputed) |
| `app_v2.py:2672-2675` | Same pattern inside calibration finetune path |

Every training run built from these Scalers is training on leakage. This has to die in Phase 2.

### B. Legacy TF preprocessing (dormant but reachable)

| File:Line | Detail |
|-----------|--------|
| `climate_twin/src/data_preprocessing.py:120-136` | `normalize_and_split` computes min/max on `X_train` after slicing off the last `test_size=5` samples. Correct SPLIT policy in principle, but hardcoded to annual-aggregate scale and never exposes zones. |
| `climate_twin/src/data_preprocessing.py:267-283` | `normalize_and_split_daily` — same. Writes `data/scalers_daily.json`. Not used by the running app but the file may still exist on disk. |
| `climate_twin/src/utils.py` | `save_scalers` / `load_scalers` — TF-era serializer, still imported by `data_preprocessing.py`. |
| `climate_twin/utils.py:29-45` | `save_scalers`, `load_scalers`, `denormalize_grid` — a THIRD copy of the same helpers, referenced by `app_v2.py:36-37` (fallback to `data/scalers.json` if present). |

### C. Denormalization on the display path

| File:Line | Formula |
|-----------|---------|
| `app_v2.py:548` | `# Convert normalized uncertainty back to physical units.` (comment only, formula in surrounding lines) |
| `app_v2.py:3010-3011` | `rain_obs = truth_rain * rr + scalers["rain_min"]`; `rain_pred = pred_rain * rr + scalers["rain_min"]` where `rr = scalers["rain_max"] - scalers["rain_min"]` |
| `app_v2.py:3030,3033` | Same formula applied to persistence + climatology baselines |
| `app_v2.py:3036-3037` | Same formula for temperature |
| `climate_twin/utils.py:37-43` | `denormalize_grid` — `grid * (max_val - min_val) + min_val` |
| `climate_twin/src/utils.py:26-30` | Duplicate `denormalize_grid`. |

**Denorm formula in use:** min-max scaling → `x_norm = (x - min) / (max - min)`, `x = x_norm * (max - min) + min`. This is a linear map that assumes bounded targets. The canonical `norm_stats.json` records mean/std as well, but the actual training path uses **min-max, not z-score**. Two-normalization-format drift: fitted stats include mean/std that are never used.

## Special: sensitivity map + climatology paths

| File:Line | Detail |
|-----------|--------|
| `app_v2.py:2669` | Comment: `# Build region-normalized cube` (calibration path builds its OWN cube here — see §A rows 2672-2675) |
| `app_v2.py:412` | `np.clip(clim_anom, -7.0, 7.0)` — clipping magic number on climatology anomaly |
| `app_v2.py:689,1537` | `np.clip(pred_rain_ann / denom, 0.45, 2.35)` — sensitivity denominator clip; not normalization per se but a scale-dependent bound |
| `app_v2.py:698` | `np.clip(daily rain, 0.0, 350.0)` — physical cap that varies wildly by zone (Konkan/NE India can exceed 500 mm/day; this cap silently truncates them) |

## Summary — Phase 1/2 must

- **Delete** the on-the-fly min/max computation in `app_v2.py:3151-3154`, `3545-3547`, `2672-2675`. All normalization goes through the frozen `regions/india_zones.yaml` (per-zone mean/std/percentiles from train years only).
- **Delete** the three duplicate `denormalize_grid` implementations. One canonical implementation in `train/data/transforms.py`.
- **Switch from min-max to z-score** for the model input (per the plan's design). Keep min-max only for display-scale mapping if needed.
- **Compute per-zone stats from train years only**, once, at Phase 1 close. Store in `regions/india_zones.yaml` and hash into `zone_mask.sha256`.
- **Ban** any code path that reads the full-year aggregate and recomputes stats on it — that is the current leakage hazard. Every Streamlit widget that touches normalization must call `DS.load_norm_stats(region)` and nothing else.
- **Add a norm-source contract** to each checkpoint: `norm_source_hash` = SHA-256 of the exact stats used. Load-time refuses on mismatch (same pattern as `manifest_sig` + `variables` in Phase 5d).

## Physical clipping constants (`app_v2.py` display path)

These are downstream of normalization but shape how forecasts are shown. Enumerated once so Phase 1 can decide zone-by-zone bounds:

| Line | Clip range | Applied to |
|-----:|------------|-----------|
| 400-401 | `[0.0, 600.0] mm`, `[5.0, 55.0] °C` | rain + temp map inputs |
| 412 | `[-7.0, 7.0]` | climatology anomaly |
| 545 | `[0.0, 12000.0]` | annual rain (mm/year) |
| 546, 606 | `[5.0, 55.0]` | temp maps |
| 689, 1537 | `[0.45, 2.35]` | rain/denom |
| 691 | `[-6.0, 6.0]` | temp anomaly |
| 698, 1538 | `[0.0, 350.0]` | daily rain (loses NE/Konkan extremes) |
| 1369 | `[0.72, 1.0]` | distance decay weight |

Every one of these is India-wide and zone-blind. The zone yaml will supply per-zone `rain_hard_max` / `temp_hard_min` / `temp_hard_max` that replace these.
