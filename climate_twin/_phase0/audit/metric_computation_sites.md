# Metric-Computation Sites

Phase 0a audit. Every place in the running app where a skill/error metric is computed. Phase 1 needs this because *every* zone-aware metric must go through the new tiered validation engine — leaving a legacy metric site alive is a slow way to make the "per-zone by default" promise a lie.

## Canonical implementations (`climate_twin/training/`)

| File | Line | Symbol | What it computes |
|------|-----:|--------|------------------|
| `training/metrics.py` | 12 | `masked_rmse` | RMSE on land cells, NaN-safe |
| `training/metrics.py` | 44 | `mae`, `bias`, `pearson_r` | Basic first- and second-moment scores |
| `training/metrics.py` | 75 | `csi(threshold=0.01)` | Critical Success Index at a rain threshold; **threshold is a hardcoded 0.01 default** |
| `training/metrics.py` | 89 | `ensemble_calibration` | Fraction of truth cells in the [p10, p90] band |
| `training/metrics.py` | 98 | `compute_all_metrics` | Bundle: rmse, mae, bias, r, csi (single threshold), coverage |
| `training/baselines.py` | 37 | `compute_baseline_metrics` | Persistence + climatology metrics via `compute_all_metrics` |

**Zone-awareness status:** none. All operate on `(H, W)` or `(N, H, W)` arrays with a single 2-D mask.

## Call sites in the training loop

| File | Line | Call | Notes |
|------|-----:|------|-------|
| `training/loops.py` | 371 | `compute_all_metrics(val_preds[i, c], val_truths[i, c], mask_np)` | per-sample per-channel loop; caps at 50 samples (line 369) |
| `training/loops.py` | 457 | `ensemble_calibration(p10[i, 0], p90[i, 0], truth_sample[i, 0], mask_np)` | **channel-0 only — silently ignores tmax/tmin/insat calibration** |
| `training/loops.py` | 465 | `compute_baseline_metrics(val_targets[0], last_obs, train_targets, mask_np)` | baseline computed once per round |
| `training/ensemble.py` | 250 | `compute_all_metrics(p50[i, c], truth[i, c], mask)` | Deep ensemble path duplicates the per-sample loop |
| `training/reward_calibration.py` | 159 | `evaluate_calibration(model, val_x, val_y, mask)` | separate metric path; **rolls its own RMSE inline** |

## Call sites in the UI (`climate_twin/app_v2.py`)

| Line | Call |
|-----:|------|
| 2931 | `MET.compute_all_metrics(pred_rain, truth_rain, mask)`  — Deep Think validation |
| 2932 | `MET.compute_all_metrics(persistence, truth_rain, mask)` — persistence baseline |
| 2933 | `MET.compute_all_metrics(climatology, truth_rain, mask)` — climatology baseline |
| 2934 | `MET.compute_all_metrics(pred_temp, truth_temp, mask)` — temp channel |
| 2935 | `MET.compute_all_metrics(pers_temp, truth_temp, mask)` |
| 2936 | `MET.ensemble_calibration(p10[0], p90[0], truth_rain, mask)` |

## Ad-hoc metric implementations to consolidate

Phase 1 will replace these; noted so nothing gets orphaned:

| File | Line | Ad-hoc |
|------|-----:|--------|
| `climate_twin/utils.py` | 46 | `masked_rmse` — duplicate of `training/metrics.masked_rmse` |
| `climate_twin/src/utils.py` | 33 | `masked_rmse` — TF-era duplicate |
| `climate_twin/src/ensemble.py` | 71 | `_masked_rmse` — TF-era duplicate |
| `climate_twin/app_v2.py` | 1768 | inline `def mrmse(a, p)` inside a plot helper |
| `climate_twin/app_v2.py` | 3815 | inline `def _mrmse(h)` inside a diagnostic helper |
| `climate_twin/bench_baseline.py` | 72-90 | inline sensitivity/percentile stats used to build the demo animation, not metrics per se, but read from the same tensors |
| `climate_twin/training/reward_calibration.py` | 165-190 | rolls its own RMSE, CSI, and calibration coverage in `evaluate_calibration`, ignoring `metrics.py` |

## Threshold conventions in use

Every place a "wet vs dry" or "significant rainfall" threshold is applied:

| File:Line | Threshold | Convention |
|-----------|-----------|------------|
| `training/metrics.py:75` | 0.01 (normalised) | fixed CSI default |
| `training/metrics.py:98` (compute_all_metrics) | 0.01 (normalised) | inherits default |
| `training/reward_calibration.py:180` | 0.01 (normalised) | duplicate |
| `app_v2.py:2931-2936` (calls) | uses defaults → 0.01 | inherits |

**IMD official thresholds are used nowhere.** No code path knows about the light/moderate/heavy/very-heavy/extremely-heavy IMD categories. Phase 1 introduces them.

**Zone percentiles are used nowhere** — the plan calls for p90/p95/p99 per zone from train data. Not present in any current code.

## Aggregation policy currently in force

- All metric aggregation is via `np.mean` across land cells and across the sample dimension (with an implicit "cells are equally weighted" assumption).
- **No area-weighting** (each 0.25° cell has ~28×28 km² area at 6.5°N but ~28×22 km² at 38.5°N — the difference is ~20% and is currently ignored).
- **No zone-weighted global aggregate** — the plan's "global number is a weighted aggregate, not the headline" is not enforced.
- **No bootstrap CI** anywhere. Every metric is a point estimate.
- **No paired significance test** (Wilcoxon or otherwise). "Better than persistence" is asserted from raw RMSE difference.

## Insufficient-data handling

Currently: metrics on empty masks return `NaN`; `NaN` is silently averaged into the mean, contaminating the score. The plan requires "insufficient data → 'insufficient data', not a number." No current code path enforces `min_cells`.

## Summary — Phase 1 must replace

- The single-threshold CSI at 0.01: replaced by both IMD-absolute categories and per-zone percentiles.
- The per-sample averaging in `loops.py:369-371`: replaced by tiered validation (Tier 1 cheap loss, Tier 2 per-zone, Tier 3 full cross-product, Tier 4 calibration + significance).
- The channel-0-only ensemble calibration: extended to all channels, per zone.
- The ad-hoc masked_rmse duplicates: deleted; every caller routes through `train/eval/metrics.py`.
- Missing bootstrap CIs / significance tests: added in `train/eval/significance.py`.
- Missing `min_cells` gate: added at the metric-emit boundary; below → the string `"insufficient data"`.
