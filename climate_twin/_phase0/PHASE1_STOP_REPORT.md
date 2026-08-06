# Phase 1 STOP-Gate Report — Zone Registry

**Date:** 2026-08-06 IST
**Status:** ✅ Frozen. Ready for Phase 2 approval.
**Scope followed:** zone registry + rasterised mask + soft membership + per-zone stats fit on TRAIN YEARS ONLY + split-coverage + QC maps + honest-benchmark. **No training code written. No Phase 2 code touched.**

---

## 1. Deliverables — where things are

| Deliverable | Path | Frozen signature |
|---|---|---|
| Zone YAML (single source of truth) | `climate_twin/regions/india_zones.yaml` | — |
| Rasterised hard mask | `climate_twin/regions/zone_mask.npy` | `242f813af71b` |
| Soft membership (129×135×9) | `climate_twin/regions/zone_membership.npy` | (in same sig) |
| Combined mask + yaml signature | `climate_twin/regions/zone_mask.sha256` | `242f813af71b` |
| Build log | `climate_twin/regions/build_mask.log.json` | — |
| Zone accessor (`get_zones()`) | `climate_twin/regions/india_zones.py` | — |
| Package init | `climate_twin/regions/__init__.py` | — |
| Per-zone stats (train-only, no leakage) | `climate_twin/regions/zone_stats.json` | `0a097b298a06` |
| Stats signature | `climate_twin/regions/zone_stats.sha256` | `0a097b298a06` |
| Hard mask QC | `climate_twin/regions/qc/zones_hard.png` | — |
| Soft-membership argmax QC | `climate_twin/regions/qc/zones_membership_argmax.png` | — |
| Annual rainfall QC | `climate_twin/regions/qc/zone_annual_rain.png` | — |
| Split-coverage table | `climate_twin/regions/qc/split_coverage.md` | — |
| **Per-zone benchmark (number-to-beat)** | `_phase0/benchmark_persistence_climatology.json` | — |

---

## 2. Zone assignment (9 zones, hard mask)

Rectangles were widened during Phase 1c iteration to close seams — every land cell inside a state polygon is now assigned to exactly one zone. Final counts:

| ID | Key | Cells | Label |
|---:|-----|------:|-------|
| 1 | northwest | 730 | Northwest India |
| 2 | west_central | 809 | West Central India |
| 3 | central_northeast | 772 | Central Northeast (Ganga plains) |
| 4 | northeast | 459 | Northeast India |
| 5 | south_peninsular | 442 | South Peninsular |
| 6 | western_ghats | 534 | Western Ghats (orographic strip) |
| 7 | thar_arid | 204 | Thar Desert (arid) |
| 8 | himalayan | 526 | Himalayan (northern) |
| 9 | tamilnadu_ne | 171 | Tamil Nadu (Northeast monsoon) |
| — | **total assigned** | **4,647** | — |
| — | inside-state, unassigned | **0** | (all closed) |
| — | ocean/outside-India | 12,768 | — |

All 9 zones exceed the `min_cells_per_split = 5` gate.

## 3. Zone-per-cell soft membership

Membership tensor shape: `(129, 135, 9)`, dtype `float32`, rows sum to 1.0.

- Interior cells: one-hot (weight = 1.0 for their zone).
- Boundary band (`soft_transition_cells = 2` → 0.5°): the cell's own zone gets ≥0.5 weight; the remainder is distributed to whatever neighbour zones fall inside the 2-cell band, inverse-distance weighted. Total "outside" influence is capped at 0.5 so interior identity is never lost.

Verification: the argmax of the soft membership matches the hard mask everywhere except cells sitting exactly on a boundary edge — visible as tiny "bleed" at zone borders in `qc/zones_membership_argmax.png`. This is exactly the intended behaviour ("the atmosphere has no hard boundary at 24.0°N").

## 4. Per-zone stats — fit on TRAIN YEARS ONLY

Train years: **1951-2022 (72 years, n_train_days = 26 298)**.
No validation or test year contributed a single sample to these stats.

Rain (mm/day) statistics per zone:

| Zone | rain μ | rain σ | rain p90 | rain p95 | rain p99 | loss weight |
|------|-------:|-------:|---------:|---------:|---------:|------------:|
| northwest | 1.947 | 8.382 | 3.470 | 12.42 | 42.04 | 1.210 |
| west_central | 2.664 | 10.358 | 6.005 | 17.31 | 48.26 | 0.792 |
| central_northeast | 3.535 | 10.999 | 10.24 | 20.66 | 52.13 | 0.703 |
| northeast | 6.202 | 15.313 | 20.20 | 32.99 | 68.23 | 0.363 |
| south_peninsular | 2.622 | 9.110 | 5.735 | 15.36 | 44.66 | 1.024 |
| western_ghats | 3.514 | 12.132 | 6.435 | 21.68 | 55.76 | 0.578 |
| thar_arid | 0.944 | 6.100 | 0.000 | 4.87 | 26.31 | 2.285 |
| himalayan | 2.713 | 8.718 | 6.055 | 14.66 | 44.01 | 1.119 |
| tamilnadu_ne | 2.778 | 9.576 | 4.815 | 15.42 | 46.42 | 0.927 |

**Sanity checks pass:**
- Northeast has the highest σ (15.31) — matches expectation (Mawsynram-adjacent cells drive the tail).
- Thar has the lowest μ (0.944) and p90 = 0 — the arid signal is genuinely near-zero.
- Loss weights (inverse-variance, mean-normalised, clipped to [0.2, 5.0]) protect the thin/arid signal: Thar carries **2.29×** the weight of an average zone, Northeast **0.36×**.
- Every zone's mean is bounded by its σ ratio in a way consistent with heavy-tailed rainfall statistics.
- TN-NE μ (2.78) resembles other peninsular zones on annual average, but its **seasonality is inverted** — this can only be diagnosed by the per-DOY climatology, not the summary stats. The full 366-DOY climatology per variable per zone is stored in `zone_stats.json`.

Also written to `zone_stats.json`:
- Per-zone per-variable `norm.mean`, `norm.std` (train-only z-score parameters)
- 366-DOY climatology for `rain`, `tmax`, `tmin`, `insat_lst` (weighted mean per DOY, NaN when <10 valid pooled obs)
- IMD absolute-category shares per zone (fraction of days in each category)

## 5. Split coverage — all 9 zones adequately covered in all 3 splits

Full table at `regions/qc/split_coverage.md`. Highlights:

- Train (1951-2022): between 6.4M and 31.8M valid cell-days per zone
- Val (2023): between 89 K (TN-NE, smallest zone) and 441 K (west-central) valid cell-days
- Test (2024-2025): between 178 K (TN-NE) and 883 K (west-central) valid cell-days

**Every zone shows the ✓ flag** — no zone is below `min_cells_per_split` in any split. TN-NE is the smallest and merits close attention during Phase 3 validation, but has enough data.

## 6. Honest benchmark — number-to-beat for Phase 2

Written to `_phase0/benchmark_persistence_climatology.json`. Weighted per-zone RMSE (soft membership) on the 2024-2025 holdout, computed with proper masking (finite-target only, weight > 0.05):

### 6.1 Rain (mm/day)

| Zone | Persistence RMSE | Climatology RMSE | Winning baseline |
|------|-----------------:|-----------------:|------------------|
| northwest | 12.15 | **8.99** | climatology |
| west_central | 14.55 | **10.99** | climatology |
| central_northeast | 13.51 | **10.11** | climatology |
| northeast | 17.42 | **13.73** | climatology |
| south_peninsular | 13.40 | **9.97** | climatology |
| western_ghats | 13.97 | **11.36** | climatology |
| thar_arid | 10.25 | **7.63** | climatology |
| himalayan | 10.37 | **7.86** | climatology |
| tamilnadu_ne | 13.87 | **10.46** | climatology |

**Climatology beats persistence in every zone for rain** — as expected on daily-scale rainfall. A Phase-2 model that doesn't beat these numbers is failing.

### 6.2 tmax (°C)

| Zone | Persistence RMSE | Climatology RMSE | Winning baseline |
|------|-----------------:|-----------------:|------------------|
| northwest | **2.04** | 2.55 | persistence |
| west_central | **1.71** | 2.14 | persistence |
| central_northeast | **1.89** | 2.22 | persistence |
| northeast | **2.12** | 2.16 | persistence |
| south_peninsular | **1.54** | 1.87 | persistence |
| western_ghats | **1.27** | 1.72 | persistence |
| thar_arid | **1.88** | 2.36 | persistence |
| himalayan | **2.25** | 2.93 | persistence |
| tamilnadu_ne | **1.33** | 1.52 | persistence |

**Persistence beats climatology in every zone for tmax** — daily temperature has strong day-to-day autocorrelation. This is the harder target: any Phase-2 model must show that its skill on tmax is better than just predicting yesterday's value.

### 6.3 tmin (°C)

Same pattern as tmax (persistence wins). Full numbers in the JSON.

### 6.4 What "beating the benchmark" means

For each zone × variable, the Phase-2 model's RMSE must be **strictly lower than min(persistence_rmse, climatology_rmse)** — i.e. lower than the *best* baseline for that cell in that regime. This is 9 zones × 3 vars × 1 pair = **27 hard win conditions**. Anything shy of that must be explicitly explained per-zone in the Phase-2 report (e.g. "TN-NE tmax regressed by 0.05 °C because the small-sample calibration…").

## 7. Design choices frozen before Phase 2

Per the rebuild plan's rule "DO NOT TUNE THE PROTOCOL", the following are FROZEN and go into `zone_mask.sha256` + `zone_stats.sha256`:

- 9 zones (keys, ids, priority order)
- lat/lon rectangles per zone (post-seam-closure)
- state-list per zone
- `soft_transition_cells = 2` (0.5° blend band)
- `min_cells_per_split = 5`
- IMD absolute rainfall categories (no_rain, very_light, light, moderate, heavy, very_heavy, extremely_heavy)
- Per-zone physics bounds (rain_max_mm_day, tmax_c, tmin_c ranges)
- Loss weighting mode = `inverse_variance`, clip = [0.2, 5.0]
- Per-zone climatology (train-years-only per-DOY means)
- Per-zone p50/p90/p95/p99 thresholds (train-years-only weighted percentiles)

Changing any of these AFTER training starts will cause the SHA-256 to drift; `india_zones.py` will refuse to serve the state and every prior metric becomes incomparable. This is intentional.

## 8. Small things noted for Phase 2

- **The RuntimeWarning "Mean of empty slice"** during climatology fit was harmless — some DOYs have <10 valid pooled cells and are correctly filled with NaN.
- **Weighted percentiles** in `fit_zone_stats.py` currently approximate with unweighted percentiles on cells with soft-membership > 0.05. Interior cells (~99% of soft mass) are exact; boundary cells get slight distortion. Documented; can be tightened in Phase 3 if a per-zone p99 drift is observed.
- **The 7 tmax<tmin regrid-boundary cells** from Phase 0 fall in the `central_northeast` zone (Sundarbans is West Bengal / Odisha coast, id 3). They will fire the Phase 5e tmax≥tmin hinge penalty during training — the model will be pushed to satisfy ordering; the observations are left untouched.
- **INSAT LST** was fit only where finite (n = 3.77M cell-days total for india cube), so most zones' insat_lst stats are computed on the same shared 2020-2021 partial window. It is present in `zone_stats.json` but a round including insat_lst will only include years where the coverage matrix says it exists — Phase 5c's coverage-aware rounds handles this and will pick up the new stats automatically.

## 9. Phase 2 entry — waiting on approval

Everything the rebuild plan required at the Phase-1 hard STOP is present. Before Phase 2 starts, please confirm:

1. **The QC map (`regions/qc/zones_hard.png`) matches your expectation of the 9 zones.** If any boundary looks wrong (e.g. you want Konkan pulled out of west_central, or Ladakh/J&K separated from Himachal), say so now — this is the only chance to adjust before the zone signature is baked into every checkpoint.

2. **The benchmark numbers above are the honest number-to-beat.** No Phase-2 checkpoint may claim victory over any zone × variable without a lower RMSE than the winning baseline for that cell in that regime.

3. **The Phase 2 config-first training core** may begin building on top of `climate_twin.regions.get_zones()` — the mask signature `242f813af71b` and stats signature `0a097b298a06` are now the contract.

**STOP. Awaiting approval to begin Phase 2.**
