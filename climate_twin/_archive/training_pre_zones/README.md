# `_archive/training_pre_zones/` — archived training area

Archived on **2026-08-06 IST** as part of the Phase 0 training-area rebuild.

## What is here

```
_archive/training_pre_zones/
├── training/          — the previous walk-forward training package
│   ├── loops.py           RoundConfig + train_one_round
│   ├── model.py           ClimateTwinModel (ConvLSTM, 2-channel default)
│   ├── metrics.py         masked_rmse, csi(threshold=0.01), ensemble_calibration
│   ├── baselines.py       persistence + climatology
│   ├── ensemble.py        DeepEnsemble, TTA, iterative refinement
│   ├── reward_calibration.py  CRPS calibration finetune
│   ├── checkpoints.py     save/load + Phase-5d variable-list contract
│   ├── registry.py        filesystem-first model registry
│   ├── schedule.py        walk-forward ScheduleConfig, MAX_ROUNDS=500
│   ├── state.py           SQLite run history (runs.db, 9 MB)
│   ├── viz.py             post-round diagnostic plots
│   ├── checkpoints/       (empty at archive time)
│   └── runs/              (empty at archive time)
├── models/            — model registry root (empty at archive time)
├── config/            — rounds*.yaml walk-forward schedules
│   ├── rounds.yaml
│   ├── rounds_india.yaml
│   └── rounds_cauvery.yaml
└── tests/
    └── test_phase5.py — 136 checks that pass against this training system
```

Total size: ~10 MB (dominated by `training/runs.db`).

## Why archived

The rebuild plan (approved 2026-08-06 IST) replaces this system because it is
**statistically invalid on India's ~70× rainfall gradient**:

- **One global normalisation.** Min-max scaling applied uniformly. Rainfall
  cells scaled by `[0, ~979 mm/day]` — arid-zone signal (Thar <150 mm/yr) is
  crushed into the bottom 2 % of the range, and its gradient contribution to
  training loss is negligible next to wet-zone cells.
- **One global RMSE.** Loss and validation metrics are averaged over all
  land cells with no area or inverse-variance weighting. A model that
  overfits monsoon-dominated pixels and underfits arid pixels reports the
  same RMSE as a balanced model.
- **One global threshold for CSI.** Fixed at 0.01 (normalised units). No
  awareness of IMD absolute categories (light / moderate / heavy / very
  heavy / extremely heavy) and no per-zone percentiles.
- **No zone conditioning in the model.** The `ClimateTwinModel` ConvLSTM
  has no way to know that Tamil Nadu peaks Oct-Dec (inverted seasonality vs
  the rest of India) — it must learn a compromise.
- **Leakage hazards.** `app_v2.py:3151-3154` recomputes min/max on the full
  year span every training run — including validation and test years.
- **~180 hardcoded hyperparameters** scattered across ~20 files, with
  divergent values (`seq_length` = 5 / 10 / 30 depending on where you look;
  `epochs` = 25 / 30 / 50; two duplicate Huber-δ constants).

Full audit at `climate_twin/_phase0/audit/hardcoded_hyperparameters.json`.

## Compatibility

Every checkpoint written by this system uses a **2-channel** input adapter
(rain + tmax). The current processed cube carries **4 channels** (rain,
tmax, tmin, insat_lst). Loading an archived checkpoint into the new model
will **fail the Phase 5d variable-list contract** — which is the correct
behaviour. Do not force-restore.

## For the regression benchmark

The four archived registry entries under `models_registry/` (in the older
`_archive_pre_official_data/`) report RMSE values on the old normalised
annual aggregate; those numbers **cannot be compared** to metrics on the
new daily 4-channel cube.

The honest "number to beat" for the new system will be established at the
close of Phase 1: **per-zone persistence + climatology skill on the
2024-2025 holdout, computed against the new 1951-2025 cube**. Any new
model must beat that per zone, or state explicitly which zones regressed
and why.

## Do not restore

The rebuild plan defines specific hard rules the new system will enforce
(zone hash on every checkpoint, insufficient-data → string not number,
per-zone bootstrap CIs, no leakage in normalisation, config-first
determinism). Restoring the archived system would silently break all of
those. If you need a specific piece of logic (e.g. the CRPS calibration
finetune formula), reimplement it inside the new `climate_twin/train/`
tree under a zone-aware contract.
