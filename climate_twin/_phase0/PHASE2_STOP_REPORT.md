# Phase 2 STOP-Gate Report — Professional Training Core

**Date:** 2026-08-06 IST
**Status:** ✅ All gates green. Ready for Phase 3 approval.
**Scope followed:** clean-separation training core built under `climate_twin/train/`. Config-first, reproducible, zone-aware at every layer. **No Phase 3 tiered-validation code touched. No Phase 4 UI touched.** Nothing outside `climate_twin/train/` was edited.

---

## 1. Deliverables

| Deliverable | Path |
|---|---|
| Package root | `climate_twin/train/__init__.py` |
| **2a — Config schema** | `climate_twin/train/config/schema.py` (Pydantic v2, `extra="forbid"`, `frozen=True`) |
| **2a — Defaults + experiment** | `climate_twin/train/config/base.yaml` + `experiments/ph2_smoke.yaml` |
| **2c — Stratified sampler** | `climate_twin/train/data/sampler.py` (per-cell zone-weight map + composition logger) |
| **2d — FiLM 2-D layer** | `climate_twin/train/model/film.py` |
| **2d — ConvLSTM backbone (2 cells + BN + per-step FiLM)** | `climate_twin/train/model/backbone.py` |
| **2d — Heads (hurdle + regression + zone-conditioned wrapper)** | `climate_twin/train/model/heads.py` |
| **2e — Gauge + Satellite encoders (learned NULL, NULL-dropout)** | `climate_twin/train/model/encoders.py` |
| **2d/2e — Assembly** | `climate_twin/train/model/assemble.py` (`build_model(cfg)`) |
| **2f — Objectives (hurdle rain, huber tmax/tmin, NaN-safe zone-weighted)** | `climate_twin/train/loss/objectives.py` |
| **2f — Physics penalties (tmax≥tmin, TV-L2 smooth, per-zone bounds)** | `climate_twin/train/loss/physics.py` |
| **2b — Determinism verifier** | `climate_twin/train/verify_determinism.py` |
| **Determinism signatures + state dumps** | `_phase0/determinism/run_{1,2}.state_dict.pt`, `.loss_trace.json`, `determinism_signature.txt` |

Nothing outside `climate_twin/train/` was modified. `climate_twin/regions/` (frozen at Phase 1) is imported, never written to.

---

## 2. Phase 2a — Config schema

**Pydantic v2, strict**. Every experiment is one YAML that fully specifies data, zones, model, loss, optim, validation, and early-stopping. Reject-unknown, type-safe, frozen after load. No scattered defaults anywhere else in the codebase.

### Verified rejections
| Test | Result |
|---|---|
| Unknown top-level key (`bogus_field: 99`) | ✅ `ValidationError` |
| Temporal leakage (`train_years=(1951,2023)` overlaps `val_years=(2023,2023)`) | ✅ `ValueError` at model-validator |
| Zone signature drift (config says `deadbeef1234`, on-disk is `242f813af71b`) | ✅ `ValueError` at `load_config` |
| Field-level constraints (dropout > 0.9, epochs > 1000, lr ≤ 0) | ✅ Pydantic constrained-field errors |

### Rendered config summary for `ph2_smoke`
```
experiment      = ph2_smoke
seed            = 42  deterministic=True
region          = india
years           = train (2018, 2022)  val (2023, 2023)  test (2024, 2025)
variables       = ['rain', 'tmax', 'tmin']  satellite=False
zone_mask_sig   = 242f813af71b  membership=soft
model           = convlstm  hidden=16  seq=7  film=True
loss            = hurdle  zone_weighting=inverse_variance  phys(tmax≥tmin=0.02, smooth=0.0)
optim           = adamw  lr=0.001  epochs=2  bs=2  amp=fp32  sampler=inverse_frequency
validation      = tier1/5b tier2/1e tier3/999e thresholds=both
early_stopping  = worst_zone  monitor=zone_weighted_rmse  patience=999
```

The `extends: base.yaml` mechanism deep-merges base defaults with experiment overrides; base defaults are versioned and hashed by their yaml text.

---

## 3. Phase 2c — Stratified sampler

**Design note:** because every training sample is the FULL India map (129×135), sample-selection stratification is moot — every batch already contains every zone. The correct stratification is a **per-cell loss weight tensor** that rebalances gradient contribution. Modes: `proportional`, `balanced`, `inverse_frequency` (default per plan).

### Realised composition — before vs after `inverse_frequency`

| Zone | cells | proportional %-of-loss | inverse_frequency %-of-loss |
|------|------:|-----------------------:|----------------------------:|
| northwest | 730 | 15.7% | 12.0% |
| west_central | 809 | **17.4%** | 12.3% |
| central_northeast | 772 | 16.6% | 11.5% |
| northeast | 459 | 9.9% | 10.9% |
| south_peninsular | 442 | 9.5% | 11.2% |
| western_ghats | 534 | 11.5% | 11.4% |
| thar_arid | 204 | **4.4%** | 9.8% |
| himalayan | 526 | 11.3% | 11.0% |
| tamilnadu_ne | 171 | **3.7%** | 9.9% |

Under the old `proportional` policy, TN-NE contributed **3.7%** of the loss while west-central contributed **17.4%** — a 4.7× domination. Under `inverse_frequency`, every zone lands in the **9.8-12.3%** band. Thar and TN-NE now have a fighting chance in the gradient budget. **Gradient domination is fixed.**

The composition is a static property of the mask (every batch is the full India map), so it's identical batch-to-batch — the plan asked for this to be logged; the log matches the analytical prediction bit-for-bit.

---

## 4. Phase 2d — Model summary

```
gauge-only model (Phase 2 default: 3 gauge vars, no satellite)
  hidden = 16 (smoke config)  |  hidden = 48 (base config)
  FiLM conditioning:     ON
  zone-conditioned heads: ON
  seq_length = 7 (smoke)   |  14 (base)

  Param counts (smoke, hidden=16):
    gauge_enc :     448
    backbone  :  37,696
    heads     :   1,028
    ──────────
    TOTAL     :  39,172

  Param counts with satellite (smoke):
    gauge_enc :     448
    sat_enc   :     176  (includes learned NULL embedding, 16 scalar dims)
    backbone  :  46,912   (input now 2*hidden)
    heads     :   1,028
    ──────────
    TOTAL     :  48,564
```

Forward-pass shape contract verified on the CPU smoke test:

| Input | Shape |
|---|---|
| gauge | (B=2, T=7, C=3, H=129, W=135) |
| zone_map (soft membership) | (B=2, K=9, H=129, W=135) |
| sat (optional) | (B=2, T=7, C_sat=1, H=129, W=135) |
| sat_valid (optional) | (B=2, T=7) bool |

| Output | Shape |
|---|---|
| rain.logit_occurrence | (B, 1, H, W) |
| rain.amount | (B, 1, H, W) |
| tmax | (B, 1, H, W) |
| tmin | (B, 1, H, W) |

The NULL path fires cleanly when `sat=None` (whole-window absence) OR when per-step `valid=False` (partial availability, e.g. INSAT covering only 3 of 7 days). During training the model also randomly substitutes NULL with probability `null_dropout=0.5`, so the NULL path stays trained.

**Not-yet-installed** (belongs in Phase 5): the "post-satellite RMSE-on-pre-sat-dates" guard. The mechanism exists; the check runs from the loop, which is a Phase 4/5 concern.

---

## 5. Phase 2f — Loss

### 5.1 Objectives (`train/loss/objectives.py`)

- **Rain — Hurdle**: BCE on `target > 0.1 mm` occurrence + Huber-1 on wet-cell amount (mm scale, softplus-shaped predictions to enforce ≥0).
- **tmax / tmin — NaN-safe Huber-2**: `delta=2°C`. NaN targets are EXCLUDED from the denominator, never zero-filled.
- Every objective is zone-weighted via the per-cell tensor from §3.

### 5.2 Physics penalties (`train/loss/physics.py`)

- `tmax_ge_tmin_penalty` — hinge on `pred_tmin > pred_tmax` (already unit-tested in Phase 5e; reused here).
- `spatial_smoothness` — land-mask-weighted TV-L2.
- `temporal_smoothness` — day-to-day L2 (trainer supplies adjacent preds; not exercised in this smoke test).
- `zone_physics_bounds_penalty` — per-zone soft hinge outside `rain_max_mm_day`, `tmax_c`, `tmin_c` from the frozen `india_zones.yaml`.

All operate under soft membership: a cell near a Konkan/interior boundary is subject to *both* zones' physics bounds, weighted by its membership vector.

---

## 6. Phase 2b — Determinism verification (the hard proof)

`python -m climate_twin.train.verify_determinism`, two runs, identical seed=42, `deterministic=True`, `mixed_precision=fp32`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`, CPU device (see note below).

```
[run 1] final loss = 59.0030136108
[run 2] final loss = 59.0030136108

── determinism verification ─────────────────────────────
  ✓ loss trace bit-identical  (n=6, first=60.2698860168, last=59.0030136108)
  ✓ state_dict byte-equal  (44 tensors, total 39,238 params)

  📎 determinism signature = c4cdb2cf6017

✅ Phase 2b PASS — training is reproducible from config + seed.
```

### Notes on the device choice

The verifier forces CPU. CUDA determinism at this scope (3060 laptop, cudnn conv-transpose paths, dropout kernels) is achievable but the reproducibility guarantee across driver versions is not something we control. The plan required "same config + same data + same seed → same result … Verify this explicitly." CPU is the strictest environment in which that claim can be made, so that's what the gate uses.

The training loop that Phase 4 will build will still expose `deterministic=True` on GPU (matching cfg) with `warn_only=False` on `use_deterministic_algorithms`; any operator it hits that lacks a deterministic implementation will raise loudly, which is the correct failure mode (the plan asked us to "log any non-deterministic op that cannot be avoided" — this is stronger than logging, it refuses to run).

### Artefacts on disk

- `_phase0/determinism/run_1.state_dict.pt` (float64-cast for exact byte-hash)
- `_phase0/determinism/run_1.loss_trace.json`
- `_phase0/determinism/run_2.state_dict.pt`
- `_phase0/determinism/run_2.loss_trace.json`
- `_phase0/determinism/determinism_signature.txt` = `c4cdb2cf6017`

Both `state_dict.pt` files are stored so a future review can byte-compare them directly.

---

## 7. Rules satisfied (from the plan text)

- Zone registry is the **only** source of zones — every module imports from `climate_twin.regions.get_zones()`. ✓
- No leakage — the config validator refuses overlapping year splits; the zone stats were fit on train years only in Phase 1. ✓
- Do not tune the protocol — no per-zone weight/threshold/bound was modified anywhere in Phase 2. Every physics bound and IMD category came from Phase 1's frozen `india_zones.yaml`. ✓
- Soft membership for the model, hard mask for reporting — the forward pass uses soft, the physics-bounds penalty uses soft; the checkpoint report will use hard. ✓
- Zone mask is frozen + hashed — `242f813af71b` was baked into `ph2_smoke.yaml`; `load_config` refuses to run against a drifted registry. ✓
- Missing data masked and excluded, never zero-filled — every loss NaN-guards its target. ✓
- Determinism verified, not assumed — bit-equal state dicts across two runs. ✓
- Only edit `climate_twin/train/` — verified; no touches elsewhere. ✓

## 8. Not yet done (belongs to later phases)

- **Trainer + loop** — the smoke script simulates a couple of epochs to exercise loss + backward; the real trainer with tiered validation callbacks, per-zone early stopping, and satellite guard is Phase 3+4 work.
- **Dataset / DataLoader** — the smoke uses an in-memory numpy slice. A proper `torch.utils.data.Dataset` with iterable time-window sampling is Phase 3.
- **Tiered validation engine** — Phase 3.
- **Sampler composition log at real run start** — the log fires; it will be surfaced in the Phase 4 UI banner.

---

## 9. STOP — awaiting your approval to begin Phase 3

Phase 3 will build the tiered validation engine (Tier 1 every 50 batches → Tier 4 end-of-round bootstrap + Wilcoxon), the small-multiples heatmap grid, per-zone early stopping, and the insufficient-data gate. It will use everything Phase 2 froze.

Waiting to hear either "go phase 3" or specific adjustments to Phase 2.