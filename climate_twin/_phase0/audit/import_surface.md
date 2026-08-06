# Import Surface — `climate_twin.training`

Phase 0a audit. Every module in the running app that imports from `climate_twin.training`. The Phase 0d stub package must expose exactly these symbols so `app_v2.py` still imports cleanly after the archive.

## Live app (`climate_twin/app_v2.py`)

| Line | Import statement | Purpose in the app |
|-----:|------------------|--------------------|
| 2409 | `from training import registry as REG` | Continue-training tab |
| 2410 | `from training import viz as VIZ` | Diagnostics rendering |
| 2488 | `from training import registry as REG` | Model library table |
| 2668 | `from training import reward_calibration as CAL` | Calibration finetune |
| 2770 | `from training import registry as REG` | Validate-mode model select |
| 2771 | `from training import metrics as MET` | Validate-mode metric compare |
| 2772 | `from training import viz as VIZ` | Post-validation panels |
| 3073 | `from training import registry as REG` | Save-model button |
| 3074 | `from training import viz as VIZ` | Live console + curves |

**Distinct top-level names imported:** `registry`, `viz`, `reward_calibration`, `metrics`.

## Demo / dev scripts (not part of live app)

| File | Line | Import statement |
|------|-----:|------------------|
| `climate_twin/_demo_partb.py` | 5 | `from training.loops import RoundConfig, TrainingProgress, train_one_round` |
| `climate_twin/_demo_partb.py` | 6 | `from training.model import ClimateTwinModel` |
| `climate_twin/_demo_partb.py` | 7 | `from training import registry as REG` |

## Tests

| File | Line | Import statement |
|------|-----:|------------------|
| `climate_twin/tests/test_phase5.py` | 33 | `from training.checkpoints import save_checkpoint, load_checkpoint, check_variables, CheckpointVariableMismatch, CHECKPOINT_DIR` |
| `climate_twin/tests/test_phase5.py` | 40 | `from training.registry import save_model, load_into, ModelVariableMismatch, delete_model` |

## Minimum shim surface required after archive

The Phase 0d stub `climate_twin/training/` package must, at minimum, expose these submodules so `import climate_twin.app_v2` (and the demo/tests) don't raise `ImportError` at import time:

```
training/
├── __init__.py            # exports: REBUILD_BANNER_TEXT, TrainingRebuildInProgress
├── registry.py            # names used: (any attr access → raises TrainingRebuildInProgress)
├── viz.py                 # same
├── metrics.py             # same
├── reward_calibration.py  # same
├── loops.py               # names used at demo import: RoundConfig, TrainingProgress, train_one_round
├── model.py               # names used at demo import: ClimateTwinModel
├── checkpoints.py         # names used at test import: save_checkpoint, load_checkpoint, check_variables, CheckpointVariableMismatch, CHECKPOINT_DIR
└── (unused but harmless: baselines.py, ensemble.py, schedule.py, state.py)
```

The stub strategy: each submodule imports cleanly, exposes the required names as either lazy attributes on a module-level `__getattr__` that raises `TrainingRebuildInProgress`, or as symbols that raise on call.

Special-case for tests: `test_phase5.py` currently PASSES against the pre-Phase-0 training area. After archive it will fail with `TrainingRebuildInProgress` from within the Section 4 checkpoint-contract tests. **This is expected and correct** — those tests belong to the archived training system. Phase 0 marks them as archived by:

- Moving `test_phase5.py` into `_archive/training_pre_zones/tests/` alongside the code they test, OR
- Leaving them and letting them fail with a clear message (the raised `TrainingRebuildInProgress` is the pass criterion for "old tests refuse to run against new system").

Recommendation: leave in place, accept failure. Phase 5g will re-establish tests for the zone-aware system.

## Import-time side effects to preserve

- `training/__init__.py` in the current implementation contains only a 62-byte comment string. No import-time side effects to preserve.
- `training/registry.py` imports `torch` at module top. The stub can defer this — a raw `import torch` at module top of a stub is fine (torch is required for the app anyway).
- `training/state.py` opens `runs.db` on first use, not at import. Safe to stub.
