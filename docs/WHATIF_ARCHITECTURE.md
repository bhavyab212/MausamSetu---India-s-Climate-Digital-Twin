# What-If Scenario Engine · Architecture

One diagram, one page. Every arrow labels a real function in the
codebase.

```
                     ┌────────────────────────────────────┐
                     │ WhatIfState (whatif/ui/state.py)   │
                     │  levers + cache_key()              │
                     └──────────────┬─────────────────────┘
                                    │
                                    ▼
             ┌──────────────────────────────────────────────┐
             │ pages/30_What_If.py  (thin orchestrator)     │
             │  → whatif/ui/engine_adapter.py               │
             │     └── cached_run_for_state(state)          │
             └──────────────┬───────────────────────────────┘
                            │
   ┌────────────────────────┴─────────────────────────────────────┐
   │ L0 · DRIVERS                                                 │
   │  drivers.load_driver(DriverSpec)                             │
   │   ├── historical.get_historical(var, start, end)             │
   │   ├── ensemble.get_ensemble(var, day)     (MC-Dropout)       │
   │   ├── perturbation.apply_perturbation     (Method 1)         │
   │   ├── analogs.build_analog_pool + find_analogs (Method 2)    │
   │   └── ssp.load_ssp_driver                 (Part 7)           │
   └──────────────┬───────────────────────────────────────────────┘
                  │  xr.DataArray  (time, lat, lon)  IST-aware
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ L1 · INDICES  (whatif/indices/)                              │
   │   et0_hargreaves · gdd · degree_days · heat_stress ·         │
   │   extremes · dry_spell · onset · spi · spei · aridity        │
   │   long_term.return_period_shift · time_of_emergence          │
   │  INDEX_REGISTRY holds versions + citations                    │
   └──────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ L2 · BIOPHYSICAL  (whatif/biophysical/)                      │
   │   crop_water.kc_curve · stage_of_day · etc_series            │
   │   soil.awc_mm_per_m · taw · raw                              │
   │   water_balance.water_balance                                │
   │     (FAO-56 single-Kc daily bucket)                          │
   └──────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ L3 · SECTORS  (whatif/sectors/)                              │
   │   agriculture.yield_water_limited (FAO-33 multi-stage)       │
   │   agriculture.yield_baseline      (climatology forcing)      │
   │   sowing_window.optimize_sowing_window                       │
   │   adaptations.load_adaptation     (Part 7)                   │
   │   to_district + DistrictRegistry  (resolution ceiling)       │
   │   run_agriculture_scenario · run_decision_scenario           │
   │   run_long_term_scenario                                     │
   └──────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ L4 · ECONOMICS  (whatif/economics/)                          │
   │   prices.load_prices · valuation.value_agriculture           │
   │   payoff.build_payoff_matrix                                 │
   │   decision.regret_matrix · minimax · var_cvar · recommend    │
   │   cost_loss.value_curve         (Murphy 1977)                │
   │   backtest.walk_forward_backtest  (leakage-guarded)          │
   │   sensitivity.tornado                                        │
   │   npv.adaptation_npv     (7 % / 8 % / 12 %)  (Part 7)        │
   └──────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ UI  (whatif/ui/panels/*)                                     │
   │   context · levers · payoff · recommendation · analogs ·     │
   │   sensitivity · backtest · long_term · provenance · save     │
   │  copy library at whatif/ui/copy/  (verb-lint enforced)       │
   └──────────────────────────────────────────────────────────────┘
```

## Definitions

- **Driver** — a source of climate data on the master 0.25° grid: an
  observed historical record, an ensemble prediction, a perturbation
  of a base, an analog-year selection, or an SSP projection.
- **Index** — a scalar or reduced-shape quantity computed from a
  driver: ET0, SPI-3, GDD, longest dry spell, Rx1day return level.
  Every index carries a version + citation in `INDEX_REGISTRY`.
- **Sector** — the composition of indices + a physical model into a
  human-relevant quantity: for agriculture, that's yield in t/ha at
  district resolution.
- **Economic outcome** — physical quantities monetised at cited
  prices, returned as `{q10, q50, q90}` with a climatology baseline
  counterpart. Never a bare rupee number.
- **Scenario** — one (driver × decision × sector) run through the
  pipeline. Its `WhatIfState` YAML replays byte-identically.
- **Decision** — a lever the user or a decision-support consumer
  chooses: which crop, which sow date, which adaptation.

## Rules the architecture enforces

1. Layers only call downward: L4 never touches L0 directly.
2. Every DataArray carries `attrs["units"]`, `attrs["source_chain"]`,
   `attrs["quantile"]` — the chain accumulates as data flows up.
3. IMD sentinels (`-999.0`, `99.9`) are masked at L0; nothing above
   ever sees them.
4. IST timezone-aware timestamps end-to-end.
5. Fits (SPI, SPEI, R95p, GEV) use `TRAIN_YEARS = (1971, 2010)` only;
   `assert_train_only` fires otherwise.
6. Long Term uses a `1971-2000` baseline (IPCC AR6) — different from
   the SPI TRAIN_YEARS on purpose; both coexist.
7. Short Term uncertainty representation is three-pass q10/q50/q90;
   Long Term is a multi-model ensemble. `RepresentationMismatch`
   raises if the two are ever mixed.
