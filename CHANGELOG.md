# Changelog

All notable changes to MausamSetu are recorded here.

## Unreleased — What-If Scenario Engine (Parts 0–8)

The What-If Scenario Engine ships a five-layer, provenance-tagged
climate → rupees pipeline for India-scale decision support. Every
number on screen is traceable to a formula, a dataset version, and a
code SHA. The engine ships under `climate_twin/whatif/` with two
Streamlit tabs (Short Term, Long Term) at `climate_twin/pages/30_What_If.py`.

### Added — by Part

- **Part 0** — Retired the legacy What-If tab; scaffolded the
  multipage layout, `whatif/` package tree, and the port ledger for
  every reused Streamlit routine.
- **Part 1** — L0 drivers (historical, ensemble, perturbation stub,
  analog stub, SSP stub), IST-aware DataArray contract, provenance
  ledger (YAML per run, git-SHA + dirty flag), three-pass q10/q50/q90
  quantile plumbing, `DriverSpec`, `RegionSpec`, and the scenario
  orchestrator skeleton.
- **Part 2** — L1 climate indices with `INDEX_REGISTRY`: ET0
  (Hargreaves-Samani, FAO-56 §3), GDD (McMaster & Wilhelm 1997), CDD /
  HDD, hot-day count + IMD heatwave, Rx1day / Rx5day / R95p, longest
  dry spell + ETCCDI CDD, IMD Kerala monsoon onset, mixed-Gamma SPI
  (McKee 1993), Fisk-family SPEI (Vicente-Serrano 2010), UNEP aridity.
  Fits confined to `TRAIN_YEARS = (1971, 2010)` via
  `assert_train_only` + `LeakageError`.
- **Part 3** — L2 water balance (FAO-56 single-Kc daily bucket) and
  L3 agriculture sector (FAO-33 multi-stage yield-response, flowering
  heat overlay, mandatory climatology baseline pass). Crop registry
  (`crops.yaml`, four crops, cited). Sowing-window optimiser.
  APY validation stub.
- **Part 4** — L4 economics: `EconomicOutcome` (no bare rupees),
  `PayoffMatrix` (decision × climate state), Savage regret, minimax,
  VaR / CVaR, stochastic dominance. Murphy 1977 cost-loss with the
  full verification bundle (Brier + BSS + reliability + ROC-AUC +
  sharpness + V(p*) sweep). Walk-forward backtest with strict pre-y
  leakage guard. OFAT sensitivity tornado.
- **Part 5** — Historical analog engine (Method 2): feature
  construction on region + window, Mahalanobis pool with χ² quality
  tiers, weighted empirical outcome distribution with fixed-seed
  bootstrap CI. Delta perturbation (Method 1) with a
  `CaveatRequiredError` gate. Analog-as-forecaster walk-forward
  backtest.
- **Part 6** — UI refactor. `WhatIfState` + `cached_run_for_state`.
  Nine pure `(state, result) -> None` panels. Copy library
  (`ui/copy/`) with a verb ban in Long-Term strings.
- **Part 7** — Long Term wiring. NEX-GDDP-CMIP6 ten-GCM catalog,
  SSP1-2.6 / 2-4.5 / 3-7.0 / 5-8.5 registry with AR6 citations,
  Quantile Delta Mapping (Cannon 2018), Hawkins-Sutton uncertainty
  decomposition, return-period shift, time-of-emergence, adaptation
  registry with cited capex + opex, NPV at 7 % / 8 % / 12 % per NITI
  Aayog, `run_long_term_scenario`, `RepresentationMismatch` guard,
  live Long-Term UI panels replacing the Part-6 shell.
- **Part 8** — End-to-end integration. Ten demo scenarios under
  `whatif/demo/scenarios/`, docs (`RUNBOOK.md`,
  `WHATIF_ARCHITECTURE.md`, `DECISIONS.md`, `LIMITATIONS.md`,
  `PR_DESCRIPTION.md`), CI workflow at
  `.github/workflows/whatif_ci.yml` with three AST-level guards
  (`leakage_ast.py`, `colormap_blocklist.py`, `citation_scan.py`),
  pre-commit hooks, auto-generated one-pager HTML
  (`whatif.report.render_one_pager`), and cross-layer `test_e2e.py` +
  acceptance `test_final.py`.

### Guardrails now enforced by CI

- **Leakage** — no `.fit(...)` under `whatif/indices/` or the backtest
  references `VALID_YEARS`. AST scan on every push.
- **Bare-rupee** — `EconomicOutcome` refuses scalars in any INR field.
- **Verb lint (Long Term)** — copy in `whatif/ui/copy/long_term.py`
  cannot contain `predict`, `forecast`, `will`, or `is going to`.
- **Colormap blocklist** — no `jet` / `hsv` / `rainbow` /
  `nipy_spectral` / `gnuplot` colorscale strings anywhere under
  `whatif/ui/` or `whatif/report/`.
- **Docstring citation scan** — every module under `whatif/indices/`,
  `whatif/biophysical/`, `whatif/sectors/`, `whatif/economics/`, and
  the SSP / NEX-GDDP / downscale modules carries a
  `Primary source` or `Citation` line.
- **Resolution ceiling** — grid inputs to `value_agriculture` raise
  `ResolutionCeilingError`.
- **Perturbation caveat gate** — export refuses without
  `caveat_acknowledged=True`.
- **Adaptation cost citation** — `adaptation_npv` refuses options with
  `cost_complete=False` unless the caller opts in explicitly.

### Test count

- 111 non-slow whatif tests pass (`pytest -m "not slow"`), covering
  unit, integration (`test_e2e.py`), and acceptance (`test_final.py`).
- One `slow`-marked AppTest cold-start test is opt-in via `-m slow`.

### Shipping artifacts (Rule 3 · honest failures)

- Demo scenario `09_backtest_failed_rule.yaml` ships with
  `V(forecast) ≤ 0` — the failed backtest is a shipping artifact, not
  a bug.
- Demo scenario `03_agri_no_strong_analog_bundelkhand.yaml` exercises
  the no-strong-analog fallback banner.

### Deferred / known gaps

- Energy sector runner (peak-load response, coal displacement) — YAML
  ships with `status: deferred`.
- Water sector reservoir routing — next release.
- Health sector (WBGT × labour productivity) — next release.
- Disaster sector damage functions — only the return-period shift
  diagnostic is plumbed today.
- Landing card on `Home.py` — the app's entry point is `app_v2.py`;
  no `Home.py` file exists in this branch. Landing-card wiring
  deferred to a future UI pass.

### Non-changes

- `MSatu/` (paused SvelteKit clone) untouched.
- No commits to `main`; PR-ready state at `feat/whatif-engine`.
- No pushes performed.
- No data, cubes, or secrets committed. All caches under
  `climate_twin/.whatif_cache/`, `.whatif_runs/`, `.whatif_scenarios/`
  remain gitignored.
