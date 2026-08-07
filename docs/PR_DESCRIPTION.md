# What-If Scenario Engine (Parts 0–8)

> An audit-tagged pipeline that turns climate data into rupees for
> India-scale decision support, with a Short-Term and Long-Term tab
> and honesty guardrails enforced by CI.

## Problem

Climate variables do not reach rupees in one step. A monolithic black
box that jumps from rainfall to a district officer's recommendation
is not defensible in a review, and every review this project would
face is a serious one. The engine has to expose its work at every
layer or nobody trusts the last layer.

## Solution — five-layer impact chain

```
L0 driver   →  L1 index  →  L2 biophysical  →  L3 sector  →  L4 economics
historical      ET0            water balance     agriculture   payoff matrix
ensemble        SPI-3          root-zone         yield         cost-loss V(f)
analog          Rx1day         Ks / Dr / DP      NPV           regret / VaR
perturbation    onset          FAO-56 bucket     baseline      tornado
SSP + QDM       return-period                    to-district   scenario card
```

Every layer versioned, every DataArray tagged with `source_chain`,
`units`, `quantile`. Every scenario replays byte-identically from its
YAML. See `docs/WHATIF_ARCHITECTURE.md`.

## What ships

- **Short Term tab** — historical analogs, ensemble ± perturbation,
  payoff matrix over decisions × climate states, recommendation card
  with q10 / q50 / q90 net revenue, sensitivity tornado, walk-forward
  backtest with red-header failure tag.
- **Long Term tab** — SSP1-2.6 / 2-4.5 / 3-7.0 / 5-8.5 with
  NEX-GDDP-CMIP6 10-GCM ensemble, QDM downscaling, Hawkins-Sutton
  uncertainty decomposition, return-period shift, time-of-emergence,
  adaptation NPV at 7 % / 8 % / 12 %.
- **Provenance drawer** — every number reconstructible from a single
  YAML block; the drawer shows layer versions, dataset SHAs, code SHA,
  and offers `Download YAML` + `Replay`.
- **Auto-generated one-pager** —
  `whatif.report.render_one_pager(path)` writes a self-contained
  Print-to-PDF HTML pulling live values from the engine. Never
  drifts from the UI.

## Demo scenarios (`climate_twin/whatif/demo/scenarios/`)

Ten YAMLs loadable from the UI dropdown:

| # | File | Purpose |
| - | --- | --- |
| 01 | `01_agri_vidarbha_paddy_kharif_shortterm_baseline.yaml` | Reference case, three crops × tercile analog states |
| 02 | `02_agri_vidarbha_droughtlike_analog.yaml` | Low-rain tercile forces bajra recommendation |
| 03 | `03_agri_no_strong_analog_bundelkhand.yaml` | **No-strong-analog banner** (Rule 10) |
| 04 | `04_agri_perturbation_rain_minus_20pct.yaml` | Method-1 delta with caveat acknowledged |
| 05 | `05_agri_wheat_rabi_heatstress_gujarat.yaml` | Wheat flowering heat overlay |
| 06 | `06_disaster_chennai_return_period_shift_ssp245.yaml` | LT · Chennai · Rx1day return-period shift |
| 07 | `07_agri_longterm_vidarbha_ssp126_vs_ssp585_2050.yaml` | LT · adaptation payoff matrix |
| 08 | `08_energy_delhi_heatwave_2022.yaml` | Energy sector · **status: deferred** |
| 09 | `09_backtest_failed_rule.yaml` | **Shipped failed backtest** (Rule 3 honesty artifact) |
| 10 | `10_stress_test_dirty_tree.yaml` | Dev-only · dirty-tree export gate |

## Honesty guardrails enforced by CI

- **Leakage guard** — AST scan refuses `.fit(...)` referencing
  `VALID_YEARS` under `whatif/indices/` or the backtest module.
- **Verb lint** — `whatif/ui/copy/long_term.py` cannot contain
  `predict / forecast / will / is going to`.
- **Colormap blocklist** — no `jet / hsv / rainbow / nipy_spectral`
  under `whatif/ui/` or `whatif/report/`.
- **Bare-rupee guard** — `EconomicOutcome` refuses scalars.
- **Docstring citation scan** — every module under
  `whatif/indices/`, `/biophysical/`, `/sectors/`, `/economics/`, and
  the LT drivers carries `Primary source` / `Citation`.
- **Resolution ceiling** — grid-cell inputs to `value_agriculture`
  raise `ResolutionCeilingError`.
- **Perturbation gate** — export refuses without `caveat_acknowledged=True`.
- **Adaptation cost citation** — `adaptation_npv` refuses uncited options.

CI at `.github/workflows/whatif_ci.yml`. Pre-commit at
`.pre-commit-config.yaml`.

## Known limitations (`docs/LIMITATIONS.md`)

- INSAT record ~2 years; fused-encoder contribution is fundamentally
  small.
- Reference ET0 is Hargreaves-Samani, not Penman-Monteith (awaits ERA5).
- APY district yields have known reporting lags; the validator refuses
  to fabricate numbers.
- NEX-GDDP-CMIP6 is remote — panels label the LT chart illustrative
  when `NEX_GDDP_ROOT` isn't set. Same policy as Part 3's soil-default
  banner: never fabricate.
- Van den Dool 1994 governs analog scarcity; many region × windows
  will have no `strong` match.
- Multi-model spread widens with LT horizon — physics, not a bug.
- Landing card on `Home.py` deferred (the app entry is `app_v2.py`;
  no `Home.py` file in this branch).

## Reviewer checklist

Please confirm:

- [ ] Cold-start install per `docs/RUNBOOK.md` succeeds.
- [ ] Demo scenario 01 loads and matches the one-pager's numbers.
- [ ] Demo scenario 03 shows the no-strong-analog banner.
- [ ] Demo scenario 09 renders the failed backtest with the red tag.
- [ ] Verb lint passes: `pytest climate_twin/whatif/tests/test_ui.py::test_long_term_copy_bans_forbidden_verbs -q`.
- [ ] Colormap guard passes: `python climate_twin/whatif/tests/guards/colormap_blocklist.py`.
- [ ] Full test suite: `pytest climate_twin/whatif/tests -m "not slow" -q` shows 111+ passed.

## When ready to open

Bhavya runs:

```bash
git push -u origin feat/whatif-engine
gh pr create \
    --title "What-If Scenario Engine (Parts 0-8)" \
    --body-file docs/PR_DESCRIPTION.md
```

No push has happened. The branch is ready; the merge is your call.

## History

Eight-commit Part-by-Part history preserved (no squash). See
`git log --oneline feat/whatif-engine`.
