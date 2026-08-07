# MausamSetu · What-If Scenario Engine — Run Book

The document a new operator opens first. Written for someone who has
never touched the codebase but is technical enough to install Python
and run `git`.

## 1. Cold-start install

Windows (PowerShell or Git Bash):

```bash
git clone <repo-url> MausamSetu
cd MausamSetu
python -m venv venv
./venv/Scripts/pip install --upgrade pip
./venv/Scripts/pip install -r requirements.txt
```

Verify:

```bash
./venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
./venv/Scripts/python -m pytest climate_twin/whatif/tests -m "not slow" -q
```

You should see **100 passed** or better (as of Part 7). If pytest
reports fewer than the CI count, stop and read `docs/LIMITATIONS.md`.

## 2. Data mounts

The engine assumes:

| Purpose | Path (Windows) | Notes |
| --- | --- | --- |
| IMD rain / tmax / tmin | `L:\MausamSetu\data\Rainfall`, `\max_temp`, `\min_temp` | yearly `.GRD` or netCDF |
| Processed India cube | `L:\MausamSetu\data\processed\india.nc` | built by `climate_twin/data/build_cube.py` |
| CWC sub-basin shapefile | `L:\MausamSetu\Subbasin\Subbasin.shp` | WGS-84 LCC |
| Zone registry | `climate_twin/regions/` | built by `climate_twin/regions/build_mask.py` |
| NEX-GDDP-CMIP6 mirror | `NEX_GDDP_ROOT` env var → any dir | optional; falls back to `s3://nex-gddp-cmip6/` if `fsspec + s3fs` are installed |
| Cache | `climate_twin/.whatif_cache/` | gitignored, regenerated on demand |
| Scenario saves | `climate_twin/.whatif_scenarios/` | gitignored, user-owned |
| Provenance runs | `climate_twin/.whatif_runs/` | gitignored, one YAML per run |

If a path differs on your machine, edit `climate_twin/whatif/config/paths.py`
— every downstream module imports the constants from that one file.

## 3. First run

```bash
./venv/Scripts/streamlit run climate_twin/app_v2.py
```

Open http://localhost:8501 in a browser. Streamlit's multipage router
picks up `climate_twin/pages/30_What_If.py` automatically. In the left
sidebar, click **What If — Scenario Engine**.

The Short Term tab loads. Levers on the left, panels on the right.

**Sanity check**: press **Run scenario** with default levers.
Vidarbha paddy on 2020-06-15 should produce a payoff matrix, a
recommendation card, and a filled Historical analog panel.

**Load a demo**: expand the "Load" area at the bottom, pick
`demo_01_agri_vidarbha_paddy_baseline`, hit Load. The page rebuilds
with the demo state. Every one of the 10 demo YAMLs at
`climate_twin/whatif/demo/scenarios/` is loadable this way.

If the page fails to render:
- **`FileNotFoundError` on `zone_stats.json`** → the zone registry
  hasn't been built; run `python -m climate_twin.regions.build_mask`.
- **`numpy._core._exceptions._ArrayMemoryError` on 1.78 GiB** → the
  historical driver tried to materialise the full 75-year cube; the
  fix (already in Part 1) is `.isel(time=idx).values` — confirm
  `climate_twin/whatif/drivers/historical.py` matches HEAD.
- **`ModuleNotFoundError: climate_twin`** → Streamlit's `pages/*.py`
  gets its own `sys.path`; the top of `30_What_If.py` prepends the
  project root. If that line was edited out, restore it.

## 4. Refreshing data

- **New IMD year lands**: drop the yearly file into `data/Rainfall/`
  etc., then rerun `python -m climate_twin.data.build_cube` to
  rebuild the processed cube. The pickle caches under
  `.whatif_cache/historical/` regenerate on next read.
- **New ensemble artifact**: your training run wrote a new checkpoint
  under `climate_twin/train/registry/models/india/`. The ensemble
  driver picks the newest compatible artifact (matching zone signature
  + manifest signature) automatically.
- **NEX-GDDP mirror update**: pull the new year's files into
  `$NEX_GDDP_ROOT`. The catalog in
  `climate_twin/whatif/drivers/nex_gddp_catalog.yaml` is static; edit
  the coverage block if a new SSP or model is added.

## 5. Adding a crop

1. Edit `climate_twin/whatif/sectors/crops.yaml`. Every value **must**
   carry a citation. The Pydantic validator refuses missing keys.
2. The YAML's SHA-256 changes. It flows into every subsequent
   scenario's provenance record automatically — you do NOT need to
   update anything else.
3. Run:
   ```bash
   ./venv/Scripts/python -m pytest climate_twin/whatif/tests/test_agriculture.py -q
   ```
   Every test should still pass. Tests that touched the old crop set
   use `load_crop(key)` so they migrate automatically.

## 6. Adding an adaptation

1. Edit `climate_twin/whatif/sectors/adaptations.yaml`. Every option
   needs `capex_inr_per_ha` (or `capex_inr_per_kwh`) with a
   `citation`. Options without a citation land with
   `cost_complete=False` and the NPV runner will refuse them until you
   fix the citation.
2. Run `pytest climate_twin/whatif/tests/test_long_term.py -q` — the
   `test_adaptations_yaml_ships_four_cited_options` test will fail
   loudly if any shipped option lost its citation.

## 7. Adding a decision rule to the backtest

1. Author a small class in your own module that satisfies the
   `DecisionRule` protocol: `rule_id: str`, `__call__(history_before_y, y) -> float`.
2. Write a leakage test alongside: use `walk_forward_backtest` and
   verify the rule never sees year `y` in its input. The harness
   already filters, but the test proves the contract.

## 8. Adding a demo scenario

1. Copy the YAML template from `climate_twin/whatif/demo/scenarios/01_*.yaml`.
2. Set an `id`, `status: shipped`, and a self-contained state block.
3. Add a replay test in `climate_twin/whatif/tests/test_final.py::test_every_demo_yaml_loads`.
4. Verify: `pytest climate_twin/whatif/tests/test_final.py -q`.

## 9. Failure modes — the top ten

| Symptom | Root cause | Fix / where |
| --- | --- | --- |
| `−999.0` or `99.9` bleed into the numbers | IMD sentinels not masked | `whatif/config/constants.py::IMD_SENTINELS`; every reader applies `mask_sentinels()` before returning |
| Timestamps show `Z` / `+00:00` | tz-naive datetime somewhere | search for `datetime.now()` without `tz=…`; every timestamp in `whatif/` must be IST-aware |
| Cold-start SPI is impossibly slow | fit not cached, running per cell | the fit is cached under `.whatif_cache/fits/`; the diagnostic page gates SPI-3 behind a compute button |
| Scenarios differ between machines | code SHA drifted (dirty tree) | provenance drawer surfaces `dirty`; commit + rerun |
| Ensemble driver returns q10=q50=q90 | no compatible checkpoint → fallback | `ensemble` driver stamps `source="mausamsetu_ensemble_fallback:cube"` on this path so provenance never lies |
| NEX-GDDP raises `NEXGDDPUnavailable` | no local mirror + no `fsspec + s3fs` | set `NEX_GDDP_ROOT` or `pip install fsspec s3fs`; LT panels surface an "illustrative until data on disk" banner meanwhile |
| Perturbation export refuses | Rule 6 gate | tick the caveat checkbox in the levers panel |
| Analog panel shows "no strong analog" | pool has no fair-or-better match | this is honest reporting (Rule 7 + Rule 10); confidence drops to Low, no automatic hiding |
| Backtest V(forecast) ≤ 0 | rule fails to beat climatology | ship it (Rule 3); the red-header tag is a feature |
| Landing card missing on Home | Home.py doesn't exist in this branch; entry is `app_v2.py` — deferred item, see PR description |
