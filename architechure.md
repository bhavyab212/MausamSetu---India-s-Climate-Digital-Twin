# MausamSetu — Architecture

> **Status:** Locked for hackathon build
> **Last updated:** 2026-07-25

---

## 1. High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js 14)                       │
│  App Router · TypeScript · shadcn/ui · Tailwind · Zustand · SWR │
│  Plotly.js · Mapbox GL JS · deck.gl · Framer Motion             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + JSON (SWR cache)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                           │
│  Uvicorn · Pydantic · async endpoints · CORS · GZip             │
│  Serves: grid, timeseries, forecast, scenario, metrics, alerts  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Python imports
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CORE LIB (mausamsetu/)                      │
│  data/ · preprocess/ · model/ · assimilate/ · storyline/        │
│  impacts/ · metrics/ · config.py                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ xarray / netCDF4 / torch
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  data/processed/cauvery.nc        (8.55 MB, 2020-2023)          │
│  checkpoints/forecaster_best.pt   (6.7 MB ConvLSTM)             │
│  External: IMD .grd · INSAT HDF5 · IMDAA NetCDF (raw)           │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Repo Layout (target after Next.js migration)

```
L:\MausamSetu\
├── mausamsetu/                    # Python core (unchanged)
│   ├── config.py
│   ├── data/                      # imd.py, insat.py, synthetic.py
│   ├── preprocess/                # build_dataset.py, dataset.py
│   ├── model/                     # convlstm.py, forecaster.py, loss.py, train.py, predict.py
│   ├── assimilate/                # enkf.py
│   ├── storyline/                 # scenario.py
│   ├── impacts/                   # hydrology.py, heat.py, rupee_risk.py
│   ├── metrics/                   # metrics.py, baselines.py
│   └── dashboard/
│       ├── api/                   # NEW — FastAPI app
│       │   ├── main.py            # app entrypoint, CORS, mounts
│       │   ├── routes/            # /grid, /forecast, /scenario, ...
│       │   └── schemas/           # Pydantic response models
│       └── _legacy_streamlit/     # old Streamlit code (backup, deprecated)
├── web/                           # NEW — Next.js 14 frontend
│   ├── app/                       # App Router
│   │   ├── page.tsx               # Overview (Command Center)
│   │   ├── map/page.tsx
│   │   ├── forecast/page.tsx
│   │   ├── scenarios/page.tsx
│   │   ├── impacts/page.tsx
│   │   ├── validation/page.tsx
│   │   └── layout.tsx             # top nav + theme
│   ├── components/                # shadcn + custom
│   ├── lib/                       # api client, hooks, utils
│   ├── stores/                    # Zustand
│   ├── styles/
│   └── public/
├── data/processed/cauvery.nc
├── checkpoints/forecaster_best.pt
├── scripts/                       # run_demo.py, download helpers
├── tests/                         # test_end_to_end.py
├── prd.md · architechure.md · rules.md · phases.md · design.md · memory.md
└── README.md
```

## 3. Backend API Contract

Base URL: `http://localhost:8011/api/v1` (MausamSetu default; configurable via `MAUSAMSETU_API_PORT`)

All endpoints return JSON. Numeric arrays are native JSON (GZip compressed
in transport). Numeric fields carry a unit either in the field name
(`rainfall_mm_per_day`) or in a sibling `unit` field. Timestamps are ISO 8601
with the IST offset (`+05:30`). Missing / sentinel values (`-999`, `99.9`,
NaN) are serialized as `null`.

### 3.1 Endpoint table (expanded, canonical for the FastAPI service)

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | Liveness + readiness: `{status, model_loaded, datacube_days, ist_time}` |
| GET  | `/state/current?date=YYYY-MM-DD` | Basin-level KPIs for the requested day (defaults to latest datacube day). |
| GET  | `/data/dates` | Full list of 1,461 available ISO dates. |
| GET  | `/data/variables` | Datacube variables with `unit`, `description`, `dtype`, and dimensions. |
| GET  | `/data/grid/{date}?var=rain` | 2D grid `{lat, lon, values, unit, date_ist}` for one variable/day. |
| GET  | `/data/timeseries?lat=&lon=&days=&end_date=` | Per-pixel time series with units. |
| GET  | `/forecast/{date}?horizon=7` | ConvLSTM MC-Dropout forecast returned as `{p10, p50, p90}` in physical units (mm/day, °C). |
| GET  | `/forecast/assimilation/{date}?var=rain` | EnKF trace `{observed, model, corrected}` with units. |
| GET  | `/scenarios/presets` | IPCC AR6 preset list from `config.IPCC_DELTAS`. |
| POST | `/scenarios/run` | Run a storyline in one of two modes; see 3.2. |
| GET  | `/impacts/hydrology?start_date=&days=` | SCS-CN basin inflow series (`m³/day`). |
| GET  | `/impacts/heat?start_date=&days=` | Heat-stress / extreme-heat day counts. |
| GET  | `/impacts/rupee?start_date=&days=&delta_temp_c=&delta_rain_pct=` | ₹-crore risk map + basin total. |
| GET  | `/validation/metrics?variable=rain&lead_day=` | Full 2023 test-split metrics for ours / persistence / climatology (cached at startup). |
| GET  | `/validation/metrics/date/{date}?variable=rain` | Per-window drill-down for the same forecast metric bundle. |
| GET  | `/alerts?date=YYYY-MM-DD` | Date-scoped derived alerts (rainfall + heat) with threshold provenance. |
| GET  | `/reports/templates` | Report templates catalogued in `design/references/09-reports.md`. |
| GET  | `/settings/thresholds` | Resolved threshold configuration (defaults + env overrides). |

### 3.2 Scenario contract

`POST /scenarios/run` accepts a discriminated request body:

```jsonc
{
  "baseline_mode": "historical" | "forecast",
  "start_date": "2023-06-01",
  "days": 30,                     // required in historical mode
  "horizon": 7,                   // used only in forecast mode
  "delta_temp_c": 2.0,
  "delta_rain_pct": -20,
  "seasonal_months": [6, 7, 8, 9],
  "label": "monsoon-drop-20"
}
```

- `historical` — baseline is `cauvery.nc` sliced over `[start_date, start_date+days-1]`.
- `forecast`   — baseline is the ConvLSTM `p50` forecast initialised at
  `start_date` for `horizon` days (in physical units).

Response bundles baseline, scenario, delta, and impact channels
(hydrology, heat, ₹-risk) with units and provenance.

### 3.3 Alerts contract

Alert thresholds resolve from `settings_service`:

- `rain_warning_mm_per_day` — default = 2020–2021 basin-mean daily rainfall p95
- `rain_critical_mm_per_day` — default = 2020–2021 basin-mean daily rainfall p99
- `heat_warning_c` — default = `config.HEAT_STRESS_TEMP` (40 °C)
- `heat_critical_c` — default = `config.EXTREME_HEAT_TEMP` (45 °C)

Environment overrides: `MAUSAMSETU_ALERT_RAIN_WARNING_MM`,
`MAUSAMSETU_ALERT_RAIN_CRITICAL_MM`, `MAUSAMSETU_ALERT_HEAT_WARNING_C`,
`MAUSAMSETU_ALERT_HEAT_CRITICAL_C`. Each alert emits its evaluated date,
observed value, threshold, threshold source, and data source.

## 4. Frontend Data Flow

- SWR hooks in `web/lib/hooks/` (e.g. `useGrid(date, var)`) → FastAPI
- Zustand stores in `web/stores/` for cross-page state (selected date, selected cell, scenario params)
- Plotly.js for charts, deck.gl / Mapbox for map — server components fetch initial data, client components handle interaction

## 5. Model Pipeline

```
Raw (IMD .grd / INSAT .h5)
   ↓ data/*.py
Regridded to 0.25° master grid
   ↓ preprocess/build_dataset.py
cauvery.nc  (rain, tmax, tmin, insat_lst, insat_rain + anomalies + climatology)
   ↓ preprocess/dataset.py
PyTorch Dataset (6-day input → 7-day target)
   ↓ model/train.py
forecaster_best.pt  (ConvLSTM, 1.67M params)
   ↓ model/predict.py (MC-Dropout, 20 members)
Ensemble forecast (p10/p50/p90)
   ↓ assimilate/enkf.py
EnKF-corrected state
   ↓ storyline/scenario.py
Counterfactual runs
   ↓ impacts/*.py
Hydrology (SCS-CN) / Heat / ₹ risk
```

## 6. Loss Function (physics-informed)

```
L = α·MSE_rain + β·BCE_occurrence + γ·MSE_temp + δ·L_physics + ε·L_smooth

L_physics  = penalty(rain < 0 | tmax < tmin)
L_smooth   = TV regularization on spatial output
```

## 7. Assimilation (EnKF)

- 20-member ensemble
- State x = [rain, tmax, tmin] over grid
- Obs y = IMD daily gridded (perturbed with obs error)
- Kalman gain K = PHᵀ (HPHᵀ + R)⁻¹
- Verified: **43% RMSE reduction** post-assimilation

## 8. Deployment (hackathon)

- Local dev: `uvicorn` on :8000 + `next dev` on :3000
- Optional demo build: `next build && next start` behind FastAPI static mount
- No cloud required for the demo; laptop-only

## 9. Extensibility Hooks

- New pilot regions: change `config.py` bounds → rebuild `cauvery.nc` under a new name
- New variables: add reader in `data/`, register in `preprocess/build_dataset.py`
- New model: keep the `predict()` contract (returns dict of arrays with same shape); train under `model/`
- New impact: drop a module in `impacts/` and expose via a `/scenario` field
