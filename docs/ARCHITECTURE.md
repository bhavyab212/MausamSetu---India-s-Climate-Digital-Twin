# MausamSetu — System Architecture

## Big Picture

```
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA & INGESTION                                            │
│  IMD gridded .grd → mausamsetu.data.imd                              │
│  INSAT HDF5 .h5   → mausamsetu.data.insat                            │
│  Synthetic fake   → mausamsetu.data.synthetic                        │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 2: PREPROCESSING (build_dataset.py)                            │
│  1. Crop to Cauvery basin (10-14.5°N, 75.5-79.5°E)                    │
│  2. Regrid everything to 0.25° master grid                            │
│  3. Compute climatology (train years only)                            │
│  4. Anomaly = raw − climatology                                       │
│  5. Standardize (z-score using train stats)                           │
│  → OUTPUT: data/processed/cauvery.nc                                  │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 3: TWIN CORE (the heartbeat)                                   │
│                                                                       │
│   ┌───────────────┐         ┌────────────────┐                        │
│   │  ConvLSTM     │         │  EnKF          │                        │
│   │  Forecaster   │◄────────│  Assimilation  │                        │
│   │  6 in → 7 out │  x_a    │  x_a = x_f +   │                        │
│   │  + Hurdle rain│         │      K·(y-x_f) │                        │
│   │  + MC-Dropout │         │  ensemble=20   │                        │
│   └───────┬───────┘         └────────┬───────┘                        │
│           │                          │                                 │
│           │ forecast x_f             │ observation y                   │
│           ▼                          ▼                                 │
│   +───────────────────────────────────────────+                       │
│   | STATE  x_a  (best estimate at time t)     |                       │
│   +───────────────────────────────────────────+                       │
│           │                                                            │
│           ▼                                                            │
└────────────┬──────────────────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 4: STORYLINE & IMPACTS                                         │
│  • Storyline engine: PGW + IPCC AR6 deltas → perturbed state         │
│  • Hydrology: SCS-CN → basin inflow (non-linear response)             │
│  • Heat: threshold day-counters + wet-bulb globe temp                 │
│  • ₹ risk: aggregate cost per pixel (surrogate for district)          │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 5: DASHBOARD (Streamlit + Plotly + Folium)                     │
│  1. Live Map  2. Forecast+Uncertainty  3. What-If Studio              │
│  4. Extremes  5. Validation Console    6. Data & Model Cards           │
└───────────────────────────────────────────────────────────────────────┘
```

## Digital Twin Properties (all 4 implemented)

| # | Property         | Implementation                                         |
|---|------------------|--------------------------------------------------------|
| 1 | Representation   | Gridded state on 0.25° Cauvery grid (19×17 pixels)     |
| 2 | Synchronization  | EnKF assimilation loop (ensemble Kalman filter)         |
| 3 | Predictivity     | ConvLSTM forecaster with hurdle rain heads              |
| 4 | Counterfactuals  | Storyline engine with IPCC AR6 pathways + custom knobs  |

## Data Sources

| Source     | Variable                | Res     | Cadence | Format |
|------------|-------------------------|---------|---------|--------|
| IMD Pune   | Gridded Rainfall         | 0.25°  | Daily  | .grd binary |
| IMD Pune   | Gridded Tmax / Tmin      | 1.0°   | Daily  | .GRD binary |
| MOSDAC (ISRO) | INSAT LST / SST / IMC | ~4 km  | 30 min | .h5 HDF5 |
| NCMRWF     | IMDAA reanalysis (opt.)  | 12 km  | Hourly | GRIB/NetCDF |
