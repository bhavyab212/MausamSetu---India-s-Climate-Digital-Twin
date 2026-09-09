# MausamSetu — Climate Digital Twin of India

MausamSetu (मौसम सेतु, “weather bridge”) is a Streamlit climate-analysis dashboard for India. It combines IMD gridded rainfall and temperature data with zone-aware processing, trained climate models, interactive exploration, validation, climate spirals, and a What-If storyline.

The active application is **Streamlit only**. The current entrypoint is:

```text
climate_twin/app_v2.py
```

The separate `MSatu/` and `web/` applications are not part of the Streamlit runtime.

## Current dashboard tabs

The sidebar in `app_v2.py` contains:

- **Home** — current cube-day summary, seven-day historical tail, temperature and rainfall cards.
- **Explorer** — daily rainfall and maximum-temperature maps, comparison views, and animation.
- **What If** — Storyline temperature-change analysis with PAST / PRESENT / FUTURE rainfall maps and a rainfall-sensitivity map.
- **Climate Spirals** — annual climate visualization.
- **Deep Analytics** — historical trends and analytical charts.
- **🔥 Training** — training dashboard, when the training assets are available.
- **🔍 Validation** — model validation dashboard.
- **🤖 RL Agent** — reservoir / policy analysis, when the RL assets are available.

The Explorer year selector is intentionally capped at **2028** for the day-wise view.

## Requirements

- Windows 10/11 or a compatible Linux environment
- Python 3.11 or 3.12 recommended
- A working virtual environment
- Streamlit
- Python dependencies listed in `requirements.txt`
- The processed climate data described below

PyTorch is required for model-dependent features such as Training, Validation, and some future/ensemble views. CPU execution is supported where the selected feature permits it.

## Installation on Windows

Open PowerShell or Git Bash at the repository root:

```bash
cd L:/MausamSetu
python -m venv venv
```

Activate the environment.

### Git Bash

```bash
source venv/Scripts/activate
```

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repository already includes `venv/`, use its interpreter directly instead of creating another environment:

```bash
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Data layout

The dashboard reads the processed cubes from the project-level `data/processed/` directory:

```text
data/
└── processed/
    ├── india.nc
    ├── cauvery.nc
    ├── india_norm_stats.json
    ├── cauvery_norm_stats.json
    ├── manifest.yaml
    └── manifest.sig
```

The application also uses these assets under `climate_twin/`:

```text
climate_twin/
├── India_States_2024.geojson
├── sensitivity_map.npy
├── convlstm.weights.h5
├── convlstm_daily.weights.h5
├── unet.weights.h5
├── scalers.json
├── scalers_daily.json
└── regions/
    ├── india_zones.yaml
    ├── zone_mask.npy
    ├── zone_membership.npy
    └── zone_stats.json
```

The processed India cube currently covers daily data from **1951 through 2025**. The Home page displays the current calendar date and uses the same day-of-year from the latest available cube year when today’s exact year is not present.

### Building the processed cube

If `data/processed/india.nc` is missing, build it from the available raw IMD/INSAT inputs using the project’s data pipeline. The exact command depends on the raw-data setup; inspect `climate_twin/data/` and `climate_twin/data_source.py` first.

The application will show a clear missing-data message rather than silently inventing a cube.

## Run the dashboard

From `L:/MausamSetu`:

```bash
./venv/Scripts/streamlit.exe run climate_twin/app_v2.py --server.headless true --server.port 8501 --server.fileWatcherType none
```

Then open:

```text
http://localhost:8501
```

The command above disables Streamlit’s file watcher. Use it when you want a stable run and restart manually after code changes.

For normal development with automatic reload:

```bash
./venv/Scripts/streamlit.exe run climate_twin/app_v2.py --server.port 8501
```

If port `8501` is already occupied on Windows, find and stop the process:

```powershell
Get-NetTCPConnection -LocalPort 8501 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

Or choose another port:

```bash
./venv/Scripts/streamlit.exe run climate_twin/app_v2.py --server.port 8502
```

## How the current What-If tab works

The sidebar’s **What If** entry is wired directly to `_render_tab3()` in `climate_twin/app_v2.py`.

It provides:

1. A temperature-change slider from **−2.0°C to +3.0°C**.
2. PAST rainfall climatology for **1975–1990**.
3. PRESENT rainfall climatology for **2010–2024**.
4. FUTURE rainfall calculated from the present field and `sensitivity_map.npy`.
5. A rainfall sensitivity map in `mm per +1°C`.

The What-If display uses the existing Plotly visualization helpers and a common color range across the three rainfall panels.

## Performance and caching

Expensive derived arrays are cached with Streamlit’s `@st.cache_data`:

- Home hero values and the seven-day cube tail
- Home / Deep Analytics climatology statistics
- Explorer daily rainfall and Tmax grids
- What-If past/present rainfall fields
- What-If future rainfall fields for each slider step
- Per-year trend series

Caches are cleared when the active region changes so an India-shaped result cannot leak into a Cauvery view.

If a code change appears not to show in the browser:

1. Stop the running Streamlit process.
2. Start it again with the command above.
3. Hard-refresh the browser with `Ctrl+Shift+R`.

## Project structure

```text
L:/MausamSetu/
├── climate_twin/
│   ├── app_v2.py                 # Streamlit entrypoint
│   ├── data_source.py            # Processed-cube access
│   ├── perf.py                   # Optional timing instrumentation
│   ├── src/                      # Plotting, model, zones, utilities
│   ├── regions/                  # Zone masks and zone statistics
│   ├── train/                    # Training and validation dashboard code
│   ├── rl/                       # RL/reservoir analysis code
│   ├── whatif/                   # Scenario-engine package and tests
│   └── pages/_archive/            # Archived multipage experiments
├── data/processed/               # Live processed climate cubes
├── Subbasin/                     # Cauvery/sub-basin GIS data
├── checkpoints/                  # Model checkpoints
├── scripts/                      # Data and utility scripts
├── notebooks/                    # Exploratory notebooks
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Testing

Run the current What-If test suite:

```bash
./venv/Scripts/python.exe -m pytest climate_twin/whatif/tests -m "not slow" -q
```

Run all tests:

```bash
./venv/Scripts/python.exe -m pytest -q
```

Static What-If guard checks:

```bash
./venv/Scripts/python.exe climate_twin/whatif/tests/guards/leakage_ast.py
./venv/Scripts/python.exe climate_twin/whatif/tests/guards/colormap_blocklist.py
./venv/Scripts/python.exe climate_twin/whatif/tests/guards/citation_scan.py
```

## Common problems

### `ModuleNotFoundError: climate_twin`

Run Streamlit from the repository root and use the project interpreter:

```bash
cd L:/MausamSetu
./venv/Scripts/streamlit.exe run climate_twin/app_v2.py
```

### `FileNotFoundError: zone_stats.json`

Build or restore the zone registry under `climate_twin/regions/`. The app expects `zone_stats.json`, `zone_mask.npy`, and `zone_membership.npy`.

### `numpy ArrayMemoryError`

Do not materialize the full multi-decade cube when selecting one day. The processed data readers use lazy xarray indexing. Restart Streamlit after changing the reader so cached results are discarded.

### `Sensitivity map not found`

The expected file is:

```text
climate_twin/sensitivity_map.npy
```

The loader also checks `climate_twin/data/sensitivity_map.npy` and the repository-root `data/sensitivity_map.npy` for compatibility.

### Streamlit starts but the browser shows old content

The stable command uses `--server.fileWatcherType none`, so changes require a restart. Stop the process, start it again, and press `Ctrl+Shift+R` in the browser.

### SciPy / NumPy compatibility warning

Some environments report a warning when SciPy was built for a different NumPy range. The dashboard may still run, but for a clean environment reinstall the pinned project dependencies in a fresh virtual environment.

## Data and model safety

Do not commit:

- raw climate data
- NetCDF cubes
- model checkpoints
- generated caches
- `.env` files or credentials

The live data and model assets are local runtime inputs. Keep them outside commits unless the project explicitly requires a small, documented fixture.

## Scope

This repository is Streamlit-focused. The separate web/API applications were removed from the working tree. The remaining `climate_twin/whatif/` package, training code, RL code, data pipeline, and documentation are intentionally kept because they may be needed by the current dashboard or future development.

## Team

**MausamSetu — BAH 2026**

Climate Digital Twin of India using national climate data and reproducible, inspectable analysis.
