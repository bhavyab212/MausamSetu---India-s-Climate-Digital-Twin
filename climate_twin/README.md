---
title: ISRO Climate Digital Twin
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.1
app_file: app_v2.py
pinned: false
python_version: 3.10.13
---

# 🛰️ ISRO Climate Digital Twin — V2
> **Internship Project | Indian Space Research Organisation (ISRO)**  
> *High-Fidelity Spatiotemporal Climate Modelling for India using Deep Learning Ensembles & Digital Twin Architecture*

---

## 📌 Project Overview

This project implements an advanced **Digital Twin of India's Climate System**, designed to simulate, analyze, and visualize multi-decadal meteorological data (1975–2075). Aligned with the architectural principles of modern planetary twins (e.g., Destination Earth/EGUsphere frameworks), this Digital Twin fuses deep learning with physical climatology constraints.

**Key Capabilities:**
1. **Historical Reconstruction:** High-fidelity modeling of historical IMD data (0.25° resolution).
2. **Predictive State Generation:** Using a dual-architecture AI engine (U-Net + ConvLSTM) to predict fluid temporal evolution and spatial structures.
3. **Interactive 3D Visualization:** Real-time data streaming to a WebGL-powered CesiumJS global twin.

---

## 🏗️ Architecture & System Framework

This project employs a cutting-edge ensemble methodology to fuse spatial and temporal data. 

*To view presentation-ready diagrams (16:9, DestinE-style, interactive — for PowerPoint):*
- **[docs/ppt/isro_ppt_index.html](./docs/ppt/isro_ppt_index.html)** — Six slides: architecture, workflow, data stream, ±ΔT what-if, ensemble diagnostics, app logic.

*Legacy full-page diagrams:*
- **[isro_architecture_premium.html](./isro_architecture_premium.html)** — End-to-end framework and neural network flow.
- **[isro_pipeline_premium.html](./isro_pipeline_premium.html)** — Data pipeline from NetCDF to dashboard.

### Core AI Components
| Component | Function | Specification |
|---|---|---|
| **U-Net** | Spatial Generator | Reconstructs static field hierarchies using Encoder-Decoder blocks. Predicts spatial distribution patterns. |
| **ConvLSTM** | Temporal Sequencer | Captures historical memory (T-30 windows) to predict the fluid temporal evolution of weather systems. |
| **Ensemble Averaging** | Fusion Node | Combines U-Net and ConvLSTM outputs to balance spatial sharpness with temporal consistency. |
| **QMBC** | Physical Calibration | **Quantile Mapping Bias Correction**: Enforces physical boundaries (e.g., preventing negative rainfall, capping thermal anomalies at ±7°C). |

---

## 📊 Dataset & Processing Pipeline

| Variable | Source | Resolution | Coverage |
|---|---|---|---|
| Maximum Temperature | IMD Gridded Dataset | 0.25° x 0.25° | Indian Landmass |
| Accumulated Rainfall | IMD Gridded Dataset | 0.25° x 0.25° | Indian Landmass |

**The Pipeline Flow:**
1. **Extraction (`data_preprocessing.py`)**: Ingesting raw NetCDF files.
2. **Masking (`zones.py`)**: Applying boolean masks to isolate Indian political boundaries and remove oceanic noise.
3. **Inference (`model.py`)**: Generating T+1 predictive tensors via the ensemble AI.
4. **Correction (`baseline.py`)**: Applying QMBC and climatological baselines.
5. **Dashboard (`app_v2.py`)**: Serving the WebGL 3D interface and analytical tabs.

---

## 🖥️ Interactive Dashboard (App V2)

The `app_v2.py` Streamlit application serves as the primary user interface for the Digital Twin, featuring 6 core tabs:

1. **Animated Indian Twin (Cesium 3D)**: A WebGL-powered 3D globe overlaying calibrated forecast tensors (Rainfall, Temp, Cloud Proxy) with timeline playback and micro-animations.
2. **Day-Wise Climate Explorer**: Drill down into specific historical or future predicted days.
3. **Historical Twin**: Annual aggregation and validation against official IMD baseline records.
4. **±1°C What-If Storyline**: Interactive sensitivity mapping showing rainfall alterations under +1°C to +3°C warming scenarios.
5. **Zone Projections**: IMD meteorological subdivision drilldowns (Northwest, Peninsular, etc.).
6. **Model Quality Assessment**: Live validation metrics (RMSE, MAE, Pearson Correlation) comparing the ensemble against Persistence and Climatology baselines.

---

## ⚙️ Setup & Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd isro-digital-twin-poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure IMD data files (.nc) are in the data/ directory
#    (Requires: temp_ind*.nc and RF25_ind*.nc)

# 4. Launch the Next-Gen Digital Twin Dashboard
streamlit run app_v2.py
```

> **Note on Cesium 3D:** The Animated Twin tab uses CesiumJS. While it works out of the box with the default ion token, for production deployment, you may need to supply your own `CESIUM_ION_TOKEN` as an environment variable.

---

## 🔬 Scientific Alignment & Performance

The ensemble model is heavily benchmarked to ensure scientific credibility for research reporting:
- Uses **Quantile Mapping Bias Correction (QMBC)** to ensure statistical distributions match historical realities (preventing "model collapse" into mean climatology).
- Achieves high Pearson correlation scores (0.80+) for 1-day ahead forecasts.
- Evaluated against IMD Annual Climate Statements and IPCC AR6 regional scenarios.

---

## 👤 Internship Details

**Organisation:** Indian Space Research Organisation (ISRO)  
**Project:** Proof of Concept — Climate Digital Twin  
**Technologies:** Python, TensorFlow/Keras, Streamlit, CesiumJS (WebGL), Plotly, Xarray, Scikit-Learn 

*Designed to demonstrate the feasibility of planetary-scale deep spatiotemporal modeling, aligned with ISRO's Earth Observation and climate monitoring missions.*
