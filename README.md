# 🇮🇳 MausamSetu — AI-Powered Digital Twin of India's Climate

**ISRO BAH 2026 Submission** | Pilot Region: **Cauvery Basin**

MausamSetu (मौसमसेतु = "Weather Bridge") is an AI-powered digital twin that fuses
IMD ground observations with INSAT satellite data to forecast rainfall & temperature,
quantify uncertainty, and simulate what-if scenarios for water/agriculture/heat-stress
adaptation — running 100% on Indian national datasets.

## 🎯 What Makes It a TRUE Digital Twin (Not Just a Forecaster)

| Property | Implementation |
|----------|----------------|
| P1 — Digital Representation | Gridded state on 0.25° Cauvery grid |
| P2 — Synchronization | **EnKF assimilation loop** with INSAT observations |
| P3 — Predictivity | ConvLSTM 7-day forecast with uncertainty (p10–p90) |
| P4 — Counterfactuals | Storyline engine (PGW deltas + IPCC AR6 scenarios) |

## 📐 Pilot Region: Cauvery Basin

- **Latitude:** 10.0°N to 14.5°N
- **Longitude:** 75.5°E to 79.5°E
- **Grid size:** 18 × 16 = 288 pixels at 0.25° (~28 km/pixel)
- Contains: KRS, Mettur, Kabini reservoirs
- Western Ghats on western edge (orographic rainfall)
- Rain-shadow on east (Tamil Nadu interior)

## 📊 Data Sources (100% Indian National)

| Source | Product | Resolution | URL |
|--------|---------|-----------|-----|
| IMD | Gridded Rainfall | 0.25° daily | imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html |
| IMD | Max Temperature | 1.0° daily | imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html |
| IMD | Min Temperature | 1.0° daily | imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html |
| ISRO MOSDAC | INSAT LST (3RIMG_L2B_LST) | ~4 km / 30 min | mosdac.gov.in |
| ISRO MOSDAC | INSAT SST (3RIMG_L2B_SST) | ~4 km / 30 min | mosdac.gov.in |
| ISRO MOSDAC | INSAT Rainfall (3RIMG_L2B_IMC) | ~4 km / 30 min | mosdac.gov.in |
| NCMRWF | IMDAA Reanalysis (optional) | 12 km hourly | rds.ncmrwf.gov.in |

## 🏗️ Architecture

```
[IMD + INSAT DATA] → [PREPROCESS] → [ConvLSTM FORECAST]
        ↑                                    ↓
        └── [EnKF ASSIMILATION] ← [OBSERVATION] ─┐
                    ↓                            │
             [STORYLINE + IMPACTS] → [DASHBOARD] ┘
```

## 🚀 Quick Start

```bash
# 1. Activate the venv
.\venv\Scripts\activate

# 2. Generate synthetic data (works without any downloads)
python -m mausamsetu.data.synthetic

# 3. Build the processed dataset
python -m mausamsetu.preprocess.build_dataset

# 4. Train the model (or use pretrained checkpoint)
python -m mausamsetu.model.train

# 5. Run the dashboard
streamlit run mausamsetu/dashboard/app.py
```

## 📅 Build Timeline

- Day 0: Setup + accounts + downloads (parallel)
- Day 1: Data readers + preprocessing → `cauvery.nc`
- Day 2-3: ConvLSTM training
- Day 4: EnKF assimilation loop
- Day 5: Storyline + impact modules
- Day 6: Dashboard + validation
- Day 7: Demo polish

## 📚 Documentation

- `docs/ARCHITECTURE.md` — System architecture
- `docs/DATA_CARD.md` — Data provenance
- `docs/MODEL_CARD.md` — Model transparency

## 👥 Team

Team MausamSetu — BAH 2026
