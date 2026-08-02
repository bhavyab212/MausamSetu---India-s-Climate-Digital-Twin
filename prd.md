# MausamSetu — Product Requirements Document

> **Status:** Draft for ISRO BAH 2026
> **Owner:** Bhavy
> **Last updated:** 2026-07-25

---

## 1. Product Name
**MausamSetu** — मौसम सेतु ("Weather Bridge") — an AI-powered Digital Twin of India's Climate.

## 2. Problem Statement (verbatim from BAH 2026)
Build an AI-powered Digital Twin of India's climate that fuses IMD gridded data with INSAT satellite products (via MOSDAC) to forecast rainfall + temperature with quantified uncertainty and simulate what-if scenarios for climate adaptation.

## 3. Pilot Region
**Cauvery basin** — 10.0-14.5°N, 75.5-79.5°E. Master grid 0.25° × 0.25° = 19 lat × 17 lon = 323 pixels.

## 4. The 4 Digital Twin Properties (must-have)
| # | Property | How MausamSetu satisfies it |
|---|----------|-----------------------------|
| P1 | **Representation** | 4D grid of rain/tmax/tmin/INSAT-LST at 0.25° daily |
| P2 | **Synchronization** | Ensemble Kalman Filter blends observations with model state |
| P3 | **Predictivity** | ConvLSTM 7-day forecast with MC-Dropout uncertainty (p10-p50-p90) |
| P4 | **Counterfactuals** | Storyline / Pseudo-Global-Warming scenarios (IPCC AR6 SSPs) |

## 5. Target Users
- **Primary:** ISRO BAH 2026 judges, IMD/MoES analysts
- **Secondary:** State disaster management, reservoir operators (KRS, Mettur, Kabini), agriculture ministries
- **Tertiary:** Climate researchers, insurance/actuarial teams

## 6. Core Data Sources
| Source | Product | Resolution | Notes |
|--------|---------|-----------|-------|
| IMD | Gridded rainfall | 0.25° daily | Fortran binary `.grd` |
| IMD | Gridded temperature (max/min) | 1.0° daily | `.GRD` |
| INSAT-3D/3DR (MOSDAC) | 3RIMG_L2B_LST | ~4-10 km | HDF5, scale_factor + add_offset |
| INSAT-3D/3DR (MOSDAC) | 3RIMG_L2B_SST | ~4-10 km | HDF5 |
| INSAT-3D/3DR (MOSDAC) | 3RIMG_L2B_IMC (rain estimate) | ~4-10 km | HDF5 |
| IMDAA (NCMRWF) | Reanalysis | 12 km | Optional; for validation |

## 7. Core User Journeys
1. **Situational awareness** — user opens Overview, sees basin state + 7-day forecast at a glance
2. **Deep dive on a cell** — user clicks any pixel on the map, sees history + forecast + anomaly for that cell
3. **What-if simulation** — user picks IPCC scenario or drags sliders, sees baseline vs scenario vs delta maps + impact rupees
4. **Trust the model** — user opens Validation, sees POD/CSI/CRPS vs persistence baseline
5. **Export** — user downloads PDF report or CSV data slice

## 8. Non-Functional Requirements
- **Latency:** map interaction < 200 ms, forecast fetch < 1.5 s, scenario recompute < 3 s
- **Accuracy:** POD ≥ 0.65, CSI ≥ 0.45 (already POD=0.71, CSI=0.52 on 2023 test)
- **Reproducibility:** train/val/test year splits fixed, no leakage; standardization stats fit on train years only
- **Availability:** works offline on 8 GB VRAM laptop for demo
- **Aesthetic:** matches high-end climate dashboards (Command Center, Stanfield, ESA Destination Earth references)

## 9. Success Metrics
- Placement in BAH 2026 shortlist
- Demonstrable improvement over persistence baseline on 2023 test (already 43% RMSE reduction via EnKF)
- Judges can operate all 4 twin properties in a live demo without assistance

## 10. Out of Scope (for hackathon)
- Real-time streaming from INSAT (batch daily is enough)
- Nation-wide coverage (Cauvery only for pilot; architecture supports expansion)
- Mobile-first UI (desktop-only is fine)
- Multi-user auth / RBAC

## 11. Open Questions
- MOSDAC INSAT account approval pending — synthetic fallback in place
- Final list of screens / dashboard scope — pending user reference images
