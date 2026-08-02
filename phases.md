# MausamSetu — Build Phases

> One-week hackathon build. Each phase has an exit criterion.
> **Last updated:** 2026-07-25

---

## Phase 0 — Foundation ✅ DONE
**Exit:** end-to-end pipeline runs, 10/10 tests pass, 43% EnKF RMSE reduction verified.

- [x] Config with Cauvery bounds + year splits
- [x] IMD `.grd` reader (rainfall 0.25°, temp 1.0°)
- [x] INSAT HDF5 reader with scale_factor + add_offset
- [x] Synthetic fallback data generator
- [x] `build_dataset.py` → `cauvery.nc` (1461 days, 2020-2023)
- [x] PyTorch Dataset (6-day input → 7-day target)
- [x] ConvLSTM forecaster (1.67M params)
- [x] Physics-informed loss + hurdle rain head
- [x] MC-Dropout prediction (20 members)
- [x] Ensemble Kalman Filter (20 members, 43% RMSE gain)
- [x] Storyline scenario apply + IPCC SSP presets
- [x] Impact modules: SCS-CN hydrology, heat days, ₹ risk
- [x] Metrics: RMSE, MAE, POD, FAR, CSI, HSS, CRPS, ACC, FSS
- [x] Baselines: persistence + climatology
- [x] End-to-end test suite (10/10 green)

## Phase 1 — Screen List Lockdown 🟡 IN PROGRESS
**Exit:** every screen has a user-provided reference image (or explicit "any" approval).

- [x] Post enumerated screen inventory to user (10 core + modals)
- [ ] User picks which screens to keep / cut
- [ ] User drops reference images per screen into `C:\Users\bhavy\Documents\v1\images\`
- [ ] Assistant confirms references received per screen
- [ ] Sign-off before any code

## Phase 2 — Design System
**Exit:** Storybook-style page in Next.js showing every reusable component matching references.

- [ ] Install Node.js + pnpm
- [ ] Scaffold `web/` with Next.js 14 + TypeScript + Tailwind
- [ ] Add shadcn/ui base components
- [ ] Extract color palette + typography + spacing from references → `tailwind.config.ts`
- [ ] Build shared components: MetricTile, MiniGauge, Sparkline, GlassCard, TopNav, StatusLight
- [ ] Design tokens documented in `design.md`

## Phase 3 — Backend API
**Exit:** all 9 endpoints in `architechure.md` return correct JSON for a picked date.

- [ ] Set up FastAPI app under `mausamsetu/dashboard/api/`
- [ ] Move Streamlit code to `_legacy_streamlit/` backup
- [ ] Implement `/dates`, `/grid`, `/timeseries`, `/forecast`, `/assimilation`, `/scenario`, `/metrics`, `/alerts`, `/health`
- [ ] Pydantic response schemas
- [ ] CORS for `http://localhost:3000`
- [ ] Gzip middleware
- [ ] Smoke test each endpoint with real `cauvery.nc` + `forecaster_best.pt`

## Phase 4 — Overview Screen (the make-or-break page)
**Exit:** side-by-side comparison with user's reference is a match.

- [ ] Top nav + IST clock
- [ ] Hero map (Plotly heatmap OR deck.gl) with basin outline
- [ ] Left rail: AI Assimilation chart + 6 metric tiles
- [ ] Right rail: 3D rainfall surface + Data Sources + 4 metric tiles
- [ ] Bottom: wind/rain/temp/humidity + risk gauge + assimilation donut + data-quality lights
- [ ] Footer: 5-stage pipeline strip + flood risk
- [ ] Real data via FastAPI
- [ ] Loading / empty / error states
- [ ] Reference comparison screenshot

## Phase 5 — Remaining Core Screens
**Exit:** each screen matches its reference and pulls real data.

- [ ] `/map` — full-viewport Cauvery map with layers + time slider + cell inspector
- [ ] `/forecast` — 7-day ensemble with uncertainty bands + small-multiple maps
- [ ] `/scenarios` — What-If Studio with baseline / scenario / delta maps + impact KPIs
- [ ] `/impacts` — Hydrology / Heat / ₹ tabs
- [ ] `/validation` — skill table + skill-vs-lead-time + scatter

## Phase 6 — Secondary Screens (if time)
**Exit:** at least Alerts + Data Explorer live.

- [ ] `/alerts` — live event feed + threshold config
- [ ] `/data` — variable browser + CSV/JSON download
- [ ] `/reports` — PDF export
- [ ] `/settings` — units + region switcher

## Phase 7 — Polish & Demo Prep
**Exit:** 5-minute demo script rehearsed twice, all screens screenshot-ready.

- [ ] Onboarding tour (first-visit overlay)
- [ ] Command palette (Cmd+K)
- [ ] Loading states across all screens
- [ ] Fix all accessibility issues (keyboard nav, contrast)
- [ ] README with quickstart
- [ ] Recording of full demo
- [ ] Longer training run on Kaggle GPU (30-50 epochs) for better skill numbers
- [ ] Real MOSDAC INSAT data ingest once account approved

## Phase 8 — Submission
**Exit:** PPTX + repo + demo video submitted to BAH 2026.

- [ ] Update PPTX with final screenshots
- [ ] Repo cleaned of dead files
- [ ] Model card / data card written
- [ ] Video walkthrough recorded
- [ ] Submission uploaded

---

## Cut-Line Rules
If we run out of time, we cut in this order (least valuable first):
1. `/settings`
2. `/reports`
3. `/data`
4. `/alerts`
5. Onboarding tour
6. Command palette
7. `/validation`
8. `/impacts` (drop Heat and ₹, keep Hydrology only)

**Never cut:** Overview, Map, Forecast, Scenarios. Those four are the twin's 4 properties in visual form.
