# MausamSetu — Full Context Handoff Prompt

> Paste the entire "Prompt to Paste" section below into a new chat.
> Everything below the prompt block is reference for you (the human) — the AI reads it via the docs it points to.
> **Last updated:** 2026-07-25

---

## 📋 PROMPT TO PASTE (copy from here to the end)

---

You are joining a project mid-flight. Read every word before responding. Do not skip ahead. Do not write code until I explicitly tell you to.

### 0. Who I am and what we're doing

I am Bhavy. I am competing in **ISRO BAH 2026** (Bharatiya Antariksh Hackathon) against IIT-level teams. My project is:

**MausamSetu** (मौसम सेतु — "Weather Bridge")
An AI-powered Digital Twin of India's Climate that fuses IMD gridded data with INSAT satellite products (via MOSDAC) to forecast rainfall + temperature with quantified uncertainty and to simulate what-if scenarios for climate adaptation.

Pilot region: **Cauvery basin** (10.0-14.5°N, 75.5-79.5°E). Master grid: 0.25° × 0.25° = 19 lat × 17 lon = 323 pixels.

The system must demonstrate the 4 properties of a digital twin:
- **P1 Representation** — 4D grid of rain/tmax/tmin/INSAT-LST at 0.25° daily
- **P2 Synchronization** — Ensemble Kalman Filter blends observations with model state
- **P3 Predictivity** — ConvLSTM 7-day forecast with MC-Dropout uncertainty (p10-p50-p90)
- **P4 Counterfactuals** — Storyline / Pseudo-Global-Warming scenarios (IPCC AR6 SSPs)

### 1. Where the code lives

- **Repo root:** `L:\MausamSetu\`
- **External raw data:** `C:\Users\bhavy\Documents\v1\DATA`
- **Reference images (dashboard mockups):** `C:\Users\bhavy\Documents\v1\images\`
- **Platform:** Windows 10, Git Bash shell, Python 3.12 in `venv/`
- **Hardware:** 2 laptops with 6 GB and 8 GB VRAM respectively
- **Timeline:** roughly 1 week to BAH 2026 submission

### 2. Documents you MUST read before doing anything

Read these six files at `L:\MausamSetu\` in order:

1. `prd.md` — what we're building and why
2. `architechure.md` — repo layout, API contract, model pipeline
3. `rules.md` — non-negotiables (no data leakage, fixed year splits, uncertainty mandatory, anti-slop checklist)
4. `phases.md` — the phased build plan; we are currently in **Phase 1** (Screen List Lockdown)
5. `design.md` — visual design system; mostly placeholder tokens waiting for reference-image confirmation
6. `memory.md` — running handoff notes with state, watch-outs, and open questions
7. `context.md` — this file (this prompt is a mirror of it)

After reading, summarize back to me in 5 bullets what you understood so I know you're actually oriented, THEN wait for my next instruction.

### 3. State of the backend (already built, all working)

- `mausamsetu/config.py` — central config with Cauvery bounds, IMD grid specs, `EXTERNAL_DATA_DIR`, year splits `TRAIN_YEARS=[2020, 2021]`, `VAL_YEARS=[2022]`, `TEST_YEARS=[2023]`
- `mausamsetu/data/imd.py` — IMD `.grd` reader (float32, Fortran-major, masks `-999` and `99.9`, Western Ghats sanity test)
- `mausamsetu/data/insat.py` — HDF5 reader with `scale_factor + add_offset` formula, regrids INSAT products to master grid
- `mausamsetu/data/synthetic.py` — Cauvery-shaped synthetic fallback when real data missing
- `mausamsetu/preprocess/build_dataset.py` — auto-detects real vs synthetic, crops, regrids, computes anomalies, standardizes (train-years-only, no leakage) → `data/processed/cauvery.nc`
- `mausamsetu/preprocess/dataset.py` — PyTorch `Dataset` (6-day input → 7-day target sliding window, 5 input channels, 3 output channels)
- `mausamsetu/model/convlstm.py`, `forecaster.py`, `loss.py`, `train.py`, `predict.py` — ConvLSTM encoder-decoder, hurdle rain head (occurrence + amount), physics-informed loss (MSE + BCE + physics penalty + smoothness), MC-Dropout inference (20 members)
- `mausamsetu/assimilate/enkf.py` — 20-member Ensemble Kalman Filter, verified **43% RMSE reduction**
- `mausamsetu/storyline/scenario.py` — `Scenario` dataclass + `IPCC_SCENARIOS` dict (SSP1-2.6 / 2-4.5 / 5-8.5) + `apply_scenario()`
- `mausamsetu/impacts/hydrology.py` — SCS-CN; verified non-linear: 20% rain drop → 42% inflow drop
- `mausamsetu/impacts/heat.py` — heatwave days, cooling degree days
- `mausamsetu/impacts/rupee_risk.py` — verified ₹29,368 cr basin loss under worst-case scenario
- `mausamsetu/metrics/metrics.py` — RMSE, MAE, POD, FAR, CSI, HSS, CRPS, ACC, FSS
- `mausamsetu/metrics/baselines.py` — persistence + climatology baselines
- `scripts/run_demo.py` — end-to-end demo runner (verifies all 4 twin properties)
- `tests/test_end_to_end.py` — **10/10 passing**

**Trained artifacts:**
- `data/processed/cauvery.nc` — 8.55 MB, 1461 days (2020-01-01 to 2023-12-31), variables: `rain, tmax, tmin, insat_lst, insat_rain, rain_anom, rain_clim, tmax_anom, tmax_clim, tmin_anom, tmin_clim`
- `checkpoints/forecaster_best.pt` — 6.7 MB ConvLSTM, **1.67M parameters**

**Verified skill on 2023 test:**
- POD = 0.71
- CSI = 0.52
- EnKF post-correction: 43% RMSE reduction vs raw model
- Beats persistence and climatology baselines

### 4. State of the frontend (about to be rebuilt)

- Streamlit dashboard exists at `mausamsetu/dashboard/command_center.py` and `mausamsetu/dashboard/components.py`
- **I have rejected the Streamlit output as insufficient visual quality**
- We are switching to a professional stack:
  - **Frontend:** Next.js 14 + TypeScript + shadcn/ui + Tailwind CSS + Plotly.js + Mapbox GL JS + deck.gl + Framer Motion + Zustand + SWR
  - **Backend:** FastAPI wrapping the existing `mausamsetu/` Python core
- The Streamlit code will be **moved** (not deleted) to `mausamsetu/dashboard/_legacy_streamlit/`
- There is a stale plan file at `.zcode/plans/plan-sess_a64d0ad9-*.md` that describes an older FastAPI + plain HTML approach — **that plan is superseded**. Follow the Next.js + FastAPI plan captured in `architechure.md` and `phases.md`.

### 5. Where we are RIGHT NOW (this is critical)

I told the previous assistant: *"we are going with node.js but before proceeding i want you to give me all the screen that will be there so that i can give you reference for each and you don't generate any slop."*

The previous assistant delivered the enumerated screen inventory. Here is the definitive list:

**Core screens (10):**
1. `/` — **Overview (Command Center)** — hero landing, everything at a glance
2. `/map` — **Map View** — full-screen interactive Cauvery map, layers, time slider, cell inspector
3. `/forecast` — **Forecast Detail** — 7-day ensemble with p10/p50/p90, small-multiples for Day+1..+7, spaghetti plot, skill panel
4. `/scenarios` — **What-If Studio** — IPCC dropdown + sliders + baseline/scenario/delta maps + impact KPI cards
5. `/impacts` — **Sector Impact Chain** — tabs for 💧 Hydrology, 🔥 Heat & Health, 💰 ₹ Risk
6. `/validation` — **Model Validation Console** — POD/FAR/CSI/CRPS table vs baselines, skill-vs-lead-time
7. `/alerts` — **Alerts & Thresholds** — live event feed + threshold config
8. `/data` — **Data Explorer** — variable browser, provenance, CSV/JSON export
9. `/reports` — **Reports & Export** — PDF templates (Weekly Basin, Scenario Analysis, Impact Assessment)
10. `/settings` — **Settings** — units, region switcher, threshold defaults

**Modals / overlays:**
- Cell Inspector (any map click)
- Scenario Comparison (side-by-side)
- Export dialog
- Onboarding tour (first-visit)
- Command palette (Cmd+K)

**Per-screen states:** loading, empty, error. Desktop-only (mobile is out of scope for judges).

**I am about to upload reference images per screen.** I will drop them into `C:\Users\bhavy\Documents\v1\images\` and paste them in chat with mappings like:

> Screen 1 Overview → this image
> Screen 2 Map → this image
> Screen 3 Forecast → any (you pick)
> Screen 7 Alerts → cut, don't build
> ...

### 6. Absolute rules — do not break these

- **No slop.** Every screen must have a user-approved reference before build starts. Say "waiting for reference" if I haven't given one.
- **Do not delete `command_center.py` or `components.py`.** Move them to `mausamsetu/dashboard/_legacy_streamlit/`.
- **No data leakage.** Standardization stats fit only on `TRAIN_YEARS`.
- **Fixed year splits.** Never shuffle across years.
- **Every forecast ships p10/p50/p90.** No bare point forecasts in the UI.
- **Every model claim needs a baseline.** Ours vs Persistence vs Climatology, always.
- **Master grid is 0.25°, 19×17.** Any other resolution must be regridded.
- **IMD missing values are `-999` and `99.9`.** Mask before any math.
- **IST clock, not local browser time.**
- **Every number in the UI has a unit label.**
- **No rainbow / jet colormap.** Use sequential blues for rain, yellow-to-red for temp, diverging blue-white-red for anomalies.
- **No color-only signals.** Pair with icon or label.
- **Never push to main.** Feature branches only. Commits only on my explicit ask.
- **Never commit** raw satellite data, `.env`, or files > 20 MB.
- **PPTX slides 1, 2, and 10 stay unchanged.**
- **English + Devanagari branding** ("MausamSetu मौसम सेतु") — do not anglicize.

### 7. My preferences (learned)

- Explain deeply before coding. I am learning as we build.
- Match my energy: direct, no fluff, no filler acknowledgments.
- Correct me when I'm wrong. Don't agree just to agree.
- Show reasoning when making recommendations.
- Use references from `C:\Users\bhavy\Documents\v1\images\` — I care about visual quality.
- I have a MOSDAC INSAT account pending approval — synthetic fallback works for now.
- Longer training on Kaggle GPU (30-50 epochs) is a stretch goal.

### 8. Success criteria for the next stretch of work

**Phase 1 — Screen Lockdown (right now):**
- I upload references per screen
- You confirm each mapping in writing
- You update `design.md` with reference paths / notes per screen
- You do NOT write any Next.js code yet

**Phase 2 — Design System (after Phase 1 sign-off):**
- Install Node.js + pnpm
- Scaffold `web/` with Next.js 14 + TypeScript + Tailwind + shadcn/ui
- Extract color / typography / spacing tokens from references into `tailwind.config.ts`
- Build shared components (`MetricTile`, `GlassCard`, `TopNav`, etc.)

**Phase 3 — FastAPI backend wrap:**
- Add `mausamsetu/dashboard/api/main.py` + routes
- Move Streamlit to `_legacy_streamlit/`
- Implement all 9 endpoints listed in `architechure.md`
- Smoke test each endpoint against real `cauvery.nc` + `forecaster_best.pt`

**Phase 4 — Overview screen only:**
- Build `/` (Command Center) end-to-end matching the reference
- Real data via FastAPI
- Loading / empty / error states
- Show me a side-by-side with the reference screenshot BEFORE moving to any other screen

**Phase 5+ — Rest of screens, only after Overview passes:**
- Follow the priority order in `phases.md`
- Cut lowest-value screens if we run out of time (order defined in `phases.md`)

### 9. What to do the moment you finish reading this

Reply with EXACTLY this structure:

```
### Orientation confirmed

**Understood in 5 bullets:**
- <bullet 1>
- <bullet 2>
- <bullet 3>
- <bullet 4>
- <bullet 5>

**Waiting for:** the reference images per screen (you'll drop them or map them screen-by-screen).

**I will not write code until you say so.**

Anything I got wrong?
```

Then STOP. Wait for me to upload / map references or give the next instruction.

---

## END OF PROMPT TO PASTE

---

## 📎 What to attach when starting the new chat

Along with the prompt above, in the very first message include:

1. **All reference images** you have collected — drag them into the chat window
2. **Screen mapping** — a list like:
   ```
   1. Overview        → image_01.png
   2. Map             → image_02.png
   3. Forecast        → image_03.png (or "any")
   4. Scenarios       → image_04.png
   5. Impacts         → cut
   6. Validation      → image_05.png
   7. Alerts          → any
   8. Data Explorer   → cut
   9. Reports         → cut
   10. Settings       → any
   ```
3. **Any URLs** — Dribbble, Behance, Awwwards, existing product URLs — with the screen each inspires
4. **Any hard visual constraints** — e.g. "must use this exact blue" or "no 3D surface, keep it 2D"

## 🗺️ Quick file map (for your reference)

```
L:\MausamSetu\
├── prd.md                 ← what & why
├── architechure.md        ← how (repo, API, pipeline)
├── rules.md               ← non-negotiables
├── phases.md              ← phased plan, current phase = 1
├── design.md              ← visual system, awaiting references
├── memory.md              ← handoff notes
├── context.md             ← THIS FILE
├── mausamsetu/            ← Python core (all working)
│   ├── config.py
│   ├── data/              ← imd.py, insat.py, synthetic.py
│   ├── preprocess/        ← build_dataset.py, dataset.py
│   ├── model/             ← convlstm.py, forecaster.py, loss.py, train.py, predict.py
│   ├── assimilate/enkf.py
│   ├── storyline/scenario.py
│   ├── impacts/           ← hydrology.py, heat.py, rupee_risk.py
│   ├── metrics/           ← metrics.py, baselines.py
│   └── dashboard/         ← Streamlit (to be deprecated)
│       ├── command_center.py       ← move to _legacy_streamlit/
│       └── components.py           ← move to _legacy_streamlit/
├── data/processed/cauvery.nc       ← 8.55 MB, 1461 days
├── checkpoints/forecaster_best.pt  ← 6.7 MB, 1.67M params
├── scripts/run_demo.py
└── tests/test_end_to_end.py        ← 10/10 green
```

## 🧠 If the new chat drifts

If the new assistant starts:
- Writing code before you've provided references → reply "Stop. Re-read `rules.md` — anti-slop checklist. Wait for references."
- Rebuilding the Streamlit dashboard → reply "Wrong stack. Read `architechure.md`. Next.js + FastAPI, not Streamlit."
- Suggesting a different pilot region → reply "Cauvery basin only. See `prd.md` §3."
- Ignoring the year splits → reply "Read `rules.md` §1. Fixed splits, no shuffling."
- Losing the Devanagari name → reply "Branding is 'MausamSetu मौसम सेतु'. See `rules.md`."

## ✅ Sanity checks before you paste

- [ ] All six markdown docs exist at `L:\MausamSetu\` (`prd.md`, `architechure.md`, `rules.md`, `phases.md`, `design.md`, `memory.md`) plus this `context.md`
- [ ] `data/processed/cauvery.nc` still there
- [ ] `checkpoints/forecaster_best.pt` still there
- [ ] Reference images are ready in `C:\Users\bhavy\Documents\v1\images\`
- [ ] You know which screens to cut vs keep (or you're happy for me to pick "any")

You're set. Open the new chat, paste the prompt block, attach the images.

---

## Phase 1 reference lockdown — 2026-07-25

### Confirmed decisions

- Screens 01–07 are retained and their filename mappings are approved:
  - `screen-01-overview.png` → `/` Overview
  - `screen-02-map.png` → `/map` Map View
  - `screen-03-forecast.png` → `/forecast` Forecast Detail
  - `screen-04-scenarios.png` → `/scenarios` What-If Studio
  - `screen-05-impacts.png` → `/impacts` Sector Impact Chain
  - `screen-06-validation.png` → `/validation` Model Validation Console
  - `screen-07-alerts.png` → `/alerts` Alerts & Thresholds
- The approved images are canonical where they conflict with the former dark-theme placeholder in `design.md`. The canonical direction is light, translucent, and operational.
- Screens 08–10 remain in scope. Because they have no dedicated screenshots, their Phase 1 specifications are synthesized strictly from the visual system established by Screens 01–07 and must not be described as pixel-perfect decodes.
- Screenshot labels and values are preserved during visual matching. They are placeholders and will be replaced with real API/model values after the visual build is approved.
- No Next.js, Tailwind, FastAPI, or Streamlit migration work was performed in Phase 1.

### Phase 1 documentation index

- Direct decodes: `design/references/01-overview.md` through `design/references/07-alerts.md`.
- Synthesized specifications: `design/references/08-data-explorer.md`, `09-reports.md`, and `10-settings.md`.
- Consolidated design system and provenance: `design.md`.

## Provisional design decisions — future review

Everything in this section is intentionally isolated for later confirmation. These decisions unblock design documentation but do not override backend/scientific truth.

1. **Screenshot values versus real values.** Phase 1 preserves screenshot numbers and copy for visual reproduction. Implementation fixtures must mark them as mock data; production UI must later use real API/model outputs.
2. **Ensemble count conflict.** Forecast and Scenarios show `32` members; the current backend uses `20` MC-Dropout members. Use `32` only while reproducing the approved static reference, then replace it with the real configured count.
3. **Validation data conflict.** The Validation image shows a `1991–2021` training window, `2022–2024` holdout, and incompatible sample counts. The locked project split remains train 2020–2021, validation 2022, and test 2023. Replace screenshot metadata when real data integration begins.
4. **Scenario naming conflict.** The Scenarios image includes `RCP 4.5`; the implemented system uses IPCC AR6 SSP1-2.6, SSP2-4.5, and SSP5-8.5. Preserve the screenshot label only for initial visual matching, then use the implemented SSP vocabulary.
5. **Screens 08–10 are synthesized.** Data Explorer, Reports & Export, and Settings extend the standard shell and existing tokens/components. Their layouts, copy, states, and interactions require user review before implementation is considered visually locked.
6. **Sector screen shell.** Screen 05 directly defines the Water security presentation. The shared header, sector selector, advisories, decision log, and pipeline are treated as the shell for the other impact views. Heat & Health and ₹ Risk content remains undesigned by a dedicated reference and must not be claimed as pixel-decoded.
7. **Alerts threshold placement.** Threshold defaults belong in Settings. Alerts may expose a compact, role-gated contextual shortcut beside the map-layer controls that opens a popover or side sheet. This is provisional and must remain subordinate to active incident response.
8. **Overview pipeline count.** Use the six visible stages from the approved Overview image: `INGEST`, `REGRID`, `ASSIMILATE`, `FORECAST`, `IMPACT`, `RENDER`. The earlier five-stage text is superseded for this screen.
9. **Static-image state gaps.** Hover, focus, keyboard, loading, empty, error, stale-data, unavailable-source, partial-data, permissions, and reduced-motion treatments are inferred from accessibility and operational requirements because the screenshots do not show them. Review these in the Phase 2 kitchen-sink page.
10. **Font confidence.** The screenshots indicate an Inter-like neo-grotesk but cannot prove the family. Start Phase 2 with Inter and verify its glyph shapes against the references before locking typography.
11. **Translucency confidence.** Alpha values cannot be recovered exactly from flattened PNGs. The opacity and blur values in `design.md` are starting points for screenshot comparison, not measured facts.
12. **Map data fidelity.** Reference maps visually imply operational data layers. Production implementation must use the real 0.25° master grid and synchronized dates; no smoothed decorative substitute is acceptable.

### Next gate

Phase 2 may begin only after explicit approval. It may encode the extracted tokens, scaffold the frontend, and build a component kitchen-sink, but it must not treat synthesized Screens 08–10 or inferred states as reference-approved without review. Overview remains the first end-to-end production screen and must pass side-by-side comparison before other screens proceed.
