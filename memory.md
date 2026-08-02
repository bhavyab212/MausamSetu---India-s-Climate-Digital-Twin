# MausamSetu — Working Memory / Handoff Notes

> Persistent context for the next chat session. Read this first.
> **Last updated:** 2026-07-25

---

## Who / What
- **Project:** MausamSetu (मौसम सेतु) — AI Digital Twin of India's Climate
- **Competition:** ISRO BAH 2026 (up against IIT-level teams)
- **Owner:** Bhavy
- **Hardware:** 2 laptops (6 GB + 8 GB VRAM)
- **Timeline:** ~1 week to submission
- **Repo root:** `L:\MausamSetu\`

## State of the World (as of 2026-07-25)

### Backend ✅ working
- ConvLSTM trained on real IMD 2020-2023 data → `checkpoints/forecaster_best.pt` (1.67M params, 6.7 MB)
- Dataset built → `data/processed/cauvery.nc` (8.55 MB, 1461 days)
- End-to-end tests 10/10 green
- EnKF verified 43% RMSE reduction
- SCS-CN hydrology: 20% rain drop → 42% inflow drop (non-linear)
- Worst-case scenario: ₹29,368 cr basin loss
- POD=0.71, CSI=0.52 on 2023 test

### Dashboard 🟡 pivoting
- Streamlit `command_center.py` and `components.py` exist but are being **deprecated**
- User rejected Streamlit output quality
- New target stack: **Next.js 14 + TypeScript + shadcn/ui + Tailwind + Plotly.js + Mapbox GL + deck.gl + Framer Motion + Zustand + SWR** on the frontend, **FastAPI** on the backend
- Note: plan file at `.zcode/plans/plan-sess_a64d0ad9-*.md` still references the older FastAPI + plain HTML approach — that plan is **superseded** by the Next.js decision

### Data pending
- MOSDAC INSAT account approval pending — synthetic INSAT fallback in place
- Longer training on Kaggle GPU (30-50 epochs) still to run

## Where We Left Off
Assistant posted the enumerated screen inventory (10 core screens + modals). User's next task: drop reference images per screen into `C:\Users\bhavy\Documents\v1\images\` and reply with which screens to keep/cut and which reference maps to which screen.

**Do not write any Next.js code until this is done.** Explicit user rule: "don't generate any slop."

## User Preferences (learned)
- Prefers deep explanations before code
- Wants formulas + diagrams in slides
- Rejected Streamlit for aesthetic reasons
- References high-end dashboards from `C:\Users\bhavy\Documents\v1\images\`
- Willing to learn; not shy about saying when they don't understand
- Prefers English + Devanagari branding ("MausamSetu मौसम सेतु")

## Locations
- **Repo:** `L:\MausamSetu\`
- **External raw data:** `C:\Users\bhavy\Documents\v1\DATA`
- **Reference images:** `C:\Users\bhavy\Documents\v1\images\`
- **PPTX:** slides 1, 2, 10 must remain unchanged

## Key Files to Read First (next chat)
1. `prd.md` — what we're building and why
2. `architechure.md` — how it's structured
3. `phases.md` — where we are in the plan (currently Phase 1)
4. `rules.md` — non-negotiables
5. `design.md` — visual system (waiting for references)
6. This file (`memory.md`) — handoff context

## Immediate Next Actions
1. **User:** drop reference images per screen into `C:\Users\bhavy\Documents\v1\images\` and confirm the screen list
2. **Assistant:** once references confirmed, install Node.js + pnpm, scaffold `web/` folder, start Phase 2 (design system)
3. **Assistant:** build Overview screen first, get user sign-off before scaling

## Watch-Outs
- Do not delete `command_center.py` — it goes into `_legacy_streamlit/` backup, not the trash
- Standardization stats are computed on train years only; never on val/test — protects against leakage
- IMD grids: `-999` and `99.9` are missing-value sentinels, mask before math
- Master grid is 0.25°, 19×17; anything else must be regridded
- No color-only UI signals; pair with icon/label for accessibility
- IST clock, not local time
- All numbers need units

## Open Questions for Next Chat
- Which of the 10 screens does the user want to keep?
- Reference images per screen?
- Mobile support? (assumed desktop-only for judges)
- Real MOSDAC INSAT ETA?

## Chat-Session Discipline
When resuming: read this file, then `phases.md` to locate current phase, then proceed. Do not re-derive decisions already captured here — challenge them only if the user asks or new info contradicts them.
