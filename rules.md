# MausamSetu — Engineering Rules

> Read this before writing any code for this project.
> **Last updated:** 2026-07-25

---

## 1. Non-Negotiables

- **No data leakage.** Standardization stats (mean/std) fit ONLY on `TRAIN_YEARS`. Never on val or test.
- **Year splits are fixed:** `TRAIN_YEARS=[2020, 2021]`, `VAL_YEARS=[2022]`, `TEST_YEARS=[2023]`. Do not shuffle across years.
- **Every model claim needs a baseline.** Report Ours vs Persistence vs Climatology. If we can't beat persistence, we say so.
- **Uncertainty is mandatory.** Every forecast output ships with p10/p50/p90. No bare point forecasts in the UI.
- **Physics sanity.** Rain ≥ 0. Tmax ≥ Tmin. Enforce in loss AND in post-processing.
- **Fixed grid.** Master grid = 0.25°, 19 lat × 17 lon over Cauvery. Anything else is regridded, never assumed.

## 2. Code Style

### Python
- Python 3.12, type hints on all public functions
- `black` (88 col) + `ruff` (default rules) — matching existing files
- Docstrings: numpy style, short one-liners for private helpers
- No `print()` in production paths — use the `logging` module
- No wildcard imports
- Absolute imports rooted at `mausamsetu.` (never relative `..`)
- Config lives in `mausamsetu/config.py` only — no scattered constants

### TypeScript / Next.js
- TypeScript strict mode on
- ESLint (`next/core-web-vitals`) + Prettier default
- Components in PascalCase files, hooks in `use*` camelCase
- Server components by default; add `"use client"` only when we need interactivity
- Zustand stores in `web/stores/<domain>.ts`, one store per concern
- API client is centralized in `web/lib/api.ts` — no `fetch()` scattered around
- Tailwind classes ordered by prettier-plugin-tailwindcss

## 3. Data Handling

- Raw data (IMD `.grd`, INSAT `.h5`) lives OUTSIDE the repo, at `C:\Users\bhavy\Documents\v1\DATA` (dev) or wherever `EXTERNAL_DATA_DIR` points
- Processed data (`cauvery.nc`) lives in-repo under `data/processed/`
- Never commit raw satellite data or intermediates larger than 20 MB
- Missing values in IMD grids: `-999` and `99.9` → mask to NaN before any math
- Coordinates: latitude increases north, longitude increases east — always double-check when regridding

## 4. Model Rules

- Checkpoints go in `checkpoints/` with a name including the run date and metric (e.g. `forecaster_best.pt` for the current champion)
- Save training config alongside every checkpoint (JSON next to the `.pt`)
- MC-Dropout: 20 members minimum for reported uncertainty
- EnKF: 20 members, obs error variance calibrated per variable
- Never change the model without re-running validation on 2023 and updating the numbers in the UI

## 5. UI Rules (Next.js)

- **No slop.** Every screen must have a user-approved reference before build starts.
- Design tokens (colors, spacing, radii) come from Tailwind config — do not inline hex values
- Every fetched panel has explicit loading / empty / error states — no bare spinners in production screens
- All maps and charts must handle NaN cells gracefully (transparent or hatched — never black holes)
- Every number shown to the user has a unit label
- IST clock is authoritative — no local browser time drift
- Dark theme is default; light theme is a stretch goal, not required for the demo

## 6. Git / Workflow

- Never push directly to `main`. Feature branches: `feat/<screen-or-module>`
- Commits only on explicit user ask
- No commits containing raw satellite data or `.env` files
- `.gitignore` covers: `checkpoints/*.pt`, `data/processed/*.nc`, `node_modules/`, `.next/`, `venv/`, `__pycache__/`

## 7. Testing

- `tests/test_end_to_end.py` must stay green (10/10)
- Any new module in `mausamsetu/` gets at least a smoke test
- Frontend: Playwright smoke test for the Overview screen once it exists
- Skill numbers on `/validation` come from the actual `metrics.py` output, never hardcoded

## 8. Documentation

- Every module has a top-of-file docstring saying what it does and what data shape it expects
- `README.md` at project root shows the one-liner launch commands
- Changes to API contract → update `architechure.md` in the same PR/commit
- Changes to screens list → update `phases.md` and `design.md`

## 9. Anti-Slop Checklist (frontend)

Before saying a screen is done:
- [ ] Matches the user-provided reference image
- [ ] Renders correctly with real data from FastAPI (not mocks)
- [ ] Loading / empty / error states designed, not just spinner-and-hope
- [ ] Interactive elements have hover / focus / active styling
- [ ] Numbers have units, dates are IST, NaN is handled
- [ ] Keyboard nav works for primary actions
- [ ] Screenshot compared side-by-side with the reference

## 10. Demo Rules (BAH 2026)

- Everything runnable offline on a single laptop
- Fresh venv install works from `requirements.txt` in one command
- `pnpm install && pnpm dev` on the frontend
- Backing data (`cauvery.nc`, `forecaster_best.pt`) is either in-repo or downloadable via a script
- All 4 twin properties demonstrable in under 5 minutes
