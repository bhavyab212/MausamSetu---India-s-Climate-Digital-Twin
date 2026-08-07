# What-If Engine · Decision Log

One entry per material choice. Format: **Date · Choice · Rationale ·
Alternatives · Reversal cost**.

---

**2026-08-06 · Streamlit-only frontend, SvelteKit paused.**
- Rationale: Streamlit's Python-native reruns and st.cache_data give us
  reproducibility guarantees and provenance-tagged results with zero
  glue code. SvelteKit needs a separate API layer, doubles the surface,
  slows iteration.
- Alternatives: Sveltekit MSatu/ folder was in flight; Dash; a static
  export.
- Reversal cost: low. All logic is behind whatif/*; a future FE would
  reuse the same package.

**2026-08-06 · Five-layer impact chain (L0 → L4).**
- Rationale: honest audit trail. A monolithic climate-to-rupees model
  is not defensible in a district officer conversation. Layers give
  everyone their intervention point.
- Alternatives: end-to-end neural model.
- Reversal cost: high; every module assumes this shape.

**2026-08-06 · Two-axis payoff matrix as the primary artifact.**
- Rationale: matches how decisions actually get made — "if the
  climate does X, and I do Y, what is the payoff?" Savage 1951 minimax
  regret is the honest answer.
- Alternatives: single-number "best decision" recommendation.
- Reversal cost: low; the recommendation card already collapses the
  matrix into a single line.

**2026-08-07 · Hargreaves ET0 default, Penman-Monteith deferred.**
- Rationale: FAO-56 §3 recommends Hargreaves-Samani when full PM
  inputs (humidity, wind, radiation) are unavailable — which is our
  case. ERA5 ingestion is a data-pipeline task, not a What-If task.
- Alternatives: Penman-Monteith with reanalysis fills.
- Reversal cost: low; et0_penman.py is stubbed with a clean
  NotImplementedError.

**2026-08-07 · SPI TRAIN_YEARS = 1971-2010; Long Term baseline = 1971-2000.**
- Rationale: these serve different purposes. TRAIN_YEARS is the SPI
  fit window per WMO-1090; the Long-Term baseline is the IPCC AR6
  convention for delta-change downscaling. Coalescing them would be
  wrong.
- Alternatives: single baseline for everything.
- Reversal cost: low; both constants live in one place each and are
  cited.

**2026-08-07 · Three-pass q10/q50/q90 in Short Term; multi-model in Long Term.**
- Rationale: the underlying uncertainty representations differ. Short
  Term uncertainty is a probabilistic forecast at a fixed skill level;
  Long Term uncertainty is model + scenario disagreement. Mixing them
  is a category error — RepresentationMismatch enforces.
- Alternatives: force one representation everywhere.
- Reversal cost: medium; a lot of code branches on which layer this is.

**2026-08-07 · Historical analog method preferred over delta perturbation.**
- Rationale: analogs are, by construction, physically consistent (they
  happened). Delta perturbation modifies mean state without preserving
  inter-variable consistency (Räisänen & Räty 2013).
- Alternatives: perturbation as the default.
- Reversal cost: low; both flavours ship, only the default and the
  caveat gate change.

**2026-08-07 · Fixed 10-GCM NEX-GDDP ensemble.**
- Rationale: Almazroui 2020 South Asia CORDEX evaluation ranks
  models; a fixed set makes results comparable across scenarios and
  reproducible. Cherry-picking a single GCM is the second-most common
  way this class of tool fails a review.
- Alternatives: full 34-model set (compute cost); user-selectable
  ensemble (small-ensemble caveat covers the corner case).
- Reversal cost: low; catalog is a YAML.

**2026-08-07 · QDM default for extremes; delta_mean caveated on extremes.**
- Rationale: Cannon 2018 shows QDM preserves tail changes; delta_mean
  distorts them. Caveat gate matches the Method-1 perturbation
  pattern.
- Alternatives: delta_mean as default.
- Reversal cost: low.

**2026-08-07 · District as reporting resolution ceiling.**
- Rationale: matches how APY is reported and how state/district
  bureaucracies act. Emitting village-level rupees would over-claim
  precision the model does not have (Rule 5).
- Alternatives: emit at grid resolution.
- Reversal cost: low; a `to_district` bypass would require
  ResolutionCeilingError to be silenced.

**2026-08-07 · Central discount rate 8 %, sensitivity 7–12 %.**
- Rationale: NITI Aayog "Manual for Economic Appraisal of Public
  Investment Projects" pins 8 % central; Rule 7 requires reporting
  the sensitivity band alongside.
- Alternatives: single-rate NPV.
- Reversal cost: low; the discount-rate tuple is a keyword arg on
  `adaptation_npv`.

**2026-08-07 · Byte-identity replay as the reproducibility test.**
- Rationale: cache_key() over the lever set + code SHA gives us a
  guarantee stronger than "close enough." Any drift surfaces
  immediately.
- Alternatives: approximate-equality tests.
- Reversal cost: medium; tests would need to be relaxed if adopted.

**2026-08-07 · predict / forecast / will / is going to banned in Long Term copy.**
- Rationale: Long Term is a scenario tab, not a forecast tab. Verb
  lint enforces via word-boundary regex on every export in
  whatif/ui/copy/long_term.py.
- Alternatives: none — this is a positioning constraint from Bhavya's
  brief.
- Reversal cost: low; the lint is a single test.
