---
route: /validation
source_path: C:\Users\bhavy\Documents\v1\images\screen-06-validation.png
canvas: 1659 × 948 px
status: Direct reference decode
confidence_caveat: High confidence for visible hierarchy, copy, relative geometry, chart forms, and selected state; medium for sampled colors, typography, spacing, radii, and shadows because the source is raster; low for interactions or responsive behavior not shown.
placeholder_data_warning: All model names, versions, dates, sample counts, scores, errors, thresholds, statuses, and audit statements below are screenshot values preserved verbatim as placeholders, not validated model evidence.
---

## 1. Layout

**Observed [high confidence].** Desktop application shell at 1659 × 948. A persistent left rail occupies approximately x=14–239 (225 px); the main white workspace spans roughly x=254–1648 (1,394 px). Both sit over a blurred pale green landscape background with 12–16 px corner radii.

- Sidebar: brand, search, `Workspace` navigation, `Pilots`, signed-in user, Hindi product mark. `Validation` is selected with pale-blue fill, a blue left rule, and blue icon.
- Main header: title/subtitle on the left; `Export`, `Share`, `Notifications`, avatar, and green `Passing` status on the right.
- Model summary strip: four horizontal fields across about y=91–176, separated by subtle vertical rules: model, training window, holdout, and retraining date.
- Primary row, about y=181–476: `Observed vs predicted` is approximately 672 × 294 px; `Detection skill` is approximately 660 × 294 px.
- Secondary row, about y=484–786: three panels in an approximately 37/29/34 split: two spatial-error maps, baseline comparison, and drought-year stress test.
- Disclosure strip, about y=798–892: three text blocks for known limitation, training data window, and what is not modeled.
- Bottom cycle stepper, about 44 px high: `VALIDATE` is current; preceding stages are complete.

**Inferred implementation [medium confidence].** The content follows a 12-column grid with 12–16 px gutters and 12–20 px card padding. Preserve this evidence-led density and horizontal comparability. The screenshot provides no mobile or collapsed-sidebar specification.

## 2. Color system

Raster antialiasing creates multiple near-white values; samples are approximate unless explicitly described as visible text values.

- **Near-exact screenshot samples [medium]:** white/near-white surfaces `#FFFFFF` / `#FEFEFE`; primary blue around `#0877F9`; success green around `#009E67`; warning orange around `#FF8500`; deep navy text around `#101522`; muted gray around `#6B7280`; borders/grid around `#E5E7EB`.
- **Chart semantics [high]:** blue = POD/model/primary measured series; orange = FAR/warning/positive spatial error; green = CSI/pass; gray = baseline/observed or neutral state. The diverging mean-bias scale runs blue → white → orange; RMSE uses white/yellow → orange/red.
- Selected sidebar and current-cycle states use pale-blue fill with saturated blue edge. `Passing`, checks, and positive comparisons use pale green/green.
- **Inferred tokens [medium]:** `--surface`, `--surface-muted`, `--border`, `--text-primary`, `--text-secondary`, `--primary`, `--success`, `--warning`, `--danger`, plus sequential and diverging visualization ramps.
- Never treat screenshot whites or raster-blended hues as authoritative CSS values. Preserve semantic contrast and test text/chart contrast independently.

## 3. Typography

**Observed [medium confidence].** Neutral sans-serif, likely Inter or a close UI grotesk; exact family is not provable from the raster.

- Brand: ~20–22 px, 700. Page title `Validation`: ~21–23 px, 700.
- Header subtitle and summary labels: ~12–13 px, 400–500.
- Summary values (`ConvLSTM v2.3`, `1991 - 2021`, etc.): ~20–22 px, 500–600.
- Panel titles: ~14–16 px, 600–700; panel subtitles: ~10–12 px.
- Chart labels/ticks/legends: ~9–11 px; score annotations: ~10–12 px, 500–600.
- Hero comparison `+29%`: ~39–43 px, 500; year labels: ~20–23 px, 500–600.
- Disclosure headings: ~11–12 px, 600–700; body: ~10–11 px with compact 1.3–1.4 line-height.

**Inference constraint.** Use tabular numerals for scores, dates, counts, and axis ticks. Keep mathematical notation (`R²`) and minus signs intact; do not substitute approximate prose.

## 4. Spacing rhythm

**Observed/inferred [medium confidence].** Compact 4 px rhythm with frequent 8, 12, 16, 20, 24, and 32 px increments.

- Workspace inset: ~12–18 px; card gutters: ~10–14 px.
- Header actions: ~8–12 px gaps; controls around 38–42 px high.
- Model strip: ~28–32 px horizontal padding per field; ~8 px label-to-value gap.
- Card padding: ~14–20 px; chart title-to-plot gap: ~10–16 px.
- Chart annotations and legends sit in bordered or open inline rows with 8–12 px spacing.
- Radii: shell 12–16 px; panels 8–12 px; badges 3–6 px; status mark fully round.
- Borders are predominantly 1 px neutral gray. Shadows are subtle and diffuse, approximately equivalent to `0 2px 8px rgba(15, 23, 42, .06)`; this CSS is inferred, not measured.
- The disclosure strip uses vertical dividers and wide internal breathing room rather than separate floating cards.

## 5. Component inventory

1. `AppShell`, `Sidebar`, `BrandLockup`, `GlobalSearch`, `NavGroup`, `NavItem`, `PilotSelector`, `UserCard`.
2. `PageHeader`, action buttons, notification badge, avatar menu, `ValidationStatus` (`Passing`).
3. `ModelAuditSummary` with `SummaryField` and framework badge (`PyTorch`).
4. `Panel` and `PanelHeader` with subtitle/context badge.
5. `ObservedPredictedScatter`: density scatter, equality/reference line, statistic chips, outlier annotation, labeled axes.
6. `ThresholdSkillChart`: grouped bars, value labels, semantic legend, limitation note.
7. `SpatialErrorPanel`: side-by-side basin maps, separate legends, shared explanatory note.
8. `BaselineComparison`: horizontal score bars, pass/fail marks, large improvement metric.
9. `DroughtStressTest`: repeated year row, sparkline, CSI score, detection result, shared legend/note.
10. `DisclosureStrip`: icon heading and compact evidence text.
11. `CycleStepper` with completed/current states.

**State coverage [high].** Visible states include selected navigation, passing validation, baseline fail/model pass, detected deficit, known limitation warning, completed stage, and current stage.

## 6. Data visualization style

- **Observed vs predicted [high]:** blue density scatter with strongest concentration near low observed/predicted rainfall and a faint diagonal reference line. Axes are `Observed (mm/day)` and `Predicted (mm/day)`, both shown from 0 to 100. Summary chips: `R² · 0.74`, `Bias · -1.2 mm`, `n · 15,240`; orange callout: `Underprediction · extreme events >75 mm`.
- **Detection skill [high]:** grouped blue/orange/green bars by threshold. `>1 mm`: `0.82`, `0.22`, `0.67`; `>10 mm`: `0.71`, `0.31`, `0.52`; `>25 mm`: `0.58`, `0.38`, `0.41`; `>50 mm`: `0.43`, `0.45`, `0.30`. Legend: POD, FAR, CSI. Shared y-axis 0.0–1.0 supports comparison.
- **Spatial error [high]:** two Cauvery Basin choropleth/raster maps with river/boundary overlays. Mean bias diverges from `-5` through `0` to `+5`; RMSE runs `0`, `8`, `16`. These are analytical maps, not decorative illustrations.
- **Skill vs baselines [high]:** horizontal bars where lower is better: `Persistence 11.5`, `Climatology 9.8`, `MausamSetu 8.2`; gray failures and blue/green success; headline `+29%` and `↑ +16% over climatology`.
- **Drought-year stress test [high]:** small multiples for `2002`, `2009`, `2015`; blue model line versus dashed gray observed line; `CSI 0.47`, `CSI 0.44`, `CSI 0.41`; each says `✓ Deficit detected`.
- **Design rule [medium]:** retain axis units, denominators, threshold direction, legend labels, and caveats. Values are screenshot placeholders and must not be presented as independently verified performance.

## 7. Map style

Two compact analytical maps show the Cauvery Basin for `Mean bias (mm/day)` and `RMSE (mm/day)`. **Observed [high confidence].** The basin boundary is a thin muted outline, rivers are pale blue lines, and state/district subdivisions are faint. Data is rendered as continuous raster-like color fields clipped to the basin. Surrounding geography is intentionally minimal; there are no roads, labels, controls, or consumer-basemap chrome.

**Implementation inference [medium].** Use identical extent/projection and aligned dimensions for valid side-by-side comparison. Keep separate, clearly labeled legends because one scale is diverging and the other sequential. Include textual summaries for accessibility; do not rely on the color ramps alone. Exact administrative geometry and projection cannot be recovered from the screenshot.

## 8. Interactions suggested by the static image

All behavior below is inferred unless a visible state is named.

- Sidebar navigation, pilot selection, `Export`, `Share`, notifications, avatar, and search (`⌘K`) appear actionable.
- The green `Passing` chip exposes a status; it may open audit criteria, but clickability is **low confidence**.
- The `JJAS 2022-2024` control reads as a context/filter badge. Dataset/season switching is plausible but not proven.
- Scatter hover, linked brushing, point inspection, threshold tooltips, map hover values, and sparkline details would be useful but are **not evidenced**. Add only when requirements confirm them.
- The charts should expose keyboard-accessible data tables or summaries and persistent legends; hover cannot be the only means of reading scores.
- `VALIDATE` is visibly current and prior cycle stages are complete. Workflow navigation is suggested, not proven clickable.
- Exported artifacts should carry the same model/version, holdout, audit date, units, and limitation disclosures to avoid decontextualized metrics.

## 9. Copy / microcopy

Legend: **[S]** static interface/product copy; **[D]** dynamic/model/audit placeholder data. Preserve displayed values verbatim.

- [S] `MausamSetu`; [S] `AI-Powered Climate`; [S] `Digital Twin of India`; [S] `Search regions, metrics...`; [S] `⌘K`; [S] `Workspace`; [S] `Overview`; [S] `Live Map`; [S] `Forecast`; [S] `Scenarios`; [S] `Validation`; [S] `Sectors`; [S] `Alerts`; [S] `Pilots`.
- [D] `Cauvery Basin`; [D] `Krishna Basin`; [D] `Godavari Basin`; [D] `Bhavya Chaudhary`; [D] `Climate Ops`; [S] `मौसम सेतु`.
- [S] `Validation`; [D] `ConvLSTM v2.3`; [D] `JJAS 2022-2024 holdout`; [D] `Last audit 15 Jul 09:00 IST`; [S] `Export`; [S] `Share`; [S] `Notifications`; [D] `2`; [D] `BC`; [D] `Passing`.
- [S] `Model`; [D] `ConvLSTM v2.3`; [D] `PyTorch`; [S] `Training window`; [D] `1991 - 2021`; [D] `30 years · 32,850 samples`; [S] `Holdout`; [D] `2022 - 2024`; [D] `JJAS-weighted`; [S] `Retrained`; [D] `Apr 2026`; [D] `Quarterly cadence`.
- [S] `Observed vs predicted`; [D] `Daily rainfall · 15,240 pairs`; [D] `JJAS 2022-2024`; [D] `R² · 0.74`; [D] `Bias · -1.2 mm`; [D] `n · 15,240`; [S] `Predicted (mm/day)`; [S] `Observed (mm/day)`; [D] `Underprediction · extreme events >75 mm`.
- [S] `Detection skill · by rainfall threshold`; [S] `POD · FAR · CSI`; [D] `>1 mm`; [D] `0.82`; [D] `0.22`; [D] `0.67`; [D] `>10 mm`; [D] `0.71`; [D] `0.31`; [D] `0.52`; [D] `>25 mm`; [D] `0.58`; [D] `0.38`; [D] `0.41`; [D] `>50 mm`; [D] `0.43`; [D] `0.45`; [D] `0.30`; [S] `POD (detection)`; [S] `FAR (false alarm)`; [S] `CSI (overall skill)`; [D] `CSI drops at higher thresholds — extreme events remain a known limitation, mitigated by ensemble spread.`
- [S] `Spatial error · Cauvery Basin`; [S] `Where the model struggles`; [S] `Mean bias (mm/day)`; [S] `RMSE (mm/day)`; [D] `-5`; [D] `0`; [D] `+5`; [D] `0`; [D] `8`; [D] `16`; [D] `Higher error in orographic zone (Western Ghats). Expected — a known inherent difficulty for regional models.`
- [S] `Skill vs baselines`; [D] `JJAS test set · RMSE (mm/day) · lower is better`; [D] `Persistence`; [D] `11.5`; [D] `Climatology`; [D] `9.8`; [D] `MausamSetu`; [D] `8.2`; [D] `+29%`; [S] `skill improvement over persistence`; [D] `↑ +16% over climatology`.
- [S] `Drought-year stress test`; [S] `Model tested on analogue years`; [D] `2002`; [D] `CSI 0.47`; [D] `✓ Deficit detected`; [D] `2009`; [D] `CSI 0.44`; [D] `✓ Deficit detected`; [D] `2015`; [D] `CSI 0.41`; [D] `✓ Deficit detected`; [S] `Model`; [S] `Observed`; [D] `Performance degrades ~15% in extreme deficit years.`
- [S] `Known limitation`; [D] `Extreme events >75 mm/day show reduced CSI.`; [D] `Mitigated by ensemble spread and explicit uncertainty bands on every forecast.`
- [S] `Training data window`; [D] `1991 - 2021 · 30 years · 32,850 daily samples.`; [D] `Retrained quarterly. Last: April 2026.`; [S] `Next:`; [D] `July 2026.`
- [S] `What we don't yet model`; [D] `Aerosol-cloud interactions, urban heat island effects, and glacial melt contributions to river discharge.`; [D] `On roadmap for v3.`
- [S] `CYCLE 46`; [S] `INGEST`; [S] `REGRID`; [S] `ASSIMILATE`; [S] `FORECAST`; [S] `IMPACT`; [S] `VALIDATE`.

## 10. Anti-slop checks

- Keep validation evidence, limitations, and provenance visible together; do not turn the screen into a celebratory scorecard.
- Never display `Passing` without model/version, holdout context, last audit, and known limitations.
- Preserve screenshot values verbatim as placeholders. Do not recalculate, round, reconcile, or claim external verification.
- Keep threshold direction (`>`) and metrics POD/FAR/CSI distinct; never collapse them into an unlabeled “accuracy” score.
- Retain chart axes, units, legends, reference line, map scales, and the explicit “lower is better” qualifier.
- Use aligned analytical maps, not generic heatmap blobs or decorative geographic silhouettes.
- Keep failure/caveat orange and success green paired with labels/icons; color alone is insufficient.
- Avoid gradients outside evidence scales, glassmorphism, oversized KPI tiles, excessive pills, and generic AI sparkle motifs.
- Clearly label inferred CSS, hover, filtering, responsive behavior, and audit drill-down as unconfirmed.
- This decode covers Screen 6 only. Do not infer Screens 8–10 or unseen validation views.
