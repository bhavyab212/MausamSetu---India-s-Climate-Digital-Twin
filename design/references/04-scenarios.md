---
route: /scenarios
source path: C:\Users\bhavy\Documents\v1\images\screen-04-scenarios.png
canvas: 1536 × 1024 px (RGB raster)
status: Direct reference decode
confidence caveat: High confidence for visible hierarchy, copy, selected states, and large-region geometry; medium for raster-measured coordinates and dimensions; low-to-medium for font family, hidden behavior, source color tokens, map provider, blur, radii, and shadows because those are inferred from a single antialiased static image.
placeholder-data warning: Phase 1 must reproduce screenshot values verbatim as placeholders—including 32 members and RCP 4.5—even though both and all scenario IDs, perturbations, periods, impacts, findings, and user data require later replacement or live wiring.
---

# Scenarios — design-engineering decode

Coordinates use screenshot origin `(0,0)` at top-left. **Measured** values are approximate raster observations (normally ±2–4 px). **Sampled** colors are exact PNG pixels, not guaranteed source tokens. **Inferred** behavior and styling are explicitly labeled with confidence.

## 1. Layout

### Canvas and shells
- **Measured:** 1536 × 1024 px desktop dashboard over a full-bleed muted aerial/landscape image.
- **Sidebar shell:** approximately `(16,12)` to `(239,997)`, 223 × 985 px, translucent near-white with 17–20 px corners. Compared with Forecast, this capture’s sidebar is slightly narrower and starts farther right.
- **Main shell:** approximately `(254,12)` to `(1521,998)`, 1267 × 986 px, translucent near-white with about 18 px corners. Interior content runs from x≈264 to x≈1508.
- Header occupies y≈24–83. Main working area begins y≈95 and ends around y≈925. The cycle stepper sits y≈943–989.

### Three-column working layout
| Region | Approx. bounds `(x, y, w, h)` | Notes |
|---|---:|---|
| Scenario configuration | `(264,95,289,830)` | Fixed left control panel; primary CTA at bottom. |
| Spatial impact | `(562,95,611,538)` | Center top; side-by-side maps and anomaly strip. |
| Runoff response | `(562,643,344,282)` | Center-bottom left. |
| Assumptions | `(915,643,258,282)` | Center-bottom right. |
| Sector impact | `(1185,95,323,830)` | Right outcome rail with four impact cards and finding. |
| Cycle stepper | `(264,943,1244,46)` | Six evenly distributed workflow states. |

### Internal geometry
- Scenario configuration uses approximately 17 px horizontal padding. Control content starts around x=280; slider tracks extend to x≈534. Group spacing is 21–28 px, with section dividers near y≈497 and y≈693.
- Spatial maps begin around `(582,180)` and `(870,180)`, each about 277 × 345 px. Gap is approximately 11 px. The anomaly strip is around `(582,544,572,78)`.
- Sector cards are approximately `(1202,164,288,137)`, `(1202,310,288,129)`, `(1202,449,288,148)`, and `(1202,606,288,127)`. Key finding occupies roughly `(1202,742,288,157)`.
- Sidebar brand/search/navigation/pilots/user blocks follow the same vertical architecture as Forecast. Selected Scenarios row is approximately `(24,366,215,38)` with a blue left rail.
- **Inferred, high confidence:** layout is optimized around side-by-side spatial comparison; preserving comparable map size is more important than preserving the exact control-panel width at smaller desktop widths.

## 2. Color system

### Exact raster samples
The following are **Sampled** from the approved screenshot and may include antialiasing or alpha compositing.

| Use | Exact sampled pixel(s) | Suggested inferred token | Confidence |
|---|---|---|---|
| Near-white cards | `#FEFEFE` `(254,254,254)`, `#FFFFFF` | `surface-card: rgba(255,255,255,.92–.97)` | High visual / medium token |
| Primary/action blue | `#0061FE`, `#0062FE`, `#0060FE` | `accent-blue: #0062FF` | High |
| Blue slider/line | `#176EFC` plus lighter antialias samples | Same accent with alpha variants | Medium-high |
| Rainfall-change orange | `#FE7203`, `#FE7A13` | `warning-orange: #F97316` | High visual |
| Active-toggle green | `#129055`, `#139256` | `success: #10935B` | High visual |
| Critical red | `#E12210`, `#E13B27` | `critical: #DC2B1B` | Medium-high |
| Map/rainfall pale blue | samples around `#C4DCF7` | sequential blue raster scale | Medium |
| Positive green impact | `#0A8A4A` and pale green composites | `positive: #0A8A4A` | Medium-high |
| Borders/dividers | raster around `#ECEDEF` | `border-subtle: #E5E7EB` | Medium |
| Primary text | black glyph cores `#000000` | `text-primary: #111318` | Medium |

### Semantic mapping
- Blue means primary action, selection, baseline runoff, temperature perturbation, rainfall magnitude, and active workflow state.
- Orange means rainfall deficit/anomaly, delayed onset, crop/heat stress, warning, and nonlinear deficit response.
- Red is reserved for critical basin-runoff degradation and severe crop stress.
- Green indicates enabled constraints, passed assumptions, improved flash-flood probability, and completed workflow stages.
- **Inferred, medium confidence:** glass surfaces use background blur around 12–20 px. Ensure contrast is based on the worst backdrop, not the pale portion visible here.
- Avoid conflating semantic orange with the map’s negative-anomaly ramp: both are intentional but must remain labeled and unit-bearing.

## 3. Typography

- **Inferred family, medium confidence:** Inter-like neutral sans; compact proportions, regular body, semibold headings. Devanagari uses a compatible fallback.

| Role | Approx. size / line height | Weight | Notes |
|---|---:|---:|---|
| Page title | 23–25 / 30 px | 650–700 | `Scenarios`. |
| Brand | 22–23 / 27 px | 700 | Sidebar. |
| Panel title | 15–17 / 21 px | 650–700 | Configuration, Spatial impact, Sector impact. |
| Large outcome | 34–39 / 42 px | 450–600 | `−36%`; other impacts are 30–34 px. |
| Control/body | 12–14 / 18 px | 450–600 | Labels, inputs, tabs, buttons. |
| Chart/map label | 10–12 / 14 px | 400–550 | Axes, city names, legend. |
| Microcopy | 10–11 / 14 px | 400–500 | ranges, runtime, subtitles. |

- Key finding body appears approximately 12 px with 18 px line height for a readable compact paragraph.
- Primary CTA is around 13–14 px semibold in white. Numeric inputs and impact values should use tabular numerals (**Inferred, medium confidence**).
- Hierarchy is achieved through weight and spacing, not an oversized scale. Do not inflate headings or reduce analytical content density.

## 4. Spacing rhythm

- **Measured rhythm:** dominant increments are 4, 8, 12, 16, 20, 24, and 32 px. Main-panel gutters are approximately 8–12 px.
- Main shell inset is about 10–12 px. Card inner padding is about 16–20 px. Configuration controls use about 16 px between label and range/value rows and 28–33 px between perturbations.
- Segmented control is approximately 255 × 34 px; each segment is 80–90 px wide. Header actions are 38–41 px high.
- Sliders use a roughly 3 px track and 14–16 px circular thumb with white halo/shadow. Value chips are about 62–88 × 28 px with 6–8 px radius.
- Preset chips are approximately 90–96 × 30 px and separated by 5–7 px.
- Form selects/stepper are approximately 139 × 30 px. Toggles are about 30 × 18 px.
- Primary CTA is about `(277,844,258,33)`, with 6–8 px radius. The smaller-than-44 px visual height requires a larger invisible hit target or increased implementation height for accessibility (**Inferred recommendation**).
- **Inferred radii:** shell 18–20 px; primary panels 10–12 px; nested impact cards 7–9 px; inputs/chips 6–8 px; avatars/toggles circular.
- **Inferred shadows, low-medium confidence:** restrained 1 px translucent borders plus `0 2px 8px rgba(15,23,42,.07)`; main shell has a broader, softer ambient shadow. Avoid thick borders and high-contrast drop shadows.

## 5. Component inventory

### App chrome
- `AppSidebar / desktop / scenarios-selected`: brand, command search, workspace navigation, pilot selection, user card, localized wordmark.
- `PageHeader / scenario-ready`: title/context/status left; Export, Share, Notifications, account, and New scenario actions right.
- `NotificationButton / unread`; `AvatarButton`; `PrimaryButton / new-scenario`.

### Scenario configuration
- `ScenarioConfigPanel` with title, scenario ID, Reset action, grouped controls, scope, ensemble controls, toggles, CTA, and runtime estimate.
- `PerturbationSlider / temperature`: blue track, numeric chip, range hint.
- `PerturbationSlider / rainfall`: orange track and value.
- `PerturbationSlider / onset-shift`: orange track and delayed-day value.
- `PresetChip / drought | flood | pathway`: colored dot plus label; none appears strongly selected.
- `SelectField / region | period`; `NumericStepper / members`; `Toggle / enabled`; `RunScenarioButton`.

### Spatial and analytic components
- `SpatialImpactCard` with `ComparisonModeTabs / side-by-side-selected`, two linked `ImpactMap` panes, and `DivergingAnomalyLegend`.
- `RunoffResponseCard`: dual-line chart, annotation callout, explanatory footer.
- `AssumptionsCard`: five checkmarked validation statements.
- `SectorImpactPanel`: four `ImpactMetricCard` variants—critical, severe, elevated, improved—each with icon, delta, before/after, sparkline, and status badge.
- `KeyFindingCard`: orange left rule, narrative, Share and Export PDF links.
- `CycleStepper`: completed, active, and upcoming states.

### Required implementation states
- **Inferred, high confidence:** sliders need keyboard adjustment and value announcements; form controls need focus, validation, dirty, disabled, and loading states.
- Scenario execution needs ready, running/progress, success, warning, failed, canceled, and stale-result states. None except ready/populated is shown.
- Spatial mode tabs need selected/hover/focus states; maps need synchronized loading/error states to avoid misleading comparisons.

## 6. Data visualization style

### Spatial comparison
- Baseline and scenario maps are equal-sized and share bounds. Sequential rainfall blue is much stronger in Baseline and sparse/pale in Scenario, making the −20% rain outcome immediately visible.
- Pane labels include perturbation context above the map rather than relying on color alone. A small orange perturbation summary sits at the upper-right.
- Rainfall anomaly uses a horizontal diverging ramp: orange at −40%, cream/white at 0, and blue at +40%, with ticks at −40%, −20%, 0, +20%, +40%.

### Runoff response
- Plot occupies approximately `(581,704)` to `(888,836)`. Y-axis is `% of normal runoff`, spanning approximately 0–120 with ticks 0, 30, 60, 90, 120. X-axis runs Day 1 through Day 30.
- Baseline line is bright blue; scenario line is orange. Both are approximately 2 px with small circular markers at sample intervals. Gridlines are faint and horizontal.
- Orange trajectory drops from near normal toward roughly 52 while blue oscillates around 95–105. A white annotation callout connects `20% rainfall deficit` to `36% runoff deficit`.
- Footer states the nonlinear mechanism. Use exact values and units in accessible tooltips.

### Sector metrics
- Outcome cards are text-first: large signed change, compact baseline→scenario line, tiny same-color sparkline, semantic badge. Sparklines omit axes and are trend cues, not precise readouts.
- Red/orange/green semantics are reinforced by labels (`Critical`, `Severe`, `Elevated`, `Improved`). Preserve this redundancy.
- **Inferred, medium confidence:** impact cards should animate/update only after a completed run; avoid interpolating between semantically different scenarios in a way that implies physical temporal evolution.
- Use a true minus sign where supported, but Phase 1 visual copy must match screenshot appearance. Keep `%`, `°C`, `MCft`, and day units adjacent to values.

## 7. Map style

- Maps show the Cauvery basin with a thin green outline over a pale cartographic terrain/road basemap. Water is very pale blue. Labels include state/region and selected cities.
- Baseline and Scenario panes each use approximately 277 × 345 px and identical crop, projection, zoom, and color domain. This lock is critical for honest side-by-side comparison.
- Visible geographic labels include `Kodagu`, `Hassan`, `Mysuru`, `Mettur`, `Salem`, `Tiruchirappalli`, `Thanjavur`, `KERALA`, `TAMILNADU`, and `Arabian Sea` (transcribed according to raster legibility). Labels are subdued gray/blue so rainfall remains primary.
- Rainfall is a soft interpolated raster/heat surface, not administrative polygons. Strongest blues form a north–south corridor in Baseline; Scenario is mostly pale with scattered light-blue areas.
- Anomaly strip uses the basin silhouette with orange negative anomalies and limited blue positive patches, followed by a compact diverging legend.
- No zoom controls, map attribution, pins, scale bar, or compass are visible. Do not invent them in the direct Phase 1 decode; legal map attribution may still be required outside/under the crop in implementation.
- **Inferred interactions, medium confidence:** map pan/zoom and hover should be synchronized between panes; Side by side, Delta only, and Overlay change rendering mode without changing scenario inputs.

## 8. Interactions suggested by the static image

- `New scenario` starts a clean scenario or opens a creation flow. Reset restores current scenario inputs to defaults; confirmation is appropriate only if changes are dirty. **Inferred.**
- Temperature, rainfall, and onset sliders support pointer and keyboard adjustment and update the nearby value chip. Range hints define bounds. Inputs should clamp to shown ranges.
- Preset chips apply coherent bundles: 2016 Drought, 2019 Flood, or RCP 4.5. Do not rename or replace RCP 4.5 in Phase 1.
- Region and period controls open selects. Members uses minus/plus stepping and preserves the screenshot placeholder `32`. MC-Dropout and Physics constraints are enabled toggles.
- Run scenario begins computation; the `Est. runtime · 2:35` line sets progress expectations. Disable duplicate submission while running and expose cancel/retry. **Inferred.**
- Comparison tabs switch Side by side, Delta only, and Overlay. Map camera/hover should remain linked where multiple panes are shown.
- Sector cards may support drill-down, but no chevrons or click affordances are visible; treat them as read-only unless product requirements confirm navigation.
- Key finding Share and Export PDF are explicit text actions. Header Share/Export apply to the broader scenario/result.
- Workflow stepper shows Configure and Perturb completed, Propagate active, and Impact/Render/Compare upcoming. It reads as status, not necessarily navigation.
- Responsive behavior is not shown. **Do not infer mobile behavior or Screens 8–10 from this image.**

## 9. Copy / microcopy

Classification: **[S]** interface/static copy; **[D]** screenshot context/data that must be reproduced verbatim as a Phase 1 placeholder, then replaced or live-wired later. All legible copy is inventoried below; punctuation and values follow the screenshot.

### Sidebar
- [S] `MausamSetu`; [S] `AI-Powered Climate`; [S] `Digital Twin of India`
- [S] `Search regions,`; [S] `metrics...`; [S] `⌘K`
- [S] `Workspace`; [S] `Overview`; [S] `Live Map`; [S] `Forecast`; [S] `Scenarios`; [S] `Validation`; [S] `Sectors`; [S] `Alerts`
- [S] `Pilots`; [D] `Cauvery Basin`; [D] `Krishna Basin`; [D] `Godavari Basin`
- [D] `BC`; [D] `Bhavya Chaudhary`; [D] `Climate Ops`; [S] `मौसम सेतु`

### Header
- [S] `Scenarios`; [D] `Cauvery Basin`; [S] `What-if simulator`; [D] `Ready to run`
- [S] `Export`; [S] `Share`; [S] `Notifications`; [D] `2`; [D] `BC`; [S] `New scenario`

### Scenario configuration
- [S] `Scenario configuration`; [S] `Scenario`; [D] `CAV-2026-047`; [S] `Reset`
- [S] `Climate perturbations`; [S] `Temperature anomaly`; [S] `Range:`; [D] `-3 to +5`; [D] `+2.0 °C`
- [S] `Rainfall change`; [S] `Range:`; [D] `-50 to +50`; [D] `-20 %`
- [S] `Monsoon onset shift`; [S] `Range:`; [D] `-15 to +15`; [D] `+5 days late`
- [S] `Presets`; [D] `2016 Drought`; [D] `2019 Flood`; [D] `RCP 4.5`
- [S] `Scope`; [S] `Region`; [D] `Cauvery Basin`; [S] `Period`; [D] `JJAS 2026`
- [S] `Ensemble`; [S] `Members`; [D] `32`; [S] `MC-Dropout`; [S] `Physics constraints`
- [S] `Run scenario`; [S] `Est. runtime`; [D] `2:35`

### Spatial impact and map labels
- [S] `Spatial impact`; [S] `Side by side`; [S] `Delta only`; [S] `Overlay`
- [S] `Baseline`; [S] `Scenario`; [D] `T +2 °C, Rain -20%`; [D] `+2 °C · -20% rain`
- [D] `Kodagu`; [D] `Hassan`; [D] `Mysuru`; [D] `Mettur`; [D] `Salem`; [D] `Tiruchirappalli`; [D] `Thanjavur`; [D] `KERALA`; [D] `TAMILNADU`; [D] `Arabian Sea`
- [S] `Rainfall (mm)`; [D] `0`; [D] `25`; [D] `50`; [D] `75`; [D] `100+`
- [S] `Rainfall anomaly`; [D] `-40%`; [D] `-20%`; [D] `0`; [D] `+20%`; [D] `+40%`

### Runoff and assumptions
- [S] `Runoff response`; [S] `non-linear`; [S] `% of normal runoff`
- [D] `120`; [D] `90`; [D] `60`; [D] `30`; [D] `0`; [D] `Day 1`; [D] `Day 5`; [D] `Day 10`; [D] `Day 15`; [D] `Day 20`; [D] `Day 25`; [D] `Day 30`
- [D] `20% rainfall deficit`; [D] `36% runoff deficit`; [S] `Amplification via evapotranspiration + soil moisture threshold`
- [S] `Assumptions`; [D] `Perturbations applied uniformly across basin`; [D] `Model coupled response (SCS-CN hydro)`; [D] `32-member Monte Carlo ensemble`; [D] `Physics constraints active`; [D] `Baseline: JJAS 2026 median forecast`

### Sector impact and finding
- [S] `Sector impact`; [S] `Downstream consequence`
- [S] `Basin runoff`; [D] `-36%`; [D] `520 → 374 MCft`; [D] `Critical`
- [S] `Crop stress days`; [D] `+14`; [D] `8 → 22 days`; [D] `Severe`
- [S] `Heatwave days`; [D] `+8`; [D] `3 → 11 days`; [S] `Population exposed`; [D] `12.4M`; [D] `Elevated`
- [S] `Flash flood probability`; [D] `-42%`; [D] `High → Moderate`; [D] `Improved`
- [S] `Key finding`; [D] `A 20% rainfall deficit combined with +2 °C warming produces a disproportionate 36% drop in basin runoff — soil moisture crosses the drying threshold.`
- [S] `Share`; [S] `Export PDF`

### Cycle
- [D] `CYCLE 46`; [S] `CONFIGURE`; [S] `PERTURB`; [S] `PROPAGATE`; [S] `IMPACT`; [S] `RENDER`; [S] `COMPARE`

## 10. Anti-slop checks

- Build only the approved Scenarios composition. Do not infer Screens 8–10, unsupported drill-downs, extra KPIs, or mobile layouts.
- Preserve Phase 1 placeholder values verbatim, explicitly including `32` members and `RCP 4.5`. Do not modernize the pathway label, “fix” values, or substitute live data during visual matching.
- Keep all scenario IDs, climate perturbations, dates/periods, model assumptions, impacts, findings, runtime, cycle state, and identity data replaceable; none are product constants.
- Preserve the three-column hierarchy: controls left, scientific comparison/response center, downstream consequences right. Do not collapse the outcome rail into generic stat tiles at the reference viewport.
- Lock side-by-side maps to the same viewport, projection, color domain, and basemap. Never auto-scale each map independently to exaggerate or hide differences.
- Distinguish exact raster samples from inferred tokens. Do not present sampled composite pixels as authoritative source design values.
- Keep blue primary, orange perturbation/deficit, red critical, and green success/improvement. Avoid rainbow palettes, decorative gradients, glass glare, neon glow, 3D charts, and excessive shadow.
- Retain labels, signs, units, before→after values, and status words; do not communicate impact through color or sparkline alone.
- Keep map imagery quiet enough for rainfall and basin boundary to dominate. Do not add prominent roads, pins, control stacks, or satellite saturation absent from the reference.
- Use restrained line widths, faint gridlines, small markers, and compact annotations. Maintain honest y-axis and anomaly scales.
- Make sliders, presets, selects, steppers, toggles, tabs, buttons, and text links keyboard operable with visible focus. Announce current slider values, run status, and map/chart equivalents to assistive technology.
- Before sign-off at 1536 × 1024, compare shell/card bounds (±4 px), gutters (±2 px), map synchronization, selected states, all copy and punctuation, slider positions, toggle states, chart signs/units, and workflow status.
