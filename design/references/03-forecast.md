---
route: /forecast
source path: C:\Users\bhavy\Documents\v1\images\screen-03-forecast.png
canvas: 1536 × 1024 px (RGB raster)
status: Direct reference decode
confidence caveat: High confidence for visible hierarchy, copy, and large-region geometry; medium for coordinates and dimensions measured from the raster; low-to-medium for font family, hidden states, exact vector colors, blur, radii, and shadows because those are inferred from antialiased pixels.
placeholder-data warning: Phase 1 must reproduce screenshot values verbatim as placeholders—including 32 members. RCP 4.5 is visible only in the approved Scenarios reference, not on this screen. Replace all placeholder operational, model, timestamp, forecast, attribution, skill, and user data only in a later data-integration phase.
---

# Forecast — design-engineering decode

Measurements use screenshot coordinates with origin `(0,0)` at the canvas top-left. Values marked **Measured** are raster observations (normally ±2–4 px); **Sampled** colors are exact screenshot pixels, not guaranteed source tokens; **Inferred** details are implementation hypotheses and include confidence.

## 1. Layout

### Canvas and shells
- **Measured:** 1536 × 1024 px desktop composition. A full-bleed, desaturated aerial/landscape image sits behind two translucent shells.
- **Sidebar:** approximately `(10,9)` to `(245,989)`, 235 × 980 px. It has a 19–21 px corner radius, pale translucent fill, and a faint edge/shadow. Internal horizontal padding is about 20 px.
- **Main shell:** approximately `(263,9)` to `(1522,1016)`, 1259 × 1007 px, with an 18–20 px radius and translucent near-white surface. Main content begins near x=276, giving 13 px shell inset; right edge is near x=1505, giving 17 px inset.
- **Primary grid:** one full-width header/KPI region, one full-width hero chart, a two-column analytic row, a three-column diagnostic row, then a full-width cycle stepper.

### Main-region geometry
| Region | Approx. bounds `(x, y, w, h)` | Notes |
|---|---:|---|
| Page heading and actions | `(276, 25, 1228, 66)` | Heading left; four action/account clusters right. |
| KPI strip | `(277, 96, 1227, 95)` | Four nearly equal metrics divided by 1 px vertical rules. |
| Rainfall forecast card | `(276, 199, 1229, 317)` | Hero card; chart plot occupies roughly y=286–447. |
| Daily spatial card | `(276, 526, 638, 217)` | Seven equal map frames in one row. |
| Ensemble spread card | `(922, 526, 583, 217)` | Spaghetti plot plus right histogram; summary footer. |
| Monsoon pulse | `(277, 754, 401, 202)` | Gauge left, three facts and state chip right. |
| Forecast attribution | `(686, 754, 388, 202)` | Five ranked horizontal bars. |
| Model skill | `(1083, 754, 421, 202)` | 2 × 3 metric grid plus positive footer. |
| Cycle stepper | `(276, 966, 1229, 40)` | Six stages; current stage centered-left. |

### Sidebar geometry
- Brand block is roughly `(31,28)` to `(218,96)`; search control `(29,117)` to `(230,175)`, about 201 × 58 px.
- Workspace label starts near y=208. Seven navigation rows run from y≈236 through y≈525 at an approximately 41 px pitch. Selected Forecast row is about `(16,324,216,36)` with a 3 px blue left indicator.
- Pilots begins around y=564; radio-style basin rows are centered near y=609, 648, and 687.
- User card is approximately `(29,827,201,68)`. Hindi wordmark sits near `(42,931)`.
- **Inferred, high confidence:** sidebar and main shell are independent scroll/containment regions, but no scrollbar is visible in this static state.

### Alignment behavior
- Main cards use 8–10 px gutters horizontally and vertically. The lower cards align to the hero card’s outer edges.
- Titles inside cards generally start 17–22 px from the left and 15–19 px from the top.
- **Inferred, medium confidence:** at narrower desktop widths, the main cards should reflow rather than scale the dense chart text; the screenshot itself specifies only the 1536 px canvas.

## 2. Color system

### Exact raster samples
These are **Sampled** RGB values found in the approved PNG; antialiasing, transparency, background imagery, and compression make them observations rather than canonical tokens.

| Use | Exact sampled pixel(s) | Suggested inferred token | Confidence |
|---|---|---|---|
| Near-white card fill | `#FEFEFE` `(254,254,254)`, `#FFFFFF` | `surface-card: rgba(255,255,255,.90–.96)` | High visual / medium token |
| Primary forecast blue | `#0D6BFA` `(13,107,250)` and nearby variants | `accent-blue: #0B6BFA` | Medium |
| Uncertainty band | `#CBDFFD` `(203,223,253)`, `#CADDFD` | `forecast-band: rgba(31,111,235,.16–.22)` | High visual |
| Selected-nav pale blue | samples around `#BFD8FD` | `accent-blue-soft: #E8F1FF` over translucent shell | Medium-low due compositing |
| Threshold orange | samples `#F3B57C`, `#F3BE7C` | `warning-orange: #F59E0B` at reduced opacity | Medium-low |
| Positive green | `#358E53` `(53,142,83)` and pale `#D0EBDF` | `success: #159455`; `success-soft` | Medium |
| Primary text | `#000000` core glyph pixels | `text-primary: #111318` | Medium; antialiased cores reach black |
| Secondary gray | raster variants around `#9EA2AC` | `text-secondary: #5F6673` rendered through translucent surface | Low-medium |
| Dividers/grid | samples around `#E8E8E8` / `#E9EAEE` | `border-subtle: #E5E7EB` | Medium |

### Semantic use
- Blue owns selection, forecast lines/points, observed markers, progress, active pipeline stage, and linked text.
- Orange is reserved for the extreme threshold and therefore reads as warning/attention, not a second categorical series.
- Green means confidence/health/positive comparison or active monsoon state.
- Grays provide climatology, grid lines, disabled/upcoming pipeline states, metadata, and borders.
- **Inferred, medium confidence:** surfaces use glass-like translucency (`backdrop-blur` equivalent around 12–20 px) rather than flat opaque white; retain adequate text contrast if the background image changes.

## 3. Typography

- **Inferred family, medium confidence:** a neutral UI sans resembling Inter, SF Pro, or a close grotesk. Glyph proportions and double-storey forms most closely suggest Inter. Devanagari likely falls back to a compatible system font.
- **Rendering:** predominantly regular and semibold; no decorative display face; tight dashboard line heights.

| Role | Approx. size / line height | Weight | Evidence |
|---|---:|---:|---|
| Page title | 24–26 / 31 px | 650–700 | Forecast heading at `(291,34)`. |
| Brand | 22–23 / 27 px | 700 | Sidebar wordmark. |
| KPI value | 27–30 / 34 px | 600–700 | `7 days`, `32`, `48 mm`, `High`. |
| Card title | 15–17 / 21 px | 650–700 | All card headings. |
| Body/control | 12–14 / 18 px | 450–600 | Tabs, buttons, nav, metadata. |
| Chart labels | 11–12 / 15 px | 400–550 | Axis labels, legends, annotations. |
| Microcopy | 10–11 / 14 px | 400–500 | subtitles and pipeline details. |
| Gauge numeral | 32–34 / 38 px | 550–650 | `78`. |

- Primary labels are near-black; metadata is a cool gray. Dynamic emphasis uses weight before color, except blue links/state and green positive values.
- **Inferred, medium confidence:** tabular numerals should be enabled for KPI and chart values to prevent jitter during updates.
- Preserve compact labels but avoid shrinking below approximately 11 px CSS at this canvas; the screenshot is already information-dense.

## 4. Spacing rhythm

- **Measured base rhythm:** most local gaps resolve to a 4 px family: 4, 8, 12, 16, 20, 24, and 32 px. Main card gutters are approximately 8–10 px.
- Main shell inset is approximately 13–17 px; card padding is usually 16–20 px; sidebar padding is about 20 px.
- Nav rows are approximately 36 px high with 8–10 px vertical gaps implied by their 41 px pitch. Icons occupy about 20–22 px; icon-to-label gap is 14–16 px.
- Buttons are approximately 38–42 px high, with 12–16 px horizontal padding. Segmented controls are about 34 px high. Chips are 28–32 px high.
- Hero chart header-to-plot gap is approximately 17 px. Plot labels sit within a compact 10–14 px margin.
- **Inferred radii:** shell 20 px; cards 10–12 px; search/buttons 8–10 px; chips 6–8 px; circular controls 999 px.
- **Inferred shadows, low-medium confidence:** cards use a faint ambient shadow around `0 2px 8px rgba(15,23,42,.08)` plus a 1 px translucent border; shells use a broader `0 8px 30px rgba(15,23,42,.10)`. Do not add dark, high-elevation shadows.
- Hairline dividers are approximately 1 px. The selected-nav indicator is approximately 3 px wide with rounded ends.

## 5. Component inventory

### Navigation and chrome
- `AppSidebar / desktop / forecast-selected`: brand, search with shortcut, grouped nav, pilot selector, user card, localized wordmark.
- `SidebarNavItem / default | selected`: outline icon, label; selected adds pale-blue background, blue icon, and left rail.
- `PilotRadio / selected | unselected`: filled blue or gray dot plus basin label.
- `HeaderAction / icon-leading`: Export and Share.
- `NotificationButton / unread`: bell plus blue count badge.
- `AccountMenu / avatar`: circular `BC` avatar plus chevron.

### Summary and controls
- `KpiStrip` containing four `KpiCell` variants: plain, tag-suffixed, linked-subline, status-dot.
- `SegmentedControl / metric`: three choices, Rainfall selected.
- `ModelSelect`: model label plus chevron.
- `ComparisonButton`: outlined comparison action.

### Analytics
- `ForecastChartCard`: title/subtitle, controls, threshold line, observed points, p50 line, uncertainty band, climatology line, today marker, peak callout, legend.
- `SpatialForecastStrip`: seven `ForecastMapFrame` items; selected/played range indicated by a blue underline spanning the first two frames in the static image.
- `EnsembleSpreadCard`: 32-member spaghetti plot, central/median trace, day-7 horizontal distribution histogram, three summary metrics.
- `MonsoonPulseCard`: 270°-style circular score gauge, facts, live status, active-phase chip.
- `AttributionCard`: ranked feature labels with normalized blue bars and percentages.
- `ModelSkillCard`: six metrics and green comparative outcome.
- `CycleStepper`: completed, active, and upcoming states.

### State notes
- **Inferred, high confidence:** all charts need loading, empty, unavailable, and stale-data states even though only populated states are pictured.
- **Inferred, medium confidence:** controls need hover, focus-visible, pressed, open, and disabled variants. The static screenshot does not authorize appearance beyond restrained extensions of the shown palette.

## 6. Data visualization style

### Basin-mean forecast
- Plot bounds are approximately `(324,286)` to `(1480,446)`. Y-axis spans 0–60 mm with ticks at 0, 15, 30, 45, and 60.
- Observations appear as blue circular markers with white rings and a thin connecting line through the historical side. A vertical dashed Today divider sits near x=868.
- The forecast p50 is a saturated blue line, approximately 2 px, with circular points. The p10–p90 interval is a layered pale-blue ribbon; perceived layering suggests inner and outer alpha bands rather than a single fill (**Inferred, medium confidence**).
- Climatology is a gray dashed horizontal reference at 18 mm; the extreme threshold is orange dashed at 40 mm. Both are directly labeled to avoid legend-only decoding.
- Peak annotation is a small white callout connected to the highlighted point. Note a screenshot inconsistency: the callout and KPI say `Jul 19`, while visual point/date alignment may appear nearer `Jul 20`; Phase 1 reproduces the screenshot rather than silently correcting it.
- The raster visibly includes `Jul 20` twice on the x-axis. Preserve that Phase 1 placeholder/transcription exactly; resolve against real dates later.

### Ensemble spread and summaries
- Approximately 32 fine blue traces use low opacity (roughly 10–25%), overlaid by a stronger central line. Avoid categorical colors; density is communicated through alpha.
- Right-side distribution uses horizontal blue bars with no heavy frame. It shares the visual blue language but not necessarily an axis scale with the line plot.
- Footer summary metrics are separated by hairline vertical rules.

### Gauge, ranking bars, and skill metrics
- Gauge uses a light-gray track and a bright-blue arc, with the score centered. No gradient is evident.
- Attribution bars have very pale tracks and solid blue fills; percentages align in a right column. Bars encode relative importance and should start at a common baseline.
- Skill metrics are text-first rather than miniature charts. Positive comparison uses green and an up-right arrow.
- **Inferred accessibility requirement:** never rely on blue/orange/green alone; retain labels, line styles, status words, and values. Tooltips should expose series, date, value, units, and uncertainty range.

## 7. Map style

- Seven small maps depict the Cauvery basin as a green boundary over a pale terrain/base-map texture. Rainfall is a semi-transparent blue raster/heat field, from very pale cyan to deep royal blue.
- Each map viewport is approximately 78–82 × 79 px, beginning near `(293,592)` with 8–10 px gaps. Date and millimeter value are centered beneath each viewport.
- Geographic labels are intentionally absent at thumbnail scale; basin silhouette and intensity field carry recognition. The cards avoid visible map controls, attribution, pins, or legends.
- **Inferred, medium confidence:** all frames use identical bounds, projection, crop, and color domain so temporal comparison is valid. Do not auto-fit each day independently.
- Basemap should stay low saturation/low contrast. Boundary is a thin green stroke, approximately 1 px. Rainfall layer uses soft, interpolated edges; it is not a choropleth.
- The blue underline beneath the first two thumbnails suggests an active scrub/progress range, not map data (**Inferred, low-medium confidence**).
- Clicking a frame is explicitly suggested by microcopy; expanded view should preserve date, value, common scale, basin boundary, and source/units.

## 8. Interactions suggested by the static image

- Search opens a region/metric command palette; the visible keyboard shortcut suggests keyboard-first access. **Inferred.**
- Sidebar rows navigate modules; pilot radios switch the active basin and refresh all page data. Preserve focus and selection state.
- Export likely opens format/options or downloads the current view; Share likely copies/opens a share flow; Notifications opens an unread panel; account chevron opens profile/session actions. **Inferred.**
- Rainfall/Max temp/Min temp tabs swap the hero metric while retaining horizon/model context. ModelSelect changes the forecast model; comparison action overlays or contrasts persistence.
- Chart hover/focus should reveal exact date, observed/p50, p10–p90, climatology, thresholds, and units. The Today and peak callouts should remain noninteractive reference annotations unless selected.
- Spatial frames are clickable to expand, as stated onscreen; keyboard activation and a visible selected frame are required. The underline may indicate scrub progress.
- Ensemble plot hover may highlight a member and linked histogram bin; summary values update with horizon selection. **Inferred, medium confidence.**
- Pipeline steps read as process status. Completed steps are green checks, Assimilate is active blue, and future stages are neutral. Do not imply that upcoming steps are clickable without product confirmation.
- Responsive behavior is not evidenced. **Do not infer mobile layouts or Screens 8–10 from this reference.**

## 9. Copy / microcopy

Classification: **[S]** interface/static copy; **[D]** screenshot data or context that must remain verbatim as a Phase 1 placeholder and later be replaced/wired. Symbols, capitalization, punctuation, date duplication, and units below reflect the image.

### Sidebar
- [S] `MausamSetu`; [S] `AI-Powered Climate`; [S] `Digital Twin of India`
- [S] `Search regions,`; [S] `metrics...`; [S] `⌘K`
- [S] `Workspace`; [S] `Overview`; [S] `Live Map`; [S] `Forecast`; [S] `Scenarios`; [S] `Validation`; [S] `Sectors`; [S] `Alerts`
- [S] `Pilots`; [D] `Cauvery Basin`; [D] `Krishna Basin`; [D] `Godavari Basin`
- [D] `BC`; [D] `Bhavya Chaudhary`; [D] `Climate Ops`; [S] `मौसम सेतु`

### Header and KPI strip
- [S] `Forecast`; [D] `Cauvery Basin`; [D] `ConvLSTM v2.3`; [D] `Updated 09:34 IST`
- [S] `Export`; [S] `Share`; [S] `Notifications`; [D] `2`; [D] `BC`
- [S] `Forecast horizon`; [D] `7 days`; [D] `+7d`
- [S] `Ensemble members`; [D] `32`; [D] `MC-Dropout`
- [S] `Peak intensity`; [D] `48 mm`; [D] `Jul 19`; [S] `in`; [D] `4 days`
- [S] `Confidence`; [D] `High`

### Hero chart
- [S] `Basin-mean rainfall forecast`; [D] `Cauvery Basin`; [S] `daily aggregates`
- [S] `Rainfall`; [S] `Max temp`; [S] `Min temp`; [D] `ConvLSTM v2.3`; [S] `vs Persistence`
- [S] `mm`; [D] `60`; [D] `45`; [D] `30`; [D] `15`; [D] `0`
- [S] `Extreme threshold`; [D] `(40 mm)`; [S] `Climatology`; [D] `(1991–2020)`; [D] `(18 mm)`
- [D] `Peak`; [D] `48 mm`; [D] `Jul 19`; [S] `Today`
- X-axis, left to right: [D] `Jul 08`; [D] `Jul 09`; [D] `Jul 10`; [D] `Jul 11`; [D] `Jul 12`; [D] `Jul 13`; [D] `Jul 14`; [D] `Jul 15`; [D] `Jul 16`; [D] `Jul 17`; [D] `Jul 18`; [D] `Jul 19`; [D] `Jul 20`; [D] `Jul 20`; [D] `Jul 21`; [D] `Jul 22`
- [S] `Observed`; [S] `p50 forecast`; [S] `p10–p90 band`; [S] `Climatology`

### Daily spatial and ensemble
- [S] `Daily forecast`; [S] `spatial`; [S] `click any frame to expand`
- [D] `Jul 16` / [D] `32 mm`; [D] `Jul 17` / [D] `34 mm`; [D] `Jul 18` / [D] `44 mm`; [D] `Jul 19` / [D] `48 mm`; [D] `Jul 20` / [D] `41 mm`; [D] `Jul 21` / [D] `28 mm`; [D] `Jul 22` / [D] `22 mm`
- [S] `Ensemble spread`; [D] `32 members`; [S] `mm`; [D] `60`; [D] `45`; [D] `30`; [D] `15`; [D] `0`
- [D] `Jul 16`; [D] `Jul 17`; [D] `Jul 18`; [D] `Jul 19`; [D] `Jul 20`; [D] `Jul 21`; [D] `Jul 22`
- [S] `Day-7 distribution`; [D] `(Jul 22)`; [D] `0`; [D] `20`; [D] `40`; [D] `60`; [D] `80`; [S] `mm`
- [S] `Median`; [D] `22 mm`; [S] `Spread`; [D] `±12 mm`; [S] `Coherence`; [D] `87%`

### Diagnostic cards and cycle
- [S] `Monsoon pulse`; [S] `Live`; [D] `78`; [S] `/100`; [S] `Onset`; [D] `Jun 3`; [S] `Season-to-date`; [D] `412 mm`; [S] `% of normal`; [D] `108%`; [S] `Currently:`; [D] `Active surge phase`
- [S] `Forecast attribution`; [S] `via Integrated Gradients`; [D] `Yesterday's rainfall @ Coorg` / [D] `78%`; [D] `Arabian Sea SST +1.2°C` / [D] `62%`; [D] `Indian Ocean Dipole index` / [D] `45%`; [D] `Soil moisture · Kabini basin` / [D] `41%`; [D] `500 hPa geopotential · Deccan` / [D] `33%`
- [S] `Model skill`; [S] `this forecast`; [S] `RMSE (7d)` / [D] `8.2 mm`; [S] `MAE (7d)` / [D] `5.3 mm`; [S] `ACC` / [D] `0.68`; [S] `CSI (>10mm)` / [D] `0.52`; [S] `POD` / [D] `0.71`; [S] `FSS` / [D] `0.61`; [S] `Beats persistence by` / [D] `29%`; [S] `beats climatology by` / [D] `16%`
- [D] `CYCLE 46`; [S] `INGEST`; [S] `REGRID`; [S] `ASSIMILATE`; [S] `FORECAST`; [S] `IMPACT`; [S] `RENDER`

## 10. Anti-slop checks

- Reproduce only this approved Forecast screen; do not extrapolate Screens 8–10, add unsupported panels, or invent mobile states.
- Keep the two-shell composition, dense analytic hierarchy, aligned card grid, restrained glass effect, and landscape backdrop. Do not turn it into a generic dark dashboard or use oversized marketing typography.
- Preserve all Phase 1 values verbatim, especially `32 members`, model/version, timestamps, dates, units, thresholds, and the screenshot’s duplicated `Jul 20`. Do not “correct” placeholders during visual implementation.
- `RCP 4.5` is not visible here; do not add it to Forecast UI. It must remain verbatim where shown in the approved Scenarios reference.
- Distinguish measured facts from inferred tokens. Do not claim sampled raster colors are source hex values; opacity and backdrop alter pixels.
- Keep blue as the only dominant data/interaction accent. Orange is threshold/warning; green is positive/status. Avoid gratuitous gradients, neon colors, rainbow chart series, and decorative glow.
- Use common scales and bounds across all seven map thumbnails. Do not independently auto-fit or recolor frames.
- Keep lines thin, grids faint, uncertainty translucent, labels direct, and legends compact. Do not add 3D charts, heavy axes, or excessive tick density.
- Retain units next to values (`mm`, `%`, days), line-style semantics, and textual status so meaning does not rely on color.
- Provide focus-visible states, keyboard access, hit targets near 40 px where possible, and chart/map text alternatives. Verify contrast over the variable photographic background.
- Treat all names, user identity, operational timestamps, basin values, forecast results, attributions, scores, and cycle state as placeholder data—not constants.
- Before sign-off, compare at 1536 × 1024: shell/card bounds (±4 px), gutters (±2 px), baseline alignment, clipping, exact copy, selected states, chart dates, map order, and z-order of ribbon/lines/points/annotations.
