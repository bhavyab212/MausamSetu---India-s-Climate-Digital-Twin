---
route: /overview
source_path: 'C:\Users\bhavy\Documents\v1\images\screen-01-overview.png'
canvas: 1536 × 1024 px
status: Direct reference decode
confidence_caveat: 'High confidence for visible structure and copy; medium for coordinates and colors sampled from the raster; low-to-medium for font family, hidden behavior, and token intent. Route is inferred from the visible navigation label.'
placeholder_data_warning: 'All dates, times, measurements, names, statuses, chart values, and alert text below are Phase 1 placeholders copied verbatim from the screenshot, not validated live data or product requirements.'
---

## 1. Layout

**Measurement convention:** approximate screenshot pixels, origin `(0,0)` at canvas top-left. Values are direct visual measurements unless marked **Inferred**.

- Desktop canvas: `1536 × 1024`.
- Background: full-canvas soft-focus landscape/weather image with a pale wash. Two foreground glass shells sit about `12–16 px` from the canvas edges.
- Base dashboard shell:
  - Left navigation: approximately `(16,12)`, `238 × 1000`, corner radius about `20 px`.
  - Main workspace: approximately `(281,12)`, `1240 × 1000`, corner radius about `20 px`.
  - Inter-shell gutter: about `27 px`.
- Sidebar geometry:
  - Brand block starts near `(38,38)`; search box near `(35,117)`, about `197 × 61`.
  - Primary navigation begins near `y=218`; rows are approximately `42 px` high. Selected row is around `(26,241)`, `210 × 45`, with a `3–4 px` teal leading indicator.
  - Pilot list begins near `y=578` after a divider.
  - User card is anchored near `(34,852)`, about `199 × 77`; language/product tile near `(34,944)`, about `199 × 47`.
- Main header: title block near `(307,36)`; action cluster spans approximately `x=1062–1501`, `y=31–72`.
- KPI row: five cards from `x≈306` to `1486`, `y≈109–271`, each approximately `229 × 162`, with `18–19 px` gaps. The reservoir card has a slightly stronger green outline than its siblings.
- Primary content grid begins near `y=291`:
  - Live climate state/map card: `(301,291)`, about `709 × 456`.
  - 7-day forecast card: `(1028,291)`, about `473 × 264`.
  - Sector status card: `(1028,569)`, about `473 × 178`.
  - Model performance card: `(301,758)`, about `342 × 198`.
  - System health card: `(657,758)`, about `353 × 198`.
  - Recent alerts card: `(1028,758)`, about `473 × 198`.
- Pipeline/status strip: approximately `(288,969)`, `1220 × 38`, divided into seven horizontally distributed stages.
- **Inferred responsive intent, medium confidence:** this is a fixed dense desktop composition. At narrower widths, do not merely squeeze it; collapse the sidebar and stack KPI/content cards while retaining the map as the primary wide module.

## 2. Color system

**Exact raster samples** are measured screenshot pixels, not guaranteed source tokens; translucency, backdrop blur, antialiasing, and compression affect them.

| Role | Exact sample from screenshot | Likely implementation token (inferred) | Confidence |
|---|---:|---:|---|
| Sidebar interior | `#FAFAF9` at `(100,100)` | warm white/glass surface | High sample, medium token |
| Main shell interior | `#FAFAFA` at `(300,20)` | neutral white/glass surface | High sample, medium token |
| Workspace field | `#F7F7F4` at `(400,200)` | off-white surface | High sample, medium token |
| Search field | `#F3F2F0` at `(305,109)` | muted input surface/border blend | High sample, low token |
| Selected-nav accent | `#2B7E77` at `(25,264)` | teal-700 around `#0F766E` | High sample, medium token |
| Map water/light blue | `#ECF3FA` at `(511,250)` | blue-50/100 | High sample, low token |
| Neutral card interior | `#FBFBFA` at `(507,161)` | white at high opacity | High sample, medium token |
| Green-tinted card field | `#F0F4F3` at `(1311,160)` | emerald tint | High sample, low token |
| Positive status dot | `#057847` at `(1457,621)` | emerald-700 | High sample, high role |
| Critical status dot | `#DB0508` at `(1457,718)` | red-600/700 | High sample, high role |
| Alert row pale red | `#FAF2F1` at `(1057,822)` | red-50 | High sample, medium token |
| Alert row pale amber | `#FBF8F5` at `(1057,864)` | amber-50 | High sample, medium token |
| Alert row pale blue/gray | `#F4F6F5` at `(1057,908)` | slate/blue-50 | High sample, low token |

- Core semantic palette: near-black primary text; gray secondary text; teal for shell navigation and positive operational context; bright blue for rainfall/live/selection; green for stable/positive; orange for heat/moderate/warning; red for elevated/critical.
- Rainfall ramps from very pale cyan through sky/royal blue to dark indigo. Representative raster blues include `#B0D9FD`, `#87C7FD`, `#669CCB`, and `#5E98C9`; these are exact image colors but not enough to define the original gradient stops.
- Card borders appear `1 px`, green-gray at low opacity. The Mettur KPI card uses a clearer green stroke to establish emphasis.
- Glass treatment: roughly `85–95%` white surface opacity plus backdrop blur; **inferred** blur radius `16–28 px`. Shadows are broad, low contrast, roughly `0 8px 24px rgba(15, 23, 42, .08)` plus a faint inner/edge highlight.
- Maintain semantic separation: blue is data/selection, teal is product chrome, green is success. Do not flatten all three into one generic accent.

## 3. Typography

- **Font family estimate:** a modern UI grotesk such as Inter, SF Pro, or a close variable sans. Letter shapes are not sufficient to identify the exact family; confidence low-to-medium. Devanagari fallback must support `मौसम सेतु` cleanly.
- Main page title “Overview”: approximately `24 px`, `600–700`, line height `30 px`.
- Brand “MausamSetu”: approximately `22 px`, `650–700`; tagline `11–12 px`, `400`, `15 px` line height.
- Card section titles: `15–16 px`, `600`; KPI eyebrow labels: `12–13 px`, uppercase, `500–600`, slight tracking.
- KPI numerals: approximately `45–48 px`, `400–500`, compact line height near `1`; units are `16–18 px`, `500–600`, baseline aligned.
- Reservoir percentage inside ring: about `31 px`, `500`; chart labels and secondary metadata: `10–12 px`, `400–500`.
- Navigation and buttons: `13–14 px`, `450–550`; selected navigation is not dramatically heavier than unselected.
- Operational table values: `12–13 px`; metric numbers about `20 px`, `500`.
- Numeric behavior **inferred:** use tabular numerals for timestamps, metrics, chart axes, percentages, and pipeline counters. Preserve decimal precision exactly during Phase 1 (`32.4`, `0.52`, `8.2`, etc.).
- Text color hierarchy: primary near `#111827`; secondary around gray-600; tertiary around gray-500. Colored text is reserved for data semantics and links, not decorative headings.

## 4. Spacing rhythm

- Dominant spacing unit is approximately `8 px`; common increments are `4, 8, 12, 16, 20, 24, 32`.
- Outer shell inset: `12–16 px`. Main panel padding: `20–25 px`. Sidebar horizontal padding: about `18–20 px`.
- Card gaps: `18–19 px` horizontally in the KPI row; `10–14 px` within dense content regions; about `11–13 px` between stacked right-column cards.
- Card padding: generally `18–20 px`; dense rows use `10–14 px` vertical padding.
- Radii: shell `≈20 px`; standard card `≈12–14 px`; buttons/inputs `≈10–12 px`; avatar and status indicators fully circular.
- Standard control heights: header buttons `≈40 px`; search `≈61 px`; nav row `≈43 px`; segmented controls `≈31 px`; compact icon buttons `≈34–40 px`.
- Divider strokes are `1 px` with large reductions in opacity. Avoid repeated heavy boxes; hierarchy comes from spacing, tint, and subtle strokes.
- Sparkline cards reserve roughly `30 px` at the bottom; the map footer timeline reserves about `50 px`; KPI content aligns to a consistent baseline despite different visualization types.

## 5. Component inventory

- `GlassShell`: variants `sidebar`, `workspace`; backdrop-image-aware translucent surface.
- `BrandLockup`: droplet mark, product name, two-line descriptor; compact mark-only variant is needed by the map route.
- `GlobalSearch`: search icon, placeholder, keyboard shortcut suffix.
- `SidebarNavGroup` and `SidebarNavItem`: default, hover, selected-with-leading-rail; icon + label.
- `PilotSelector`: default and selected-dot states.
- `UserIdentityCard`: avatar initials, name, role, settings action.
- `LanguageBrandTile`: icon plus Devanagari label.
- `PageHeader`: title, basin/update metadata, action buttons, avatar menu.
- `HeaderActionButton`: icon + label; notification variant with badge; avatar/dropdown variant.
- `KpiCard`: variants `sparkline`, `ring`, `highlighted`; slots for eyebrow, value, unit, semantic delta, icon, mini-chart.
- `MetricSparkline`: blue, orange, and green variants; terminal dot and optional shaded fill.
- `ProgressRing`: central percentage plus external delta.
- `PanelCard`: standard title/header action/body/footer anatomy.
- `SegmentedControl`: rainfall/temperature/wind/LST; active tab; adjacent layers icon button.
- `BasinRainfallMap`: state/river outlines, clipped raster rainfall layer, labels, reservoir markers, legend, zoom/geolocate controls.
- `MapTimeScrubber`: play button, start/end labels, slider, selected date/today label.
- `ForecastLineChart`: observed/forecast line, confidence band, peak annotation, climatology reference, axis labels.
- `InsightBanner`: warning dot + sentence.
- `StatusList`: icon, category, status text, status dot; semantic variants stable/low-risk/moderate/elevated.
- `MetricMatrix`: labeled model scores with optional info tooltip and positive footer.
- `HealthTable`: label/value/status rows.
- `AlertList`: semantic dot, event/domain/value string, timestamp, view-all action.
- `PipelineStepper`: complete, active, and pending stages; cycle label.

## 6. Data visualization style

- KPI sparklines are thin (`≈1.5–2 px`), minimally gridded, and use a terminal circular marker. Color follows metric semantics: rainfall blue, temperature orange, forecast/model health green or blue-green.
- Reservoir level uses a thick circular progress ring: pale track, green progress, rounded caps, centered `87%`; ring diameter about `88 px`, stroke about `9–11 px`.
- Forecast chart occupies roughly `(1044,339)` to `(1484,504)`. It uses a bright blue smoothed line, four marked points, a broad pale-blue uncertainty band, faint horizontal gridlines, and a dotted/gray `18 mm` climatology reference. The maximum receives a floating white annotation.
- Phase 1 forecast values visible from the chart: peak `48 mm` on `Jul 19`; x-axis `Jul 16`, `17`, `18`, `19`, `20`, `21`, `22`; y-axis `0`, `15`, `30`, `45`, `60 mm`; reference `Climatology (18 mm)`.
- Model performance is a 3×2 metric matrix rather than a chart. Keep score abbreviations and precision unchanged until backed by a schema.
- Semantic status visualization combines icon, colored word, and dot; color is never the only signal.
- **Inferred implementation:** charts should support tooltips, keyboard focus, and exact values, but the screenshot only proves hover-style peak annotation and selectable date behavior—not chart library, interpolation method, or confidence calculation.
- Do not introduce gradients, 3D columns, oversized legends, or decorative gauges absent from the reference.

## 7. Map style

- The overview embeds a regional map inside a dashboard card; it is not a full-screen map route.
- Basemap is very pale, low-saturation, and labels only major states/water bodies. Thin cyan rivers and green administrative/basin boundaries remain visible beneath the rainfall layer.
- Rainfall appears as a gridded/raster heat layer concentrated in a northwest-to-southeast band, using hard-enough grid cells to communicate model resolution while retaining slight translucency.
- Basin boundary is green with a subtle dashed/segmented character; “Cauvery Basin” is a larger green two-line label near center-bottom.
- Reservoir markers are hollow teal/green rings with adjacent two-line labels and percentages; Mettur, KRS, and Kabini share one reusable marker pattern.
- Legend is a compact floating white card in the lower-left of the map, horizontal blue ramp, title `Rainfall (mm)`, ticks `0`, `25`, `50`, `75`, `100+`.
- Controls are compact white square buttons on the right: zoom in, zoom out, and geolocate/target.
- The map’s bottom date scrubber is integrated into the card, separated by a light divider. Keep the map clipping radius aligned with the enclosing card.
- **Inferred, medium confidence:** use a projected geographic layer with vector labels/lines and raster precipitation tiles. Do not infer provider, tile source, projection, or actual coordinates from this screenshot alone.

## 8. Interactions suggested by the static image

- Search opens region/metric discovery; `⌘K` suggests a keyboard command shortcut.
- Sidebar rows navigate routes; pilot basin rows switch active basin/context. Selected state uses both a blue/teal dot and emphasized text.
- Header actions suggest file export, share flow, notifications tray, and account menu. Badge `2` is dynamic.
- KPI cards may drill into metric detail; sparklines and the reservoir ring likely expose tooltips. This is **inferred**, not visibly proven.
- Metric segmented control swaps the map layer between Rainfall, Temperature, Wind, and LST; layers button likely opens layer configuration.
- Map supports pan/zoom, geolocation, reservoir selection, and rainfall-cell inspection. Only controls and markers are visible; exact click behavior is inferred.
- Map date control supports play/pause and dragging between `Jul 12` and `Jul 18`; `Jul 15 · Today` is the current selection.
- Forecast dropdown switches forecast variable; the peak annotation suggests hover/focus; `View all →` opens a complete alert feed.
- Info icons beside model performance/system health suggest tooltips explaining metrics or data freshness.
- Pipeline stages communicate complete/current/pending state; whether they are clickable is unknown and should not be assumed in Phase 1.
- Accessibility implication: every icon-only control needs a programmatic label; segmented controls need selected state; chart/map information needs a nonvisual data equivalent; status must retain text labels.

## 9. Copy / microcopy

All strings are transcribed as displayed and classified `[S]` static UI copy or `[D]` dynamic/placeholder data. Punctuation, capitalization, precision, and abbreviations are Phase 1 reference values.

- Brand/sidebar: `[S] MausamSetu`; `[S] AI-Powered Climate`; `[S] Digital Twin of India`; `[S] Search regions, metrics…`; `[S] ⌘K`; `[S] Workspace`; `[S] Overview`; `[S] Live Map`; `[S] Forecast`; `[S] Scenarios`; `[S] Validation`; `[S] Sectors`; `[S] Alerts`; `[S] Pilots`; `[D] Cauvery Basin`; `[D] Krishna Basin`; `[D] Godavari Basin`; `[D] BC`; `[D] Bhavya Chaudhary`; `[D] Climate Ops`; `[S] मौसम सेतु`.
- Header/actions: `[S] Overview`; `[D] Cauvery Basin`; `[D] Updated 09:34 IST`; `[S] Export`; `[S] Share`; `[S] Notifications`; `[D] 2`; `[D] BC`.
- KPI 1: `[S] BASIN RAINFALL 24H`; `[D] 32.4`; `[S] mm`; `[D] ↑ +78% vs normal`.
- KPI 2: `[S] PEAK TMAX`; `[D] 31.8`; `[S] °C`; `[D] → @ Tiruchirappalli`.
- KPI 3: `[S] RESERVOIR (METTUR)`; `[D] 87%`; `[D] ↑ +18%`.
- KPI 4: `[S] FORECAST PEAK`; `[D] 48`; `[S] mm`; `[D] → Jul 19`.
- KPI 5: `[S] MODEL SKILL`; `[D] 0.52`; `[S] CSI`; `[D] ↑ +29% vs baseline`.
- Live map header/controls: `[S] Live climate state`; `[D] Live`; `[S] Rainfall`; `[S] Temperature`; `[S] Wind`; `[S] LST`.
- Embedded map labels/data: `[D] KARNATAKA`; `[D] ANDHRA PRADESH`; `[D] KERALA`; `[D] TAMIL NADU`; `[D] Arabian Sea`; `[D] Bay of Bengal`; `[D] Cauvery Basin`; `[D] Kabini Reservoir`; `[D] 87%`; `[D] KRS Reservoir`; `[D] 82%`; `[D] Mettur Reservoir`; `[D] 87%`; `[S] Rainfall (mm)`; `[D] 0`; `[D] 25`; `[D] 50`; `[D] 75`; `[D] 100+`; `[D] Jul 12`; `[D] Jul 15 · Today`; `[D] Jul 18`.
- Forecast panel: `[S] 7-day forecast`; `[S] Rainfall`; `[S] mm`; `[D] 60`; `[D] 45`; `[D] 30`; `[D] 15`; `[D] 0`; `[D] Peak 48 mm · Jul 19`; `[D] Climatology (18 mm)`; `[D] Jul 16`; `[D] 17`; `[D] 18`; `[D] 19`; `[D] 20`; `[D] 21`; `[D] 22`; `[D] Monsoon surge probable Jul 18–20 · high confidence`.
- Sector status: `[S] Sector status`; `[S] Water security`; `[D] Stable`; `[S] Agriculture`; `[D] Low risk`; `[S] Heat stress`; `[D] Moderate`; `[S] Flood risk`; `[D] Elevated`.
- Model performance: `[S] Model performance`; `[S] RMSE`; `[D] 8.2`; `[S] mm`; `[S] POD`; `[D] 0.71`; `[S] FAR`; `[D] 0.31`; `[S] CSI`; `[D] 0.52`; `[S] ACC`; `[D] 0.68`; `[S] FSS`; `[D] 0.61`; `[D] Beats persistence by 29%`.
- System health: `[S] System health`; `[S] INSAT-3DR`; `[D] Online`; `[S] IMD grid feed`; `[D] Synced · 09:34`; `[S] MOSDAC`; `[D] Active`; `[S] Assimilation`; `[D] Cycle 46 · Complete`; `[S] Next cycle in`; `[D] 25 min 48 sec`; `[S] Latency`; `[D] 42 ms`.
- Recent alerts: `[S] Recent alerts`; `[S] View all →`; `[D] Heavy rain · Kodagu · >75mm expected`; `[D] 09:34`; `[D] Soil saturation · Kabini · 87%`; `[D] 09:12`; `[D] Reservoir advisory · Mettur · +40% inflow`; `[D] 08:47`.
- Pipeline: `[D] CYCLE 46`; `[S] INGEST`; `[S] REGRID`; `[S] ASSIMILATE`; `[S] FORECAST`; `[S] IMPACT`; `[S] RENDER`.

## 10. Anti-slop checks

- Reproduce the base two-shell dashboard composition; do not substitute a generic top-nav SaaS dashboard.
- Preserve five distinct KPI cards, their order, units, precision, semantic colors, and visualization forms. Do not turn every KPI into the same icon-and-number tile.
- Keep rainfall blue, heat orange, operational success green/teal, and elevated flood risk red. Do not use indiscriminate gradients or purple “AI” styling.
- Preserve the embedded map’s basin shape, rainfall corridor, reservoir markers, legend, and date scrubber. Do not use a random world map or placeholder chart.
- Keep glass effects restrained: readable white surfaces, subtle landscape bleed-through, thin borders, and low shadows. Avoid excessive blur, glow, neon, or floating-card inflation.
- Do not invent extra metrics, alerts, basins, controls, map providers, backend freshness guarantees, or hidden interactions.
- Keep all screenshot values verbatim as Phase 1 placeholders; do not “correct,” localize, recalculate, or claim they are live. Bind them to real data only after schema validation.
- Preserve information density and alignment: compact controls, modest radii, consistent card baselines, and a functional pipeline footer.
- Ensure status is conveyed by text/icon as well as color; provide focus states, keyboard operation, reduced-motion handling, chart alternatives, and minimum readable contrast.
- This decode covers only `screen-01-overview.png`; it makes no claims about Screens 8–10 or any unshown breakpoint/state.
