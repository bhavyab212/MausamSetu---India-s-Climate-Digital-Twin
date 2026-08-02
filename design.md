# MausamSetu — Reference-Locked Design System

> **Status:** Phase 1 design extraction complete; implementation not started  
> **Last updated:** 2026-07-25  
> **Canonical visual source:** approved images in `C:\Users\bhavy\Documents\v1\images\`  
> **Brand:** MausamSetu मौसम सेतु

## 1. Design direction

The approved references establish a **light, translucent climate-operations interface**, not the earlier dark command-center placeholder. The visual character is analytical, calm, geographic, and operational:

- off-white workspace over a restrained blurred environmental backdrop;
- near-white translucent cards with fine borders and low shadows;
- one dominant action/data blue;
- green, amber/orange, and crimson used only for explicit semantics;
- compact information density, strong alignment, and visible units;
- maps and evidence receive more space than decorative summary cards;
- no neon glow, rainbow palettes, cartoon weather graphics, or consumer-weather styling.

The images are canonical where they conflict with earlier placeholder guidance. Exact screenshot values are design placeholders during Phase 1 and do not override scientific or backend rules when real data is connected.

## 2. Evidence and confidence

### Direct reference decodes

| Screen | Route | Source | Decode |
|---|---|---|---|
| Overview | `/` | `screen-01-overview.png` · 1536×1024 | [`design/references/01-overview.md`](design/references/01-overview.md) |
| Map View | `/map` | `screen-02-map.png` · 1536×1024 | [`design/references/02-map.md`](design/references/02-map.md) |
| Forecast Detail | `/forecast` | `screen-03-forecast.png` · 1536×1024 | [`design/references/03-forecast.md`](design/references/03-forecast.md) |
| What-If Studio | `/scenarios` | `screen-04-scenarios.png` · 1536×1024 | [`design/references/04-scenarios.md`](design/references/04-scenarios.md) |
| Sector Impact Chain | `/impacts` | `screen-05-impacts.png` · 1659×948 | [`design/references/05-impacts.md`](design/references/05-impacts.md) |
| Model Validation | `/validation` | `screen-06-validation.png` · 1659×948 | [`design/references/06-validation.md`](design/references/06-validation.md) |
| Alerts & Thresholds | `/alerts` | `screen-07-alerts.png` · 1536×1024 | [`design/references/07-alerts.md`](design/references/07-alerts.md) |

Each direct decode separates pixel samples from inferred reusable tokens and marks copy as static `[S]` or dynamic `[D]`. Pixel samples are exact screenshot pixels, but transparency, antialiasing, blur, and imagery mean they are not automatically CSS tokens.

### Synthesized specifications

Screens without dedicated images were explicitly authorized to extend the visual language of Screens 1–7:

| Screen | Route | Specification |
|---|---|---|
| Data Explorer | `/data` | [`design/references/08-data-explorer.md`](design/references/08-data-explorer.md) |
| Reports & Export | `/reports` | [`design/references/09-reports.md`](design/references/09-reports.md) |
| Settings | `/settings` | [`design/references/10-settings.md`](design/references/10-settings.md) |

These are marked **Synthesized specification** and use `[P]` for proposed copy. They require visual review before implementation can be called pixel-locked.

## 3. Canonical tokens

These values are the implementation candidates for Phase 2. They consolidate repeated evidence rather than averaging every raster pixel.

### 3.1 Colors

```css
/* surfaces */
--color-background:       #F7F8FA;
--color-workspace:        rgba(249, 250, 250, 0.92);
--color-card:             rgba(255, 255, 255, 0.88);
--color-card-solid:       #FEFFFE;
--color-card-muted:       #F1F6FA;
--color-border:           #E1E4E8;
--color-divider:          #D8DCE2;

/* text */
--color-text-primary:     #111318;
--color-text-secondary:   #56606B;
--color-text-tertiary:    #7A8390;
--color-text-inverse:     #FFFFFF;

/* semantic */
--color-primary:          #0061FE;
--color-primary-strong:   #0854FE;
--color-primary-soft:     #EAF2FE;
--color-positive:         #07945B;
--color-positive-soft:    #E5FCED;
--color-warning:          #F57602;
--color-warning-soft:     #FEF7EA;
--color-critical:         #ED3335;
--color-critical-strong:  #EC3736;
--color-critical-soft:    #FDEBEC;

/* climate visualization */
--color-rain-0:           #DDEBFF;
--color-rain-25:          #ABD2FF;
--color-rain-50:          #8FBEFD;
--color-rain-75:          #3D91FB;
--color-rain-100:         #0061FE;
--color-temperature-low:  #FEF3C7;
--color-temperature-high: #B91C1C;
--color-anomaly-negative: #E56604;
--color-anomaly-neutral:  #FFFFFF;
--color-anomaly-positive: #143CB9;
--color-missing:          #E5E7EB;
```

**Provenance:** Backgrounds and near-white card surfaces repeat across all seven direct references. `#0061FE` is directly sampled in Scenarios and recurs within the observed blue range of every screen. Green is sampled near `#058A54`, `#07945B`, and `#189257`; `#07945B` is the canonical shared token, while individual raster samples remain documented in their screen decodes. Warning and critical values are anchored by Validation and Alerts. The climate palettes preserve the project’s scientific color rules.

**Opacity:** Source opacity cannot be recovered exactly from a flattened PNG. `0.88` for cards and `0.92` for the workspace are implementation starting points inferred from the references, not measured alpha values. They must be checked by screenshot comparison in Phase 2.

### 3.2 Radius, spacing, borders, and shadows

```css
--radius-card:            12px;
--radius-panel:           16px;
--radius-shell:           20px;
--radius-control:         8px;
--radius-button:          8px;
--radius-pill:            9999px;

--spacing-card-padding:   16px;
--spacing-card-padding-lg:20px;
--spacing-row-gap:        12px;
--spacing-column-gap:     12px;
--spacing-workspace-inset:16px;

--border-card:            1px solid #E1E4E8;
--shadow-card:            0 8px 24px rgba(20, 30, 45, 0.08);
--shadow-floating:        0 12px 32px rgba(20, 30, 45, 0.12);
```

The references use an 8 px rhythm with common 8, 12, 16, 20, 24, and 32 px increments. Cards consistently fall near 10–14 px radius; 12 px is canonical. Pills are reserved for statuses and compact filters—not generic cards or every button.

### 3.3 Typography

The screenshot font is an Inter-like neutral neo-grotesk. Exact font identification is not possible from raster images, so Phase 2 should begin with **Inter** and verify letter shapes side-by-side.

```text
typography.hero    44–48 px / 500–600 / tight line height / tabular numerals
typography.page    22–24 px / 650 / 1.2
typography.title   15–17 px / 600 / 1.3
typography.label   12–14 px / 500–600 / 1.35
typography.body    12–14 px / 400 / 1.5
typography.caption 10–12 px / 400–500 / 1.4
typography.eyebrow 10–11 px / 600 / 0.06em tracking
typography.data    tabular numerals; units remain attached
```

Sentence case is standard. Uppercase is limited to process steps, state labels, alert eyebrows, and compact data labels. Technical coordinates and identifiers may use a mono fallback; body copy must not.

## 4. Shared shells

### 4.1 Standard dashboard shell

Screens 01 and 03–07 establish the standard shell:

- translucent left sidebar: approximately 224–240 px;
- 8–16 px gap between sidebar and workspace;
- large translucent workspace filling the remaining desktop viewport;
- page header approximately 64–76 px high;
- card gutters approximately 10–16 px;
- bottom six-stage pipeline strip approximately 41–54 px high.

Canvas differences (1536×1024 versus 1659×948) change available density, not the base visual language.

### 4.2 Full-map shell override

Screen 02 is an explicit route override:

- approximately 69 px icon rail;
- approximately 70 px top command bar;
- map bleeds through the remaining viewport;
- controls are floating panels over the map;
- no standard 240 px sidebar or card-grid workspace.

Do not average this exception into the standard shell.

## 5. Shared component vocabulary

- `AppSidebar` — product lockup, command search, workspace navigation, pilots, profile, Hindi mark.
- `CompactMapRail` — Map-only icon navigation.
- `PageHeader` — title, context/freshness metadata, Export, Share, Notifications, avatar.
- `GlassCard` — standard analytical surface.
- `MetricTile · sparkline | gauge | delta | confidence`.
- `StatusPill · live | positive | warning | critical | neutral`.
- `SegmentedControl` — metrics, map layers, comparison modes.
- `InlineLegend` — labels and units adjacent to the visualization.
- `BasinMap` — pale editorial basemap, green basin boundary, cyan rivers.
- `GridCellOverlay` — discrete 0.25° cells; never falsely smoothed.
- `CellInspector` — current values, forecast, anomaly, actions.
- `TimelineDock` — observed/forecast distinction, current-time marker, playback.
- `UncertaintyChart` — p50 line plus p10–p90 band.
- `SpatialSmallMultiples` — synchronized daily basin maps.
- `EnsembleSpreadPlot` — low-opacity members plus central estimate/distribution.
- `SkillComparison` — MausamSetu vs Persistence vs Climatology.
- `ScenarioSlider`, `ScenarioPreset`, `ScenarioRunState`.
- `SectorTab`, `ImpactMetricCard`, `AdvisoryCard`, `DecisionLog`.
- `AlertRow`, `IncidentBanner`, `ResponseRoster`, `BroadcastComposer`.
- `PipelineFooter` — route-specific six-stage lifecycle state.

Components must preserve the reference density. “Reusable” does not mean every panel gets identical proportions.

## 6. Data visualization rules

- Rainfall: sequential light-to-deep blue.
- Temperature: sequential yellow-to-red.
- Signed anomaly/bias: diverging scale centered explicitly at zero.
- Uncertainty: p10–p90 translucent band plus solid p50 line; never a bare point forecast.
- Model claims: always show MausamSetu, Persistence, and Climatology.
- Gridlines: thin, low-contrast gray.
- Axes: compact 10–12 px labels with units.
- Direct labels and annotations are preferred to detached legends.
- Color is always paired with labels, signs, icons, checks/crosses, patterns, or line styles.
- Missing values use neutral hatch plus a “Missing” label; never black or silent omission.
- No rainbow/jet, 3D charts, ornamental gradients, unlabeled sparklines as evidence, or smoothing that hides the 0.25° grid.

## 7. Map rules

- Pale editorial/administrative basemap with low visual competition.
- Near-white land, pale-blue water, thin cyan rivers, restrained roads and boundaries.
- Cauvery basin boundary in green.
- Rainfall rendered as visible model cells or softened raster that does not imply finer native resolution.
- Legends always include variable and unit.
- City/state labels remain legible but secondary to data.
- Selected cells use a blue outline and remain spatially connected to their inspector.
- Comparison maps share extent, scale, and cursor state.
- Replay states must be unmistakably historical, never visually confused with Live.

## 8. Motion and interaction

Motion is restrained and functional:

- page transition: 180–220 ms ease-out, subtle fade/translate;
- card/control state: 120–160 ms;
- map data crossfade: approximately 250–300 ms;
- no bounce, elastic motion, particles, neon pulses, or decorative parallax;
- live pulse is allowed only when paired with explicit “Live” text;
- all icon-only actions require labels and visible focus;
- minimum effective target: 44×44 px;
- charts, maps, sliders, timelines, and grid cells must support keyboard access;
- high-consequence actions such as escalation or broadcast require confirmation and audit records.

Focus uses a 2 px primary-blue ring with sufficient offset and contrast on light surfaces.

## 9. Screen-specific overrides

### `/` — Overview

Dominant asymmetric live map, five KPI tiles, compact forecast, sector status, performance, health, alerts, and a six-stage `INGEST → REGRID → ASSIMILATE → FORECAST → IMPACT → RENDER` footer.

### `/map` — Map View

Full-screen map shell with narrow rail, floating layers and cell inspector, wind streamlines, synoptic track, timeline dock, and assimilation status.

### `/forecast` — Forecast Detail

Full-width uncertainty chart, seven spatial small multiples, ensemble spread, monsoon pulse, attribution, and forecast-specific skill panel.

### `/scenarios` — What-If Studio

Three-column layout: configuration, spatial/runoff evidence, and downstream impacts. Key-finding card uses a warning-soft surface and orange rule.

### `/impacts` — Sector Impact Chain

The reference directly defines the **Water security** tab shell. Shared header, sector selector, advisories, decision log, and pipeline remain stable when other sector content is later designed. Hidden Heat & Health and ₹ Risk content is not pixel-decoded.

### `/validation` — Model Validation

Evidence-dense layout with scatter, threshold skill, matched spatial-error maps, baselines, stress tests, and persistent known limitations. “Passing” must link to criteria, not imply perfection.

### `/alerts` — Alerts & Thresholds

A single dominant critical-red incident banner is the screen exception. Red must not spread to ordinary controls. Threshold configuration is a provisional secondary control near map layers or in Settings, never a permanent competitor to incident response.

### `/data`, `/reports`, `/settings`

These use the standard shell and existing tokens/components. Their exact layouts are synthesized and require user review before they are considered visually locked.

## 10. Accessibility and content integrity

- WCAG AA text contrast target: at least 4.5:1.
- No color-only state communication.
- Every number displays a unit unless dimensionless and clearly named.
- IST is explicit for operational timestamps.
- English + Devanagari branding is retained.
- Dynamic values must be API/model-derived in implementation.
- Phase 1 screenshot values may be reproduced for visual matching only and must be tagged in fixtures/mocks so they cannot be mistaken for verified outputs.
- Loading, empty, error, unavailable-source, partial-data, stale-data, and permission states are required for every screen, although their exact appearance is inferred because static references do not show them.

## 11. Phase boundary

Phase 1 documentation is complete. Do **not** encode these tokens in `web/tailwind.config.ts`, scaffold Next.js, move Streamlit files, or build application components until Phase 2 is explicitly authorized. The first visual implementation gate is the shared kitchen-sink/design-system view, followed by an Overview side-by-side comparison before any other production screen.