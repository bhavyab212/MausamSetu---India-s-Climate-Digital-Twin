# Screen 08 — Data Explorer

- **Route:** `/data`
- **Status:** Synthesized specification — no dedicated reference screenshot
- **Visual basis:** Shared shell and visual language decoded from Screens 01, 03, 04, 05, 06, and 07; map treatments from Screens 01, 02, 04, 06, and 07
- **Approval boundary:** The user authorized this screen to be derived from Screens 01–07. It is not a pixel-for-pixel decode and must be reviewed before implementation.
- **Data rule:** Copy below is proposed (`[P]`). Values shown in implementation must come from provenance metadata and the real dataset/API.

## 1. Layout

Use the standard desktop dashboard shell: approximately 240 px left navigation, 16 px shell gap, and a translucent main workspace with 16–24 px internal inset. Do not use the Map screen’s 69 px rail; that is a route-specific override.

Proposed 1536 × 1024 composition:
- Header: `x≈264–1508, y≈20–84`.
- Variable/provenance summary strip: `x≈264–1508, y≈94–176`, four compact cells.
- Variable browser: `x≈264–530, y≈186–926`.
- Data preview: `x≈540–1110, y≈186–622`.
- Provenance and quality: `x≈1120–1508, y≈186–622`.
- Time-series/table preview: `x≈540–1508, y≈632–926`.
- Six-stage cycle footer: `x≈264–1508, y≈942–991`.

The browser remains a stable left column. Selecting a variable updates preview, provenance, and table without navigating away. This is an inferred arrangement, chosen because it reuses the Scenarios three-column density and Validation’s evidence-first hierarchy.

## 2. Color system

Inherit the canonical light/translucent system:
- Workspace/background: `#F7F8FA`.
- Card: `#FEFFFE` / `rgba(255,255,255,0.88)`.
- Border: `#E1E4E8`.
- Primary text: `#111318`.
- Secondary text: `#56606B`.
- Tertiary text: `#7A8390`.
- Primary/action blue: `#0061FE`.
- Positive: `#07945B`.
- Warning: `#F57602`.
- Critical: `#ED3335`.

Variable swatches follow domain rules: sequential blue for rainfall, yellow-to-red for temperature, and blue–white–red for signed anomalies. Missing cells use a neutral hatch plus “Missing”; never black fill or color alone.

## 3. Typography

Use the shared Inter-like neo-grotesk stack pending font confirmation:
- Page title: 22–24 px / 600–700.
- Summary values: 22–28 px / 600.
- Panel titles: 15–17 px / 600.
- Browser rows and body: 12–14 px / 400–500.
- Coordinates, dimensions, timestamps, filenames, and schema fields: 11–12 px tabular numerals; a mono fallback may be used only for technical values.
- Eyebrows: 10–11 px / 600 with restrained tracking.

Sentence case is the default. Units remain attached to every numeric value.

## 4. Spacing rhythm

Use the common 8 px base rhythm:
- Workspace inset: 16 px.
- Card gaps: 10–12 px.
- Card padding: 16–20 px.
- Browser row: 40–44 px.
- Filter/control height: 36–40 px.
- Card radius: 12 px.
- Control radius: 8 px.
- Pill radius: 9999 px only for true status/filter chips.
- Border: 1 px.
- Shadow: restrained `0 8px 24px rgba(20,30,45,0.08)`; no glow.

## 5. Component inventory

- `AppSidebar`, `PageHeader`, `HeaderActions`, `PipelineFooter`.
- `DatasetSummaryStrip` with date range, grid, variables, and source state.
- `VariableBrowser` with search, domain groups, variable rows, units, and availability.
- `VariableRow · selected | available | partial | unavailable`.
- `PreviewModeTabs · map | chart | table`.
- `SpatialPreview` reusing basin map, legend, timeline, and cell inspection patterns.
- `ProvenanceCard` with source, processing lineage, spatial/temporal resolution, update time, and license.
- `QualityCard` with missingness, range checks, stale status, and warnings.
- `SchemaTable`, `DataTable`, `DateRangeControl`, `ExportMenu`.
- `FormatChoice · CSV | JSON` and an export confirmation dialog.
- Loading skeleton, empty selection, unavailable variable, partial-data warning, and export-failure states.

## 6. Data visualization style

Default to a spatial preview for gridded variables, a line/area chart for a selected cell or basin aggregate, and a compact table for raw values. Reuse the Map screen’s discrete 0.25° cells rather than a smoothed heat blob. Every legend states variable and unit. Tooltips include timestamp, grid cell, value, missing/quality state, and source.

Quality is shown with labels and counts, not decorative gauges. Missingness may use a small bar or grid, but must report both count and percentage. Avoid visualizations that imply interpolation where none exists.

## 7. Map style

Reuse the pale editorial basemap, green Cauvery boundary, cyan rivers, restrained place labels, and variable-specific raster clipped to the 19 × 17 master grid. Default extent is the Cauvery basin. Map controls remain compact and secondary. A selected cell uses the blue outline/handle treatment from Screen 02 and synchronizes with the table and time series.

No dedicated screenshot exists; exact map panel dimensions, label density, and control placement are provisional.

## 8. Interactions suggested by the established system

- Search/filter variables by name, source, domain, and availability.
- Selecting a variable updates all evidence panels atomically.
- Preview tabs retain variable, date, aggregation, and selected cell.
- Timeline supports keyboard stepping and valid-date snapping.
- Provenance steps disclose input source, transformations, regridding, masking, and derived-variable formulas.
- Export requires variable, date range, spatial scope, format, estimated row count, and file size before confirmation.
- CSV/JSON exports include units, coordinate reference, missing-value representation, and provenance metadata.
- Disabled/unavailable variables explain why and what source is pending.

## 9. Copy / microcopy

All strings are proposed, not transcribed:
- `[P] Data Explorer`
- `[P] Cauvery Basin · 0.25° daily grid`
- `[P] Export data`
- `[P] Variables`
- `[P] Search variables…`
- `[P] Rainfall`; `[P] Maximum temperature`; `[P] Minimum temperature`; `[P] INSAT LST`; `[P] Rainfall anomaly`; `[P] Climatology`
- `[P] Spatial preview`; `[P] Time series`; `[P] Data table`
- `[P] Provenance`; `[P] Source`; `[P] Processing`; `[P] Resolution`; `[P] Coverage`; `[P] Updated`
- `[P] Data quality`; `[P] Complete`; `[P] Partial`; `[P] Source unavailable`; `[P] Missing cells`
- `[P] Date range`; `[P] Spatial scope`; `[P] Selected cell`; `[P] Entire basin`
- `[P] CSV`; `[P] JSON`; `[P] Include provenance metadata`; `[P] Prepare export`
- `[P] Select a variable to inspect its spatial field, history, and provenance.`

Real dates, counts, availability, values, sources, and units are dynamic and must not be hard-coded.

## 10. Anti-slop checks

- Never claim this screen has a dedicated approved screenshot.
- Do not invent new colors, radii, shadows, navigation, or chart grammar.
- Preserve the standard shell; do not copy the Map route’s narrow rail.
- Expose provenance and quality beside the preview, not behind an obscure modal.
- Do not smooth the 0.25° grid or hide missing cells.
- Every number has a unit; every export has scope and provenance.
- No rainbow palette, generic database icon wall, oversized empty hero, or decorative KPI cards.
- Mark implementation screenshots for user review before considering this screen visually locked.
