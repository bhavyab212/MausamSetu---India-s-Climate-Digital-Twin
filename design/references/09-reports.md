# Screen 09 — Reports & Export

- **Route:** `/reports`
- **Status:** Synthesized specification — no dedicated reference screenshot
- **Visual basis:** Shared shell and card language from Screens 01, 03, 04, 05, 06, and 07; export actions and evidence disclosure from Screens 03–07
- **Approval boundary:** The user authorized synthesis from Screens 01–07. This is not a screenshot decode and requires visual review before implementation.
- **Data rule:** Proposed copy uses `[P]`. Generated reports must use real API/model values in the implementation phase.

## 1. Layout

Use the standard desktop shell with an approximately 240 px sidebar and light translucent workspace.

Proposed 1536 × 1024 composition:
- Header: `x≈264–1508, y≈20–84`.
- Report-template row: `x≈264–1508, y≈94–274`, three equal cards.
- Recent reports: `x≈264–930, y≈284–926`.
- Report configuration/preview: `x≈940–1508, y≈284–926`.
- Six-stage cycle footer: `x≈264–1508, y≈942–991`.

The three templates are Weekly Basin, Scenario Analysis, and Impact Assessment. Selecting one updates the right configuration panel; the recent-report list remains visible for operational continuity. Preview is a document thumbnail/outline, not a fake full PDF rendered at unreadable scale.

## 2. Color system

Inherit the canonical tokens: off-white workspace, near-white cards, `#E1E4E8` borders, near-black text, `#0061FE` primary action, `#07945B` completed, `#F57602` warning, and `#ED3335` failed/critical.

Template cards may use domain icons and a 3 px semantic top rule but must not introduce new gradients. Report status always combines color with text and icon: Draft, Generating, Ready, Failed, or Expired.

## 3. Typography

Use the shared Inter-like stack:
- Page title: 22–24 px / 600–700.
- Template title: 16–18 px / 600.
- Generated report title: 14–16 px / 600.
- Metadata/body: 12–14 px.
- Dates, file sizes, page counts, and IDs: 11–12 px tabular numerals.
- Button text: 12–13 px / 600.

Document hierarchy appears in the preview through restrained title/subtitle/body scaling. Do not simulate a different brand inside reports.

## 4. Spacing rhythm

Use the 8 px base system:
- Workspace inset: 16 px.
- Template gaps: 12 px.
- Card padding: 16–20 px.
- Recent-report row: 56–64 px.
- Form group gap: 20–24 px.
- Control height: 36–40 px.
- Card radius: 12 px; button/control radius: 8 px; status pill only: full.
- Border: 1 px; shadow restrained and consistent with other screens.

## 5. Component inventory

- `AppSidebar`, `PageHeader`, `HeaderActions`, `PipelineFooter`.
- `ReportTemplateCard · weekly | scenario | impact`.
- `ReportStatusPill · draft | generating | ready | failed | expired`.
- `RecentReportRow` with report ID, title, scope, created time, author, size, and actions.
- `ReportConfigurationPanel`.
- `SectionChecklist` for report contents.
- `DateRangeControl`, `BasinSelect`, `ScenarioSelect`, `UncertaintyToggle`, `BaselineToggle`.
- `ReportOutlinePreview`.
- `GenerateReportButton`, `DownloadButton`, `ShareButton`, `DeleteDraftButton`.
- Export dialog with PDF as the primary format and explicit inclusion options.
- Generating progress, no-reports empty state, failed generation, stale inputs, and unavailable scenario states.

## 6. Data visualization style

This screen previews report structure rather than introducing novel analytics. Embedded report charts reuse source-screen grammar:
- p10–p90 band plus p50 for forecasts.
- Ours vs Persistence vs Climatology for model claims.
- Sequential blue rainfall maps, yellow-to-red temperature maps, diverging anomaly maps.
- Direct labels, units, legends, and source notes.

Preview thumbnails must not omit uncertainty or baselines merely to look cleaner. The full generated PDF must include provenance, generation time in IST, and model/data versions.

## 7. Map style

Any preview map reuses the pale editorial Cauvery basemap, green basin boundary, cyan rivers, restrained labels, and clipped 0.25° data grid. Do not create a new print-only cartographic language. In the PDF, legends and units remain legible at print scale and are not overlaid on critical data.

No dedicated screenshot exists; preview size and exact print layout are provisional.

## 8. Interactions suggested by the established system

- Selecting a template applies a recommended section set while preserving editable options.
- Configuration changes update the outline and a clear report-input summary.
- Generation is an explicit action with estimated runtime and input snapshot.
- A generated report is immutable; changes create a new version rather than silently replacing the old file.
- Download and Share retain report ID/version and audit entries.
- Delete applies to drafts only by default and requires confirmation.
- Failed generation exposes the failed stage and retry action.
- Keyboard and screen-reader users can inspect section ordering, status, and progress.

## 9. Copy / microcopy

All strings are proposed:
- `[P] Reports & Export`
- `[P] Generate decision-ready reports from the current basin state.`
- `[P] Weekly Basin Report`; `[P] Scenario Analysis`; `[P] Impact Assessment`
- `[P] Forecast, uncertainty, alerts, and sector status.`
- `[P] Baseline, scenario, delta, assumptions, and downstream effects.`
- `[P] Hydrology, heat and health, and ₹ risk evidence.`
- `[P] Recent reports`; `[P] View all`
- `[P] Configure report`; `[P] Report outline`
- `[P] Basin`; `[P] Date range`; `[P] Scenario`; `[P] Sections`
- `[P] Include p10/p50/p90 uncertainty`; `[P] Include persistence and climatology baselines`; `[P] Include provenance and limitations`
- `[P] Generate PDF`; `[P] Est. runtime`; `[P] Generating`; `[P] Ready`; `[P] Failed`
- `[P] Download`; `[P] Share`; `[P] Duplicate`; `[P] Delete draft`
- `[P] No reports yet. Select a template to create the first report.`

Report IDs, timestamps, authors, versions, sizes, durations, basin/scenario names, and status are dynamic.

## 10. Anti-slop checks

- Never represent this as a pixel-decoded screen.
- Keep exactly the three required templates; do not invent a gallery of decorative exports.
- Reports must include uncertainty, units, provenance, limitations, and baselines where claims are made.
- Do not render tiny illegible fake PDF pages as the primary interaction.
- Do not introduce gradients, colored document covers, or unrelated illustration styles.
- Do not allow regeneration to overwrite an auditable report version silently.
- Preserve IST timestamps and English + Devanagari branding in generated artifacts.
- PDF implementation must not modify the protected presentation slides 1, 2, or 10.
- Require user visual review before this synthesized screen is considered locked.
