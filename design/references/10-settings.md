# Screen 10 — Settings

- **Route:** `/settings`
- **Status:** Synthesized specification — no dedicated reference screenshot
- **Visual basis:** Shared shell, form controls, toggles, selectors, and disclosure patterns from Screens 03, 04, 05, 06, and 07
- **Approval boundary:** Authorized synthesis from Screens 01–07; not a pixel-for-pixel reference decode. Visual review is required before implementation.
- **Data rule:** Proposed copy uses `[P]`. Existing saved values must be loaded from configuration/API rather than copied from this document.

## 1. Layout

Use the standard desktop shell and keep settings quieter than analytical screens.

Proposed 1536 × 1024 composition:
- Header: `x≈264–1508, y≈20–84`.
- Settings navigation: `x≈264–520, y≈94–926`.
- Active settings form: `x≈530–1110, y≈94–926`.
- Context/help and change summary: `x≈1120–1508, y≈94–926`.
- Six-stage footer: `x≈264–1508, y≈942–991`.

Proposed categories are General, Region & units, Alert defaults, Data & model, and Accessibility. The user requirement specifically covers units, region switching, and threshold defaults; additional categories organize existing system-level concerns without adding unrelated preferences. A sticky save bar appears only with unsaved changes.

## 2. Color system

Use the same light/translucent neutrals and semantic colors as all standard-shell screens. Blue indicates selected category, focused control, and primary save action. Green confirms saved state. Orange marks values diverging from recommended defaults. Red is reserved for invalid or safety-critical settings and destructive reset actions.

Do not use theme pickers or dark-theme previews in Phase 1; approved images are canonical and light.

## 3. Typography

Use the shared Inter-like stack:
- Page title: 22–24 px / 600–700.
- Settings section title: 16–18 px / 600.
- Field label: 12–14 px / 500–600.
- Helper/error text: 11–12 px.
- Numeric thresholds and units: tabular numerals.
- Category labels: 13–14 px / 500.

Labels remain sentence case. Every numeric setting displays its unit in the control and helper text.

## 4. Spacing rhythm

Use the common 8 px base:
- Workspace/card gaps: 10–12 px.
- Panel padding: 20–24 px.
- Form group spacing: 24–32 px.
- Label-to-control: 8 px.
- Helper text: 6–8 px below control.
- Control height: 38–42 px.
- Settings navigation row: 42–46 px.
- Card radius: 12 px; control radius: 8 px; border: 1 px.
- Dividers separate groups; avoid wrapping every field in its own card.

## 5. Component inventory

- `AppSidebar`, `PageHeader`, `HeaderActions`, `PipelineFooter`.
- `SettingsNav`, `SettingsSection`, `SettingsGroup`.
- `UnitSelect` for rainfall, temperature, wind, and storage/flow presentation.
- `RegionSelect` with Cauvery as the current pilot and availability status for other basins.
- `TimezoneField` fixed to IST in this phase.
- `ThresholdField` with value, unit, severity, persistence/window, and scope.
- `ThresholdPreset · recommended | custom`.
- `ToggleField`, `SelectField`, `NumberField`, `ResetSectionButton`.
- `ChangeSummary`, `UnsavedChangesBar`, `SaveSettingsButton`, `DiscardButton`.
- Validation error, save error, permissions/read-only, and settings-loaded states.

## 6. Data visualization style

Settings is form-led. Do not add decorative charts. Alert-default controls may include a compact threshold preview that shows labeled severity bands and a sample value; it must be explanatory, not a live claim. A change-summary diff uses old value → new value with units and labels.

## 7. Map style

No permanent map is required. Region selection may show a small static basin thumbnail using the approved pale basemap, green basin outline, and minimal labels. It must not become an interactive competing workspace. If no thumbnail materially aids basin choice, omit it.

## 8. Interactions suggested by the established system

- Category selection changes the form while preserving unsaved changes.
- Region switching previews downstream effects and requires confirmation because it changes dashboard scope.
- Unit changes preview formatted examples before saving but never alter stored scientific data.
- Threshold defaults validate ordered severity levels, units, ranges, persistence windows, and geographic scope.
- Saving threshold changes records actor, timestamp, previous values, and new values.
- Reset restores documented recommended defaults and requires confirmation if custom thresholds exist.
- IST is visible and non-editable unless the product scope changes.
- Save/discard controls appear only when the form is dirty; navigation warns before losing edits.
- Role-restricted settings explain required permission instead of silently disabling controls.

## 9. Copy / microcopy

All strings are proposed:
- `[P] Settings`
- `[P] Configure units, pilot region, and operational defaults.`
- `[P] General`; `[P] Region & units`; `[P] Alert defaults`; `[P] Data & model`; `[P] Accessibility`
- `[P] Pilot region`; `[P] Cauvery Basin`; `[P] Region changes update maps, forecasts, scenarios, and alerts.`
- `[P] Timezone`; `[P] India Standard Time (IST)`; `[P] Fixed for this deployment`
- `[P] Rainfall unit`; `[P] Temperature unit`; `[P] Wind unit`; `[P] Storage and flow units`
- `[P] Alert thresholds`; `[P] Recommended defaults`; `[P] Custom`
- `[P] Advisory`; `[P] Elevated`; `[P] Severe`; `[P] Critical`
- `[P] Persistence window`; `[P] Geographic scope`; `[P] Preview alert level`
- `[P] Reset section`; `[P] Discard changes`; `[P] Save settings`
- `[P] Unsaved changes`; `[P] Settings saved`; `[P] Could not save settings`
- `[P] This change will be recorded in the audit log.`

Actual region availability, units, threshold values, user permissions, and saved state are dynamic.

## 10. Anti-slop checks

- Never claim a dedicated screenshot exists for Settings.
- Do not invent personalization, billing, team management, or unrelated account settings.
- Preserve the approved light visual direction; no theme switcher in this phase.
- Do not create a card for every field or use oversized toggles.
- Units affect presentation only; they must not mutate stored source data.
- IST remains explicit and fixed.
- Thresholds require units, ordered severity, scope, persistence, validation, and audit history.
- Do not place threshold configuration permanently in the Alerts incident surface; Settings owns defaults, while Alerts may expose a provisional context-specific shortcut.
- Require visual review before treating the synthesized layout as locked.
