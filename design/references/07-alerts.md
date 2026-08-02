---
route: /alerts
source_path: C:\Users\bhavy\Documents\v1\images\screen-07-alerts.png
canvas: 1536 × 1024 px
status: Direct reference decode
confidence_caveat: High confidence for visible hierarchy, copy, incident state, relative geometry, and map/chart forms; medium for sampled colors, type metrics, spacing, radii, and shadows because the source is raster; low for unseen workflows, threshold controls, and responsive behavior.
placeholder_data_warning: All incident names, dates, times, places, populations, forecasts, thresholds, agencies, statuses, message text, reach counts, and operational actions below are screenshot values preserved verbatim as placeholders, not live warnings or validated emergency guidance.
---

## 1. Layout

**Observed [high confidence].** Desktop incident-command workspace at 1536 × 1024. A persistent left rail occupies approximately x=10–237 (227 px); the main white workspace spans approximately x=246–1524 (1,278 px). Both surfaces have 12–16 px corners over a blurred pale-green landscape backdrop.

- Sidebar: brand, search, `Workspace` navigation, `Pilots`, signed-in user, and Hindi product mark. `Alerts` is selected with a pale-blue row, blue left rule, and red count badge.
- Main header: title and three-part status summary at left; `Export`, `Share`, `Notifications`, avatar, and chevron at right.
- Incident command banner: approximately x=258–1511, y=85–185. Red left edge and outline establish the active severe state. Incident identity occupies the left half, three impact metrics occupy the middle, and two incident actions stack at right.
- Incident metadata line: a compact unboxed row beneath the banner.
- Core workspace: two columns from about y=222–743. Left is ~643 px wide; right is ~594 px wide, separated by ~12 px.
  - Left upper: `Incident zone` map, about 643 × 326 px.
  - Left lower: `Incident timeline`, about 643 × 190 px.
  - Right upper: `Live alert feed`, about 594 × 292 px.
  - Right lower: `Response coordination`, about 594 × 217 px.
- Lower workspace from about y=751–958: left `Outgoing broadcast · draft` combines a message editor and distribution list; right `Decision log` contains a filter row and horizontal event timeline.
- Bottom cycle strip: approximately 47 px high; `ALERT` is current and red while prior stages are complete.

**Inferred implementation [medium confidence].** Use a fixed desktop shell, 12-column grid, 10–14 px card gutters, and 14–18 px workspace padding. Incident response, map awareness, live feed, coordination, and outbound communication dominate the hierarchy. No responsive layout is evidenced.

## 2. Color system

Raster antialiasing creates many near-white variants. Values below are approximate visual samples rather than extracted production tokens.

- **Near-exact visible samples [medium]:** surfaces `#FFFFFF` / `#FEFEFE`; primary blue around `#0877F9`; incident/severe red around `#F51B24`; elevated/warning orange around `#FF8500`; live/success green around `#009E67`; neutral gray around `#6B7280`; deep navy text around `#101522`; borders around `#E5E7EB`.
- **Incident hierarchy [high]:** red is reserved for active/severe/escalation/current-alert emphasis; orange for elevated/pending/onset; blue for advisory/active response/navigation; green for live, notified, acknowledged, and completed; gray for informational/history.
- Rainfall raster uses a sequential pale cyan → medium blue → saturated navy scale labeled `0`, `25`, `50`, `75`, `100+`.
- Broadcast draft uses a pale cream/yellow surface; selected navigation uses pale blue; agency status chips use pale semantic tints.
- **Inferred tokens [medium]:** `--surface`, `--surface-muted`, `--border`, `--text-primary`, `--text-secondary`, `--primary`, `--live`, `--warning`, `--critical`, and a sequential `--rainfall-*` scale.
- Do not use red decoratively. It must retain its incident/severity/escalation meaning and always be paired with explicit text or iconography.

## 3. Typography

**Observed [medium confidence].** Neutral sans-serif, likely Inter or a similar UI grotesk; the exact family cannot be established from pixels.

- Brand: ~20–22 px, 700. Page title `Alerts & Incident Command`: ~20–22 px, 700.
- Incident title: ~16–18 px, 600–700; `ACTIVE INCIDENT`: ~10–11 px, 600–700 with red emphasis.
- Impact metrics (`340,000`, `4,200`, `59 facilities`): ~22–25 px, 500–600; labels ~10–11 px.
- Panel titles: ~13–15 px, 600–700; subtitles and metadata ~10–11 px.
- Feed headline: ~10–12 px, 500–600; secondary line ~9–10 px.
- Timeline milestone labels: ~9–10 px; dates and descriptions ~8–10 px.
- Broadcast body and distribution rows: ~8–10 px with compact line-height. Buttons/navigation: ~11–13 px, 500–600.
- Geographic labels are uppercase with tracking for states and mixed case for towns/rivers.

**Inference constraint.** Use tabular numerals for times, counts, rainfall values, subscriber totals, and character counts. Preserve casing and punctuation of emergency copy; do not rely on small size alone to subordinate required warnings.

## 4. Spacing rhythm

**Observed/inferred [medium confidence].** Compact 4 px base rhythm, commonly 4, 8, 12, 16, 20, 24, and 32 px.

- Shell gap: ~10 px; workspace inset: ~12–16 px; major vertical gaps: ~8–12 px.
- Header controls: ~8–12 px apart and ~38–42 px high.
- Incident banner: ~24–30 px horizontal padding; metric columns separated by vertical rules; action stack uses ~8 px gap.
- Panels: ~12–16 px internal padding with 1 px borders and ~8–10 px radii.
- Feed rows: ~50–58 px high with a narrow semantic-color rule and ~10–12 px horizontal gaps.
- Agency rows: ~34–38 px high. Timeline milestones distribute evenly along a thin horizontal axis.
- Radii: shell 12–16 px; panels 8–10 px; buttons 5–7 px; badges/chips 3–6 px; event markers fully round.
- Shadows are faint and diffuse, approximately equivalent to `0 2px 8px rgba(15, 23, 42, .06)` plus a light border. This is an inferred CSS equivalent, not a measured declaration.

## 5. Component inventory

1. `AppShell`, `Sidebar`, `BrandLockup`, `GlobalSearch`, `NavGroup`, `NavItem`, `PilotSelector`, `UserCard`.
2. `PageHeader`, action buttons, notification badge, avatar menu, incident summary line.
3. `IncidentCommandBanner`: state label, incident identity, projected peak, lead time, severity, impact metrics, escalation and management actions.
4. `IncidentMetaBar`: start time, duration, agencies notified, response coordinator.
5. `IncidentMapPanel`: layer tabs, map canvas, raster overlay, incident boundary, labels, rainfall legend, zoom/geolocation controls, scale bar.
6. `IncidentTimeline`: completed/current/future milestones, forecast peak, descriptions, projection horizon.
7. `LiveAlertFeed`: live indicator, severity rail/dot, timestamp, alert title/detail, acknowledgement/escalation actions, view-all link.
8. `ResponseCoordination`: agency icon, agency/role, status chip, status dot.
9. `BroadcastComposer`: draft/version state, provenance, body, language, character count, sensitivity, save/send actions.
10. `DistributionList`: channel, endpoint, audience/reach, aggregate reach.
11. `DecisionLog`: category filters, timeline event, audit-trail link.
12. `CycleStepper` with completed/current incident state.

**Threshold configuration placement [low-confidence/provisional].** No threshold configuration control is visible in the screenshot. If product requirements add one, place it provisionally inside an incident-management or alert-policy settings flow reached from `Manage incident`, never in the primary command banner and never above acknowledge, coordinate, escalate, or broadcast actions. Threshold configuration is subordinate to active incident response and must not compete with the severe incident hierarchy.

## 6. Data visualization style

- **Impact metrics [high]:** three direct figures in the command banner: `Population at risk 340,000`, `Est. evacuations 4,200`, `Critical infra exposed 59 facilities`. They are decision-support estimates, not decorative KPIs.
- **Rainfall field [high]:** blocky gridded intensity raster clipped/overlaid within the red Kodagu incident boundary. The sequential legend reads `Rainfall · mm/day` with `0`, `25`, `50`, `75`, `100+`. Darker blue means higher intensity.
- **Incident timeline [high]:** a linear temporal sequence from detection through advisories and current response to onset and peak. Completed points are green, current response is red with a halo, onset is orange, and projected peak is red outlined. Future uncertainty is implied by a dashed connector.
- **Feed [high]:** ranked temporal events with semantic severity rails: severe red, elevated orange, advisory blue, information gray.
- **Decision log [high]:** thin horizontal time axis with color-coded event points and concise authority/action labels.
- **Style rules [medium]:** keep units adjacent to data, preserve forecast/projected language, show timestamps and update time, and pair every color with a label. Do not smooth the rainfall raster into an aesthetically pleasing but less faithful blob. All displayed values remain placeholders.

## 7. Map style

**Observed [high confidence].** The map is an operational terrain/topographic view centered on Kodagu district. A bright red district/incident boundary contains a semi-transparent square-cell rainfall raster. Pale blue river lines and labels include `Kushalnagar`, `Cauvery (River)`, and nearby hydrology; black point labels mark `Madikeri`, `Virajpet`, `Somwarpet`, `Hassan`, and `Bannur`. Regional labels include `KARNATAKA` and `KERALA`.

- Basemap: low-saturation terrain/land-cover texture, faint roads/boundaries, restrained labels.
- Incident geometry: vivid red outline with no heavy fill, keeping the rainfall field readable.
- Controls: vertical `+`, `−`, and locate/target control at lower left; `10 km` scale bar.
- Layer tabs: `Rainfall` selected; `Wind` and `Terrain` available but their unseen content must not be invented.
- Legend: inset at lower right in a white floating box; blue horizontal ramp and explicit `mm/day` unit.

**Inferred implementation [medium confidence].** Keep attribution and accessible map summaries in production even though attribution is not legible in the screenshot. Preserve labels above raster where necessary, avoid saturated consumer-map colors, and provide non-map equivalents for rainfall maxima, affected area, and key places. Projection, tile provider, exact raster resolution, and layer behavior are unknown.

## 8. Interactions suggested by the static image

All behavior is inferred unless the screenshot exposes a state.

- `Escalate to NDMA` is the primary critical action; `Manage incident →` is secondary. Escalation should require confirmation, show recipients/consequences, and write to the audit log.
- `Rainfall`, `Wind`, and `Terrain` read as map layer tabs. Only Rainfall is documented; do not invent hidden layer visuals or values.
- Map controls suggest zoom and recenter. Pan, point/raster hover, and geographic drill-down are plausible but unconfirmed.
- Feed actions `Acknowledge`, `Escalate`, and `View all →` appear actionable. Acknowledgement needs actor/time feedback and escalation needs confirmation.
- `Save draft` and `Send broadcast` act on the outgoing message. Sending should present channel/reach confirmation and preserve draft/version provenance.
- Sidebar, pilots, search (`⌘K`), export, share, notifications, avatar, decision filters, and audit trail appear actionable.
- The current `Response coordination` timeline state and red `ALERT` cycle stage suggest live workflow progress; clickability is not proven.
- **Provisional threshold interaction [low]:** if introduced, threshold editing belongs behind `Manage incident` or dedicated policy settings, requires role control and audit history, and must remain subordinate to acknowledge/escalate/coordinate/broadcast work during an active event. The screenshot itself specifies no threshold UI.

## 9. Copy / microcopy

Legend: **[S]** static interface/product copy; **[D]** dynamic incident/forecast/operational placeholder data. Preserve displayed values verbatim.

- [S] `MausamSetu`; [S] `AI-Powered Climate`; [S] `Digital Twin of India`; [S] `Search regions, metrics...`; [S] `⌘K`; [S] `Workspace`; [S] `Overview`; [S] `Live Map`; [S] `Forecast`; [S] `Scenarios`; [S] `Validation`; [S] `Sectors`; [S] `Alerts`; [D] `3`; [S] `Pilots`.
- [D] `Cauvery Basin`; [D] `Krishna Basin`; [D] `Godavari Basin`; [D] `Bhavya Chaudhary`; [D] `Climate Ops`; [S] `मौसम सेतु`.
- [S] `Alerts & Incident Command`; [D] `1 active incident`; [D] `3 severe alerts`; [D] `Response ongoing`; [S] `Export`; [S] `Share`; [S] `Notifications`; [D] `5`; [D] `BC`.
- [S] `ACTIVE INCIDENT`; [D] `Heavy rainfall warning`; [D] `Kodagu district`; [D] `Karnataka`; [D] `Peak intensity projected Jul 19, 14:00 IST`; [D] `in 4 days`; [S] `Severity:`; [D] `Severe`; [S] `Population at risk`; [D] `340,000`; [S] `Est. evacuations`; [D] `4,200`; [S] `Critical infra exposed`; [D] `59 facilities`; [S] `Escalate to NDMA`; [S] `Manage incident →`.
- [S] `Started`; [D] `08:47 IST`; [S] `Duration`; [D] `45 minutes`; [D] `3 agencies notified`; [S] `Response coordinator`; [D] `MausamSetu`.
- [S] `Incident zone`; [D] `Kodagu district`; [S] `Live rainfall intensity`; [D] `updated 09:34 IST`; [S] `Live`; [S] `Rainfall`; [S] `Wind`; [S] `Terrain`; [D] `KARNATAKA`; [D] `KERALA`; [D] `Bannur`; [D] `Kushalnagar`; [D] `Madikeri`; [D] `Virajpet`; [D] `Somwarpet`; [D] `Hassan`; [D] `Cauvery (River)`; [S] `Rainfall · mm/day`; [D] `0`; [D] `25`; [D] `50`; [D] `75`; [D] `100+`; [D] `10 km`.
- [S] `Live alert feed`; [S] `Live`; [S] `View all →`; [D] `09:34`; [D] `Severe`; [D] `Heavy rain warning`; [D] `Kodagu district · >75 mm/day expected Jul 18-20`; [S] `Acknowledge`; [S] `Escalate`; [D] `09:12`; [D] `Elevated`; [D] `Soil saturation`; [D] `Kabini sub-basin at 87% saturation, still rising`; [S] `Acknowledge`; [S] `Escalate`; [D] `08:47`; [D] `Advisory`; [D] `Reservoir inflow`; [D] `Mettur expected +40% inflow Jul 19-21`; [S] `Acknowledge`; [S] `Escalate`; [D] `08:23`; [D] `Info`; [D] `Assimilation cycle 46 complete`; [D] `42 ms latency`; [D] `Nominal`; [D] `08:00`; [D] `Info`; [D] `Daily model report generated`; [D] `JJAS outlook and scenario pack ready`.
- [S] `Incident timeline`; [D] `projection to Jul 21`; [S] `Detected`; [D] `Jul 15, 08:47`; [D] `Assimilation flagged surge`; [S] `Advisories issued`; [D] `Jul 15, 09:22`; [D] `3 agencies notified`; [S] `Response coordination`; [D] `Jul 15, 09:34`; [D] `Live monitoring active`; [S] `Onset window`; [D] `Jul 18, 06:00`; [D] `First heavy bands expected`; [D] `PEAK · 82 mm/hr`; [D] `Jul 19, 14:00`; [D] `Highest expected intensity`.
- [S] `Response coordination`; [D] `5 agencies engaged`; [D] `NDMA`; [D] `National Command`; [D] `Acknowledged by Additional Secretary`; [D] `Notified · 09:22`; [D] `TN State DMA`; [D] `Active coordination · 3h 12m response`; [D] `Acknowledged · 09:15`; [D] `Karnataka DMA`; [D] `Field teams mobilized to Kodagu`; [D] `Active response`; [D] `IMD Bengaluru`; [D] `Ground observations flowing every 15 min`; [D] `Data sharing · Live`; [D] `District Collectors · 3`; [D] `Kodagu, Hassan, Mysuru`; [D] `Advisory issued · Pending confirmation`.
- [S] `Outgoing broadcast`; [D] `draft`; [D] `Kisan Suvidha SMS + WhatsApp + Public advisory API`; [S] `Save draft`; [S] `Send broadcast`; [D] `Draft · v2`; [D] `Auto-generated by MausamSetu`; [D] `The Cauvery Basin is under an active monsoon surge. Kodagu district should prepare for flash-flood risk between Jul 18 and Jul 20 with peak intensity of 82 mm/hr expected on Jul 19. Follow local advisories from Karnataka DMA. Emergency line: 1077.`; [S] `Language`; [D] `English + Kannada`; [S] `Character count`; [D] `287 / 320`; [S] `Sensitivity`; [D] `Public`.
- [S] `Distribution`; [D] `SMS · Kisan Suvidha`; [D] `24,382 subscribers`; [D] `WhatsApp · TN broadcast`; [D] `8,712 subscribers`; [D] `Twitter · @MausamSetu`; [D] `public feed`; [D] `IVR · Emergency line`; [D] `district officers`; [D] `Public advisory API`; [D] `12 partner apps`; [S] `Total estimated reach`; [D] `33,094 direct + 2.4M via partner apps`.
- [S] `Decision log`; [S] `Actions taken by authorities using MausamSetu today`; [S] `All`; [S] `Water`; [S] `Agriculture`; [S] `Heat`; [S] `Flood`; [D] `08:12 IST`; [D] `TN Irrigation Dept`; [D] `Sluice opening planning`; [D] `Initiated at Mettur`; [D] `08:47 IST`; [D] `Karnataka DMA`; [D] `Kodagu flash-flood`; [D] `watch issued`; [D] `09:15 IST`; [D] `BWSSB Bengaluru`; [D] `Water distribution`; [D] `schedule normalized`; [D] `09:22 IST`; [D] `Agriculture Dept KA`; [D] `Mandya paddy sowing`; [D] `advisory drafted`; [S] `Full audit trail · 24h view`.
- [S] `CYCLE 46`; [S] `INGEST`; [S] `REGRID`; [S] `ASSIMILATE`; [S] `FORECAST`; [S] `IMPACT`; [S] `ALERT`.

## 10. Anti-slop checks

- Preserve incident response as the dominant hierarchy: identify, assess, acknowledge, coordinate, escalate, communicate, and audit.
- Treat every visible incident value and broadcast statement as placeholder data. Do not publish it, “correct” it, or present it as live emergency guidance.
- Keep threshold configuration provisional and subordinate. No threshold control is visible; do not add one to the banner or let configuration compete with active response actions.
- Do not invent the hidden `Wind` or `Terrain` layers, unseen incident-management screens, automation rules, or alert thresholds.
- Keep severe red scarce and semantic; pair red/orange/green/blue/gray with explicit labels and icons.
- Retain update times, forecast qualifiers, units, source/provenance, audience reach, agency state, and audit path. Avoid decontextualized KPI cards.
- Preserve the blocky rainfall raster, district outline, operational basemap, legend, labels, scale, and controls; do not substitute a generic heatmap illustration.
- Critical actions require confirmation, authorization, feedback, and auditability. Never make `Escalate` or `Send broadcast` a silent one-click side effect.
- Avoid glassmorphism, decorative gradients, oversized hero numbers, generic AI sparkle motifs, and excess pill styling absent from the reference.
- This decode covers Screen 7 only. Do not infer Screens 8–10 or any unseen routes/states.
