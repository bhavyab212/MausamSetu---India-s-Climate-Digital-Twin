---
route: /sectors
source_path: C:\Users\bhavy\Documents\v1\images\screen-05-impacts.png
canvas: 1659 × 948 px
status: Direct reference decode
confidence_caveat: High confidence for visible hierarchy, copy, relative geometry, and state; medium for sampled color, type metrics, spacing, radii, and shadows because the source is a raster screenshot; low for behavior not directly shown.
placeholder_data_warning: All names, dates, times, quantities, percentages, statuses, model outputs, and operational recommendations below are screenshot values preserved verbatim as placeholders, not live or validated data.
---

## 1. Layout

**Observed [high confidence].** Desktop application shell at 1659 × 948. A fixed-looking left rail occupies approximately x=10–247 (237 px); the main white workspace begins around x=261 and extends to x=1648 (about 1,387 px). Outer gutters are roughly 10–14 px. Both surfaces have large, soft corners (about 12–16 px) over a blurred pale green landscape backdrop.

- Sidebar: brand block at top; search; `Workspace` navigation; `Pilots`; signed-in user; Hindi product mark at bottom. The selected `Sectors` row is a pale-blue pill with a blue left rule.
- Main header: approximately 70 px high. Title/subtitle at left; `Export`, `Share`, `Notifications`, avatar, and chevron at right.
- Sector tab shell: four equal cards in one row, approximately x=273–1561, y=88–160, with 14–20 px gaps. `Water security` is the only visible selected reference: blue outline/bottom accent and blue icon. This decode must not imply or invent the hidden content of the other tabs.
- Primary content: two-column grid from about y=176–603. Left `Reservoir network` panel is about 646 × 427; right column is about 705 px wide and splits vertically into `Supply vs demand · 30 days` and `Water balance · today`.
- Lower content: full-width `Stakeholder advisories` panel (about 182 px high), then `Decision log` (about 113 px high).
- Bottom cycle stepper: persistent strip about 42 px high, spanning the main workspace.

**Inferred implementation [medium confidence].** Use a 12-column desktop grid with a ~48/52 split for the primary panels, 12–16 px inter-panel gaps, and content padding around 20 px. The screenshot establishes only this desktop composition; responsive behavior is not evidenced.

## 2. Color system

Raster antialiasing introduces many near-white pixels, so the values below separate visible samples from inferred reusable tokens.

- **Exact/near-exact visible samples [medium]:** canvas/card white `#FFFFFF` and near-white `#FEFEFE`; primary blue around `#0877F9`; deep navy text around `#101522`; success green around `#009E67`; warning orange around `#FF8500`; critical red around `#F51B24`; cyan/teal around `#0AAFC0`; neutral rules/backgrounds around `#E6E9EC` / `#F5F7F8`.
- **Inferred tokens [medium]:** `--surface`, `--surface-subtle`, `--border`, `--text-primary`, `--text-secondary`, `--accent-blue`, `--success`, `--warning`, `--critical`, `--info`.
- Selected navigation and tab states use a very pale blue fill plus saturated blue edge. Statuses use color redundantly with text (`Nominal`, `Stable`, `Rising · +18%`, `Acknowledged`, `Pending`).
- Chart fills are translucent: blue and orange area bands appear around 8–15% opacity; reservoir network links use pale blue/gray ribbons with bright blue flow dots.
- **Do not** derive production tokens from screenshot whites alone; compression/background blending makes `#FEFEFE`, `#FDFDFD`, and `#FFFFFF` visually equivalent here.

## 3. Typography

**Observed [medium confidence].** A neutral sans-serif is used throughout; likely Inter or a close UI grotesk, but the family cannot be proven from pixels. Text is mostly dark navy rather than absolute black.

- Brand: approximately 20–22 px, 700.
- Page title `Sectors`: approximately 21–23 px, 700; subtitle 13–14 px, 400–500.
- Panel titles: approximately 14–16 px, 600–700.
- Sector labels: approximately 15–16 px, 600; sector summaries 11–12 px.
- Primary metrics: reservoir percentages roughly 22–25 px, 600; smaller values and labels 10–12 px.
- Body/microcopy: 10–12 px with about 1.35–1.5 line-height. Buttons and nav labels are around 12–13 px, 500–600.
- Small uppercase chips (`PRIORITY`, `INFO`, `OPPORTUNITY`) are around 8–9 px, 600–700 with slight tracking.

**Inference constraint.** Preserve hierarchy and density, not guessed font metrics. Use tabular numerals for times and quantitative values if the chosen family supports them.

## 4. Spacing rhythm

**Observed/inferred [medium confidence].** The interface follows a compact 4 px base rhythm: 4, 8, 12, 16, 20, 24, and 32 px recur.

- Main shell gap: ~14 px; main content inset: ~12–20 px.
- Header controls: ~8–12 px between buttons; buttons around 38–42 px high with 10–14 px horizontal padding.
- Panels: ~12 px gaps; card padding roughly 16–20 px; internal title-to-content gap 8–12 px.
- Sector cards: roughly 16 px internal horizontal spacing; icon cell ~54 px wide.
- Advisories use three near-equal columns separated by vertical rules and 12–18 px padding.
- Radii: shell 12–16 px; panels/cards 8–12 px; chips 3–6 px; circular indicators fully round.
- Borders: predominantly 1 px light gray; selected accents 2–3 px.
- Shadows: subtle, diffuse card shadow approximately `0 2px 8px rgba(15, 23, 42, .06)` plus border; outer shell shadow is broader. These are inferred equivalents, not measured CSS.

## 5. Component inventory

Reusable components visible in the direct reference:

1. `AppShell`, `Sidebar`, `BrandLockup`, `GlobalSearch`, `NavGroup`, `NavItem`, `PilotSelector`, `UserCard`.
2. `PageHeader`, icon/text `ActionButton`, notification badge, avatar menu.
3. `SectorTabShell` and `SectorTab` with icon, status dot, title, summary, and selected state. Only `Water security` supplies visible selected-tab body content.
4. `Panel`, `PanelHeader`, contextual badge/counter.
5. `ReservoirNetwork`: reservoir node, capacity ring, state chip, animated-flow-style connector, discharge label, delta destination card.
6. `TimeSeriesAreaChart`, legend, today marker, metric footer.
7. `WaterBalanceSankey`: inputs, stock cards, flow ribbons, output labels.
8. `AdvisoryList`, `AdvisoryCard`, category icon, severity chip, recipient/status row, details link, panel actions.
9. `DecisionLog`, filter chips, timeline event, audit link.
10. `CycleStepper` with completed/current/future states.

**State coverage [high].** Visible states include selected, nominal/stable/rising, priority/info/opportunity, acknowledged/pending, completed/current/future, and unread notification count.

## 6. Data visualization style

- **Reservoir network [high]:** three circular storage gauges: `KRS · Karnataka` at `79%` / `91 MCft`, `Kabini · Karnataka` at `72%` / `68 MCft`, and `Mettur · Tamil Nadu` at `87%` / `142 MCft`. Pale branching ribbons converge on Mettur; blue dots imply directional flow. Labels include `River discharge 340 m³/s`, `Rising · +18%`, `+ 142 MCft`, and `Cauvery Delta` / `Agricultural command area · 800,000 hectares`.
- **Supply/demand [high]:** blue supply and orange demand lines with faint area fills, y-axis `MCft/day` and 0–200 scale, x-axis `Day 1 Jul 15` through `30 Aug 14`, vertical `TODAY` marker. Summary `Current surplus · +32% ↑`; footer `Peak demand · Aug 3`, `Min supply · Aug 12`, `Days of buffer · 15`.
- **Water balance [high]:** compact Sankey/alluvial graphic. Inputs: `Rainfall 32.4 mm`, `Groundwater 8 mm`, `Transfer 0 mm`; stocks: `Basin storage 87%`, `Soil moisture 74%`; outputs: `Irrigation · 45%`, `Evapotranspiration · 30%`, `Domestic · 15%`, `Environmental flow · 10%`.
- **Style rules [medium]:** thin axes, sparse grid, direct labels, color plus text, rounded stock blocks, restrained animation implications. Exact values must remain placeholders. Do not imply mathematical consistency beyond what the screenshot shows.

## 7. Map style

No geographic map is present in this screen. **High confidence.** `Reservoir network` is a schematic topology, not a geospatial map: do not place nodes according to real coordinates or add basemap layers. The tiny water/river iconography and labeled administrative locations communicate geography semantically only. If reused, distinguish this component from the product’s map component in naming, accessibility text, and interaction model.

## 8. Interactions suggested by the static image

All behavior is inferred unless the screenshot exposes a state.

- `Water security`, `Agriculture`, `Heat stress`, and `Flood risk` read as tabs. **Visible fact:** Water security is selected. **Inference:** selecting another tab would replace the body; hidden bodies are undocumented and must not be invented.
- Sidebar rows, pilot basin rows, header actions, avatar, advisory actions, filters, `View details →`, `Full audit trail · 24h view`, and cycle stages appear actionable.
- Search displays shortcut `⌘K`; likely opens/focuses global search.
- `Notifications` has badge `2`; likely opens a notification surface.
- `New advisory` is primary creation; `View history` is secondary review.
- Advisory status/action semantics should support keyboard focus and not rely solely on color.
- Chart/tooltips, reservoir-node hover, connector animation, and timeline drill-down are plausible but **low-confidence inferences**; implement only if product requirements confirm them.
- Cycle stage `IMPACT` is current, prior stages are complete, `VALIDATE` is future; this suggests workflow navigation but does not prove clickability.

## 9. Copy / microcopy

Legend: **[S]** static interface/product copy; **[D]** dynamic or operational placeholder data. Preserve displayed values verbatim.

- [S] `MausamSetu`; [S] `AI-Powered Climate`; [S] `Digital Twin of India`; [S] `Search regions, metrics...`; [S] `⌘K`; [S] `Workspace`; [S] `Overview`; [S] `Live Map`; [S] `Forecast`; [S] `Scenarios`; [S] `Validation`; [S] `Sectors`; [S] `Alerts`; [S] `Pilots`.
- [D] `Cauvery Basin`; [D] `Krishna Basin`; [D] `Godavari Basin`; [D] `Bhavya Chaudhary`; [D] `Climate Ops`; [S] `मौसम सेतु`.
- [S] `Sectors`; [D] `Cauvery Basin`; [D] `4 sectors monitored`; [D] `Updated 09:34 IST`; [S] `Export`; [S] `Share`; [S] `Notifications`; [D] `2`; [D] `BC`.
- [S] `Water security`; [D] `Reservoirs +18% inflow`; [S] `Agriculture`; [D] `Soil moisture adequate`; [S] `Heat stress`; [D] `2 days >38°C forecast`; [S] `Flood risk`; [D] `Kodagu, Hassan zones`.
- [S] `Reservoir network`; [D] `Cauvery Basin`; [S] `Live inflow`; [D] `3 reservoirs monitored`; [D] `KRS · Karnataka`; [D] `Nominal`; [D] `79%`; [D] `91 MCft`; [D] `River discharge`; [D] `340 m³/s`; [D] `Kabini · Karnataka`; [D] `Stable`; [D] `72%`; [D] `68 MCft`; [D] `Mettur · Tamil Nadu`; [D] `Rising · +18%`; [D] `87%`; [D] `142 MCft`; [D] `+ 142 MCft`; [D] `Cauvery Delta`; [D] `Agricultural command area · 800,000 hectares`.
- [S] `Supply vs demand`; [D] `30 days`; [S] `Current surplus`; [D] `+32% ↑`; [S] `MCft/day`; [S] `TODAY`; [D] `Day 1`; [D] `Jul 15`; [D] `5`; [D] `10`; [D] `15`; [D] `20`; [D] `25`; [D] `30`; [D] `Aug 14`; [S] `Supply`; [S] `Demand`; [S] `Peak demand`; [D] `Aug 3`; [S] `Min supply`; [D] `Aug 12`; [S] `Days of buffer`; [D] `15`.
- [S] `Water balance`; [D] `today`; [S] `Inputs`; [S] `Rainfall`; [D] `32.4 mm`; [S] `Groundwater`; [D] `8 mm`; [S] `Transfer`; [D] `0 mm`; [S] `Stocks`; [S] `Basin storage`; [D] `87%`; [S] `Soil moisture`; [D] `74%`; [S] `Outputs`; [S] `Irrigation`; [D] `45%`; [S] `Evapotranspiration`; [D] `30%`; [S] `Domestic`; [D] `15%`; [S] `Environmental flow`; [D] `10%`.
- [S] `Stakeholder advisories`; [D] `3 active recommendations`; [D] `Issued 09:22 IST`; [S] `New advisory`; [S] `View history`; [D] `Irrigation · Tamil Nadu`; [S] `PRIORITY`; [D] `Open Mettur sluice gates by Jul 20 to accommodate projected +40% inflow. Recommended discharge: 12,000 cusecs.`; [S] `Recipient`; [D] `TN Public Works Dept`; [D] `Acknowledged`; [S] `Response time`; [D] `3h 12m`; [S] `View details →`.
- [D] `Urban water · Bengaluru`; [S] `INFO`; [D] `Reservoir buffer stable. Current supply covers 15 days at normal demand. No consumer action required. Communicate stability publicly.`; [D] `BWSSB`; [D] `Acknowledged`; [D] `47m`; [S] `View details →`.
- [D] `Agriculture · Mandya district`; [S] `OPPORTUNITY`; [D] `Soil moisture optimal for kharif paddy sowing. Recommend farmer advisory issued via Kisan Suvidha SMS network. Window: Jul 16-22.`; [D] `KAD + AgriDept KA`; [D] `Pending`; [S] `Response time`; [D] `-`; [S] `View details →`.
- [S] `Decision log`; [S] `Actions taken by authorities using MausamSetu today`; [S] `All`; [S] `Water`; [S] `Agriculture`; [S] `Heat`; [S] `Flood`; [D] `08:47 IST`; [D] `TN Irrigation Dept`; [D] `Sluice opening planning`; [D] `Initiated at Mettur`; [D] `09:15 IST`; [D] `Karnataka DMA`; [D] `Kodagu flash-flood`; [D] `watch issued`; [D] `BWSSB Bengaluru`; [D] `Water distribution`; [D] `schedule normalized`; [D] `09:22 IST`; [D] `Agriculture Dept KA`; [D] `Mandya paddy sowing`; [D] `advisory drafted`; [S] `Full audit trail · 24h view`.
- [S] `CYCLE 46`; [S] `INGEST`; [S] `REGRID`; [S] `ASSIMILATE`; [S] `FORECAST`; [S] `IMPACT`; [S] `VALIDATE`.

## 10. Anti-slop checks

- Keep the desktop information density; do not inflate cards, typography, or whitespace into a generic marketing dashboard.
- Treat `Water security` as the visible tab-shell reference only. Do not invent Agriculture, Heat stress, or Flood risk panel contents.
- Keep reservoir topology schematic and supply/demand/Sankey visuals distinct; do not replace them with generic KPI cards.
- Preserve every screenshot number/date/name/status verbatim as placeholder data; do not “correct,” normalize, or silently recompute it.
- Distinguish exact visible evidence from inferred CSS/behavior. Font family, CSS shadows, animation, hover, responsive layout, and hidden states are not directly known.
- Retain semantic status text and icons; never encode severity, stage, or acknowledgement by color alone.
- Keep operational actions (`New advisory`, advisory details, audit trail) visually subordinate to monitored impact context.
- Use real chart semantics, units, legends, and accessible descriptions rather than decorative SVG waves.
- Do not introduce gradients, glassmorphism, oversized hero metrics, excessive pills, or gratuitous illustrations absent from the reference.
- This decode covers Screen 5 only. Do not infer Screens 8–10 or any unseen routes/states.
