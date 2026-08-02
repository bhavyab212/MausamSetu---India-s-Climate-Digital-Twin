---
route: /map
source_path: 'C:\Users\bhavy\Documents\v1\images\screen-02-map.png'
canvas: 1536 × 1024 px
status: Direct reference decode
confidence_caveat: 'High confidence for visible structure and copy; medium for coordinates and colors sampled from the raster; low-to-medium for font family, hidden behavior, geographic precision, and token intent. The /map route is inferred from the Live Map title and file name.'
placeholder_data_warning: 'All dates, times, coordinates, measurements, names, percentages, forecast values, statuses, storm identifiers, and scenario labels below are Phase 1 placeholders copied verbatim from the screenshot, not validated live data or product requirements.'
---

## 1. Layout

**Measurement convention:** approximate screenshot pixels, origin `(0,0)` at canvas top-left. Values are direct visual measurements unless marked **Inferred**.

- Desktop canvas: `1536 × 1024`. A regional map fills almost the entire background beneath floating controls and panels.
- **Route-specific shell override:** this screen uses a narrow icon rail plus a full-screen map workspace. It is a Live Map route override, **not** the base dashboard shell shown in the overview reference. Do not force the overview’s `≈238 px` labeled sidebar or card-grid workspace onto this route.
- Narrow rail: approximately `(14,13)`, `71 × 995`, radius about `17 px`. Brand mark occupies the top `≈90 px`; eight primary icon destinations are vertically distributed from `y≈117` to `560`; the selected map item has a blue left rail and pale-blue field. User initials and a vertical Devanagari product mark sit near the bottom.
- Floating header: approximately `(99,13)`, `1420 × 75`, radius about `13 px`.
  - Title/context block starts at `(122,28)`.
  - Action cluster runs from `x≈1041` to `1504`: Export, Share, Notifications, avatar menu.
- Map viewport begins around `x=94`, `y=84` and extends through the right/bottom canvas; controls float over it rather than consuming grid columns.
- Layer panel: approximately `(101,108)`, `228 × 445`, with `20 px` internal padding. Ten layer rows plus an opacity control.
- Basin/search switcher: approximately `(1062,107)`, `455 × 51`; five basin scopes and a search action.
- Selected grid-cell detail panel: approximately `(1075,306)`, `350 × 465`, radius about `14 px`. It overlays the map and is the dominant inspector.
- Map controls:
  - Zoom/geolocate/north stack near `(102,592)`, about `44 × 180`.
  - Rainfall legend near `(158,680)`, about `209 × 94`.
  - Storm callout near `(1314,212)`, about `167 × 54`.
  - Reservoir callouts around Kabini `(469,304)`, KRS `(587,493)`, and Mettur `(926,305)`.
- Timeline/replay deck: approximately `(127,790)`, `1012 × 198`, radius about `14 px`. Playback controls occupy the top-left, current timestamp is centered, scenarios/calendar are top-right, and the observed/forecast date rail spans the lower two-thirds.
- Assimilation status card: approximately `(1189,829)`, `295 × 80`.
- Bottom metric strip: approximately `(1039,937)`, `408 × 53`, three equal mini-cards; separate full-screen control near `(1466,937)`, `52 × 53`.
- **Inferred responsive intent, medium confidence:** preserve map primacy. On smaller screens, collapse the rail and layer panel into drawers; convert the inspector and timeline into dismissible bottom sheets rather than shrinking all overlays simultaneously.

## 2. Color system

**Exact raster samples** are measured screenshot pixels, not guaranteed source tokens; translucent overlays, basemap imagery, antialiasing, and compression affect them.

| Role | Exact sample from screenshot | Likely implementation token (inferred) | Confidence |
|---|---:|---:|---|
| Outer rail/background edge | `#EEEFF1` at `(16,16)` | cool gray-100 surface | High sample, medium token |
| Rail interior | `#F7FAFC` at `(50,100)` | translucent white/slate-50 | High sample, medium token |
| Header edge/surface | `#ECEEF1` at `(101,20)` | cool glass border | High sample, low token |
| Header interior | `#F1F6F9` at `(200,50)` | white-blue glass | High sample, medium token |
| Map pale water/field | `#CCDFEE` at `(103,110)` | basemap water blue-gray | High sample, low token |
| Map neutral field | `#E9F2F9` at `(200,150)` | cool map background | High sample, low token |
| Selected layer row | `#E0F2FD` at `(294,178)` | sky-100 selection | High sample, medium token |
| Off toggle | `#BEC3CF` at `(295,247)` | slate-300 | High sample, high role |
| Inspector surface | `#F9FAFB` at `(1120,400)` | white/slate-50 at high opacity | High sample, medium token |
| Inspector/map pale blue | `#EFF7FB` at `(1270,445)` | blue-50 | High sample, low token |
| Primary data/action blue | `#0062FD` at `(1255,535)` | blue-600 near `#0061FD` | High sample, high role |
| Timeline surface | `#F4F8FA` at `(200,745)` | translucent cool white | High sample, medium token |
| Timeline active line | `#0063FA` at `(604,876)` | blue-600 | High sample, high role |
| Selected date/today | `#0061FD` at `(604,943)` | blue-600 | High sample, high role |
| Canvas edge | `#EBF0F5` at `(1510,1000)` | cool-gray background | High sample, low token |

- Primary accent is a saturated electric blue around exact raster values `#0061FD–#0063FA`; it marks active navigation, enabled toggles, selected dates, rainfall bars, map selection, and links.
- Rainfall uses a stepped light-blue-to-deep-blue ramp. Representative exact raster colors include `#B5D4FD`, `#9DC6FD`, `#8FBEFD`, and `#0061FE`; these samples are affected by the basemap and should not be treated as original ramp stops.
- Positive/live/complete states are green/teal; temperature uses orange; inactive controls are cool gray. Current analysis deliberately uses multiple semantic colors: rainfall blue, Tmax orange, Tmin blue, INSAT LST orange.
- Overlay cards are cool white at high opacity with subtle blue-gray borders. **Inferred:** `background: rgba(248, 251, 253, .90–.96)`, backdrop blur `12–20 px`, border `1 px rgba(148, 163, 184, .20–.35)`.
- Shadows are broad and faint, approximately `0 6px 20px rgba(15, 23, 42, .10)`; the selected inspector has enough separation to remain readable above the detailed map.
- Do not recolor the basemap into brand blue wholesale. Data blue must remain more saturated than water, roads, boundaries, and atmospheric streamlines.

## 3. Typography

- **Font family estimate:** modern UI grotesk such as Inter, SF Pro, or a close sans; exact identification is not possible from the raster. Confidence low-to-medium. Devanagari fallback is required for `मौसम सेतु`.
- Header title “Live Map”: about `19–20 px`, `650–700`, line height `24 px`. Context line: `12–13 px`, `400–500`.
- Panel titles “Layers” and “Mysuru”: `15–18 px`, `600–700`; section labels “Current analysis,” “7-day forecast,” and “Anomaly”: `11–12 px`, `500–600`.
- Layer labels, basin tabs, header actions, and control labels: `11–13 px`, `450–550`.
- Inspector metrics: numerals around `18–20 px`, `600`; unit suffixes `10–12 px`, `500`. Forecast bar values: `10–11 px`, `600`; dates: `9–10 px`.
- Timeline current time: about `16–18 px`, `550–650`; date ticks and legends `10–11 px`; selected `TODAY` badge `11–12 px`, uppercase, `600–700`.
- Map labels preserve geographic hierarchy: states use `13–18 px` uppercase with wide tracking; cities `11–14 px`; rivers `10–11 px` blue; seas `14–16 px` blue with stacked labels where necessary.
- Storm identifier `L-14 · 998 hPa`: about `13 px`, `650`; descriptor `monsoon depression`: about `10 px`, `400`.
- **Inferred:** use tabular numerals for timestamps, coordinates, measurements, percentages, forecast values, timeline dates, and cycle countdowns. Do not alter displayed precision in Phase 1.
- Primary text is near black; metadata is gray-600; links/selection are blue; semantic measurements retain their visible category colors.

## 4. Spacing rhythm

- Dominant spacing unit is approximately `8 px`, with recurring increments `4, 8, 12, 16, 20, 24`.
- Canvas inset for rail/header: `13–16 px`; gap between rail and map/header: about `14 px`.
- Header horizontal padding: about `22 px`; actions are `≈99–126 px` wide and `≈44 px` high with `8–12 px` gaps.
- Floating panel padding: `16–20 px`; layer rows: `≈34 px` high; toggle width `≈27 px`, height `≈17 px`.
- Basin tabs: approximately `73–80 px` wide, `38 px` high. The search segment is about `47 px` wide.
- Standard overlay radius: `12–15 px`; rail/header radius `14–18 px`; selected row and buttons `8–10 px`; chips/callouts `10–13 px`; status/timeline markers circular.
- Inspector section gaps: `18–22 px`; analysis metrics are distributed across four equal columns; forecast bars occupy about `100 px` of chart height.
- Timeline deck uses `20–26 px` side padding. Date markers repeat at approximately `58 px` intervals; the selected date has a `25–28 px` ring and a vertical guide.
- Borders are mostly `1 px` and low contrast. Separate dense overlays by surface opacity and shadow, not heavy outlines.
- Keep map labels/callouts clear of inspector and timeline safe areas; collision handling is part of the layout, not optional polish.

## 5. Component inventory

- `MapRouteShell`: variants `desktop-rail`, `collapsed-mobile`; route-specific full-screen map shell.
- `IconRail`: brand mark, icon-only nav items, selected-leading-rail state, user avatar, vertical brand tile.
- `MapHeader`: page title/context/live state, timestamp, export/share/notification/account actions.
- `HeaderActionButton`: icon + label; notification badge; avatar/dropdown variant.
- `LayerPanel`: title/settings action, layer rows, opacity slider.
- `LayerToggleRow`: icon, label, switch; variants enabled, disabled, selected-row.
- `OpacitySlider`: numeric percentage and draggable thumb.
- `BasinScopeTabs`: Cauvery/Krishna/Godavari/Mahanadi/All India plus search.
- `InteractiveClimateMap`: vector basemap, rainfall raster, streamlines, rivers, reservoirs, settlements, boundaries, storm track, grid selection.
- `MapNavigationControls`: zoom in/out, locate/target, north/orientation.
- `RasterLegend`: title, stepped gradient, ticks.
- `ReservoirCallout`: reservoir icon, name, percentage, radial mini-gauge; variants normal/selected.
- `StormCallout`: cyclone glyph, identifier/pressure, descriptor; paired storm center and dotted trajectory.
- `GridCellSelection`: outlined cell/crosshair and connector line to inspector.
- `GridCellInspector`: header/close, coordinates, current-analysis metrics, forecast bars, anomaly, status chip, action footer.
- `ForecastBars`: observed/forecast daily columns, value labels, date labels, selected peak bar.
- `TimelineDeck`: play, speed menu, current timestamp, scenario presets, calendar, daily scrubber, observed/forecast legend, selected-today marker.
- `ScenarioButton`: Now, replay preset, date picker variants.
- `AssimilationToast`: status dot, process/cycle/result, countdown, collapse control.
- `LiveMetricStrip`: Rainfall/Wind/Humidity mini-cards with sparklines.
- `FullscreenButton`: icon-only expansion control.

## 6. Data visualization style

- The primary visualization is a layered geospatial canvas rather than a conventional dashboard chart. Visual hierarchy: rainfall raster first, selected grid cell and inspector second, atmospheric/rivers/boundaries third, basemap labels last.
- Selected grid-cell inspector shows four current metrics in a horizontal matrix:
  - Rainfall `32.4 mm` in blue.
  - Tmax `28.1 °C` in orange.
  - Tmin `21.3 °C` in blue.
  - INSAT LST `29.8 °C` in orange.
- Seven-day forecast is a compact column chart with pale blue bars and a saturated blue peak bar. Values remain directly above bars; dates remain below. No visible y-axis or grid is necessary because each value is labeled.
- Phase 1 bar sequence: `16`, `22`, `30`, `48`, `36`, `24`, `18` for `Jul 16` through `Jul 22`; unit `mm/day`. `Jul 19` is selected/peak.
- Anomaly is shown as text rather than another chart: `+14.2 mm above climatology (+78%)`, green and upward-oriented. Monsoon state is paired with an `Active` chip.
- Reservoir callouts use small blue radial gauges plus percentage labels: Kabini `72%`, KRS `79%`, Mettur `87%`.
- Bottom metrics use tiny blue sparklines without axes; current values remain prominent: rainfall `32.4`, wind `18`, humidity `86`.
- Timeline uses filled blue points for observed dates and hollow blue points for forecast dates, backed by a thin horizontal track. Today gets a larger double-ring marker, vertical guide, and blue `TODAY` badge.
- Atmospheric wind/pressure visualization uses fine semi-transparent streamlines. Storm track is cyan/dotted with small nodes; center is a saturated blue dot with concentric white/blue circulation rings.
- **Inferred implementation:** tooltips, keyboard-accessible data points, textual chart equivalents, and reduced-motion support are required, but the screenshot does not establish a charting library, interpolation, forecast methodology, or update protocol.

## 7. Map style

- Geographic extent focuses southern India and adjacent Arabian Sea/Bay of Bengal, with the Cauvery basin centered. The viewport shows approximately `75°E–80°E` and a visible `14°N` label; raster values must not be reverse-engineered into authoritative coordinates.
- Basemap is pale and cool, with low-saturation terrain texture, fine gray roads, blue waterways, green terrain/administrative hints, and subdued place labels. Water areas are light blue.
- State names are letterspaced uppercase; city labels are mixed case with small node markers; rivers are blue labels following linework. The selected Mysuru area is marked by a blue grid-cell/crosshair.
- Rainfall overlay is a visibly gridded, semi-transparent raster in a north-south band from coastal/western Karnataka into Tamil Nadu. Cells range from pale cyan to deep blue and allow map details to show through.
- Basin/state boundary linework uses blue and green, approximately `1–2 px`. Rivers are thin saturated cyan/blue. Wind flow is a field of fine curved white/light-blue streamlines.
- Enabled layers visible in the panel and map: Rainfall, Wind flow, Reservoirs, Rivers. Disabled controls shown: Max temperature, Min temperature, INSAT LST, Soil moisture, Flood risk zones. Overlay opacity is `72%`.
- Reservoir callouts are floating white glass cards connected spatially to markers; they combine icon, name, percentage, and radial fill rather than generic pin icons.
- The depression at sea uses concentric wind lines, central pressure marker, cyan dotted path, and a compact callout. Do not replace it with a decorative weather icon detached from geography.
- Legend remains bottom-left above the timeline: title `Rainfall · mm/day`, horizontal blue ramp, ticks `0`, `25`, `50`, `75`, `100+`.
- **Inferred, medium confidence:** use a GPU-capable map with composable vector, raster, and particle/streamline layers. Provider, tile source, coordinate reference system, forecast model, and exact geographic geometries are unknown and must not be inferred from the screenshot.

## 8. Interactions suggested by the static image

- Icon rail navigates product routes; selected Live Map state uses both blue rail and blue globe icon. Tooltips/accessible names are essential because labels are hidden.
- Header Export and Share initiate output/collaboration flows; Notifications opens a tray; avatar opens account options. Badge `2` is dynamic.
- Layer switches toggle independent overlays. Settings likely opens layer configuration. The opacity slider changes the composite overlay globally or rainfall layer specifically; scope is **inferred** and must be confirmed.
- Basin tabs change map extent/context; search finds places or map entities.
- Map supports pan, zoom, geolocation/orientation, hover/focus inspection, and selectable grid cells/reservoirs. The blue Mysuru cell and connector prove a selected map feature; exact selection mechanics are inferred.
- Inspector close dismisses selection. `Time series`, `Compare`, and `Export` imply deeper temporal analysis, side-by-side comparison, and selected-cell export.
- Forecast bars suggest hover/focus details and date selection; `Jul 19` is visibly emphasized.
- Playback control and `1x` menu animate time. Timeline markers select a day; drag/scrub is strongly implied. Observed and forecast points use different fill states.
- `Now`, `Replay 2018 flood`, `Replay 2016 drought`, and calendar imply scenario/time-mode switching. Replays must be clearly distinguished from live/forecast data.
- Assimilation card can collapse via chevron. Full-screen button expands the map workspace.
- Accessibility implications: map-only content needs a structured data/table alternative; toggles need announced state; timeline needs keyboard stepping and explicit dates; playback needs pause and reduced motion; inspector focus should move predictably and return to the selected map feature.

## 9. Copy / microcopy

All strings are transcribed as displayed and classified `[S]` static UI copy or `[D]` dynamic/placeholder data. Punctuation, capitalization, precision, and abbreviations are Phase 1 reference values.

- Rail/brand: `[D] BC`; `[S] मौसम सेतु`.
- Header: `[S] Live Map`; `[D] Cauvery Basin`; `[D] Live`; `[D] 09:34:12 IST`; `[S] Export`; `[S] Share`; `[S] Notifications`; `[D] 2`; `[D] BC`.
- Layers: `[S] Layers`; `[S] Rainfall`; `[S] Wind flow`; `[S] Max temperature`; `[S] Min temperature`; `[S] INSAT LST`; `[S] Reservoirs`; `[S] Rivers`; `[S] Soil moisture`; `[S] Flood risk zones`; `[S] Overlay opacity`; `[D] 72%`.
- Basin scope: `[D] Cauvery`; `[D] Krishna`; `[D] Godavari`; `[D] Mahanadi`; `[S] All India`.
- Geographic/map labels: `[D] 14°N`; `[D] KARNATAKA`; `[D] ANDHRA PRADESH`; `[D] TAMIL NADU`; `[D] KERALA`; `[D] Arabian Sea`; `[D] Bay of Bengal`; `[D] Nilgiris`; `[D] Kodagu`; `[D] Hassan`; `[D] Bengaluru`; `[D] Mysuru`; `[D] Salem`; `[D] Coimbatore`; `[D] Tiruchirappalli`; `[D] Thanjavur`; `[D] Mettur`; `[D] Hemavathi`; `[D] Kabini`; `[D] Shimsha`; `[D] Bhavani`; `[D] Pykara`; `[D] Amaravati`; `[D] 75°N`; `[D] 76°E`; `[D] 77°E`; `[D] 78°E`; `[D] 79°E`; `[D] 80°E`.
- Reservoir/storm callouts: `[D] Kabini`; `[D] 72%`; `[D] KRS`; `[D] 79%`; `[D] Mettur`; `[D] 87%`; `[D] L-14 · 998 hPa`; `[D] monsoon depression`.
- Legend: `[S] Rainfall · mm/day`; `[D] 0`; `[D] 25`; `[D] 50`; `[D] 75`; `[D] 100+`.
- Inspector identity: `[D] Mysuru`; `[S] Grid cell`; `[D] 12.25 °N, 76.50 °E`.
- Current analysis: `[S] Current analysis`; `[S] Rainfall`; `[D] 32.4`; `[S] mm`; `[S] Tmax`; `[D] 28.1`; `[S] °C`; `[S] Tmin`; `[D] 21.3`; `[S] °C`; `[S] INSAT LST`; `[D] 29.8`; `[S] °C`.
- Forecast inspector: `[S] 7-day forecast`; `[S] mm/day`; `[D] 16`; `[D] Jul 16`; `[D] 22`; `[D] Jul 17`; `[D] 30`; `[D] Jul 18`; `[D] 48`; `[D] Jul 19`; `[D] 36`; `[D] Jul 20`; `[D] 24`; `[D] Jul 21`; `[D] 18`; `[D] Jul 22`.
- Inspector status/actions: `[S] Anomaly`; `[D] ↑ +14.2 mm above climatology (+78%)`; `[D] Monsoon surge active`; `[D] Active`; `[S] Time series`; `[S] Compare`; `[S] Export`.
- Timeline controls/context: `[D] 1x`; `[D] Jul 15, 2026`; `[D] 09:34:12 IST`; `[D] Today`; `[S] Now`; `[D] Replay 2018 flood`; `[D] Replay 2016 drought`.
- Timeline dates/legend: `[D] Jul 08`; `[D] Jul 09`; `[D] Jul 10`; `[D] Jul 11`; `[D] Jul 12`; `[D] Jul 13`; `[D] Jul 14`; `[D] Jul 15`; `[D] Jul 16`; `[D] Jul 17`; `[D] Jul 18`; `[D] Jul 19`; `[D] Jul 20`; `[D] Jul 21`; `[D] Jul 22`; `[D] TODAY`; `[S] Observed`; `[S] Forecast`.
- System state: `[D] Assimilation`; `[D] Cycle 46`; `[D] Complete`; `[S] Next cycle in`; `[D] 25 min 48 s`.
- Bottom metric strip: `[S] Rainfall (mm)`; `[D] 32.4`; `[S] Wind (km/h)`; `[D] 18`; `[S] Humidity (%)`; `[D] 86`.

## 10. Anti-slop checks

- Implement the narrow rail/full-screen map as a route-specific override. It is not the base overview dashboard shell; do not add a wide labeled sidebar or constrain the map to a card grid.
- Preserve the map as the primary surface and the exact overlay hierarchy: rail/header, layer panel, basin tabs, grid-cell inspector, timeline, system status, metric strip.
- Keep enabled/disabled layer states and `72%` opacity as Phase 1 placeholders. Do not silently activate every layer for visual richness.
- Preserve the gridded rainfall raster, rivers, wind streamlines, reservoir gauges, storm center/track, selected Mysuru cell, and observed-versus-forecast timeline treatment. Do not substitute a generic map with random pins.
- Keep electric blue for active/data states, green for live/positive/complete, orange for heat, and gray for disabled. Avoid purple AI gradients, neon glows, 3D bars, or excessive glass blur.
- All screenshot values remain verbatim Phase 1 placeholders, including apparently unusual labels such as `75°N`; do not “correct,” recalculate, geocode, or represent them as live until source data and coordinate formatting are validated.
- Do not invent map provider, model source, projection, exact layer blend mode, storm science, backend cadence, export format, comparison behavior, or scenario semantics.
- Prevent overlay collisions at this canvas and responsive sizes. Map labels may yield to panels, but essential controls, inspector content, and timeline state must remain legible.
- Provide non-color state cues, icon labels, focus visibility, keyboard map/timeline controls, pause/reduced-motion behavior, and a nonvisual equivalent for geospatial/chart data.
- This decode covers only `screen-02-map.png`; it makes no claims about Screens 8–10 or any unshown hover, loading, error, empty, mobile, or permissions state.
