"""
pages/30_What_If.py — Scenario Engine entry point.

Streamlit's multipage router auto-registers this file as a top-level
page because it sits at ``pages/*.py``. The numeric prefix ``30_``
orders it AFTER Home / Explorer and BEFORE the training + validation
pages that will migrate here in subsequent parts.

Part 1 delivers:
    * Two-tab shell (Short Term / Long Term).
    * A hidden "Engine self-check" expander inside the Short Term tab
      that calls ``load_driver(historical, tmean, 2020-06-01, all_india)``
      and reports shape, NaN %, min/max, units — a live L0 smoke test.

Part 2 extends the Short Term self-check with L1 index diagnostics:
    * ET0 (Hargreaves) — daily map for 2020-06-15.
    * Longest dry spell — map for June 2020.
    * SPI-3 for June 2020 (gated behind a compute button; fit cached).
    * INDEX_REGISTRY table with version + citation.

No user-facing scenario controls yet; those arrive in Part 6.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure `climate_twin.*` package imports resolve when Streamlit invokes
# this page directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────────────────────────────
# NOTE (Part 6): This file is the Parts-1–5 diagnostic page, imported
# as a module so the ``?dev=1`` gate on pages/30_What_If.py can call
# individual _render_* helpers without duplicating them. Its top-level
# runtime (set_page_config / title / caption / _tab_short with _tab_long)
# is retired — the main page owns those now.
# ──────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────
# Diagnostic helpers — kept local to the page so the rest of the app
# never picks up the heavy plotly / xarray import chain.
# ────────────────────────────────────────────────────────────────────────


def _map_from_da(
    da,                                   # xr.DataArray with (lat, lon) or a numpy 2-D array
    *,
    title: str,
    unit: str,
    colorscale: str,
    vmin: float | None = None,
    vmax: float | None = None,
    zmid: float | None = None,
    height: int = 380,
) -> go.Figure:
    """Small self-contained heatmap for the diagnostic panel.

    We deliberately avoid ``plotly_vis.plot_map`` — that singleton owns
    app-wide caches and re-uses colorscale keys geared to the legacy
    tabs. The diagnostic wants a plain, explicit map."""
    import xarray as xr

    if isinstance(da, xr.DataArray):
        arr = np.asarray(da.values, dtype=float)
        lat = np.asarray(da["lat"].values, dtype=float) if "lat" in da.coords else None
        lon = np.asarray(da["lon"].values, dtype=float) if "lon" in da.coords else None
    else:
        arr = np.asarray(da, dtype=float)
        lat = lon = None

    finite = np.isfinite(arr)
    if vmin is None and finite.any():
        vmin = float(np.nanpercentile(arr[finite], 2))
    if vmax is None and finite.any():
        vmax = float(np.nanpercentile(arr[finite], 98))

    kw = dict(
        z=arr,
        colorscale=colorscale,
        colorbar=dict(title=unit, thickness=12, len=0.85),
        zmin=vmin,
        zmax=vmax,
        hovertemplate="lat %{y:.2f}<br>lon %{x:.2f}<br>%{z:.2f} " + unit + "<extra></extra>",
    )
    if zmid is not None:
        kw["zmid"] = zmid
    if lat is not None and lon is not None:
        kw["x"] = lon
        kw["y"] = lat

    fig = go.Figure(data=go.Heatmap(**kw))
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=13)),
        margin=dict(l=8, r=8, t=32, b=8),
        height=height,
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange=True),
        xaxis=dict(constrain="domain"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_l0_smoketest():
    from climate_twin.whatif.config.region import RegionSpec
    from climate_twin.whatif.drivers.driver import DriverSpec, load_driver

    spec = DriverSpec(
        mode="historical",
        var="tmean",
        dates=(date(2020, 6, 1), date(2020, 6, 1)),
        region=RegionSpec(kind="all_india"),
    )
    da = load_driver(spec)

    arr = da.values
    finite = np.isfinite(arr)
    n_total = arr.size
    n_nan = int(np.isnan(arr).sum())
    nan_pct = 100.0 * n_nan / max(n_total, 1)

    cols = st.columns(4)
    cols[0].metric("shape", "×".join(str(int(x)) for x in da.shape))
    cols[1].metric("NaN %", f"{nan_pct:.1f}%")
    if finite.any():
        cols[2].metric("min", f"{float(arr[finite].min()):.2f} {da.attrs.get('units','')}")
        cols[3].metric("max", f"{float(arr[finite].max()):.2f} {da.attrs.get('units','')}")

    attrs = {
        "quantile": da.attrs.get("quantile", "—"),
        "source": da.attrs.get("source", "—"),
        "source_version": da.attrs.get("source_version", "—"),
        "units": da.attrs.get("units", "—"),
        "time[0]": str(da["time"].to_index()[0]),
        "lat range": f"{float(da.lat[0]):.2f} → {float(da.lat[-1]):.2f}",
        "lon range": f"{float(da.lon[0]):.2f} → {float(da.lon[-1]):.2f}",
    }
    st.markdown("**attrs / coords**")
    st.code("\n".join(f"{k:16s}  {v}" for k, v in attrs.items()), language="text")
    st.success("✓ L0 driver online — historical / all-India / tmean.")


def _render_et0_map():
    """ET0 Hargreaves — single-day map for 2020-06-15."""
    from climate_twin.whatif.drivers.historical import get_historical
    from climate_twin.whatif.indices.et0_hargreaves import et0_hargreaves

    day = date(2020, 6, 15)
    tmax = get_historical("tmax", day, day)
    tmin = get_historical("tmin", day, day)
    et0 = et0_hargreaves(tmax, tmin).isel(time=0)

    fin = np.isfinite(et0.values)
    if fin.any():
        mean = float(np.nanmean(et0.values))
        p5 = float(np.nanpercentile(et0.values[fin], 5))
        p95 = float(np.nanpercentile(et0.values[fin], 95))
    else:
        mean = p5 = p95 = float("nan")

    # Blues → Oranges divergent, centred on the India-mean ET0 that day.
    div_colorscale = [
        [0.00, "#1f4e79"],
        [0.25, "#5b9bd5"],
        [0.50, "#f5f5f5"],
        [0.75, "#f0a04b"],
        [1.00, "#c25c1c"],
    ]
    fig = _map_from_da(
        et0,
        title=f"ET0 (Hargreaves) — {day.isoformat()}",
        unit="mm/day",
        colorscale=div_colorscale,
        vmin=p5, vmax=p95, zmid=mean,
    )
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("mean ET0", f"{mean:.2f} mm/day")
    c2.metric("p5", f"{p5:.2f} mm/day")
    c3.metric("p95", f"{p95:.2f} mm/day")
    st.caption(
        "Hargreaves-Samani 1985 (FAO-56 §3 fallback). "
        f"attrs.source_chain = `{et0.attrs.get('source_chain', '—')}`  |  "
        f"units = `{et0.attrs.get('units', '—')}`"
    )


def _render_dry_spell_map():
    """Longest dry spell — map for June 2020 (30 days)."""
    from climate_twin.whatif.drivers.historical import get_historical
    from climate_twin.whatif.indices.dry_spell import (
        DRY_THRESHOLD_MM,
        longest_dry_spell,
    )

    start, end = date(2020, 6, 1), date(2020, 6, 30)
    rain = get_historical("rain", start, end)
    lds = longest_dry_spell(rain)

    fig = _map_from_da(
        lds,
        title=f"Longest consecutive dry-day run — June 2020 (< {DRY_THRESHOLD_MM} mm/day)",
        unit="days",
        colorscale="YlOrBr",     # sequential; explicitly NOT jet
        vmin=0, vmax=30,
    )
    st.plotly_chart(fig, use_container_width=True)
    fin = np.isfinite(lds.values)
    if fin.any():
        cols = st.columns(3)
        cols[0].metric("mean", f"{float(np.nanmean(lds.values)):.1f} days")
        cols[1].metric("p90", f"{float(np.nanpercentile(lds.values[fin], 90)):.0f} days")
        cols[2].metric("max", f"{float(np.nanmax(lds.values)):.0f} days")
    st.caption(
        f"Rain-day = ≥ {DRY_THRESHOLD_MM} mm (IMD). Run-length over the 30 days "
        "in June 2020. Method: consecutive-run-length."
    )


def _render_spi3_map():
    """SPI-3 for June 2020, evaluated against a TRAIN_YEARS fit.

    The fit is expensive on first run (~8.4 M per-cell gamma MLEs)
    so we gate it behind an explicit button and cache-check first.
    """
    from climate_twin.whatif.drivers.historical import get_historical
    from climate_twin.whatif.indices.reference import TRAIN_YEARS
    from climate_twin.whatif.indices.spi import (
        _fit_cache_path,
        fit_spi_from_cube,
        spi,
    )

    cache = _fit_cache_path(3)
    cache_exists = cache.exists()

    if cache_exists:
        st.caption(
            f"SPI-3 fit cached at `{cache.relative_to(_PROJECT_ROOT) if str(cache).startswith(_PROJECT_ROOT) else cache}` "
            f"— fit period {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]}."
        )
        do_run = True
    else:
        st.warning(
            f"SPI-3 fit not cached yet. First run does 12 × {129 * 135:,} "
            f"per-cell Gamma MLEs over {TRAIN_YEARS[1] - TRAIN_YEARS[0] + 1} "
            "years — expect ~5–10 minutes."
        )
        do_run = st.button("Compute SPI-3 fit (one-shot; cached afterwards)", key="spi3_fit_btn")

    if not do_run:
        return

    with st.spinner("Fitting mixed-Gamma SPI-3 over TRAIN_YEARS…"):
        fit = fit_spi_from_cube(accum_months=3)

    # SPI needs at least 3 months of history to yield June's value.
    start = date(2020, 4, 1)
    end = date(2020, 6, 30)
    rain = get_historical("rain", start, end)
    with st.spinner("Applying fit to Apr–Jun 2020…"):
        spi3 = spi(rain, fit)

    # Pick June's SPI (monthly grid → the June row)
    times = pd.DatetimeIndex(spi3["time"].values)
    june_mask = (times.year == 2020) & (times.month == 6)
    if not june_mask.any():
        st.error("SPI-3 output has no June 2020 entry — accumulation window mismatch.")
        return
    june_idx = int(np.flatnonzero(june_mask)[-1])
    june_spi = spi3.isel(time=june_idx)

    # Divergent RdBu centred on 0
    fig = _map_from_da(
        june_spi,
        title=f"SPI-3 — June 2020 (fit: {fit.train_years[0]}–{fit.train_years[1]})",
        unit="σ",
        colorscale="RdBu",
        vmin=-3.0, vmax=3.0, zmid=0.0,
    )
    st.plotly_chart(fig, use_container_width=True)

    arr = june_spi.values
    fin = np.isfinite(arr)
    if fin.any():
        cols = st.columns(4)
        cols[0].metric("mean", f"{float(np.nanmean(arr)):+.2f} σ")
        cols[1].metric("p10", f"{float(np.nanpercentile(arr[fin], 10)):+.2f} σ")
        cols[2].metric("p90", f"{float(np.nanpercentile(arr[fin], 90)):+.2f} σ")
        cols[3].metric("dry cells (<-1σ)", f"{100.0 * float(np.mean(arr[fin] < -1.0)):.1f} %")
    st.caption(
        f"Fit version `{fit.version}` · reference period "
        f"{fit.train_years[0]}–{fit.train_years[1]} (TRAIN_YEARS). "
        f"attrs.source_chain = `{june_spi.attrs.get('source_chain', '—')}`."
    )


def _render_agriculture_diagnostic():
    """Agriculture sector — paddy over Vidarbha, sow 2020-06-15.

    Runs the full L0→L2→L3 chain on the observed IMD driver as a
    three-fold synthetic quantile bundle, then displays:
        * a map of Ya (t/ha) on the master grid;
        * a district (states-fallback) table of Ya/Ymax;
        * a q10/q50/q90 expected-yield table;
        * a ranked sowing-window DataFrame;
        * a Validation card with the latest APY comparison, or a
          "not-yet-ingested" banner if APY isn't on disk.
    """
    from datetime import date as _date

    from climate_twin.whatif.biophysical.water_balance import water_balance
    from climate_twin.whatif.config.region import RegionSpec
    from climate_twin.whatif.drivers.historical import get_historical
    from climate_twin.whatif.indices.et0_hargreaves import et0_hargreaves
    from climate_twin.whatif.sectors import (
        DistrictRegistry,
        apy_available,
        build_deterministic_bundle,
        get_last_validation,
        optimize_sowing_window,
        to_district,
        yield_water_limited,
    )
    from climate_twin.whatif.sectors.crops import load_crop, list_crops

    st.caption(
        "Region: Vidarbha (bbox 18–22 °N, 76–82 °E — falls back to the "
        "declared bbox because a district shapefile isn't bundled yet). "
        "Crop: paddy_kharif. Sow date: 2020-06-15. Driver: observed IMD "
        "rain / tmax / tmin over the crop's full duration."
    )
    st.caption(
        f"Registered crops: {', '.join(list_crops())}"
    )

    crop = load_crop("paddy_kharif")
    sow = _date(2020, 6, 15)
    end = sow.fromordinal(sow.toordinal() + crop.total_days - 1)

    region = RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0))

    with st.spinner(f"Loading IMD rain/tmax/tmin over {sow.isoformat()} → {end.isoformat()}…"):
        rain = get_historical("rain", sow, end)
        tmax = get_historical("tmax", sow, end)
        tmin = get_historical("tmin", sow, end)

    # Restrict to the Vidarbha bbox
    from climate_twin.whatif.config.region import apply_region
    rain = apply_region(rain, region)
    tmax = apply_region(tmax, region)
    tmin = apply_region(tmin, region)

    with st.spinner("ET0 (Hargreaves) → water balance → yield…"):
        et0 = et0_hargreaves(tmax, tmin)
        wb = water_balance(crop, rain, et0, sow, region, irrigation=None)
        yg = yield_water_limited(crop, wb, tmax=tmax)

    # ── Provenance banner ──
    warn = wb.attrs.get("soil_warning") or ""
    if warn:
        st.warning(f"⚠️ {warn}")
    st.code(
        "\n".join([
            f"crop:               {crop.key} ({crop.common_name})",
            f"crop.registry:      {crop.registry_version}  sha={crop.registry_sha256}",
            f"agriculture.version: {yg.attrs.get('version', '?')}",
            f"water_balance.ver:  {wb.attrs.get('version', '?')}",
            f"soil.source:        {wb.attrs.get('soil_source', '?')} ({wb.attrs.get('soil_source_version', '?')})",
            f"quantile:           {yg.attrs.get('quantile', '?')}",
            f"source_chain:       {yg.attrs.get('source_chain', '?')}",
        ]), language="text",
    )

    # ── Ya map ──
    st.markdown("**Ya — absolute yield (t/ha) on the master grid**")
    fig = _map_from_da(
        yg["Ya"],
        title=f"Paddy Ya — sow {sow.isoformat()} (Vidarbha bbox)",
        unit="t/ha",
        colorscale="YlGn",
        vmin=0.0, vmax=float(crop.ymax_t_per_ha),
    )
    st.plotly_chart(fig, use_container_width=True)

    fin = np.isfinite(yg["Ya"].values)
    if fin.any():
        arr = yg["Ya"].values[fin]
        cols = st.columns(4)
        cols[0].metric("mean Ya", f"{float(arr.mean()):.2f} t/ha")
        cols[1].metric("p10 Ya", f"{float(np.percentile(arr, 10)):.2f} t/ha")
        cols[2].metric("p90 Ya", f"{float(np.percentile(arr, 90)):.2f} t/ha")
        cols[3].metric("mean Ya/Ymax", f"{float(np.nanmean(yg['Ya_over_Ymax'].values)):.2f}")

    # ── District (states-fallback) table ──
    st.markdown("---")
    st.markdown("**Ya/Ymax by zone (states-as-districts fallback)**")
    try:
        # Build a zone-based fallback registry, then intersect each mask
        # with the yield grid's (lat, lon) subset.
        reg_full = DistrictRegistry.states_fallback()
        # Subset each zone mask to the yield_ds axis
        target_lat = yg["lat"].values
        target_lon = yg["lon"].values
        lat_axis = rain["lat"].values          # source master grid — same as awc
        lon_axis = rain["lon"].values          # (already subset to bbox)
        # For states-fallback we need masks on the *yield* grid, so
        # slice by matching indices.
        from climate_twin.regions import get_zones
        Z = get_zones()
        hard = Z.hard_mask                     # (129, 135) on master grid

        # Build lat/lon → index maps for the master grid
        from climate_twin.whatif.config.region import master_axes
        m_lat, m_lon = master_axes()
        lat_idx = [int(np.argmin(np.abs(m_lat - la))) for la in target_lat]
        lon_idx = [int(np.argmin(np.abs(m_lon - lo))) for lo in target_lon]
        lat_idx = np.array(lat_idx)
        lon_idx = np.array(lon_idx)
        sub_hard = hard[np.ix_(lat_idx, lon_idx)]

        mapping = {}
        for zone in Z.zones:
            m = (sub_hard == zone.id)
            if m.any():
                mapping[(zone.key, zone.key)] = m
        reg = DistrictRegistry(mapping=mapping)
        reg.min_area_km2 = DistrictRegistry.MIN_DISTRICT_AREA_KM2
        df_dist = to_district(yg, reg)
        st.dataframe(
            df_dist.style.format({
                "Ya_t_per_ha": "{:.2f}",
                "Ya_over_Ymax": "{:.2f}",
                "valid_frac": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
    except Exception as e:
        st.info(
            f"Zone-fallback aggregation unavailable ({type(e).__name__}: {e}). "
            "A proper district shapefile will replace this in Part 4."
        )

    # ── q10/q50/q90 expected-yield triple ──
    st.markdown("---")
    st.markdown("**Sowing-window optimiser — Ey across q10/q50/q90**")
    st.caption(
        "The observed driver is played back through the pipeline as a "
        "three-quantile bundle. Once forecast ensembles are wired in "
        "Part 5, q10 / q50 / q90 will differ; today they collapse to "
        "the observed series."
    )
    from datetime import date as _dc
    candidates = [
        _date(2020, 6, 1),  _date(2020, 6, 6),  _date(2020, 6, 11),
        _date(2020, 6, 16), _date(2020, 6, 21), _date(2020, 6, 26),
        _date(2020, 7, 1),  _date(2020, 7, 6),  _date(2020, 7, 11),
    ]
    try:
        # For the optimiser we need each sow date's crop-duration window
        # fully in the driver. Re-load a wider driver here.
        latest_end = max(_date.fromordinal(d.toordinal() + crop.total_days - 1)
                         for d in candidates)
        rain2 = apply_region(get_historical("rain", candidates[0], latest_end), region)
        tmax2 = apply_region(get_historical("tmax", candidates[0], latest_end), region)
        tmin2 = apply_region(get_historical("tmin", candidates[0], latest_end), region)
        bundle = build_deterministic_bundle(rain2, tmax2, tmin2)
        with st.spinner(f"Optimising over {len(candidates)} candidate sow dates…"):
            ranked = optimize_sowing_window(crop, region, bundle, candidates)
        st.dataframe(
            ranked.style.format({
                "Ey": "{:.2f}", "Ey_over_Ymax": "{:.2f}",
                "p_good_year": "{:.2f}",
                "Ya_q10": "{:.2f}", "Ya_q50": "{:.2f}", "Ya_q90": "{:.2f}",
                "worst_case_Ya": "{:.2f}",
                "heat_stress_days_flower_q50": "{:.1f}",
                "baseline_Ya": "{:.2f}",
                "value_vs_baseline": "{:+.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
    except Exception as e:
        st.error(f"❌ sowing-window optimiser failed — {type(e).__name__}: {e}")

    # ── Validation card ──
    st.markdown("---")
    st.markdown("**Validation — APY comparison**")
    v = get_last_validation("paddy_kharif")
    if not v["ok"]:
        st.info(
            "APY snapshot not yet ingested — model unvalidated.\n\n"
            f"reason: {v['reason']}"
        )
    else:
        st.success(f"APY validation from `{v['parquet_path']}`")
        cols = st.columns(4)
        s = v["scores"]
        cols[0].metric("districts", v["n_districts"])
        cols[1].metric("Pearson r", f"{s.get('pearson_r', float('nan')):.2f}")
        cols[2].metric("RMSE", f"{s.get('rmse_t_ha', float('nan')):.2f} t/ha")
        cols[3].metric("MPE", f"{s.get('mpe_pct', float('nan')):.1f} %")


def _render_analogs_diagnostic():
    """L5 diagnostic — analog engine over Vidarbha, JJAS window.

    Two flavours are rendered side-by-side:
      * Method 2 (analogs)      — the "strong" flavour. Historical
                                   years similar to a target year in a
                                   physically-motivated feature vector,
                                   with distance + quality tier badges.
      * Method 1 (perturbation) — the delta-change sensitivity strip.
                                   Ships the physical-inconsistency
                                   caveat banner as required by Rule 4.

    The target year is a fixed 2020 (observed) — Part 6 will replace
    this with the live forecast when the ensemble → forecast adapter
    lands.
    """
    from datetime import date as _date

    import plotly.graph_objects as go

    from climate_twin.whatif.config.region import RegionSpec
    from climate_twin.whatif.drivers.analog_features import (
        DEFAULT_FEATURES,
        AnalogSpec,
        build_feature_matrix,
    )
    from climate_twin.whatif.drivers.analog_outcomes import _compute_weights
    from climate_twin.whatif.drivers.analogs import (
        build_analog_pool,
        find_analogs,
        target_from_year,
    )
    from climate_twin.whatif.drivers.perturbation import (
        CaveatRequiredError,
        PerturbationSpec,
        apply_perturbation,
    )

    region = RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0))
    spec = AnalogSpec(
        region=region,
        window="JJAS",
        features=DEFAULT_FEATURES,
        metric="mahalanobis",
    )

    # 1) Feature matrix — heavy on first call (walks 40 years of the
    # cube). Cached to CACHE_DIR/analog_features/ so subsequent calls
    # are ~O(ms). Gate on a button so the page doesn't recompute on
    # every rerun.
    st.caption(
        "Region: Vidarbha bbox (18–22 °N, 76–82 °E). Window: JJAS. "
        "Feature vector: rain_total, rx5day, cdd, tmax_anom, tmin_anom, "
        "onset_offset, spi3_regional. Metric: Mahalanobis (χ² tiers "
        "on df=7)."
    )
    target_year = st.number_input(
        "Target year (observed)", value=2020, min_value=2011, max_value=2022,
        step=1, key="analog_target_year",
    )
    k = int(st.slider("k analogs", 3, 20, 10, key="analog_k"))
    weighting = st.selectbox(
        "weighting", ["uniform", "inv_distance", "softmax"],
        index=1, key="analog_weighting",
    )

    do_run = st.button("Compute analogs (may take ~30 s first time)",
                         key="analog_go")
    if not do_run:
        st.info("Click **Compute analogs** to run the feature builder + "
                 "retrieval. Results are cached after the first run.")
        return

    with st.spinner("Building feature matrix over TRAIN_YEARS + target…"):
        # Exclude the target year and everything after it (Rule 1).
        excl = tuple(range(int(target_year),
                            spec.train_years[1] + 20))
        # Build a pool that has TRAIN_YEARS only.
        pool = build_analog_pool(
            spec, exclude_years=excl, pool_years=spec.train_years,
        )

    if not pool.years():
        st.warning(
            "Analog pool is empty — feature construction dropped every "
            "candidate year (missing data in TRAIN_YEARS window). "
            "Investigate the cube coverage."
        )
        return

    with st.spinner(f"Computing target features for {int(target_year)}…"):
        try:
            target = target_from_year(int(target_year), spec)
        except Exception as e:
            st.error(f"target_from_year failed — {type(e).__name__}: {e}")
            return

    matches = find_analogs(target, pool, k=k)
    if not matches:
        st.warning("No analog matches returned.")
        return

    # 1) Analog table
    st.markdown("**Top-k analog matches**")
    rows = []
    for m in matches:
        rows.append({
            "Year": m.year,
            "Distance": f"{m.distance:.2f}",
            "Quality": m.quality,
            "rain_total_std": f"{float(m.features_analog['rain_total_std']):+.2f}",
            "tmax_mean_anom": f"{float(m.features_analog['tmax_mean_anom']):+.2f} °C",
            "onset_offset_days": f"{float(m.features_analog['onset_offset_days']):+.1f} d",
        })
    df_a = pd.DataFrame(rows)
    st.dataframe(df_a, use_container_width=True, hide_index=True)

    strong = [m for m in matches if m.quality == "strong"]
    fair = [m for m in matches if m.quality == "fair"]
    poor = [m for m in matches if m.quality == "poor"]
    st.caption(
        f"{len(strong)} strong, {len(fair)} fair, {len(poor)} poor. "
        "Rule 7: never lie about analog count — poor matches are "
        "shown, not hidden. Van den Dool 1994 warns that at seasonal "
        "horizons in a 40-year record, strong analogs are scarce."
    )
    narrative = ", ".join(f"{m.year} (d={m.distance:.2f}, {m.quality})"
                            for m in matches[:3])
    st.info(f"This year most resembles: {narrative}.")

    # 2) Weight distribution
    st.markdown("---")
    st.markdown(f"**Weight distribution — {weighting}**")
    weights_by_year = _compute_weights(matches, weighting, 1.0)
    years_sorted = sorted(weights_by_year.keys())
    fig_w = go.Figure(go.Bar(
        x=[str(y) for y in years_sorted],
        y=[weights_by_year[y] for y in years_sorted],
        marker_color="#3b7ad9",
    ))
    fig_w.update_layout(
        title=dict(text=f"weight w_i per analog year — {weighting}",
                   x=0.02, font=dict(size=13)),
        xaxis=dict(title="analog year"), yaxis=dict(title="weight"),
        height=280, margin=dict(l=8, r=8, t=42, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # 3) Delta-perturbation sensitivity strip (Method 1)
    st.markdown("---")
    st.markdown("**Delta-perturbation sensitivity strip (Method 1)**")
    st.warning(
        "⚠️ Method 1 perturbation — physically inconsistent by "
        "construction (Räisänen & Räty 2013). Prefer analogs (Method 2) "
        "for reported scenarios."
    )
    rain_scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    tmax_shifts = [0.0, 1.0, 2.0]
    # Build a base dataset from observed data for the target-year JJAS
    from climate_twin.whatif.config.region import apply_region
    from climate_twin.whatif.drivers.historical import get_historical

    with st.spinner("Loading target-year JJAS observed base…"):
        start = _date(int(target_year), 6, 1)
        end = _date(int(target_year), 9, 30)
        rain_o = apply_region(get_historical("rain", start, end), region)
        tmax_o = apply_region(get_historical("tmax", start, end), region)
        tmin_o = apply_region(get_historical("tmin", start, end), region)
    base_ds = xr.Dataset({"rain": rain_o, "tmax": tmax_o, "tmin": tmin_o},
                          attrs={"source_chain": f"observed-JJAS-{int(target_year)}"})

    grid = np.full((len(tmax_shifts), len(rain_scales)), np.nan, dtype=np.float64)
    for i, shift in enumerate(tmax_shifts):
        for j, rs in enumerate(rain_scales):
            try:
                out = apply_perturbation(
                    base_ds,
                    PerturbationSpec(
                        rain_scale=float(rs), tmax_shift_c=float(shift),
                        caveat_acknowledged=True,     # UI has shown the caveat
                    ),
                )
                # Cheap summary: mean rainfall × Δtmax as a proxy metric —
                # the sector-level ₹/ha response would be Part-6 heavy.
                grid[i, j] = float(np.nanmean(out["rain"].values))
            except CaveatRequiredError:
                grid[i, j] = float("nan")
    fig_p = go.Figure(go.Heatmap(
        z=grid,
        x=[f"× {v:.1f}" for v in rain_scales],
        y=[f"+{v:.0f} °C" for v in tmax_shifts],
        colorscale="RdBu",
        colorbar=dict(title="mean rain (mm/d)", thickness=12),
    ))
    fig_p.update_layout(
        title=dict(text="Perturbation strip — mean JJAS rain response",
                   x=0.02, font=dict(size=13)),
        xaxis=dict(title="rain_scale"),
        yaxis=dict(title="tmax_shift"),
        height=240, margin=dict(l=8, r=8, t=42, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.caption(
        "This strip is diagnostic only. The Part-6 UI will replace it "
        "with a decision-level Δ net-revenue heatmap driven by the "
        "sector runner, still gated behind the caveat."
    )

    # 4) Provenance drill-down extension
    st.markdown("---")
    st.markdown("**Provenance drill-down — Part 5**")
    lines = [
        f"drivers.analog_features   {spec.version}  sig={spec.signature()}",
        f"drivers.analogs           analogs-v1",
        f"drivers.analog_outcomes   analog-outcomes-v1",
        f"drivers.perturbation      perturbation-v1 (Method 1)",
        f"drivers.state_builder     state-builder-v1",
        f"analog metric             {spec.metric}",
        f"analog window             {spec.window}",
        f"pool size (train-only)    {len(pool.years())} years",
        f"excluded from pool        {list(excl)[:5]}{'…' if len(excl) > 5 else ''}",
    ]
    st.code("\n".join(lines), language="text")


def _render_decisions_diagnostic():
    """L4 diagnostic — payoff matrix over three decisions × three climate
    states, using observed IMD data for 2020, then the decision-layer
    utilities: recommendation, cost-loss value curve, walk-forward
    backtest, tornado, provenance drill-down.
    """
    from datetime import date as _date

    from climate_twin.whatif.config.region import RegionSpec
    from climate_twin.whatif.drivers.driver import DriverSpec
    from climate_twin.whatif.economics import (
        SHIPPED_SETUPS,
        ClimateState,
        Decision,
        HistoricalFrequencyRule,
        OnsetAnomalyRule,
        expected_value_with_uncertainty,
        prices_registry_sha256,
        prices_registry_version,
        recommend,
        synthesise_history,
        tornado,
        value_curve,
        walk_forward_backtest,
    )
    from climate_twin.whatif.report import format_inr, payoff_heatmap
    from climate_twin.whatif.sectors import run_decision_scenario

    region = RegionSpec(kind="bbox", bbox=(18.0, 76.0, 22.0, 82.0))
    sow_iso = "2020-06-15"

    # Driver spec anchors the sector runner (rain, tmax, tmin will be
    # pulled from historical for these dates + region)
    driver_spec = DriverSpec(
        mode="historical",
        var="rain",                       # sector runner replaces per-var
        dates=(_date(2020, 6, 1), _date(2020, 11, 30)),
        region=region,
    )

    decisions = [
        Decision(label="Paddy sow 2020-06-15",  kind="crop",
                 params=(("crop", "paddy_kharif"), ("sow_date", sow_iso))),
        Decision(label="Bajra sow 2020-06-15",  kind="crop",
                 params=(("crop", "bajra_kharif"), ("sow_date", sow_iso))),
        Decision(label="Fallow",                 kind="fallow",
                 params=(("crop", "paddy_kharif"),)),
    ]
    states = [
        ClimateState(label="rain × 0.8", weight=0.28,
                     perturbation=(("rain_scale", 0.8),)),
        ClimateState(label="rain × 1.0", weight=0.51,
                     perturbation=(("rain_scale", 1.0),)),
        ClimateState(label="rain × 1.2", weight=0.21,
                     perturbation=(("rain_scale", 1.2),)),
    ]

    with st.spinner("Running L2+L3+L4 across 3 decisions × 3 climate states…"):
        bundle = run_decision_scenario(
            decisions, states, region,
            driver_spec=driver_spec,
            crop_key="paddy_kharif",
            season="2024-25",
            sow_date_iso=sow_iso,
        )

    pm = bundle["payoff_matrix"]

    # 1) Payoff heatmap
    st.markdown("**Payoff matrix — net revenue ₹/ha (q50)**")
    st.plotly_chart(payoff_heatmap(pm), use_container_width=True)

    # 2) Recommendation card
    rec = bundle["recommendation"]
    st.markdown("**Recommendation**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best-EV decision", rec["argmax_EV_label"])
    c2.metric("EV q50", format_inr(rec["EV_q50"]))
    c3.metric("Worst-case (q50)", format_inr(rec["worst_case_q50"]))
    c4.metric("Δ vs baseline (EV)", format_inr(rec["delta_vs_baseline_EV"]))
    st.caption(
        f"EV band q10 → q90: {format_inr(rec['EV_q10'])} → "
        f"{format_inr(rec['EV_q90'])}. Minimax-regret pick: "
        f"**{rec['argmin_maxRegret_label']}** (max regret "
        f"{format_inr(rec['maxRegret_at_argmin'])}). "
        f"VaR₁₀ = {format_inr(rec['VaR_10'])}, "
        f"CVaR₁₀ = {format_inr(rec['CVaR_10'])}. "
        f"Baseline EV: {format_inr(rec['baseline_EV'])}."
    )
    if rec["delta_vs_baseline_EV"] > 0:
        st.success(
            f"Beats climatology by {format_inr(rec['delta_vs_baseline_EV'])}/ha "
            "in expectation."
        )
    else:
        st.warning(
            f"Does not beat climatology "
            f"({format_inr(rec['delta_vs_baseline_EV'])}/ha in expectation)."
        )

    # 3) Cost–loss value curve (synthetic forecast for now — real
    # ensemble hookup lands in Part 5)
    st.markdown("---")
    st.markdown("**Cost–loss value curve** — "
                "preventive irrigation if dry-spell forecast")
    hist = synthesise_history(seed=13)
    setup = SHIPPED_SETUPS["preventive_irrigation_if_dry_spell_forecast"]
    p_clim = float(hist["event"].mean())
    # Turn onset_anomaly into a soft forecast probability
    fcast_prob = (0.35 + 0.05 * hist["onset_anomaly_days"]).clip(0.02, 0.98)
    vc = value_curve(setup, fcast_prob, hist["event"], p_clim)
    import plotly.graph_objects as go
    fig_vc = go.Figure()
    fig_vc.add_trace(go.Scatter(
        x=vc.thresholds, y=vc.V, mode="lines+markers",
        line=dict(color="#2a6bcc", width=2),
        name="V(p*)",
    ))
    fig_vc.add_hline(y=0, line=dict(color="#999", dash="dot"))
    fig_vc.add_hline(y=1, line=dict(color="#666", dash="dot"))
    fig_vc.update_layout(
        title=dict(text=f"Relative economic value V(p*) — "
                         f"setup={setup.action_id}", font=dict(size=13), x=0.02),
        xaxis=dict(title="threshold p*"),
        yaxis=dict(title="V (0 = climatology, 1 = perfect)",
                     range=[-0.2, 1.05]),
        height=280, margin=dict(l=8, r=8, t=42, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_vc, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V_best", f"{vc.V_best:+.3f}")
    c2.metric("best threshold", f"{vc.best_threshold:.2f}")
    c3.metric("Brier", f"{vc.brier:.3f}")
    c4.metric("BSS vs clim", f"{vc.brier_skill:+.3f}")

    # 4) Walk-forward backtest — publish honestly
    st.markdown("---")
    st.markdown("**Walk-forward backtest** — shipped rules over VALID_YEARS")
    rules_to_show = [HistoricalFrequencyRule(), OnsetAnomalyRule()]
    rows = []
    for rule in rules_to_show:
        try:
            bt = walk_forward_backtest(
                rule, setup, hist, write_parquet=False,
            )
            rows.append({
                "rule": rule.rule_id,
                "V_forecast": bt.V_forecast,
                "ME_forecast": bt.ME_forecast,
                "ME_climatology": bt.ME_climatology,
                "ME_perfect": bt.ME_perfect,
                "Brier": bt.brier,
                "BSS": bt.brier_skill,
                "n_years": len(bt.per_year),
            })
        except Exception as e:
            rows.append({
                "rule": rule.rule_id, "V_forecast": float("nan"),
                "error": f"{type(e).__name__}: {e}",
            })
    bt_df = pd.DataFrame(rows)
    st.dataframe(
        bt_df.style.format({
            "V_forecast": "{:+.3f}",
            "ME_forecast": "{:.1f}",
            "ME_climatology": "{:.1f}",
            "ME_perfect": "{:.1f}",
            "Brier": "{:.3f}",
            "BSS": "{:+.3f}",
        }),
        use_container_width=True, hide_index=True,
    )
    negatives = [r for r in rows if r.get("V_forecast", 0) is not None
                 and np.isfinite(r.get("V_forecast", float("nan")))
                 and r["V_forecast"] <= 0]
    if negatives:
        st.info(
            f"{len(negatives)} rule(s) failed to beat climatology (V ≤ 0). "
            "Published as-is — hiding negative results would defeat the point."
        )

    # 5) Tornado on the best-EV decision
    st.markdown("---")
    st.markdown("**Tornado — top drivers of net revenue (best-EV decision)**")
    best_dec = pm.decisions[rec["argmax_EV"]]

    def _scenario_fn(levers):
        # Fresh call to the sector runner with the passed levers merged
        # onto the best decision's params + our fixed climate state
        # (median rain × 1.0).
        merged = {
            "crop": best_dec.params_dict().get("crop", "paddy_kharif"),
            "sow_date": best_dec.params_dict().get("sow_date", sow_iso),
            "season": "2024-25",
            "overrides": dict(levers.get("overrides") or {}),
        }
        # Anchor climate state on rain*1.0 — a fair comparator
        merged["overrides"].setdefault("rain_scale", 1.0)
        from climate_twin.whatif.sectors import run_agriculture_scenario
        result = run_agriculture_scenario(driver_spec, merged)
        eo = result.get("economic_outcome")
        if eo is None:
            raise RuntimeError("scenario_fn: no economic_outcome returned")
        return eo

    try:
        tor = tornado(_scenario_fn, base_levers={}, delta_pct=0.20)
        st.dataframe(
            tor.rows.head(5).style.format({
                "delta_signed": "{:+.2f}",
                "net_p50_up": "{:.0f}",
                "net_p50_dn": "{:.0f}",
                "range": "{:.0f}",
                "signed_range": "{:+.0f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.info(tor.sentence())
    except Exception as e:
        st.warning(
            f"tornado skipped — {type(e).__name__}: {e}. "
            "This is fine: the tornado depends on the sector runner "
            "picking up every override key. Details will be robust "
            "once Part 5's ensemble driver lands."
        )

    # 6) Provenance drill-down
    st.markdown("---")
    st.markdown("**Provenance drill-down**")
    prov_sample = pm.provenance[0]["valuation"] if pm.provenance else {}
    lines = [
        f"code_hash            : (from scenarios/provenance.py)",
        f"prices_registry.ver  : {prices_registry_version()}",
        f"prices_registry.sha  : {prices_registry_sha256()}",
        f"valuation.version    : {prov_sample.get('valuation_version', '?')}",
        f"MSP season           : {prov_sample.get('msp_season', '?')}",
        f"MSP ₹/qt             : {prov_sample.get('msp_inr_per_qt', '?')}",
        f"cost ₹/ha            : {prov_sample.get('cost_of_cultivation_inr_per_ha', '?')}",
        f"discounts (mois+trn) : "
        f"{prov_sample.get('moisture_discount_pct', 0) + prov_sample.get('transport_marketing_pct', 0):.3f}",
        f"crop registry sha    : {prov_sample.get('crop_registry_sha256', '?')}",
    ]
    st.code("\n".join(lines), language="text")


def _render_registry_table():
    from climate_twin.whatif.indices import INDEX_REGISTRY

    rows = []
    for name, spec in INDEX_REGISTRY.items():
        rows.append({
            "index": name,
            "version": spec.version,
            "inputs": ", ".join(spec.inputs),
            "units": spec.units,
            "needs_fit": "✓" if spec.needs_fit else "",
            "citation": spec.citation,
        })
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"{len(df)} L1 indices registered. Each `version` is pinned in "
        "provenance; bumping the formula bumps the version."
    )


# (retired) Runtime block that once drew ``st.tabs(...)`` and the
# "Engine self-check" expander lives in pages/30_What_If.py's ?dev=1
# gate. This module only exports the _render_* helpers.
__all__ = [
    "_render_l0_smoketest",
    "_render_et0_map",
    "_render_dry_spell_map",
    "_render_spi3_map",
    "_render_registry_table",
    "_render_agriculture_diagnostic",
    "_render_decisions_diagnostic",
    "_render_analogs_diagnostic",
]
