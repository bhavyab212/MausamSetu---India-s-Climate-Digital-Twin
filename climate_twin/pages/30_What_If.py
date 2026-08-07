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


st.set_page_config(
    page_title="What If — Scenario Engine · MausamSetu",
    page_icon="❓",
    layout="wide",
)

st.title("What If — Scenario Engine")
st.caption(
    "Scenario-driven climate reasoning: perturb history, resample analog "
    "years, or project SSP futures, then propagate through indices, "
    "biophysical models, sectors, and economics."
)

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


_tab_short, _tab_long = st.tabs(["Short Term", "Long Term"])

with _tab_short:
    st.info("Scenario controls come online in Part 6.")

    with st.expander("Engine self-check", expanded=False):
        st.markdown("#### L0 — Historical driver")
        st.caption(
            "Live L0 driver smoke test. Calls `load_driver` for "
            "historical all-India tmean on 2020-06-01 and reports the "
            "returned DataArray's shape, NaN%, range, and attrs. If this "
            "fails, the backend is not wired correctly."
        )
        try:
            _render_l0_smoketest()
        except Exception as e:
            st.error(f"❌ L0 self-check failed — {type(e).__name__}: {e}")

        st.divider()
        st.markdown("#### L1 — Climate indices")

        st.markdown("**ET0 (Hargreaves) — 2020-06-15**")
        try:
            _render_et0_map()
        except Exception as e:
            st.error(f"❌ ET0 map failed — {type(e).__name__}: {e}")

        st.markdown("---")
        st.markdown("**Longest dry spell — June 2020**")
        try:
            _render_dry_spell_map()
        except Exception as e:
            st.error(f"❌ dry-spell map failed — {type(e).__name__}: {e}")

        st.markdown("---")
        st.markdown("**SPI-3 — June 2020 vs TRAIN_YEARS fit**")
        try:
            _render_spi3_map()
        except Exception as e:
            st.error(f"❌ SPI-3 map failed — {type(e).__name__}: {e}")

        st.divider()
        st.markdown("#### INDEX_REGISTRY")
        try:
            _render_registry_table()
        except Exception as e:
            st.error(f"❌ registry table failed — {type(e).__name__}: {e}")

with _tab_long:
    st.info("Coming online in Part 6.")
