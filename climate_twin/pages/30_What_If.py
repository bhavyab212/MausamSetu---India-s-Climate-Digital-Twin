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

No user-facing controls yet; those arrive in Part 6.
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

_tab_short, _tab_long = st.tabs(["Short Term", "Long Term"])

with _tab_short:
    st.info("Coming online in Part 6.")

    with st.expander("Engine self-check", expanded=False):
        st.caption(
            "Live L0 driver smoke test. Calls `load_driver` for "
            "historical all-India tmean on 2020-06-01 and reports the "
            "returned DataArray's shape, NaN%, range, and attrs. If this "
            "fails, the backend is not wired correctly."
        )
        try:
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
            st.code("\n".join(f"{k:16s}  {v}" for k, v in attrs.items()),
                     language="text")

            st.success("✓ L0 driver online — historical / all-India / tmean.")
        except Exception as e:
            st.error(f"❌ self-check failed — {type(e).__name__}: {e}")

with _tab_long:
    st.info("Coming online in Part 6.")
