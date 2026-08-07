"""whatif.ui.panels.context — top-of-page context strip."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

from ..copy import context as _copy

_IST = timezone(timedelta(hours=5, minutes=30))


def _short_hash(s: str | None, n: int = 8) -> str:
    if not s:
        return "—"
    return str(s)[:n]


def render_context_strip(state, result: dict | None) -> None:
    """A single narrow strip at the top of the tab. Purpose: any
    screenshot must be self-describing from the strip alone."""
    cols = st.columns([1.1, 1.0, 1.1, 1.8])

    region_str = _copy.REGION_BADGE_FMT.format(
        region_kind=state.region_kind, region_id=state.region_id or "—",
    )
    with cols[0]:
        st.caption("Region")
        st.markdown(f"**{region_str}**")

    with cols[1]:
        st.caption("Season / Window")
        season_line = _copy.WINDOW_BADGE_FMT.format(
            season=state.season, window="JJAS",
        )
        st.markdown(f"**{season_line}**")

    with cols[2]:
        st.caption("Horizon")
        label = (_copy.HORIZON_SHORT if state.horizon == "short_term"
                  else _copy.HORIZON_LONG)
        st.markdown(f"**{label}**")

    with cols[3]:
        st.caption("Provenance")
        prov = (result or {}).get("provenance") if isinstance(result, dict) else None
        # Provenance can be a dict or a list of per-cell records
        code_hash = "—"
        run_id = "—"
        if isinstance(prov, list) and prov:
            v = prov[0].get("valuation", {}) if isinstance(prov[0], dict) else {}
            run_id = v.get("valuation_version", "—")
            code_hash = _short_hash(v.get("crop_registry_sha256"))
        line = _copy.DATA_PROVENANCE_TEMPLATE.format(
            run_id=_short_hash(run_id), code_hash=code_hash,
        )
        # Render an IST timestamp so screenshots can be dated
        now_ist = datetime.now(_IST).strftime("%Y-%m-%d %H:%M IST")
        st.markdown(f"<div style='font-size:0.85em;color:#888;'>{line} · {now_ist}</div>",
                     unsafe_allow_html=True)
