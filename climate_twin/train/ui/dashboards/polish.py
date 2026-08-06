"""
train.ui.dashboards.polish — Part-C visual primitives.

Everything is CPU-side and cheap so it can run inside a fragment.

    format.*    unit-aware number formatters
    status.*    ✓/⚠/⛔/— badges (colour ALWAYS paired with an icon)
    empty.*     empty-state cards + loading skeletons
    charts.*    consistent chart wrappers (fixed height, viridis/RdBu)
    profile.*   fragment-timing helper
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────
# UNIT-AWARE FORMATTERS  (Part C1)
# Every number carries a unit; every time value is human-readable.
# ─────────────────────────────────────────────────────────────────
_THIN = "\u2009"        # THIN SPACE (thousands separator)


def _isnum(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f       # not NaN


def rain(v, dp: int = 2, dash: str = "—") -> str:
    if not _isnum(v):
        return dash
    return f"{float(v):.{dp}f}{_THIN}mm/day"


def temp(v, dp: int = 1, dash: str = "—") -> str:
    if not _isnum(v):
        return dash
    return f"{float(v):.{dp}f}{_THIN}°C"


def loss(v, dp: int = 4, dash: str = "—") -> str:
    if not _isnum(v):
        return dash
    return f"{float(v):.{dp}f}"


def pct(v, dp: int = 0, dash: str = "—") -> str:
    if not _isnum(v):
        return dash
    return f"{float(v):.{dp}f}%"


def gb(mb, dp: int = 1, dash: str = "—") -> str:
    if not _isnum(mb):
        return dash
    return f"{float(mb) / 1024:.{dp}f}{_THIN}GB"


def count(v, dash: str = "—") -> str:
    """Thin-space thousands separator: 1 842 not 1,842."""
    if not _isnum(v):
        return dash
    return f"{int(round(float(v))):,}".replace(",", _THIN)


def hms(seconds, dash: str = "—") -> str:
    """0-based, never a bare 8040s. 'hms(90) -> 1m 30s'."""
    if seconds is None or not _isnum(seconds):
        return dash
    s = float(seconds)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{int(s // 60)}m{_THIN}{int(s % 60)}s"
    if s < 86400:
        return f"{int(s // 3600)}h{_THIN}{int((s % 3600) // 60)}m"
    d = int(s // 86400); h = int((s % 86400) // 3600)
    return f"{d}d{_THIN}{h}h"


def delta_arrow(curr, prev) -> str:
    """↓ (green) / ↑ (red) / → (grey) — used for metric-tile deltas."""
    if not _isnum(curr) or not _isnum(prev):
        return ""
    d = float(curr) - float(prev)
    if abs(d) < 1e-9:
        return "→"
    return "↓" if d < 0 else "↑"


# ─────────────────────────────────────────────────────────────────
# STATUS BADGES  (Part C2)
# Colour paired with icon + text. Never colour alone.
# ─────────────────────────────────────────────────────────────────
STATUS_COLOURS = {
    "ok":     ("#28a745", "✓"),
    "warn":   ("#F4A34A", "⚠"),
    "fail":   ("#e63946", "⛔"),
    "n/a":    ("#6b7280", "—"),
}


def badge(kind: str, text: str = "") -> str:
    """Return HTML for an inline status badge. `kind` ∈ ok/warn/fail/n/a."""
    colour, icon = STATUS_COLOURS.get(kind, STATUS_COLOURS["n/a"])
    return (
        f'<span style="color:{colour};font-family:monospace;">'
        f'{icon}</span>{" " + text if text else ""}'
    )


def render_check(kind: str, text: str) -> None:
    st.markdown(badge(kind, text), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# EMPTY STATES + SKELETONS  (Part C4/C5)
# ─────────────────────────────────────────────────────────────────
def empty_state(icon: str, headline: str, body: str = "",
                height: int = 180) -> None:
    """Never blank space. Always explain what will fill this and when."""
    st.markdown(
        f'<div style="background:#0E1522;border:1px dashed #333;'
        f'border-radius:6px;padding:16px;min-height:{height}px;'
        f'display:flex;flex-direction:column;justify-content:center;'
        f'align-items:center;text-align:center;font-family:monospace;">'
        f'<div style="font-size:2em;">{icon}</div>'
        f'<div style="font-size:1.05em;color:#F4A34A;margin-top:6px;">'
        f'<b>{headline}</b></div>'
        f'{"<div style=color:#9DA6B0;margin-top:4px;font-size:0.9em;>" + body + "</div>" if body else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def skeleton_line(height: int = 12, width: str = "80%") -> None:
    st.markdown(
        f'<div style="height:{height}px;width:{width};'
        f'background:linear-gradient(90deg,#0E1522,#1a2130,#0E1522);'
        f'background-size:200% 100%;border-radius:3px;'
        f'animation:mausam-shimmer 1.4s infinite;"></div>'
        f'<style>@keyframes mausam-shimmer {{'
        f'0%{{background-position:200% 0}}100%{{background-position:-200% 0}}'
        f'}}</style>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# CHART HELPERS  (Part C3)
# All charts: fixed heights, viridis/RdBu, no rainbow, downloadable.
# ─────────────────────────────────────────────────────────────────
def zone_strip_dataframe(df: pd.DataFrame, height: int = 300):
    """Colour-graded zone×epoch strip. Uses viridis (sequential) reversed
    so lower RMSE = green-ish, higher = purple.  Insufficient cells
    render as '—' hatched-style via the .format(na_rep=…)."""
    styled = (
        df.style
          .background_gradient(cmap="viridis_r", axis=None)
          .format(precision=2, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=height)


def line_chart_fixed(data, height: int = 200) -> None:
    """Wrapper enforcing a fixed height so refreshes don't jump."""
    st.line_chart(data, height=height)


# ─────────────────────────────────────────────────────────────────
# FRAGMENT PROFILER  (Part C7)
# Wrap the body of a fragment in `with profile(name):` to record a
# millisecond timing that shows up in the ⏱ perf panel.
# ─────────────────────────────────────────────────────────────────
_PROFILE_KEY = "mausam_fragment_profile"


def _profile_bucket() -> dict[str, list[float]]:
    if _PROFILE_KEY not in st.session_state:
        st.session_state[_PROFILE_KEY] = {}
    return st.session_state[_PROFILE_KEY]


@contextmanager
def profile(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        buckets = _profile_bucket()
        buckets.setdefault(name, []).append(dt_ms)
        # Keep last 40 samples per fragment
        if len(buckets[name]) > 40:
            buckets[name] = buckets[name][-40:]


def render_profile_panel() -> None:
    """Small sidebar-friendly panel showing p50 + max ms per fragment."""
    buckets = _profile_bucket()
    if not buckets:
        st.caption("Fragment profile: no data yet.")
        return
    rows = []
    for name, samples in sorted(buckets.items()):
        if not samples:
            continue
        arr = np.asarray(samples)
        rows.append({
            "fragment": name,
            "n": int(arr.size),
            "p50 ms": round(float(np.median(arr)), 1),
            "max ms": round(float(np.max(arr)), 1),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("p50 ms", ascending=False)
    slow = df[df["max ms"] > 400]

    if not slow.empty:
        st.caption(f"⚠ {len(slow)} fragment(s) exceeded 400 ms max.")
    st.dataframe(df, use_container_width=True, hide_index=True, height=200)
