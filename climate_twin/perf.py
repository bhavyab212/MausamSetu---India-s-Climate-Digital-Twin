"""
perf.py — lightweight timing harness for the climate_twin Streamlit app.

Usage:
    from perf import timed, render_perf_panel, reset_timings

    @timed("load_artifacts")
    def load_artifacts(...): ...

At the top of every rerun call reset_timings(); render the panel from the
sidebar behind a toggle with render_perf_panel().

Everything is a no-op-cheap wrapper: it only appends (label, ms) to a list in
st.session_state, so it ships harmless.
"""
from __future__ import annotations
import time
import functools

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


def _store():
    if st is None:
        return None
    return st.session_state.setdefault("_timings", [])


def reset_timings():
    """Clear the timing buffer — call once at the very top of each rerun."""
    if st is not None:
        st.session_state["_timings"] = []


def record(label: str, dt_ms: float):
    buf = _store()
    if buf is not None:
        buf.append((label, float(dt_ms)))


def timed(label):
    """Decorator: record wall-clock ms of the wrapped call under `label`.

    Place ABOVE @st.cache_data so it measures the real per-rerun cost
    (cache lookup + any recompute on a miss).
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                record(label, (time.perf_counter() - t) * 1000.0)
        return wrap
    return deco


class timed_block:
    """Context manager for timing an inline block:

        with timed_block("figure: map"):
            fig = build_map(...)
    """
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        record(self.label, (time.perf_counter() - self._t) * 1000.0)
        return False


def render_perf_panel():
    """Sidebar toggle + slowest-first table of this rerun's timings."""
    if st is None:
        return
    with st.sidebar:
        show = st.toggle("🐢 Perf panel", value=False, key="_perf_panel_on")
    if not show:
        return
    buf = _store() or []
    with st.sidebar:
        if not buf:
            st.caption("No timings captured this rerun.")
            return
        # aggregate duplicate labels (sum + count)
        agg = {}
        for label, dt in buf:
            s, n = agg.get(label, (0.0, 0))
            agg[label] = (s + dt, n + 1)
        rows = sorted(
            ({"op": k, "ms": round(v[0], 1), "calls": v[1]} for k, v in agg.items()),
            key=lambda r: r["ms"], reverse=True,
        )
        total = sum(v[0] for v in agg.values())
        st.caption(f"Total instrumented: {total:.0f} ms")
        try:
            import psutil, os
            rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
            st.caption(f"Process RSS: {rss:.0f} MB")
        except Exception:
            pass
        st.dataframe(rows, use_container_width=True, hide_index=True)
