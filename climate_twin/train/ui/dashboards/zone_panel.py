"""
train.ui.dashboards.zone_panel — the always-visible 9-zone sidebar block.

Renders a compact India mini-map coloured by zone id + a zone-count table.
Cached — the map is a static PNG generated once per zone-mask signature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from climate_twin.regions import get_zones


ZONE_PALETTE = [
    "#0b0f1e",   # 0 unassigned
    "#F4A34A",   # 1 northwest
    "#8AB4F8",   # 2 west_central
    "#7EE787",   # 3 central_northeast
    "#B392F0",   # 4 northeast
    "#F97583",   # 5 south_peninsular
    "#79B8FF",   # 6 western_ghats
    "#FFEA7F",   # 7 thar_arid
    "#A5D6FF",   # 8 himalayan
    "#F4C7C3",   # 9 tamilnadu_ne
]


def _render_zone_map_png(sig: str, out_path: Path) -> Path:
    """Render the 9-zone mini-map to a PNG. Cached by zone_mask_sig."""
    Z = get_zones()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    mg = Z.master_grid
    lats = np.linspace(mg["extent_lat"][0], mg["extent_lat"][1], mg["n_lat"])
    lons = np.linspace(mg["extent_lon"][0], mg["extent_lon"][1], mg["n_lon"])
    cmap = ListedColormap(ZONE_PALETTE[:Z.n_zones() + 1])
    norm = BoundaryNorm(list(range(Z.n_zones() + 2)), cmap.N)

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    fig.patch.set_facecolor("#050912")
    ax.set_facecolor("#050912")
    ax.pcolormesh(lons, lats, Z.hard_mask, cmap=cmap, norm=norm, shading="nearest")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(20.0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


@st.cache_data(show_spinner=False)
def _zone_map_png(sig: str) -> str:
    # parents: [0]=dashboards, [1]=ui, [2]=train, [3]=climate_twin
    out = Path(__file__).resolve().parents[3] / "regions" / "qc" / f"zones_mini_{sig}.png"
    if not out.exists():
        _render_zone_map_png(sig, out)
    return str(out)


def render_sidebar_zone_panel(min_cells_warn: int = 250) -> None:
    """Render the persistent 9-zone panel (mini-map + table + hover hints)."""
    Z = get_zones()
    png = _zone_map_png(Z.mask_signature)
    st.sidebar.markdown("### 🗺 Nine zones")
    st.sidebar.image(png, use_container_width=True)

    rows = []
    for z in Z.zones:
        cells = int((Z.hard_mask == z.id) .sum())
        badge = "✓" if cells >= min_cells_warn else "⚠"
        rows.append({
            "id": z.id,
            "key": z.key,
            "cells": f"{cells:,}",
            "status": badge,
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    st.sidebar.dataframe(
        df, use_container_width=True, hide_index=True,
    )
    small = ", ".join(r["key"] for r in rows if r["status"] == "⚠")
    if small:
        st.sidebar.caption(f"⚠ near-threshold zones: {small}")
    st.sidebar.caption(f"zone_mask_sig: `{Z.mask_signature}`")
