"""
pages/30_What_If.py — Storyline: What If Temperature Changes?

Faithful reproduction of the original tab from the cloned Streamlit
build. Dark theme. Slider for ΔT ∈ [−2, +3] °C. Three side-by-side
India rainfall maps (PAST 1975-1990 · PRESENT 2010-2024 · FUTURE
under the requested ΔT via the sensitivity map). Sensitivity map at
the bottom.

The Parts 0-8 scenario engine (climate_twin/whatif/) remains available
in git history at commit 2fb69d1; this file no longer imports it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

st.set_page_config(
    page_title="What If Temperature Changes · MausamSetu",
    page_icon="❓",
    layout="wide",
)

CUBE_PATH = Path(r"L:\MausamSetu\data\processed\india.nc")
SENSITIVITY_MAP = Path(_PROJECT_ROOT) / "climate_twin" / "sensitivity_map.npy"


# ────────────────────────────────────────────────────────────────────
# Page-scoped dark theme + slider styling
# ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; padding-bottom: 3rem; }

      h1, h2, h3, h4 {
        color: #f5f7fb;
        letter-spacing: 0.3px;
      }
      .storyline-title {
        color: #f5f7fb;
        font-weight: 700;
        font-size: 2.3em;
        letter-spacing: 0.3px;
        margin: 4px 0 4px 0;
      }
      .storyline-sub {
        color: #b3bccc;
        font-size: 1.02em;
        margin-bottom: 20px;
      }
      .section-title {
        color: #f5f7fb;
        font-weight: 700;
        font-size: 1.5em;
        margin: 16px 0 6px 0;
        letter-spacing: 0.3px;
      }
      .section-hr {
        border: 0;
        border-top: 1px solid rgba(255,255,255,0.10);
        margin: 26px 0 10px 0;
      }

      /* Slider — red accent to match the screenshot */
      div[data-testid="stSlider"] label {
        color: #d5dbe6 !important;
        font-weight: 500;
      }
      div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: rgba(255,255,255,0.14) !important;
      }
      div[data-testid="stSlider"] div[role="slider"] {
        background: #ff4d4f !important;
        border: none !important;
        box-shadow: 0 0 12px rgba(255,77,79,0.55) !important;
      }
      /* Filled portion left of the thumb */
      div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background: #ff4d4f !important;
      }
      div[data-testid="stSlider"] [data-testid="stTickBar"] * {
        color: #b3bccc !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────
# Data — annual rainfall means + sensitivity
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading rainfall climatology…")
def _load_rainfall_epochs() -> dict:
    """Return past (1975-1990) + present (2010-2024) annual-rain means
    on the master grid, plus lat/lon axes and land mask."""
    if not CUBE_PATH.exists():
        # Fallback: procedural gradient so the page still boots
        lat = np.linspace(6.5, 38.5, 129)
        lon = np.linspace(66.5, 100.0, 135)
        base = np.clip(
            1400 - 30 * (lat[:, None] - 15) + 6 * (85 - lon[None, :]),
            0, 3200,
        ).astype(np.float32)
        return {
            "lat": lat, "lon": lon,
            "past": base * 0.98, "present": base,
            "mask": None,
            "years_past": (1975, 1990),
            "years_present": (2010, 2024),
            "source": "synthetic-fallback",
        }
    ds = xr.open_dataset(CUBE_PATH)
    try:
        rain = ds["rain"]
        annual = rain.groupby("time.year").sum("time", skipna=False)
        years = annual["year"].values
        past_mask = (years >= 1975) & (years <= 1990)
        pres_mask = (years >= 2010) & (years <= 2024)
        past = annual.sel(year=annual["year"][past_mask]).mean(
            "year", skipna=True,
        ).values.astype(np.float32)
        present = annual.sel(year=annual["year"][pres_mask]).mean(
            "year", skipna=True,
        ).values.astype(np.float32)
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        mask = ds["mask"].values.astype(bool) if "mask" in ds else None
    finally:
        ds.close()
    return {
        "lat": lat, "lon": lon,
        "past": past, "present": present, "mask": mask,
        "years_past": (1975, 1990),
        "years_present": (2010, 2024),
        "source": "IMD india.nc",
    }


@st.cache_data(show_spinner=False)
def _load_sensitivity() -> np.ndarray | None:
    if not SENSITIVITY_MAP.exists():
        return None
    return np.load(SENSITIVITY_MAP).astype(np.float32)


def _apply_mask(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Set cells outside the land mask to NaN so plotly leaves them
    transparent — the screenshots show a clean India outline against
    the dark app background."""
    if mask is None:
        return arr
    out = arr.astype(np.float32, copy=True)
    out[~mask] = np.nan
    return out


# ────────────────────────────────────────────────────────────────────
# Plotly builders — dark-theme
# ────────────────────────────────────────────────────────────────────
def _rain_map(z: np.ndarray, lat, lon, title: str, vmax: float) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=lon, y=lat,
        colorscale="Blues",
        zmin=0.0, zmax=vmax,
        colorbar=dict(
            title=dict(text="mm", side="right",
                        font=dict(color="#d5dbe6", size=11)),
            len=0.85, thickness=12,
            tickfont=dict(color="#d5dbe6", size=10),
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovertemplate="Lat %{y:.2f}<br>Lon %{x:.2f}<br>%{z:.0f} mm<extra></extra>",
        zsmooth="best",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>", x=0.02, xanchor="left",
            font=dict(color="#f0f3f8", size=14),
        ),
        xaxis=dict(
            title=dict(text="Longitude", font=dict(color="#8a94a6", size=11)),
            color="#b3bccc",
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            range=[65, 100],
        ),
        yaxis=dict(
            title=dict(text="Latitude", font=dict(color="#8a94a6", size=11)),
            color="#b3bccc",
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            range=[5, 38],
            scaleanchor="x", scaleratio=1,
        ),
        height=520,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.35)",
    )
    return fig


def _sensitivity_map(sens: np.ndarray, lat, lon,
                      mask: np.ndarray | None) -> go.Figure:
    z = _apply_mask(sens, mask)
    # Symmetric range around 0 for the diverging scale
    vabs = 45.0
    fig = go.Figure(go.Heatmap(
        z=z, x=lon, y=lat,
        colorscale="RdBu_r",
        zmin=-vabs, zmax=vabs, zmid=0.0,
        colorbar=dict(
            title=dict(text="mm/°C", side="right",
                        font=dict(color="#d5dbe6", size=11)),
            len=0.85, thickness=12,
            tickfont=dict(color="#d5dbe6", size=10),
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovertemplate="Lat %{y:.2f}<br>Lon %{x:.2f}<br>"
                       "%{z:+.1f} mm/°C<extra></extra>",
        zsmooth="best",
    ))
    fig.update_layout(
        title=dict(
            text="<b>Rainfall Sensitivity (mm per +1°C)</b>",
            x=0.02, xanchor="left",
            font=dict(color="#f0f3f8", size=13),
        ),
        xaxis=dict(
            title=dict(text="Longitude", font=dict(color="#8a94a6", size=11)),
            color="#b3bccc",
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            range=[35, 125],
        ),
        yaxis=dict(
            title=dict(text="Latitude", font=dict(color="#8a94a6", size=11)),
            color="#b3bccc",
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            range=[5, 38],
            scaleanchor="x", scaleratio=1,
        ),
        height=620,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.35)",
    )
    return fig


# ────────────────────────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='storyline-title'>Storyline: What If Temperature Changes?</div>"
    "<div class='storyline-sub'>PAST / PRESENT / FUTURE comparison</div>",
    unsafe_allow_html=True,
)

delta_t = st.slider(
    "Temperature Change (°C)",
    min_value=-2.0, max_value=3.0, value=1.0, step=0.1,
    key="wif_delta_t",
)

data = _load_rainfall_epochs()
sens = _load_sensitivity()

past = _apply_mask(data["past"], data["mask"])
present = _apply_mask(data["present"], data["mask"])

if sens is not None:
    future = _apply_mask(data["present"] + sens * float(delta_t), data["mask"])
else:
    st.warning("Sensitivity map not found on disk — the FUTURE panel is "
                "a constant copy of PRESENT until "
                "`climate_twin/sensitivity_map.npy` is available.")
    future = present.copy()

# Shared vmax across the three panels so colors are comparable.
finite_vals = np.concatenate([
    arr[np.isfinite(arr)].ravel() for arr in (past, present, future)
    if arr is not None
])
if finite_vals.size:
    p99 = float(np.nanpercentile(finite_vals, 99))
    vmax = max(400.0, min(3000.0, p99 * 1.02))
else:
    vmax = 3000.0

st.markdown(
    f"<div class='section-title'>Rainfall Under {delta_t:+.1f}°C Scenario</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(
        _rain_map(past, data["lat"], data["lon"],
                    f"PAST ({data['years_past'][0]}–{data['years_past'][1]})",
                    vmax=vmax),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with c2:
    st.plotly_chart(
        _rain_map(present, data["lat"], data["lon"],
                    f"PRESENT ({data['years_present'][0]}–{data['years_present'][1]})",
                    vmax=vmax),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with c3:
    st.plotly_chart(
        _rain_map(future, data["lat"], data["lon"],
                    f"FUTURE ({delta_t:+.1f}°C)",
                    vmax=vmax),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ── Sensitivity Map ──
st.markdown("<hr class='section-hr'/>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-title'>Sensitivity Map</div>",
    unsafe_allow_html=True,
)
if sens is not None:
    st.plotly_chart(
        _sensitivity_map(sens, data["lat"], data["lon"], data["mask"]),
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False},
    )
    st.caption(
        "🔴 Red = rainfall increases with warming. 🔵 Blue = rainfall decreases. "
        f"Data source: {data['source']} · sensitivity from IMD trend regression."
    )
else:
    st.info(
        "Sensitivity map artifact is missing. Regenerate via the Colab "
        "notebook or the training pipeline; expected at "
        "`climate_twin/sensitivity_map.npy`."
    )
