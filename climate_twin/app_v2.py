import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure `climate_twin.*` package imports resolve when Streamlit runs this
# file directly (sys.path[0] would otherwise be climate_twin/ itself).
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = str(_Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import calendar
import numpy as np
import streamlit as st
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import hashlib
import io
import base64
import streamlit.components.v1 as components
from src.visualization import Visualizer
from src.daily_temp_loader import load_daily_temp
from src.viz_plotly import PlotlyVisualizer
from src.zones import IMD_ZONES, get_zone_mask
from src.utils import ensure_data_file
import data_source as DS
from perf import timed, timed_block, reset_timings, render_perf_panel


def active_region() -> str:
    """The region selected in the sidebar (defaults to India on first load)."""
    return st.session_state.setdefault("active_region", "india")


def _sync_repo_artifacts_to_data():
    """HF Space stores some files at repo root; app expects them under data/."""
    os.makedirs("data", exist_ok=True)
    names = [
        "aggregates.npz",
        "convlstm.weights.h5",
        "convlstm_daily.weights.h5",
        "unet.weights.h5",
        "scalers.json",
        "scalers_daily.json",
        "sensitivity_map.npy",
        "India_States_2024.geojson",
    ]
    for name in names:
        dst = os.path.join("data", name)
        if os.path.exists(dst):
            continue
        if os.path.exists(name):
            import shutil
            shutil.copy2(name, dst)


_sync_repo_artifacts_to_data()

# Reset the per-rerun perf buffer as early as possible (harness is a no-op
# unless the sidebar "🐢 Perf panel" toggle is on).
reset_timings()

st.set_page_config(layout="wide", page_title="ISRO Climate Digital Twin")



# Suppress Streamlit websocket disconnections and sessionInfo toasts
# NOTE: Never hide stException — doing so makes crashes invisible and the app appears "stuck".
st.markdown("""
    <style>
    /* Legacy / fallback selectors */
    #ConnectionStatus { display: none !important; }
    /* Hide all Toast notifications (which include Bad Message Format errors) */
    [data-testid="stToast"] { display: none !important; }
    [data-testid="stToastContainer"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Inject JS via components.html so it actually executes (st.markdown ignores script tags)
components.html("""
    <script>
    const parentDoc = window.parent.document;
    
    function hideStreamlitModals() {
        const modals = parentDoc.querySelectorAll('div[role="dialog"], [data-testid="stModal"]');
        modals.forEach(m => {
            const text = (m.innerText || m.textContent || '').toLowerCase();
            if (text.includes('bad message format') || text.includes('sessioninfo')) {
                m.style.display = 'none';
                m.style.opacity = '0';
                
                // Hide the grey backdrop behind it
                const backdrop = parentDoc.querySelector('[data-testid="stModalBackdrop"]') || parentDoc.querySelector('.stModalBackdrop');
                if (backdrop) {
                    backdrop.style.display = 'none';
                    backdrop.style.opacity = '0';
                }
                
                // Also hide any parent node acting as the modal container
                const container = m.closest('[data-testid="stModal"]');
                if(container) {
                    container.style.display = 'none';
                }
            }
        });
    }
    
    // Check continuously for the rogue popup and banish it
    setInterval(hideStreamlitModals, 50);
    </script>
""", height=0, width=0)

col1, col2, col3 = st.columns([1, 14, 2])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=70)
with col2:
    st.title("ISRO Climate Digital Twin")
    st.markdown("**High-Fidelity Architecture** | 3-Model Ensemble | Daily Future Prediction")
with col3:
    st.markdown("""
        <div style="text-align: center; font-family: sans-serif; line-height: 1.1; margin-top: 5px;">
            <span style="color: #f37021; font-weight: 800; font-size: 22px;">इसरो</span>
            <span style="color: #0072c6; font-weight: 900; font-size: 23px; letter-spacing: 1px;">isro</span>
            <div style="border-top: 2px solid #f37021; margin: 4px auto 2px auto; width: 85%;"></div>
            <span style="color: #f37021; font-weight: 900; font-size: 20px; font-style: italic; letter-spacing: 2px;">nrsc</span>
        </div>
    """, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]


def days_in_month(year: int, month_idx: int) -> int:
    """month_idx 0 = January (matches MONTHS list). Handles leap-year February."""
    return int(calendar.monthrange(year, month_idx + 1)[1])


def month_day_to_doy(month_idx, day, year=None):
    """Day-of-year 1..366. Pass ``year`` for correct leap-year February."""
    if year is None:
        return sum(MONTH_DAYS[:month_idx]) + day
    total = day
    for mi in range(month_idx):
        total += days_in_month(year, mi)
    return total

def build_data_signature():
    tracked = [
        os.path.join("data", "aggregates.npz"),
        os.path.join("data", "convlstm.weights.h5"),
        os.path.join("data", "convlstm_daily.weights.h5"),
        os.path.join("data", "scalers.json"),
        os.path.join("data", "scalers_daily.json"),
    ]
    parts = []
    for p in tracked:
        if os.path.exists(p):
            stat = os.stat(p)
            parts.append(f"{p}:{int(stat.st_mtime)}:{int(stat.st_size)}")
        else:
            parts.append(f"{p}:missing")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

@timed("data: get_nc_year_data")
@st.cache_data(show_spinner=False)
def get_nc_year_data(year, region="india"):
    """Daily rainfall (days, H, W) for a region/year, read from local raw IMD .grd
    via data_source (no NetCDF, no cloud)."""
    return DS.daily_rain(region, int(year))

# ── Load Artifacts ────────────────────────────────────────────────────────────
def _legacy_load_artifacts():
    """ORIGINAL clone loader (aggregates.npz + cloud fallback). Retained for
    reference only; superseded by data_source.load_aggregates()."""
    agg = ensure_data_file('aggregates.npz')
    if not os.path.exists(agg):
        return None, None, None, None
    data = np.load(agg)
    sens_path = ensure_data_file('sensitivity_map.npy')
    sens = np.load(sens_path) if os.path.exists(sens_path) else None
    return data['rain'], data['temp'], data['mask'], sens


@timed("data: load_artifacts")
@st.cache_data(show_spinner="Building climate cube from your local IMD data...")
def load_artifacts(region="india"):
    """Region-aware artifacts, rebuilt from the user's local raw IMD grids.

    Returns (rain, temp, mask, sens) with the same shapes the app expects:
      rain (n_years, H, W), temp (n_years, H, W), mask (H, W).
    sensitivity map is India-only (129x135); None for other regions.
    """
    try:
        rain, temp, mask, _years = DS.load_aggregates(region)
    except Exception as e:
        print(f"[load_artifacts] build failed for {region}: {e}")
        return None, None, None, None

    sens = None
    if region == "india":
        sens_path = os.path.join("data", "sensitivity_map.npy")
        if os.path.exists(sens_path):
            try:
                sm = np.load(sens_path)
                if sm.shape == mask.shape:
                    sens = sm
            except Exception:
                sens = None
    return rain, temp, mask, sens

# ── Vertical navigation (sidebar, top) ────────────────────────────────────────
_TAB_LABELS = [
    "Home", "Explorer", "±1°C What-If",
    "Model Comparison", "Zone Projections",
    "Climate Spirals", "Deep Analytics",
    "🔥 Training", "🔍 Validation", "🤖 RL Agent",
]
# Icons only for labels that don't already start with an emoji.
_TAB_ICONS = {
    "Home":              "🏠",
    "Explorer":          "🌐",
    "±1°C What-If":      "❓",
    "Model Comparison":  "🧭",
    "Zone Projections":  "🗺",
    "Climate Spirals":   "🌀",
    "Deep Analytics":    "📊",
    "🔥 Training":       "",
    "🔍 Validation":     "",
    "🤖 RL Agent":       "",
}

# ── Glassmorphic global theme (injected ONCE per session for speed) ─────────
# NOTE: backdrop-filter is expensive (browser must blur the background behind
# every element on every paint). Removed on all hot elements — flat translucent
# cards read similarly and cost 10-100× less to paint.
if not st.session_state.get("_mausam_css_injected"):
    st.session_state["_mausam_css_injected"] = True
    st.markdown(
    """
    <style>
    /* ═════════════════════════════ APP BACKGROUND ═════════════════════════════ */
    .stApp {
        background:
            radial-gradient(ellipse at 20% -10%, rgba(244, 163, 74, 0.08) 0%, transparent 45%),
            radial-gradient(ellipse at 90% 100%, rgba(138, 180, 248, 0.06) 0%, transparent 45%),
            linear-gradient(180deg, #050912 0%, #0a0f1e 100%);
    }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* ═════════════════════════════ HEADINGS ═══════════════════════════════════ */
    h1, h2, h3, h4 { letter-spacing: 0.3px; color: #F0F3F8; }
    h1 { text-shadow: 0 0 30px rgba(244, 163, 74, 0.15); }

    /* ═════════════════════════════ METRIC TILES ═══════════════════════════════ */
    /* Flat translucent card — no blur, no hover transform. */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #FFD9A8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: rgba(229, 233, 240, 0.6) !important;
        font-size: 0.82em !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ═════════════════════════════ BUTTONS ════════════════════════════════════ */
    .stButton > button, .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E5E9F0;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 500;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: rgba(244, 163, 74, 0.12);
        border-color: rgba(244, 163, 74, 0.45);
        color: #FFD9A8;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(95deg,
                    rgba(244, 163, 74, 0.85) 0%,
                    rgba(255, 175, 100, 0.80) 100%);
        border: 1px solid rgba(244, 163, 74, 0.7);
        color: #0a0f1e;
        font-weight: 700;
        box-shadow: 0 4px 24px rgba(244, 163, 74, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(95deg,
                    rgba(255, 175, 100, 0.95) 0%,
                    rgba(255, 195, 130, 0.90) 100%);
        color: #050912;
    }
    .stButton > button:disabled { opacity: 0.4; cursor: not-allowed; }

    /* ═════════════════════════════ EXPANDERS ══════════════════════════════════ */
    details[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin: 8px 0;
    }
    details[data-testid="stExpander"] summary {
        padding: 12px 16px !important;
        border-radius: 12px !important;
    }

    /* ═════════════════════════════ TABS ═══════════════════════════════════════ */
    div[data-testid="stTabs"] button[role="tab"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: rgba(229, 233, 240, 0.65);
        margin-right: 4px;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: rgba(244, 163, 74, 0.12);
        border-color: rgba(244, 163, 74, 0.4);
        border-bottom: 2px solid rgba(244, 163, 74, 0.9);
        color: #FFD9A8;
        font-weight: 600;
    }

    /* ═════════════════════════════ ALERTS ═════════════════════════════════════ */
    div[data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px !important;
        border-left: 3px solid rgba(244, 163, 74, 0.7) !important;
    }

    /* ═════════════════════════════ DATAFRAMES / TABLES ═══════════════════════ */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 4px;
    }

    /* ═════════════════════════════ CODE / TEXT AREAS ═════════════════════════ */
    div[data-testid="stCodeBlock"], .stTextInput textarea, .stTextArea textarea {
        background: rgba(5, 9, 18, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }

    /* ═════════════════════════════ FORM INPUTS ═══════════════════════════════ */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #E5E9F0 !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(244, 163, 74, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(244, 163, 74, 0.15) !important;
    }
    label { color: rgba(229, 233, 240, 0.75) !important; }

    /* ═════════════════════════════ SLIDER ═════════════════════════════════════ */
    div[data-testid="stSlider"] div[role="slider"] {
        background: #F4A34A !important;
        box-shadow: 0 0 12px rgba(244, 163, 74, 0.5) !important;
    }

    /* ═════════════════════════════ PROGRESS ═══════════════════════════════════ */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #F4A34A 0%, #FFD9A8 100%) !important;
        box-shadow: 0 0 10px rgba(244, 163, 74, 0.4);
    }
    div[data-testid="stProgress"] > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border-radius: 6px;
    }

    /* ═════════════════════════════ TOAST ══════════════════════════════════════ */
    div[data-testid="stToast"] {
        background: rgba(15, 22, 41, 0.95) !important;
        border: 1px solid rgba(244, 163, 74, 0.3) !important;
    }

    /* ═════════════════════════════ SCROLLBAR ══════════════════════════════════ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.02); }
    ::-webkit-scrollbar-thumb {
        background: rgba(244, 163, 74, 0.25);
        border-radius: 6px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(244, 163, 74, 0.45); }

    /* ═════════════════════════════ SIDEBAR (from earlier) ═════════════════════ */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(160deg,
                    rgba(15, 22, 41, 0.92) 0%,
                    rgba(10, 15, 30, 0.90) 100%);
        border-right: 1px solid rgba(244, 163, 74, 0.12);
    }
    /* Header block */
    .mausam-brand {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 4px 0 14px 0;
    }
    .mausam-brand .name {
        font-weight: 700;
        font-size: 1.05em;
        letter-spacing: 1.2px;
        background: linear-gradient(90deg, #F4A34A 0%, #FFC998 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .mausam-brand .tagline {
        color: rgba(229, 233, 240, 0.55);
        font-size: 0.68em;
        letter-spacing: 0.9px;
        margin-top: 3px;
        text-transform: uppercase;
    }
    /* Hide the radio circle dot */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    /* Each radio item → flat translucent pill (no blur; sidebar-wide) */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 14px !important;
        margin: 3px 0 !important;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(244, 163, 74, 0.08);
        border-color: rgba(244, 163, 74, 0.25);
        transform: translateX(2px);
    }
    /* Active (checked) item — orange accent glass */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(95deg,
                    rgba(244, 163, 74, 0.18) 0%,
                    rgba(244, 163, 74, 0.08) 100%);
        border: 1px solid rgba(244, 163, 74, 0.45);
        box-shadow: 0 0 24px rgba(244, 163, 74, 0.15),
                    inset 0 0 12px rgba(244, 163, 74, 0.06);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #FFD9A8 !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 0.95em !important;
        letter-spacing: 0.2px;
    }
    /* Section labels */
    section[data-testid="stSidebar"] h3 {
        color: rgba(244, 163, 74, 0.85);
        font-size: 0.78em !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08);
        margin: 14px 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(
        "<div class='mausam-brand'>"
        "<div class='name'>MAUSAMSETU</div>"
        "<div class='tagline'>AI-powered climate digital twin of India</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _nav_choice = st.radio(
        "Navigation",
        _TAB_LABELS,
        format_func=lambda k: (f"{_TAB_ICONS.get(k, '•')}  {k}"
                                 if _TAB_ICONS.get(k) else k),
        key="active_tab",
        label_visibility="collapsed",
    )
    st.markdown("---")

# ── Region selector (global sidebar, below nav) ───────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Region  ·  मौसम सेतु")
    region_key = st.radio(
        "Region",
        options=list(DS.REGIONS.keys()),
        format_func=lambda k: DS.REGIONS[k]["label"],
        key="active_region",
        label_visibility="collapsed",
    )

# When the region changes, drop every region-dependent @st.cache_data result so
# no India-shaped array (129x135) leaks into a Cauvery run (19x17) and vice-versa.
if st.session_state.get("_prev_region") != active_region():
    st.cache_data.clear()
    st.session_state["_prev_region"] = active_region()

rain_data, temp_data, mask, sens_map = load_artifacts(active_region())
if rain_data is None:
    st.error("⚠️ Processed cubes not found")
    st.info(
        "The daily processed cubes are missing. Build them from the raw IMD + "
        "INSAT data with:\n\n"
        "    python -m climate_twin.data.build_cube\n\n"
        "Expected files: `data/processed/india.nc`, `data/processed/cauvery.nc`, "
        "`data/processed/manifest.yaml`."
    )
    st.stop()

# Region-aware year axis (from the cube, not a hardcoded constant).
_region_years = DS.region_info(active_region(), sig=DS.manifest_sig())["all_years"]

vis = Visualizer()
_ext = DS.REGIONS[active_region()]["extent"]
plotly_vis = PlotlyVisualizer(
    os.path.join('data', 'India_States_2024.geojson'), mask,
    lat_min=_ext["lat"][0], lat_max=_ext["lat"][1],
    lon_min=_ext["lon"][0], lon_max=_ext["lon"][1],
)
# Instrument every map build under one perf label (no behaviour change).
_orig_plot_map = plotly_vis.plot_map
def _timed_plot_map(*a, **k):
    with timed_block("figures: plotly_vis.plot_map"):
        return _orig_plot_map(*a, **k)
plotly_vis.plot_map = _timed_plot_map
# Data-year axis (derived from the loaded cube — no hardcoded 1975 baseline).
if _region_years:
    START_YEAR = int(min(_region_years))
    END_YEAR = int(max(_region_years))
else:
    START_YEAR = 2018
    END_YEAR = 2025
N_YEARS = len(rain_data)
FUTURE_END = 2075

# ── Region context panel (sidebar, under the selector) ────────────────────────
with st.sidebar:
    _info = DS.region_info(active_region(), sig=DS.manifest_sig())
    _la, _lo = _info["extent"]["lat"], _info["extent"]["lon"]
    _nc_ok = "✓" if _info["nc_exists"] else "✗"
    st.markdown(
        f"**Region:** {_info['label']}  \n"
        f"**Grid:** {_info['shape'][0]} × {_info['shape'][1]}  ·  "
        f"land {_info['land_pct']}%  \n"
        f"**Extent:** {_la[0]}–{_la[1]}°N, {_lo[0]}–{_lo[1]}°E  \n"
        f"**Data:** {os.path.basename(_info['processed_nc'])} {_nc_ok}  ·  "
        f"Years {_info['years'][0]}–{_info['years'][1]}"
    )
    st.caption("Checkpoints for this region are prefixed "
               f"`{_info['ckpt_prefix']}_round_*`.")

    # ── Data provenance (Phase 5f) — surfaces the real manifest facts ──
    with st.expander("📜 Data provenance", expanded=False):
        _sig = _info.get("manifest_sig") or "—"
        _git = (_info.get("manifest_git_hash") or "")[:12] or "—"
        _built = _info.get("manifest_generated_at_ist") or "—"
        _tr = _info.get("train_years") or (None, None)
        st.markdown(
            f"**Manifest sig:** `{_sig}`  \n"
            f"**Git hash:** `{_git}`  \n"
            f"**Built (IST):** {_built}  \n"
            f"**Train years:** {_tr[0]}–{_tr[1]}"
        )
        st.markdown("**Per-variable coverage** _(fraction of days with any valid cell)_:")
        for _v in _info["variables"]:
            _yr = _info["coverage"][_v]
            _kept = sorted(y for y, c in _yr.items() if c >= 0.90)
            _partial = sorted(y for y, c in _yr.items() if 0 < c < 0.90)
            _zero = sorted(y for y, c in _yr.items() if c == 0)
            _avg = (sum(_yr.values()) / len(_yr) * 100) if _yr else 0.0
            _line = f"- `{_v}` avg **{_avg:.1f}%**"
            if _kept:    _line += f"  ·  full: {min(_kept)}–{max(_kept)}"
            if _partial: _line += f"  ·  partial: {_partial}"
            if _zero:    _line += f"  ·  missing: {_zero}"
            st.markdown(_line)
        _readers = _info.get("manifest_readers") or {}
        if _readers:
            _reader_lines = [f"- `{k}` → {v.get('module', '?')}" for k, v in _readers.items()]
            st.markdown("**Readers:**\n" + "\n".join(_reader_lines))


# ── Fixed scales for uniform maps ─────────────────────────────────────────────
SCALES = {
    'rain_daily':  {'vmin': 0, 'vmax': 100,  'cmap': 'Blues', 'unit': 'mm'},
    'temp_daily':  {'vmin': 15,'vmax': 50,   'cmap': 'turbo', 'unit': '°C'},
    'rain_annual': {'vmin': 0, 'vmax': 3000, 'cmap': 'Blues', 'unit': 'mm'},
    'temp_annual': {'vmin': 20,'vmax': 50,   'cmap': 'turbo', 'unit': '°C'},
    'sensitivity': {'vmin':-50,'vmax': 50,   'cmap': 'RdBu',          'unit': 'mm/°C'},
    'uncertainty': {'vmin': 0, 'vmax': 5,    'cmap': 'Purples',       'unit': 'uncertainty'},
}

def render_map(grid, title, scale_key, figsize=(5, 4), custom_vmin=None, custom_vmax=None):
    """Matplotlib/cartopy map (legacy). Prefer ``plotly_annual_*`` for India geojson consistency."""
    s = SCALES[scale_key]
    masked = np.where(mask == 1, grid, np.nan)
    fig, ax = vis.create_figure(figsize=figsize)
    vmin = s['vmin'] if custom_vmin is None else custom_vmin
    vmax = s['vmax'] if custom_vmax is None else custom_vmax
    im = vis.plot_map(ax, masked, title, s['cmap'], vmin, vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"{s['unit']}")
    return fig


@timed("figures: plotly_annual_rain_map")
def plotly_annual_rain_map(grid, title):
    """India Plotly map — same style as Daily Explorer (geojson + Blues)."""
    land = np.where(mask == 1, grid, np.nan)
    p99 = float(np.nanpercentile(land, 99)) if np.any(np.isfinite(land)) else 1500.0
    vmax = max(400.0, min(3000.0, p99 * 1.05))
    return plotly_vis.plot_map(grid, title, "rain_annual", custom_range=[0.0, vmax])


@timed("figures: plotly_annual_temp_map")
def plotly_annual_temp_map(grid, title):
    land = np.where(mask == 1, grid, np.nan)
    if not np.any(np.isfinite(land)):
        return plotly_vis.plot_map(grid, title, "temp_annual", custom_range=[20.0, 45.0])
    t5 = float(np.nanpercentile(land, 5))
    t95 = float(np.nanpercentile(land, 95))
    t_min = max(18.0, t5 - 1.0)
    t_max = min(48.0, max(t_min + 4.0, t95 + 1.0))
    return plotly_vis.plot_map(grid, title, "temp_annual", custom_range=[t_min, t_max])


def plotly_sensitivity_map(grid, title):
    land = np.where(mask == 1, grid, np.nan)
    if not np.any(np.isfinite(land)):
        return plotly_vis.plot_map(grid, title, "sensitivity", custom_range=[-50.0, 50.0])
    amax = float(np.nanpercentile(np.abs(land), 98))
    amax = max(8.0, min(50.0, amax * 1.05))
    return plotly_vis.plot_map(grid, title, "sensitivity", custom_range=[-amax, amax])

import pandas as pd

def safe_land_grid(grid, fallback_value=None):
    """Ensure grid is valid over land; optional fallback_value fills residual NaNs/Infs."""
    g = np.array(grid, dtype=np.float64)
    
    # Use pandas spatial interpolation to smoothly fill missing internal columns
    df = pd.DataFrame(g)
    df.interpolate(method='linear', axis=1, limit_direction='both', inplace=True)
    df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    g = df.values

    if fallback_value is not None:
        g = np.nan_to_num(g, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)
    
    g = np.where(mask == 1, g, np.nan)
    return g

def land_stats(grid):
    vals = np.where(mask == 1, grid, np.nan)
    return {
        "mean": float(np.nanmean(vals)),
        "p95": float(np.nanpercentile(vals, 95)),
        "p99": float(np.nanpercentile(vals, 99)),
    }

def apply_physical_qc(month_idx, future_rain, future_temp, day_rain_clim, day_temp_clim):
    """
    Lightweight India-specific QA:
    - avoid unrealistically dry monsoon days
    - keep anomalies within a reasonable band around climatology
    """
    issues = []
    r = np.array(future_rain, dtype=np.float64)
    t = np.array(future_temp, dtype=np.float64)

    # Monsoon guardrail (Jun–Sep, 0-based 5–8).
    if month_idx in [5, 6, 7, 8]:
        monsoon_floor = 0.3 * day_rain_clim
        r = np.maximum(r, monsoon_floor)

    # Clip extreme rainfall and temperature.
    r = np.clip(r, 0.0, 600.0)
    t = np.clip(t, 5.0, 55.0)

    # Keep temperature anomalies in a ±7°C band (95th percentile).
    clim_anom = t - day_temp_clim
    mask_land = (mask == 1)
    anom_vals = np.where(mask_land, np.abs(clim_anom), np.nan)
    try:
        p95 = np.nanpercentile(anom_vals, 95)
    except Exception:
        p95 = 0.0
    if p95 > 7.0:
        t = day_temp_clim + np.clip(clim_anom, -7.0, 7.0)
        issues.append("Temperature anomaly clipped to ±7°C around day climatology.")

    r = safe_land_grid(r)
    t = safe_land_grid(t)
    return r, t, issues

# ── Climatology for future disaggregation ─────────────────────────────────────
@timed("data: compute_daily_climatology")
@st.cache_data(show_spinner=False)
def compute_daily_climatology():
    h, w = mask.shape
    rain_clim = np.zeros((365, h, w), dtype=np.float64)
    r_count = np.zeros(365, dtype=np.int32)
    for y in range(max(START_YEAR, END_YEAR - 9), END_YEAR + 1):
        arr = get_nc_year_data(y, active_region())
        if arr is not None:
            n = min(arr.shape[0], 365)
            rain_clim[:n] += np.nan_to_num(arr[:n], nan=0.0)
            r_count[:n] += 1
    r_count[r_count == 0] = 1
    rain_clim /= r_count[:, None, None]
    rain_clim = np.nan_to_num(rain_clim, nan=0.0, posinf=0.0, neginf=0.0)
    # Temperature: sinusoidal model (peak May, trough Jan)
    annual_mean = np.nanmean(temp_data[-10:], axis=0)
    temp_clim = np.zeros((365, h, w), dtype=np.float64)
    for d in range(365):
        temp_clim[d] = annual_mean + 8.0 * np.cos(2 * np.pi * (d - 135) / 365)
    return rain_clim, temp_clim

@timed("data: get_day_temp_climatology")
@st.cache_data(show_spinner=False)
def get_day_temp_climatology(doy, years_window=10):
    """Observed day-of-year climatology from IMD temperature files."""
    grids = []
    d_idx = max(0, min(364, int(doy) - 1))
    start = max(START_YEAR, END_YEAR - years_window + 1)
    for y in range(start, END_YEAR + 1):
        tg_year = DS.daily_tmax(active_region(), y)
        if tg_year is not None and tg_year.shape[0] > d_idx:
            grids.append(tg_year[d_idx])
    if grids:
        out = np.nanmean(np.array(grids), axis=0)
        out = np.nan_to_num(out, nan=float(np.nanmean(temp_data[-1])))
        return out
    return np.nan_to_num(temp_data[-1], nan=30.0)

@timed("data: get_day_rain_climatology")
@st.cache_data(show_spinner=False)
def get_day_rain_climatology(doy, years_window=10):
    """Observed day-of-year climatology from IMD rainfall files."""
    grids = []
    d_idx = max(0, min(364, int(doy) - 1))
    start = max(START_YEAR, END_YEAR - years_window + 1)
    for y in range(start, END_YEAR + 1):
        arr = get_nc_year_data(y, active_region())
        if arr is not None and arr.shape[0] > d_idx:
            grids.append(arr[d_idx])
    if grids:
        out = np.nanmean(np.array(grids), axis=0)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
    # Fallback: weak daily proxy from annual mean
    return np.clip(np.nan_to_num(np.nanmean(rain_data[-10:], axis=0) / 365.0, nan=0.0), 0.0, None)



def compute_zone_daywise_stats(rain_grid, temp_grid):
    lat_dim, lon_dim = mask.shape
    rows = []
    for zone_name, (lt0, lt1, ln0, ln1) in IMD_ZONES.items():
        box = get_zone_mask(lat_dim, lon_dim, lt0, lt1, ln0, ln1)
        land = np.logical_and(mask == 1, box)
        if not np.any(land):
            continue
        zr = float(np.nanmean(np.where(land, rain_grid, np.nan)))
        zt = float(np.nanmean(np.where(land, temp_grid, np.nan)))
        rows.append({"Zone": zone_name, "Rainfall (mm/day)": round(zr, 2), "Max Temp (°C)": round(zt, 2)})
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner=False)
def _get_ensemble(region, seq, h, w):
    """Load the Keras ensemble ONCE per (region, grid) and reuse across reruns."""
    from src.ensemble import EnsemblePredictor
    return EnsemblePredictor(seq, h, w, 'data')


@timed("inference: predict_future_annual")
@st.cache_data(show_spinner=False)
def predict_future_annual(target_year):
    steps = target_year - END_YEAR
    if steps <= 0:
        return None, None, None, None, None

    if N_YEARS < 5:
        return None, None, None, None, None

    seq = min(30, N_YEARS - 1)
    full = np.stack([rain_data, temp_data], axis=-1)
    rmax, rmin = float(np.nanmax(full[..., 0])), float(np.nanmin(full[..., 0]))
    tmax, tmin = float(np.nanmax(full[..., 1])), float(np.nanmin(full[..., 1]))
    rr = (rmax - rmin) or 1.0
    tr = (tmax - tmin) or 1.0
    norm = full.copy()
    norm[..., 0] = (norm[..., 0] - rmin) / rr
    norm[..., 1] = (norm[..., 1] - tmin) / tr
    window = norm[-seq:]
    ens_weights = None
    mean_uncertainty = None
    try:
        ens = _get_ensemble(active_region(), seq, mask.shape[0], mask.shape[1])

        # Validation-driven weighting to improve forecast robustness.
        X_sup, Y_sup = [], []
        for i in range(norm.shape[0] - seq):
            X_sup.append(norm[i:i + seq])
            Y_sup.append(norm[i + seq])
        if len(X_sup) >= 3:
            X_sup = np.array(X_sup)
            Y_sup = np.array(Y_sup)
            val_n = min(5, len(X_sup))
            ens_weights, _ = ens.fit_weights(X_sup[-val_n:], Y_sup[-val_n:], mask=mask)

        for _ in range(steps):
            inp = np.expand_dims(window, 0)
            mean, std = ens.predict(inp)
            pred = np.clip(mean[0], 0, 1)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            mean_uncertainty = std[0]
            window = np.append(window[1:], np.expand_dims(pred, 0), axis=0)
        rain_pred = pred[..., 0] * rr + rmin
        temp_pred = pred[..., 1] * tr + tmin
        # Physical plausibility guardrails for India forecasts.
        rain_pred = np.clip(np.nan_to_num(rain_pred, nan=0.0), 0.0, 12000.0)
        temp_pred = np.clip(np.nan_to_num(temp_pred, nan=float(np.nanmean(temp_data[-1]))), 5.0, 55.0)
        if mean_uncertainty is not None:
            # Convert normalized uncertainty back to physical units.
            rain_unc = mean_uncertainty[..., 0] * rr
            temp_unc = mean_uncertainty[..., 1] * tr
        else:
            rain_unc = None
            temp_unc = None
        return rain_pred, temp_pred, rain_unc, temp_unc, ens_weights
    except Exception as e:
        st.warning(f"Ensemble fallback: {e}")
        # Fallback: simple trend projection
        r_trend = np.nanmean(rain_data[-5:] - rain_data[-10:-5], axis=0) / 5
        t_trend = np.nanmean(temp_data[-5:] - temp_data[-10:-5], axis=0) / 5
        return rain_data[-1] + r_trend * steps, temp_data[-1] + t_trend * steps, None, None, None

@timed("inference: get_all_future_annual")
@st.cache_data(show_spinner=False)
def get_all_future_annual(target_year):
    """Efficiently computes and returns all intermediate years up to target_year."""
    steps = target_year - END_YEAR
    if steps <= 0 or N_YEARS < 5:
        return {}
    
    seq = min(30, N_YEARS - 1)
    full = np.stack([rain_data, temp_data], axis=-1)
    rmax, rmin = float(np.nanmax(full[..., 0])), float(np.nanmin(full[..., 0]))
    tmax, tmin = float(np.nanmax(full[..., 1])), float(np.nanmin(full[..., 1]))
    rr = (rmax - rmin) or 1.0
    tr = (tmax - tmin) or 1.0
    norm = full.copy()
    norm[..., 0] = (norm[..., 0] - rmin) / rr
    norm[..., 1] = (norm[..., 1] - tmin) / tr
    window = norm[-seq:]
    
    results = {}
    try:
        from src.ensemble import EnsemblePredictor
        ens = EnsemblePredictor(seq, mask.shape[0], mask.shape[1], 'data')
        
        X_sup, Y_sup = [], []
        for i in range(norm.shape[0] - seq):
            X_sup.append(norm[i:i + seq])
            Y_sup.append(norm[i + seq])
        if len(X_sup) >= 3:
            X_sup = np.array(X_sup)
            Y_sup = np.array(Y_sup)
            val_n = min(5, len(X_sup))
            ens.fit_weights(X_sup[-val_n:], Y_sup[-val_n:], mask=mask)
            
        for s in range(steps):
            inp = np.expand_dims(window, 0)
            mean, _ = ens.predict(inp)
            pred = np.clip(mean[0], 0, 1)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            window = np.append(window[1:], np.expand_dims(pred, 0), axis=0)
            
            rain_pred = pred[..., 0] * rr + rmin
            temp_pred = pred[..., 1] * tr + tmin
            rain_pred = np.clip(np.nan_to_num(rain_pred, nan=0.0), 0.0, 12000.0)
            temp_pred = np.clip(np.nan_to_num(temp_pred, nan=float(np.nanmean(temp_data[-1]))), 5.0, 55.0)
            
            results[END_YEAR + 1 + s] = (rain_pred, temp_pred)
            
        return results
    except Exception as e:
        # Fallback
        r_trend = np.nanmean(rain_data[-5:] - rain_data[-10:-5], axis=0) / 5
        t_trend = np.nanmean(temp_data[-5:] - temp_data[-10:-5], axis=0) / 5
        for s in range(steps):
            results[END_YEAR + 1 + s] = (rain_data[-1] + r_trend * (s+1), temp_data[-1] + t_trend * (s+1))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Animated Twin helper
# Builds daily rainfall + temperature grids for a chosen year and month.
# Returns: rain_stack [D,H,W], temp_stack [D,H,W]
# ══════════════════════════════════════════════════════════════════════════════
@timed("figures: build_animation_grids")
@st.cache_data(show_spinner=False)
def build_animation_grids(target_year, month_idx):
    h, w = mask.shape
    max_d = days_in_month(int(target_year), int(month_idx))
    rain_stack = np.zeros((max_d, h, w), dtype=np.float32)
    temp_stack = np.zeros((max_d, h, w), dtype=np.float32)

    # Past: observed daily files (RF25 nc + GRD temp).
    if target_year <= END_YEAR:
        arr = get_nc_year_data(int(target_year), active_region())
        if arr is None:
            raise FileNotFoundError(f"Rainfall file for {target_year} not found or corrupt.")
        n_t = int(arr.shape[0])

        for d in range(1, max_d + 1):
            doy = month_day_to_doy(month_idx, d, int(target_year))
            d_idx = doy - 1
            if d_idx >= n_t:
                # Partial-year file or short archive: blend last available with climatology
                clim_r = get_day_rain_climatology(doy)
                last_r = np.maximum(np.nan_to_num(arr[-1], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
                slice_r = np.where(
                    mask == 1,
                    0.55 * np.nan_to_num(clim_r, nan=0.0) + 0.45 * last_r,
                    np.nan,
                )
            else:
                slice_r = np.maximum(arr[d_idx], 0.0)
                slice_r = np.where(mask == 1, slice_r, np.nan)
                # Sparse / corrupt day: blend with climatology so animation does not "lose" rain
                land_ok = np.isfinite(slice_r) & (mask == 1)
                frac = float(np.mean(land_ok)) if np.any(mask == 1) else 0.0
                if frac < 0.82:
                    clim_r = get_day_rain_climatology(doy)
                    slice_r = np.where(
                        land_ok,
                        0.72 * slice_r + 0.28 * np.nan_to_num(clim_r, nan=0.0),
                        np.where(mask == 1, np.nan_to_num(clim_r, nan=0.0), np.nan),
                    )

            rain_stack[d - 1] = slice_r.astype(np.float32)

            tg_year = DS.daily_tmax(active_region(), int(target_year))
            if tg_year is None:
                raise FileNotFoundError(f"Temperature file missing for year={target_year}, doy={doy}")
            _di = min(doy - 1, tg_year.shape[0] - 1)
            temp_stack[d - 1] = tg_year[_di]

        rain_stack = np.where(mask == 1, rain_stack, np.nan)
        temp_stack = np.where(mask == 1, temp_stack, np.nan)
        return rain_stack, temp_stack

    # Future: trained annual ensemble -> day-wise disaggregation with climatology anchors.
    pred_rain_ann, pred_temp_ann, _, _, _ = predict_future_annual(int(target_year))
    if pred_rain_ann is None:
        raise ValueError(f"Failed to build future prediction for year={target_year}")

    annual_rain_clim = np.nanmean(rain_data[-10:], axis=0)
    annual_temp_clim = np.nanmean(temp_data[-10:], axis=0)
    pred_rain_ann = np.nan_to_num(pred_rain_ann, nan=0.0, posinf=0.0, neginf=0.0)
    pred_temp_ann = np.nan_to_num(pred_temp_ann, nan=float(np.nanmean(annual_temp_clim)))

    denom = np.where(np.abs(annual_rain_clim) < 1.0, 1.0, annual_rain_clim)
    rain_ratio = np.clip(pred_rain_ann / denom, 0.45, 2.35)

    temp_anomaly = np.clip(pred_temp_ann - annual_temp_clim, -6.0, 6.0)

    for d in range(1, max_d + 1):
        doy = month_day_to_doy(month_idx, d, int(target_year))
        day_rain_clim = get_day_rain_climatology(doy)
        day_temp_clim = get_day_temp_climatology(doy)

        future_rain = np.clip(safe_land_grid(day_rain_clim * rain_ratio, fallback_value=0.0), 0.0, 350.0)
        future_temp = np.clip(
            safe_land_grid(day_temp_clim + temp_anomaly, fallback_value=float(np.nanmean(day_temp_clim))),
            10.0,
            52.0,
        )

        future_rain, future_temp, _ = apply_physical_qc(
            month_idx=month_idx,
            future_rain=future_rain,
            future_temp=future_temp,
            day_rain_clim=day_rain_clim,
            day_temp_clim=day_temp_clim,
        )
        # Dry months (esp. Jan): ensemble can suppress rain too much vs climatology — keep plausible totals.
        if month_idx in (0, 1, 10, 11):
            future_rain = np.maximum(future_rain, 0.28 * day_rain_clim)

        rain_stack[d - 1] = future_rain.astype(np.float32)
        temp_stack[d - 1] = future_temp.astype(np.float32)

    # Temporal smoothing: temperature only.
    temp_smooth = np.zeros_like(temp_stack)
    for d in range(max_d):
        start_idx = max(0, d - 1)
        end_idx = min(max_d, d + 2)
        temp_smooth[d] = np.nanmean(temp_stack[start_idx:end_idx], axis=0)

    return rain_stack, temp_smooth

@st.cache_data(show_spinner=False)
def _grid_to_png_data_url(
    grid,
    cmap_name,
    vmin,
    vmax,
    interpolate_nans=True,
    rain_cut_mm=None,
):
    import scipy.ndimage as ndimage
    import pandas as pd

    g = np.array(grid, dtype=np.float64)

    if interpolate_nans == "rain":
        # Land-only, light hole-fill, mild blur — matches Daily Explorer (no heavy temp-style smooth).
        g = np.where(mask == 1, g, np.nan)
        med = float(np.nanmedian(g)) if np.any(np.isfinite(g)) else 0.0
        g_df = pd.DataFrame(g)
        g = g_df.interpolate(method="linear", axis=1, limit_direction="both", limit=3).interpolate(
            method="linear", axis=0, limit_direction="both", limit=3
        ).values
        g = ndimage.gaussian_filter(np.nan_to_num(g, nan=med), sigma=0.35)
        g = np.where(mask == 1, g, np.nan)
        cut = rain_cut_mm if rain_cut_mm is not None else max(0.06, float(vmax) * 0.032)
        # Dry / trace rain: keep a faint Blues tint on land (opaque) so Cesium never
        # "shows through" to temperature or other layers — fully transparent rain was
        # misread as a temp map when Rain mode was selected.
        floor_v = float(min(2.8, max(0.06, float(vmax) * 0.038, cut * 0.42)))
        land = mask == 1
        dry = land & (~np.isfinite(g) | (g < cut))
        g = np.where(dry, floor_v, g)
        g = np.where(land, g, np.nan)
    elif interpolate_nans is True:
        # For temperature: Fill missing data holes using robust 2D interpolation
        g_df = pd.DataFrame(g)
        g = g_df.interpolate(axis=1, limit_direction='both').interpolate(axis=0, limit_direction='both').values
        # Smooth the continuous data
        g = ndimage.gaussian_filter(g, sigma=1.2)
    elif interpolate_nans == "clouds":
        # For clouds: heavy Gaussian blur to create contiguous smoke/fog masses
        g = ndimage.gaussian_filter(np.nan_to_num(g, nan=0.0), sigma=2.8)
        g = np.where(g < 0.2, np.nan, g) # keep soft edges
    else:
        # For rainfall: Treat NaNs as 0 so Gaussian blur creates smooth glowing storm clouds
        g = ndimage.gaussian_filter(np.nan_to_num(g, nan=0.0), sigma=1.5)
        # Anything that smoothed to near zero should be completely transparent
        g = np.where(g < 1.0, np.nan, g)
        
    # Re-apply the strict India geographical mask to keep borders perfectly crisp
    g = np.where(mask == 1, g, np.nan)
    
    # Grid arrays have lat=6.5 at index 0 (bottom). Images draw index 0 at the top.
    # We must flip vertically to prevent India from being upside down!
    g = np.flipud(g)
    
    # Slight contrast stretch (skip for rainfall — it hid light-moderate rain in Cesium)
    if interpolate_nans != "rain":
        vmin = vmin + (vmax - vmin) * 0.05
        vmax = vmax - (vmax - vmin) * 0.05
    
    if cmap_name == "clouds_white":
        from matplotlib.colors import LinearSegmentedColormap
        # Custom white-to-white gradient with alpha (opacity) ramping up
        cmap = LinearSegmentedColormap.from_list("clouds_white", [
            (1.0, 1.0, 1.0, 0.00), # 0%  - Transparent
            (1.0, 1.0, 1.0, 0.45), # 33% - Wispy smoke
            (1.0, 1.0, 1.0, 0.85), # 66% - Dense fog
            (0.9, 0.9, 0.95, 1.00) # 100%- Solid stormy core
        ])
    else:
        cmap = plt.get_cmap(cmap_name).copy()
        
    cmap.set_bad((0, 0, 0, 0)) # NaNs become fully transparent
    cmap.set_under((0, 0, 0, 0))
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(g))
    rgba_u8 = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    from PIL import Image
    img = Image.fromarray(rgba_u8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _rain_frame_vmax(frame):
    """Per-day color cap so a wet-month global p99 does not wash out dry January frames."""
    land = np.where(mask == 1, frame, np.nan)
    if not np.any(np.isfinite(land)):
        return 12.0
    p88 = float(np.nanpercentile(land, 88))
    p98 = float(np.nanpercentile(land, 98))
    mx = float(np.nanmax(land))
    vm = max(3.5, p88 * 1.25, p98 * 1.08, mx * 0.78)
    return float(min(150.0, vm))


@st.cache_data(show_spinner=False)
def build_cesium_frame_urls(target_year, month_idx):
    rain_seq, temp_seq = build_animation_grids(target_year, month_idx)
    max_d = rain_seq.shape[0]

    # Downsample to every 2nd day to halve PNG generation (75 vs 150) and reduce payload
    step = max(1, max_d // 15)
    rain_ds = rain_seq[::step]
    temp_ds = temp_seq[::step]
    n_frames = len(rain_ds)

    temp_vals = np.where(mask == 1, temp_ds, np.nan)
    t_lo = float(np.nanpercentile(temp_vals, 5))
    t_hi = float(np.nanpercentile(temp_vals, 95))
    temp_vmin = max(8.0, t_lo - 1.0)
    temp_vmax = min(55.0, max(temp_vmin + 6.0, t_hi + 1.0))

    temp_urls, rain_urls, cloud_urls = [], [], []
    for i in range(n_frames):
        temp_urls.append(_grid_to_png_data_url(temp_ds[i], "turbo", temp_vmin, temp_vmax, interpolate_nans=True))
        rain_vm = _rain_frame_vmax(rain_ds[i])
        rain_cut = max(0.07, rain_vm * 0.03)
        # Rainfall: Blues + dedicated "rain" pipeline (same data philosophy as Daily Explorer)
        rain_urls.append(
            _grid_to_png_data_url(
                rain_ds[i], "Blues", 0.0, rain_vm, interpolate_nans="rain", rain_cut_mm=rain_cut
            )
        )

        # Cloud Coverage: heavily smoothed white fog/smoke proxy using rainfall data
        cc = np.where(rain_ds[i] > 0.0, rain_ds[i], np.nan)
        cloud_urls.append(_grid_to_png_data_url(cc, "clouds_white", 0.0, rain_vm * 0.6, interpolate_nans="clouds"))

    land_means = [
        float(np.nanmean(np.where(mask == 1, rain_ds[i], np.nan))) for i in range(n_frames)
    ]
    thumb_i = int(np.argmax(land_means)) if n_frames else 0
    thumb_rain = rain_urls[thumb_i] if rain_urls else ""

    return temp_urls, rain_urls, cloud_urls, thumb_rain

def render_india_cesium_3d(
    temp_urls,
    rain_urls,
    cloud_urls,
    year_label,
    month_label,
    speed_ms=180,
    thumb_rain=None,
):
    ion_token = os.getenv("CESIUM_ION_TOKEN", "")
    num_frames = len(temp_urls)
    thumb_temp = temp_urls[0] if temp_urls else ""
    if thumb_rain is None:
        thumb_rain = rain_urls[0] if rain_urls else ""
    thumb_cloud = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?service=WMS&request=GetMap&layers=MODIS_Terra_CorrectedReflectance_TrueColor&styles=&format=image/jpeg&transparent=false&version=1.1.1&srs=EPSG:4326&bbox=66.5,6.5,100,38.5&width=512&height=256"
    html = f"""
    <html><head>
      <script>window.CESIUM_BASE_URL="https://unpkg.com/cesium@1.104.0/Build/Cesium/";</script>
      <script src="https://unpkg.com/cesium@1.104.0/Build/Cesium/Cesium.js"></script>
      <link href="https://unpkg.com/cesium@1.104.0/Build/Cesium/Widgets/widgets.css" rel="stylesheet"/>
      <style>
        html,body,#cesiumContainer{{width:100%;height:100vh;margin:0;overflow:hidden;background:#050a18;font-family:Inter,Segoe UI,Arial,sans-serif;}}
        .cesium-viewer .cesium-widget-credits{{display:none!important;}}
        #spaceFx{{position:absolute;inset:0;z-index:5;pointer-events:none;
          background:
            radial-gradient(circle at 20% 15%, rgba(75,135,255,0.26) 0 18%, transparent 45%),
            radial-gradient(circle at 78% 74%, rgba(33,210,255,0.14) 0 14%, transparent 45%),
            radial-gradient(circle at 50% 100%, rgba(255,110,70,0.08) 0 16%, transparent 55%);
          animation:spacePulse 8s ease-in-out infinite;}}
        @keyframes spacePulse{{0%{{opacity:.62;}}50%{{opacity:1;}}100%{{opacity:.62;}}}}
        #top{{position:absolute;left:50%;top:12px;transform:translateX(-50%);z-index:100;width:min(820px,94vw);
          background:rgba(12,20,44,0.56);border:1px solid rgba(255,255,255,0.24);border-radius:16px;padding:12px 16px;color:#fff;backdrop-filter:blur(11px);
          box-shadow:0 0 0 1px rgba(120,180,255,0.12),0 10px 30px rgba(0,0,0,0.45),0 0 44px rgba(49,125,255,0.2) inset;}}
        #title{{text-align:center;font-size:16px;font-weight:600;margin-bottom:8px;}}
        #row1{{display:flex;align-items:center;gap:10px;}} .m{{width:34px;font-size:11px;opacity:.9;text-align:center;}}
        #daySlider{{flex:1;accent-color:#2ba7ff;}}
        .ib{{width:30px;height:30px;border:none;border-radius:8px;background:rgba(255,255,255,0.12);color:#fff;cursor:pointer;transition:.2s;}}
        .ib:hover{{background:rgba(255,255,255,0.24);}}
        #row2{{display:flex;justify-content:center;gap:10px;margin-top:8px;}}
        #chips{{display:flex;gap:8px;justify-content:center;margin-top:8px;flex-wrap:wrap;}}
        .chip{{font-size:11px;padding:4px 8px;border-radius:12px;background:rgba(255,255,255,0.12);cursor:pointer;}} .chip.on{{background:rgba(76,168,255,0.55);}}
        #menu{{position:absolute;left:50%;top:112px;transform:translateX(-50%);display:none;z-index:110;width:min(560px,94vw);background:rgba(15,26,54,0.72);border:1px solid rgba(255,255,255,0.2);border-radius:14px;padding:12px;backdrop-filter:blur(10px);}}
        #grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}}
        .card{{border:1px solid rgba(255,255,255,0.18);border-radius:10px;overflow:hidden;cursor:pointer;background:rgba(255,255,255,0.06);}} .card.active{{outline:2px solid #4ab5ff;}}
        .card img{{width:100%;height:86px;object-fit:cover;display:block;}} .card div{{font-size:12px;color:#e8f3ff;text-align:center;padding:6px 4px;}}
        #legend{{position:absolute;left:50%;bottom:12px;transform:translateX(-50%);z-index:100;width:min(560px,92vw);background:rgba(15,26,54,0.62);border:1px solid rgba(255,255,255,0.2);border-radius:14px;color:#fff;padding:9px 12px;text-align:center;}}
        #lTitle{{font-size:13px;font-weight:600;margin-bottom:6px;}} #bar{{height:12px;border-radius:7px;border:1px solid rgba(255,255,255,0.2);}}
        #ticks{{display:flex;justify-content:space-between;font-size:10px;opacity:.85;margin-top:5px;}}
        #status{{position:absolute;left:12px;bottom:12px;z-index:100;font-size:11px;background:rgba(8,15,35,0.72);color:#9fd4ff;border:1px solid rgba(255,255,255,0.15);border-radius:8px;padding:4px 8px;}}
        #sideDock{{position:absolute;left:14px;top:84px;z-index:120;width:58px;padding:8px 7px;border-radius:16px;
          background:linear-gradient(180deg,rgba(10,18,38,0.92),rgba(6,13,30,0.9));border:1px solid rgba(255,255,255,0.16);
          box-shadow:0 10px 28px rgba(0,0,0,0.5),0 0 24px rgba(56,140,255,0.22) inset;display:flex;flex-direction:column;gap:8px;}}
        .dockBtn{{height:44px;border:none;border-radius:12px;background:transparent;color:#d8ebff;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:.22s;}}
        .dockBtn:hover,.dockBtn.on{{background:rgba(74,160,255,0.24);box-shadow:0 0 16px rgba(66,145,255,0.4) inset,0 0 12px rgba(30,124,255,0.35);}}
        .dockBtn svg{{width:25px;height:25px;stroke:currentColor;stroke-width:1.8;fill:none;}}
      </style>
    </head>
    <body>
      <div id="spaceFx"></div>
      <div id="top"><div id="title">ISRO Climate DT — {month_label} {year_label} • Day <span id="dayLabel">1</span></div>
        <div id="row1"><div class="m">Day 1</div><button class="ib" id="playBtn">▶</button><button class="ib" id="pauseBtn">❚❚</button><input id="daySlider" type="range" min="1" max="{num_frames}" value="1"/><div class="m">Day {num_frames}</div></div>
        <div id="row2"><button class="ib" id="spinBtn">🌐</button><button class="ib" id="menuBtn">🗂</button><button class="ib" id="flyBtn">✈</button><button class="ib" id="fullBtn">⛶</button></div>
      </div>
      <div id="menu"><div id="grid">
        <div class="card" id="cTemp"><img src="{thumb_temp}"/><div>Maximum Temperature</div></div>
        <div class="card active" id="cRain"><img src="{thumb_rain}"/><div>Accumulated Rainfall</div></div>
        <div class="card" id="cCloud"><img src="{thumb_cloud}"/><div>Total Cloud Coverage</div></div>
      </div></div>
      <div id="legend"><div id="lTitle">Accumulated Rainfall (mm/day)</div><div id="bar" style="background:linear-gradient(90deg,#f4f8ff 0%,#d8e8ff 25%,#9ec5ff 50%,#5e9fff 75%,#1f63d8 100%)"></div><div id="ticks"><span id="tk0">0</span><span id="tk1">100</span></div><div style="font-size:10px;opacity:.8;margin-top:4px;">Powered by ISRO</div></div>
      <div id="sideDock">
        <button class="dockBtn on" id="dockGlobe" title="Rotate Globe">
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"></circle><path d="M4 12h16M12 4c3 2.5 3 13.5 0 16M12 4c-3 2.5 -3 13.5 0 16"></path></svg>
        </button>
        <button class="dockBtn" id="dockGrid" title="Flat Cinematic Map">
          <svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14"></rect><path d="M3 10h18M3 14h18M9 5v14M15 5v14"></path></svg>
        </button>
        <button class="dockBtn" id="dockCine" title="Tilted Plate Cinematic">
          <svg viewBox="0 0 24 24"><path d="M3 19l3-10h12l3 10H3Z"></path><path d="M7 9l2-4h6l2 4"></path></svg>
        </button>
      </div>
      <div id="status">Initializing…</div>
      <div id="cesiumContainer"></div>
      <script>
        const T={temp_urls}, R={rain_urls}, C={cloud_urls}, N=T.length;
        const RECT=Cesium.Rectangle.fromDegrees(66.5,6.5,100.0,38.5);
        if("{ion_token}") Cesium.Ion.defaultAccessToken="{ion_token}";
        const setS=(m)=>{{const e=document.getElementById("status"); if(e)e.innerText=m;}};
        function makeIndiaFogTexture(){{
          const c=document.createElement("canvas");
          c.width=640; c.height=420;
          const g=c.getContext("2d");
          g.clearRect(0,0,c.width,c.height);
          // Main diffuse fog mass around India.
          const grad=g.createRadialGradient(320,210,40,320,210,210);
          grad.addColorStop(0.0,"rgba(255,255,255,0.55)");
          grad.addColorStop(0.35,"rgba(255,255,255,0.34)");
          grad.addColorStop(0.72,"rgba(255,255,255,0.16)");
          grad.addColorStop(1.0,"rgba(255,255,255,0.0)");
          g.fillStyle=grad;
          g.fillRect(0,0,c.width,c.height);
          // Layered smoky blobs.
          const blobs=[[290,210,110,0.18],[360,180,92,0.16],[245,170,84,0.12],[380,245,98,0.14],[300,260,120,0.12],[430,205,76,0.10]];
          for(const b of blobs){{
            const r=g.createRadialGradient(b[0],b[1],8,b[0],b[1],b[2]);
            r.addColorStop(0,`rgba(255,255,255,${{b[3]}})`);
            r.addColorStop(1,"rgba(255,255,255,0.0)");
            g.fillStyle=r;
            g.fillRect(0,0,c.width,c.height);
          }}
          return c.toDataURL("image/png");
        }}
        function setLegend(mode){{const t=document.getElementById("lTitle"),b=document.getElementById("bar"),a=document.getElementById("tk0"),c=document.getElementById("tk1");
          if(mode==="temp"){{t.innerText="Maximum Temperature (°C)";b.style.background="linear-gradient(90deg,#30123b 0%,#466be3 15%,#28ea8d 35%,#a3ff3b 50%,#fabb21 65%,#f04129 85%,#7a0403 100%)";a.innerText="8";c.innerText="52";}}
          else if(mode==="rain"){{t.innerText="Accumulated Rainfall (mm/day)";b.style.background="linear-gradient(90deg,#f4f8ff 0%,#d8e8ff 25%,#9ec5ff 50%,#5e9fff 75%,#1f63d8 100%)";a.innerText="0";c.innerText="100";}}
          else{{t.innerText="Total Cloud Coverage";b.style.background="linear-gradient(90deg,#293754 0%,#5a6f99 35%,#98aacb 70%,#f0f4fb 100%)";a.innerText="0";c.innerText="100%";}}}}

        async function init(){{
          let tp=new Cesium.EllipsoidTerrainProvider();
          if("{ion_token}"){{try{{tp=await Cesium.createWorldTerrainAsync();}}catch(e){{}}}}
          const v=new Cesium.Viewer("cesiumContainer",{{terrainProvider:tp,baseLayerPicker:false,animation:false,timeline:false,geocoder:false,homeButton:false,sceneModePicker:false,navigationHelpButton:false,fullscreenButton:false,infoBox:false,selectionIndicator:false}});
          v.imageryLayers.removeAll();
          v.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({{url:"https://basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png",maximumLevel:19}}));
          v.scene.skyBox.show=true;v.scene.backgroundColor=Cesium.Color.fromCssColorString("#04091a");v.scene.globe.show=true;v.scene.globe.baseColor=Cesium.Color.fromCssColorString("#0a1628");v.scene.globe.enableLighting=true;v.scene.skyAtmosphere.show=true;v.scene.fog.enabled=true;try{{v.scene.fog.density=0.0002;}}catch(e){{}};
          v.scene.globe.dynamicAtmosphereLighting=true;
          v.scene.globe.dynamicAtmosphereLightingFromSun=true;
          v.scene.highDynamicRange=true;
          v.scene.postProcessStages.bloom.enabled=true;
          v.scene.postProcessStages.bloom.uniforms.contrast=105.0;
          v.scene.postProcessStages.bloom.uniforms.brightness=-0.018;
          v.scene.postProcessStages.bloom.uniforms.glowOnly=false;
          v.scene.postProcessStages.bloom.uniforms.delta=1.0;
          v.scene.postProcessStages.bloom.uniforms.sigma=1.6;
          v.scene.postProcessStages.bloom.uniforms.stepSize=2.4;
          try{{v.useBrowserRecommendedResolution=true;}}catch(e){{}}
          
          const tempLs=[],rainLs=[],cloudLs=[];
          for(let i=0;i<N;i++){{
            const t=v.imageryLayers.addImageryProvider(new Cesium.SingleTileImageryProvider({{url:T[i],rectangle:RECT}}));
            const r=v.imageryLayers.addImageryProvider(new Cesium.SingleTileImageryProvider({{url:R[i],rectangle:RECT}}));
            const c=v.imageryLayers.addImageryProvider(new Cesium.SingleTileImageryProvider({{url:C[i],rectangle:RECT}}));
            t.alpha=0; r.alpha=(i===0)?1.0:0; c.alpha=0;
            tempLs.push(t); rainLs.push(r); cloudLs.push(c);
          }}
          const indiaFogLayer=v.imageryLayers.addImageryProvider(
            new Cesium.SingleTileImageryProvider({{url:makeIndiaFogTexture(), rectangle:RECT}})
          );
          indiaFogLayer.alpha=0.0;
          function flyIndia(){{v.camera.flyTo({{destination:Cesium.Cartesian3.fromDegrees(80,22,3000000),orientation:{{heading:0,pitch:Cesium.Math.toRadians(-82),roll:0}},duration:2.2,easingFunction:Cesium.EasingFunction.CUBIC_IN_OUT}});}}
          setTimeout(flyIndia,300);
          let day=1,timer=null,speed={speed_ms},active="rain",autoRotate=true,cineMode=false;
          let viewMode="globe";
          let currentIdx=0;
          function setCard(mode){{document.getElementById("cTemp").classList.toggle("active",mode==="temp");document.getElementById("cRain").classList.toggle("active",mode==="rain");document.getElementById("cCloud").classList.toggle("active",mode==="cloud");}}
          function apply(idx){{const i=Math.max(0,Math.min(N-1,idx));for(let k=0;k<N;k++){{tempLs[k].alpha=0;rainLs[k].alpha=0;cloudLs[k].alpha=0;}}
            if(active==="temp"){{tempLs[i].alpha=0.95;rainLs[i].alpha=0.20;indiaFogLayer.alpha=0.0;}}
            if(active==="rain"){{
              rainLs[i].alpha=1.0;
              tempLs[i].alpha=0.0;
              cloudLs[i].alpha=0.0;
              indiaFogLayer.alpha=0.0;
            }}
            if(active==="cloud"){{
              cloudLs[i].alpha=0.95;
              tempLs[i].alpha=0.15;
              rainLs[i].alpha=0.05;
              indiaFogLayer.alpha=0.0;
            }}
            currentIdx=i;document.getElementById("daySlider").value=i+1;document.getElementById("dayLabel").innerText=i+1;}}
          function refresh(i){{
            const target=Math.max(0,Math.min(N-1,i));
            const start=currentIdx;
            if(start===target){{day=target+1;apply(target);return;}}
            const st=performance.now(),dur=360;
            function fade(ts){{
              const p=Math.min(1,(ts-st)/dur);
              if(active!=="cloud"){{
                const ia=Math.floor(start), ib=Math.floor(target);
                for(let k=0;k<N;k++){{tempLs[k].alpha=0;rainLs[k].alpha=0;}}
                const wa=1-p, wb=p;
                if(active==="temp"){{tempLs[ia].alpha=0.95*wa;tempLs[ib].alpha=0.95*wb;rainLs[ia].alpha=0.20*wa;rainLs[ib].alpha=0.20*wb;}}
                if(active==="rain"){{rainLs[ia].alpha=1.0*wa;rainLs[ib].alpha=1.0*wb;tempLs[ia].alpha=0;tempLs[ib].alpha=0;}}
              }}
              if(p<1){{requestAnimationFrame(fade);}}else{{day=target+1;apply(target);}}
            }}
            requestAnimationFrame(fade);
          }}
          function play(){{if(timer)return;timer=setInterval(()=>{{refresh(day%N);}},speed);}} function pause(){{if(timer){{clearInterval(timer);timer=null;}}}}
          document.getElementById("playBtn").onclick=play;document.getElementById("pauseBtn").onclick=pause;document.getElementById("flyBtn").onclick=flyIndia;
          document.getElementById("fullBtn").onclick=()=>{{const el=document.documentElement;if(el.requestFullscreen)el.requestFullscreen();}};
          document.getElementById("menuBtn").onclick=()=>{{const m=document.getElementById("menu");m.style.display=(m.style.display==="block")?"none":"block";}};
          document.getElementById("spinBtn").onclick=()=>{{
            if(viewMode!=="globe") enterGlobeMode();
            autoRotate=!autoRotate;
            document.getElementById("dockGlobe").classList.toggle("on",autoRotate);
          }};
          document.getElementById("daySlider").oninput=(e)=>refresh(parseInt(e.target.value,10)-1);
          document.getElementById("cTemp").onclick=()=>{{active="temp";setLegend("temp");setCard("temp");apply(day-1);}};
          document.getElementById("cRain").onclick=()=>{{active="rain";setLegend("rain");setCard("rain");apply(day-1);}};
          document.getElementById("cCloud").onclick=()=>{{
            active="cloud";setLegend("cloud");setCard("cloud");apply(day-1);
            if(!cloudLayer) setS("Cloud coverage source unavailable right now (NASA GIBS).");
          }};
          document.getElementById("dockGlobe").onclick=()=>{{autoRotate=!autoRotate;document.getElementById("dockGlobe").classList.toggle("on",autoRotate);}};
          function enterGlobeMode(){{
            viewMode="globe";
            v.scene.morphTo3D(0.8);
            v.scene.globe.show=true;
            if(cloudLayer) cloudLayer.alpha=(active==="cloud")?0.70:0.0;
            flyIndia();
            document.getElementById("dockGrid").classList.remove("on");
            document.getElementById("dockCine").classList.remove("on");
          }}
          function enterFlatMode(){{
            viewMode="flat";
            v.scene.morphTo2D(1.0);
            v.scene.globe.show=true;
            if(cloudLayer) cloudLayer.alpha=Math.max(cloudLayer.alpha,0.78);
            autoRotate=false;
            document.getElementById("dockGlobe").classList.remove("on");
            document.getElementById("dockGrid").classList.add("on");
            document.getElementById("dockCine").classList.remove("on");
            setTimeout(()=>{{
              try {{
                v.camera.flyTo({{
                  destination: Cesium.Rectangle.fromDegrees(-180,-80,180,80),
                  duration: 1.2
                }});
              }} catch(e) {{}}
            }}, 900);
          }}
          function enterTiltedMode(){{
            viewMode="tilted";
            v.scene.morphToColumbusView(1.0);
            v.scene.globe.show=true;
            if(cloudLayer) cloudLayer.alpha=Math.max(cloudLayer.alpha,0.76);
            autoRotate=false;
            document.getElementById("dockGlobe").classList.remove("on");
            document.getElementById("dockGrid").classList.remove("on");
            document.getElementById("dockCine").classList.add("on");
            setTimeout(()=>{{
              try {{
                v.camera.flyTo({{
                  destination: Cesium.Cartesian3.fromDegrees(84,10,22000000),
                  orientation: {{
                    heading: Cesium.Math.toRadians(0),
                    pitch: Cesium.Math.toRadians(-39),
                    roll: 0
                  }},
                  duration: 1.6
                }});
              }} catch(e) {{}}
            }}, 900);
          }}
          document.getElementById("dockGrid").onclick=()=>{{
            if(viewMode==="flat") enterGlobeMode();
            else enterFlatMode();
            apply(day-1);
          }};
          document.getElementById("dockCine").onclick=()=>{{
            cineMode=!cineMode;
            if(cineMode){{
              enterTiltedMode();
              v.scene.postProcessStages.bloom.uniforms.brightness=-0.032;
            }} else {{
              enterGlobeMode();
              autoRotate=true;
              document.getElementById("dockGlobe").classList.add("on");
              v.scene.postProcessStages.bloom.uniforms.brightness=-0.018;
            }}
          }};
          v.clock.onTick.addEventListener(()=>{{
            const t=Cesium.JulianDate.toDate(v.clock.currentTime).getTime()*0.001;
            // Gentle bloom breathing (slower / subtler than before)
            if(active==="rain"){{
              v.scene.postProcessStages.bloom.uniforms.contrast=96.0+0.55*Math.sin(t*0.28);
              v.scene.postProcessStages.bloom.uniforms.sigma=1.45+0.04*Math.cos(t*0.28);
            }} else {{
              v.scene.postProcessStages.bloom.uniforms.contrast=112.0+1.35*Math.sin(t*0.28);
              v.scene.postProcessStages.bloom.uniforms.sigma=1.66+0.065*Math.cos(t*0.28);
            }}
            if(active==="cloud"){{
              indiaFogLayer.alpha=0.42+0.09*Math.sin(t*1.05);
            }} else {{
              indiaFogLayer.alpha=0.0;
            }}
            // Slow orbit to the **right** (opposite of rotateRight): use rotateLeft with small radians.
            if(autoRotate && viewMode==="globe")v.scene.camera.rotateLeft(cineMode?0.00052:0.00036);
          }});
          setLegend("rain");setCard("rain");apply(0);setS("Ready");
        }}
        init().catch((e)=>{{console.error(e);setS("Error: "+(e&&e.message?e.message:e));}});
      </script>
    </body></html>
    """
    components.html(html, height=860, scrolling=False)


# ── Matplotlib simulated animation (replaces the Cesium 3D globe) ─────────────
@timed("figures: build_matplotlib_animation_gif")
@st.cache_data(show_spinner="Rendering animation...")
def build_matplotlib_animation_gif(region, target_year, month_idx, var, speed_ms):
    """Smooth day-by-day GIF for the chosen variable/period, region-aware.

    month_idx is 0–11 for a single month, or None for the FULL YEAR (all 365 days).
    Returns raw GIF bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from scipy.ndimage import gaussian_filter, zoom
    from PIL import Image, ImageDraw

    full_year = month_idx is None
    labels = None
    if full_year:
        rain_parts, temp_parts, labels = [], [], []
        for mi in range(12):
            rs, ts = build_animation_grids(int(target_year), mi)
            rs = np.asarray(rs); ts = np.asarray(ts)
            rain_parts.append(rs); temp_parts.append(ts)
            for d in range(rs.shape[0]):
                labels.append(f"{MONTHS[mi][:3]} {d + 1}")
        rain_seq = np.concatenate(rain_parts, axis=0)
        temp_seq = np.concatenate(temp_parts, axis=0)
        tween = 1  # 365 frames already — don't tween
    else:
        rain_seq, temp_seq = build_animation_grids(int(target_year), int(month_idx))
        tween = 2
    seq = np.asarray(rain_seq if var == "Rainfall" else temp_seq, dtype=np.float64)
    n_days = int(seq.shape[0])
    land = (mask == 1)

    # colour scale (fixed across all frames so brightness doesn't flicker)
    if var == "Rainfall":
        cmap_name, vmin = "Blues", 0.0
        vmax = max(6.0, float(np.nanpercentile(seq[:, land], 96)))
        unit = "mm/day"
    else:
        cmap_name = "turbo"
        vmin = float(np.nanpercentile(seq[:, land], 4))
        vmax = float(np.nanpercentile(seq[:, land], 96))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 10.0, 45.0
        unit = "°C"

    up = 2  # spatial upsample factor for a smooth surface
    land_up = zoom(land.astype(float), up, order=0) >= 0.5

    def _prep(frame):
        g = np.nan_to_num(np.array(frame, dtype=np.float64), nan=0.0, posinf=vmax, neginf=0.0)
        g = gaussian_filter(g, sigma=0.9)          # de-speckle
        g = zoom(g, up, order=3)                    # bicubic upsample -> smooth
        g = np.clip(g, vmin, vmax)
        return np.where(land_up, g, np.nan)         # keep only land

    day_frames = [_prep(seq[i]) for i in range(n_days)]

    def _lab(i):
        return labels[i] if labels else f"Day {i + 1}/{n_days}"

    # temporal tweening between days for smooth motion (single month only)
    frames, frame_labels = [], []
    for i in range(n_days - 1):
        for tw in range(tween):
            f = tw / tween
            frames.append(day_frames[i] * (1 - f) + day_frames[i + 1] * f)
            frame_labels.append(_lab(i))
    frames.append(day_frames[-1]); frame_labels.append(_lab(n_days - 1))

    # ---- render the static chrome ONCE (axes + ticks + colorbar + title) ----
    ext = DS.REGIONS[region]["extent"]
    extent = [ext["lon"][0], ext["lon"][1], ext["lat"][0], ext["lat"][1]]
    aspect_wh = (ext["lon"][1] - ext["lon"][0]) / (ext["lat"][1] - ext["lat"][0])
    fig_h = 4.6
    fig, ax = plt.subplots(figsize=(fig_h * aspect_wh + 1.0, fig_h), facecolor="#050912")
    ax.set_facecolor("#050912")
    cmap_obj = plt.get_cmap(cmap_name).copy()
    cmap_obj.set_bad(color="#050912")
    empty = np.full_like(frames[0], np.nan)
    im = ax.imshow(empty, origin="lower", extent=extent, cmap=cmap_obj,
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.tick_params(colors="#8899bb", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#22314f")
    ax.set_xlabel("Longitude (°E)", color="#8899bb", fontsize=8)
    ax.set_ylabel("Latitude (°N)", color="#8899bb", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(unit, color="#8899bb", fontsize=8)
    cbar.ax.tick_params(colors="#8899bb", labelsize=7)
    label = DS.REGIONS[region]["label"]
    _period = f"Full Year {int(target_year)}" if full_year else f"{MONTHS[month_idx]} {int(target_year)}"
    ax.set_title(f"{label} — {_period}  ({var})",
                 color="#e6edf7", fontsize=11, pad=8)
    fig.tight_layout()
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba()).copy()
    Hf = bg.shape[0]
    bb = ax.get_window_extent()
    x0, x1 = int(bb.x0), int(bb.x1)
    y0, y1 = int(Hf - bb.y1), int(Hf - bb.y0)
    aw, ah = max(1, x1 - x0), max(1, y1 - y0)
    plt.close(fig)

    # ---- compose frames with PIL (data image over the static chrome) ----
    norm = mcolors.Normalize(vmin, vmax)
    bg_img = Image.fromarray(bg, "RGBA")
    gif_frames = []
    for k, d in enumerate(frames):
        rgba = cmap_obj(norm(np.nan_to_num(d, nan=vmin)))
        rgba[..., 3] = np.where(np.isnan(d), 0.0, 1.0)      # ocean transparent
        data_img = (Image.fromarray((rgba * 255).astype("uint8"), "RGBA")
                    .transpose(Image.FLIP_TOP_BOTTOM)         # origin lower -> image top=north
                    .resize((aw, ah), Image.BICUBIC))
        fr = bg_img.copy()
        fr.alpha_composite(data_img, (x0, y0))
        ImageDraw.Draw(fr).text((x0 + 8, y0 + 6), frame_labels[k],
                                fill=(230, 237, 247, 255))
        gif_frames.append(fr.convert("P", palette=Image.ADAPTIVE, colors=128))

    fps = max(6, min(24, int(round(1000.0 / max(1, int(speed_ms)))) * tween))
    duration_ms = int(1000.0 / fps)
    buf = io.BytesIO()
    gif_frames[0].save(buf, format="GIF", save_all=True, append_images=gif_frames[1:],
                       duration=duration_ms, loop=0, disposal=2, optimize=True)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# TABS — active tab is chosen by the vertical nav in the sidebar (above).
# Only the ACTIVE section is rendered so the other tab bodies never execute.
# ══════════════════════════════════════════════════════════════════════════════
_active_tab = st.session_state.get("active_tab") or _TAB_LABELS[0]

# Event-driven cache refresh hint when artifacts changed.
curr_sig = build_data_signature()
prev_sig = st.session_state.get("artifact_signature")
if prev_sig is None:
    st.session_state["artifact_signature"] = curr_sig
elif prev_sig != curr_sig:
    st.session_state["artifact_signature"] = curr_sig
    st.info("Model/data artifacts changed. Refreshing dashboard caches for latest predictions.")
    st.cache_data.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Daily Explorer — Historical + Future
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _india_land_mask():
    """India land mask (129×135) for the focus/spotlight context view."""
    _, _, m, _ = DS.load_aggregates("india")
    return m


def focus_india_figure(region, field, kind, title, unit, vmin, vmax):
    """Spotlight view: place a sub-region's data on the FULL India map, keep the
    region bright, and darken the rest of India with a distance gradient (vignette).

    field: region-shaped 2-D array (physical units, NaN outside the basin).
    kind : 'rain' or 'temp' (chooses colormap).
    """
    import matplotlib.pyplot as _plt
    import plotly.graph_objects as go
    from scipy.ndimage import distance_transform_edt as _edt, zoom as _zoom

    i_mask = _india_land_mask()
    H, W = i_mask.shape
    lat, lon = DS.RAIN_LAT, DS.RAIN_LON
    _, _, rlat_idx, rlon_idx = DS.region_grid(region)

    canvas = np.full((H, W), np.nan)
    canvas[np.ix_(rlat_idx, rlon_idx)] = field
    basin = ~np.isnan(canvas)

    # data → colour
    norm = np.clip((canvas - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    cmap = _plt.get_cmap("turbo" if kind == "temp" else "Blues")
    rgb = cmap(np.nan_to_num(norm))[..., :3]

    # Context: the rest of India stays clearly VISIBLE (light grey-blue silhouette),
    # only very gently dimmed with distance; ocean is dark navy (not pure black).
    dist = _edt(~basin)
    dim = np.clip(np.exp(-dist / 45.0), 0.72, 1.0)     # 1 at basin → ~0.72 far (subtle)
    land = (i_mask == 1)
    base = np.zeros((H, W, 3), dtype=float)
    base[land] = np.array([0.34, 0.40, 0.50])          # light land silhouette (visible)
    base[~land] = np.array([0.05, 0.07, 0.12])         # dark-navy ocean, not black
    dark = base * dim[..., None]

    rgb_final = np.where(basin[..., None], rgb, dark)
    rgba = (np.dstack([rgb_final, np.ones((H, W))]) * 255).astype("uint8")

    up = 4
    rgba_up = _zoom(rgba, (up, up, 1), order=1)
    img = rgba_up[::-1]  # flip so north (high lat) is at the top for go.Image

    rows, cols = img.shape[0], img.shape[1]
    fig = go.Figure(go.Image(
        z=img,
        x0=float(lon[0]), dx=float((lon[-1] - lon[0]) / (cols - 1)),
        y0=float(lat[-1]), dy=float(-(lat[-1] - lat[0]) / (rows - 1)),
        hoverinfo="skip",
    ))
    # invisible heatmap only to supply the colorbar legend
    fig.add_trace(go.Heatmap(
        z=[[vmin, vmax]], colorscale="Turbo" if kind == "temp" else "Blues",
        zmin=vmin, zmax=vmax, opacity=0.0, showscale=True,
        colorbar=dict(title=unit, len=0.75, thickness=12),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="white"), x=0.05, y=0.96),
        height=520, margin=dict(l=10, r=10, t=46, b=10),
        paper_bgcolor="#0d1526", plot_bgcolor="#0d1526", template="plotly_dark",
    )
    fig.update_xaxes(visible=False, range=[lon[0], lon[-1]])
    fig.update_yaxes(visible=False, range=[lat[0], lat[-1]], scaleanchor="x", scaleratio=1)
    return fig


@st.fragment
def _render_tab1():
    st.header("Day-Wise Climate Explorer & Predictor")
    st.markdown(f"**Historical** (1975–{END_YEAR}): from raw IMD files. **Future** ({END_YEAR+1}–2075): ensemble prediction disaggregated to daily.")

    c1, c2, c3 = st.columns(3)
    with c1:
        yr_list = list(range(START_YEAR, FUTURE_END + 1))
        yr_idx = yr_list.index(2026) if 2026 in yr_list else min(N_YEARS - 1, len(yr_list) - 1)
        yr = st.selectbox("Year", yr_list, index=yr_idx, key='d_yr')
    with c2:
        sel_month = st.selectbox("Month", MONTHS, index=5, key='d_mon')
        mon_idx = MONTHS.index(sel_month)
    with c3:
        max_d = days_in_month(int(yr), mon_idx)
        sel_day = st.number_input("Day", min_value=1, max_value=max_d, value=1, key='d_day')

    doy = month_day_to_doy(mon_idx, sel_day, int(yr))
    date_label = f"{int(sel_day)} {sel_month} {yr}"
    is_future = yr > END_YEAR

    @st.cache_data(max_entries=120, show_spinner=False)
    def _cached_load_rain(nc_path):
        basename = os.path.basename(nc_path)
        if "ind" in basename:
            try:
                year_str = basename.split("ind")[1].split("_")[0]
                year = int(year_str)
                arr = get_nc_year_data(year)
                if arr is not None:
                    return arr
            except Exception:
                pass
        import xarray as xr
        import numpy as np
        ds = xr.open_dataset(nc_path)
        rv = list(ds.data_vars)[0]
        arr = np.asarray(ds[rv].values, dtype=np.float64)
        ds.close()
        return arr

    @st.cache_data(max_entries=500, show_spinner=False)
    def _cached_load_temp(y, d_idx, region):
        arr = DS.daily_tmax(region, int(y))
        if arr is None:
            return None
        di = int(d_idx)
        if di >= arr.shape[0]:
            di = arr.shape[0] - 1
        return arr[di]

    if is_future:
        st.info(f"**Future Prediction Mode** — Ensemble forecast for **{date_label}**")

    col_r, col_t = st.columns(2)

    _region = active_region()

    def _emit_map(field, kind, title, unit, vmin, vmax, scale_key, crange):
        """Cauvery → spotlight-on-India focus view; India → normal full map."""
        if _region == "cauvery":
            return focus_india_figure("cauvery", field, kind, title, unit, vmin, vmax)
        return plotly_vis.plot_map(field, title, scale_key, custom_range=crange)
    if not is_future:
        # ── Historical ──
        with col_r:
            arr = get_nc_year_data(yr, _region)
            if arr is not None:
                n_t = int(arr.shape[0])
                d_idx = doy - 1
                if d_idx >= n_t:
                    clim_r = get_day_rain_climatology(doy)
                    last_r = np.maximum(np.nan_to_num(arr[-1], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
                    rg = np.where(
                        mask == 1,
                        0.55 * np.nan_to_num(clim_r, nan=0.0) + 0.45 * last_r,
                        np.nan,
                    )
                else:
                    rg = np.maximum(arr[d_idx], 0.0)
                    rg = np.where(mask == 1, rg, np.nan)
                    land_ok = np.isfinite(rg) & (mask == 1)
                    frac = float(np.mean(land_ok)) if np.any(mask == 1) else 0.0
                    if frac < 0.82:
                        clim_r = get_day_rain_climatology(doy)
                        rg = np.where(
                            land_ok,
                            0.72 * rg + 0.28 * np.nan_to_num(clim_r, nan=0.0),
                            np.where(mask == 1, np.nan_to_num(clim_r, nan=0.0), np.nan),
                        )
                rain_vmax = float(np.nanpercentile(np.where(mask == 1, rg, np.nan), 99))
                st.plotly_chart(
                    _emit_map(rg, "rain", f"Daily Rainfall — {date_label}", "mm",
                              0.0, max(12.0, rain_vmax), "rain_daily",
                              [0, max(12.0, rain_vmax)]),
                    use_container_width=True,
                )
            else:
                st.warning(f"Rainfall file for {yr} not found.")
        with col_t:
            tg = _cached_load_temp(yr, doy - 1, _region)
            if tg is not None:
                tvals = np.where(mask == 1, tg, np.nan)
                t_lo = float(np.nanpercentile(tvals, 5))
                t_hi = float(np.nanpercentile(tvals, 95))
                t_min = max(8.0, t_lo - 1.0)
                t_max = min(52.0, max(t_min + 6.0, t_hi + 1.0))
                st.plotly_chart(
                    _emit_map(tg, "temp", f"Daily Max Temp — {date_label}", "°C",
                              t_min, t_max, "temp_daily", [t_min, t_max]),
                    use_container_width=True,
                )
            else:
                st.warning(f"Temperature file for {yr} not found.")
    else:
        # ── Future: ensemble + disaggregation ──
        pred_rain_ann, pred_temp_ann, rain_unc_ann, temp_unc_ann, ens_weights = predict_future_annual(yr)

        if pred_rain_ann is not None:
            # Stable daily disaggregation from trained annual ensemble output.
            annual_rain_clim = np.nanmean(rain_data[-10:], axis=0)
            annual_temp_clim = np.nanmean(temp_data[-10:], axis=0)

            pred_rain_ann = np.nan_to_num(pred_rain_ann, nan=0.0, posinf=0.0, neginf=0.0)
            pred_temp_ann = np.nan_to_num(pred_temp_ann, nan=float(np.nanmean(annual_temp_clim)))

            day_rain_clim = get_day_rain_climatology(doy)
            day_temp_clim = get_day_temp_climatology(doy)

            denom = np.where(np.abs(annual_rain_clim) < 1.0, 1.0, annual_rain_clim)
            rain_ratio = np.clip(pred_rain_ann / denom, 0.45, 2.35)
            future_rain = np.clip(safe_land_grid(day_rain_clim * rain_ratio, fallback_value=0.0), 0.0, 350.0)

            temp_anomaly = np.clip(pred_temp_ann - annual_temp_clim, -6.0, 6.0)
            future_temp = np.clip(
                safe_land_grid(day_temp_clim + temp_anomaly, fallback_value=float(np.nanmean(day_temp_clim))),
                10.0,
                52.0,
            )

            future_rain, future_temp, qc_issues = apply_physical_qc(
                month_idx=mon_idx,
                future_rain=future_rain,
                future_temp=future_temp,
                day_rain_clim=day_rain_clim,
                day_temp_clim=day_temp_clim,
            )
            if mon_idx in (0, 1, 10, 11):
                future_rain = np.maximum(future_rain, 0.28 * day_rain_clim)

            with col_r:
                rain_vmax = float(np.nanpercentile(np.where(mask == 1, future_rain, np.nan), 99))
                st.plotly_chart(
                    _emit_map(future_rain, "rain", f"Predicted Rainfall — {date_label}", "mm",
                              0.0, max(15.0, rain_vmax), "rain_daily", [0, max(15.0, rain_vmax)]),
                    use_container_width=True
                )
            with col_t:
                tvals = np.where(mask == 1, future_temp, np.nan)
                t_lo = float(np.nanpercentile(tvals, 5))
                t_hi = float(np.nanpercentile(tvals, 95))
                t_min = max(8.0, t_lo - 1.0)
                t_max = min(56.0, max(t_min + 6.0, t_hi + 1.0))
                st.plotly_chart(
                    _emit_map(future_temp, "temp", f"Predicted Max Temp — {date_label}", "°C",
                              t_min, t_max, "temp_daily", [t_min, t_max]),
                    use_container_width=True
                )

            avg_r = float(np.nanmean(future_rain))
            avg_t = float(np.nanmean(future_temp))
            st.success(f"**{date_label}** — National Avg Rainfall: **{avg_r:.1f} mm** | Max Temp: **{avg_t:.1f} °C**")
            qa_col1, qa_col2, qa_col3 = st.columns(3)
            qa_col1.metric("Rain p95 (mm/day)", f"{land_stats(future_rain)['p95']:.1f}")
            qa_col2.metric("Rain p99 (mm/day)", f"{land_stats(future_rain)['p99']:.1f}")
            qa_col3.metric("Temp p95 (°C)", f"{land_stats(future_temp)['p95']:.1f}")
            if qc_issues:
                for msg in qc_issues:
                    st.warning(f"Forecast QC: {msg}")
            if rain_unc_ann is not None and temp_unc_ann is not None:
                u_r = float(np.nanmean(np.where(mask == 1, rain_unc_ann, np.nan)))
                u_t = float(np.nanmean(np.where(mask == 1, temp_unc_ann, np.nan)))
                st.caption(f"Uncertainty (ensemble spread): Rain ±{u_r:.1f} mm | Temp ±{u_t:.2f} °C")
            if ens_weights:
                st.caption(f"Calibrated ensemble weights: {ens_weights}")

            # Zone-level drilldown and export for selected day.
            st.markdown("#### Day-wise Zone Drilldown (IMD)")
            df_zone_day = compute_zone_daywise_stats(future_rain, future_temp)
            st.dataframe(df_zone_day, use_container_width=True)

            export_df = pd.DataFrame({
                "lat_index": np.repeat(np.arange(future_rain.shape[0]), future_rain.shape[1]),
                "lon_index": np.tile(np.arange(future_rain.shape[1]), future_rain.shape[0]),
                "rain_mm_day": np.ravel(np.where(mask == 1, future_rain, np.nan)),
                "temp_c": np.ravel(np.where(mask == 1, future_temp, np.nan)),
            })
            st.download_button(
                "📥 Download Selected-Day Grid (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"isro_daily_grid_{yr}_{mon_idx+1:02d}_{int(sel_day):02d}.csv",
                mime="text/csv",
            )
            st.download_button(
                "📥 Download Selected-Day Zone Stats (CSV)",
                data=df_zone_day.to_csv(index=False).encode("utf-8"),
                file_name=f"isro_daily_zones_{yr}_{mon_idx+1:02d}_{int(sel_day):02d}.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.subheader("Animated Indian Twin")
    st.caption("Matplotlib day-by-day animation. Pick a single month or the whole year; "
               "Rainfall or Temperature; the map is masked to the active region.")

    # List of years to search for 2026. If not present, fallback to max.
    yr_list = list(range(START_YEAR, FUTURE_END + 1))
    yr_idx = yr_list.index(2026) if 2026 in yr_list else min(N_YEARS - 1, len(yr_list) - 1)

    a_year = st.selectbox(
        "Animation Year",
        yr_list,
        index=yr_idx,
        key="anim_year",
    )
    a_full_year = st.checkbox("🗓️ Animate full year (all 365 days)", value=False, key="anim_full_year",
                              help="Play the entire year day-by-day (Jan 1 → Dec 31) instead of one month. "
                                   "Longer to build — the GIF has ~365 frames.")
    a_month = st.selectbox("Animation Month", MONTHS, index=5, key="anim_month",
                           disabled=a_full_year)  # index=5 is June
    a_mon_idx = MONTHS.index(a_month)

    a_var = st.radio("Variable", ["Rainfall", "Temperature"], horizontal=True, key="anim_var")
    a_speed = st.slider("Animation Speed (ms per day)", 80, 500, 180, step=10, key="anim_speed")

    _period_arg = None if a_full_year else int(a_mon_idx)
    _period_label = "full_year" if a_full_year else a_month
    try:
        if a_full_year:
            st.caption("Building a full-year animation (~365 frames) — this takes a bit longer the first time.")
        gif_bytes = build_matplotlib_animation_gif(
            active_region(), int(a_year), _period_arg, a_var, int(a_speed)
        )
        st.image(gif_bytes, width=640)
        st.download_button(
            "⬇ Download animation (GIF)",
            data=gif_bytes,
            file_name=f"{active_region()}_{a_var.lower()}_{_period_label}_{a_year}.gif",
            mime="image/gif",
            key="anim_gif_dl",
        )
    except Exception as e:
        st.error(f"Could not build animation: {e}")

    plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Historical Twin
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tab2():
    st.header("Historical Climate Twin")
    sel_yr = st.selectbox("Select Year", list(range(START_YEAR, END_YEAR + 1)), index=N_YEARS - 1, key='h_yr')
    idx = sel_yr - START_YEAR

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plotly_annual_rain_map(rain_data[idx], f"Annual Rainfall (mm) — {sel_yr}"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plotly_annual_temp_map(temp_data[idx], f"Annual Max Temp (°C) — {sel_yr}"),
            use_container_width=True,
        )

    # National averages time series
    avg_r = [float(np.nanmean(np.where(mask == 1, rain_data[i], np.nan))) for i in range(N_YEARS)]
    avg_t = [float(np.nanmean(np.where(mask == 1, temp_data[i], np.nan))) for i in range(N_YEARS)]
    yrs = list(range(START_YEAR, END_YEAR + 1))

    fig_ts, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(yrs, avg_r, 'b-o', markersize=3); ax1.set_title("National Avg Rainfall"); ax1.set_ylabel("mm"); ax1.grid(alpha=0.3)
    ax2.plot(yrs, avg_t, 'r-o', markersize=3); ax2.set_title("National Avg Max Temp"); ax2.set_ylabel("°C"); ax2.grid(alpha=0.3)
    fig_ts.tight_layout()
    st.pyplot(fig_ts)
    plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ±1°C What-If Storyline
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tab3():
    st.header("Storyline: What If Temperature Changes?")
    st.markdown("PAST / PRESENT / FUTURE comparison")

    if sens_map is not None:
        delta_t = st.slider("Temperature Change (°C)", -2.0, 3.0, 1.0, 0.1, key='wif')

        past_rain = np.nanmean(rain_data[:16], axis=0)
        present_rain = np.nanmean(rain_data[-15:], axis=0)
        future_rain = present_rain + sens_map * delta_t

        st.subheader(f"Rainfall Under {delta_t:+.1f}°C Scenario")
        land_stack = np.where(mask == 1, np.stack([past_rain, present_rain, future_rain]), np.nan)
        p99 = float(np.nanpercentile(land_stack, 99)) if np.any(np.isfinite(land_stack)) else 1500.0
        rain_vmax = max(400.0, min(3000.0, p99 * 1.05))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                plotly_vis.plot_map(
                    past_rain, "PAST (1975–1990)", "rain_annual", custom_range=[0.0, rain_vmax]
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plotly_vis.plot_map(
                    present_rain, "PRESENT (2010–2024)", "rain_annual", custom_range=[0.0, rain_vmax]
                ),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(
                plotly_vis.plot_map(
                    future_rain, f"FUTURE ({delta_t:+.1f}°C)", "rain_annual", custom_range=[0.0, rain_vmax]
                ),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Sensitivity Map")
        st.plotly_chart(
            plotly_sensitivity_map(sens_map, "Rainfall Sensitivity (mm per +1°C)"),
            use_container_width=True,
        )
        st.info("🔴 Red = rainfall decreases with warming. 🔵 Blue = rainfall increases.")
    else:
        st.warning("Sensitivity map not found. Run the Colab notebook.")
    plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tab4():
    st.header("Model Quality Assessment")

    test_n = min(5, N_YEARS - 1)
    test_rain = rain_data[-test_n:]
    test_temp = temp_data[-test_n:]
    pers_rain = np.tile(rain_data[-(test_n + 1)], (test_n, 1, 1))
    pers_temp = np.tile(temp_data[-(test_n + 1)], (test_n, 1, 1))
    clim_rain = np.tile(np.nanmean(rain_data[:-(test_n)], axis=0), (test_n, 1, 1))
    clim_temp = np.tile(np.nanmean(temp_data[:-(test_n)], axis=0), (test_n, 1, 1))

    def mrmse(a, p):
        return float(np.sqrt(np.nanmean(np.where(mask == 1, (a - p)**2, np.nan))))
    def mmae(a, p):
        return float(np.nanmean(np.where(mask == 1, np.abs(a - p), np.nan)))

    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Persistence RMSE (Rain)", f"{mrmse(test_rain, pers_rain):.0f} mm")
        st.metric("Climatology RMSE (Rain)", f"{mrmse(test_rain, clim_rain):.0f} mm")
    with mc2:
        st.metric("Persistence RMSE (Temp)", f"{mrmse(test_temp, pers_temp):.2f} °C")
        st.metric("Climatology RMSE (Temp)", f"{mrmse(test_temp, clim_temp):.2f} °C")

    st.info("The ConvLSTM ensemble should beat both baselines. Check Zone Projections for details.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Zone Projections
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tab5():
    st.header("Zone-Level Climate Projections")
    from src.zones import IMD_ZONES, get_zone_mask
    lat_dim, lon_dim = mask.shape

    zone_rows = []
    for zname, (lt0, lt1, ln0, ln1) in IMD_ZONES.items():
        box = get_zone_mask(lat_dim, lon_dim, lt0, lt1, ln0, ln1)
        land = np.logical_and(mask == 1, box)
        if not np.any(land):
            continue
        t_now = float(np.nanmean(np.where(land, temp_data[-1], np.nan)))
        r_now = float(np.nanmean(np.where(land, rain_data[-1], np.nan)))
        t_base = float(np.nanmean(np.where(land, np.nanmean(temp_data[16:46], axis=0), np.nan)))
        r_base = float(np.nanmean(np.where(land, np.nanmean(rain_data[16:46], axis=0), np.nan)))
        zone_rows.append({
            "Zone": zname,
            "Temp Now (°C)": f"{t_now:.1f}",
            "Temp Baseline (°C)": f"{t_base:.1f}",
            "Warming (°C)": f"{t_now - t_base:+.2f}",
            "Rain Now (mm)": f"{r_now:.0f}",
            "Rain Baseline (mm)": f"{r_base:.0f}",
            "Rain Change (%)": f"{((r_now - r_base) / r_base * 100) if r_base > 0 else 0:+.1f}%",
        })

    df_z = pd.DataFrame(zone_rows)
    st.dataframe(df_z, use_container_width=True)
    st.download_button("📥 Download Zone Data (CSV)", df_z.to_csv(index=False).encode('utf-8'),
                       'isro_zone_projections.csv', 'text/csv')

    # National timeseries
    st.markdown("---")
    st.subheader("50-Year National Trend")
    yrs = list(range(START_YEAR, END_YEAR + 1))
    avg_r = [float(np.nanmean(np.where(mask == 1, rain_data[i], np.nan))) for i in range(N_YEARS)]
    avg_t = [float(np.nanmean(np.where(mask == 1, temp_data[i], np.nan))) for i in range(N_YEARS)]

    fig_ts, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(yrs, avg_r, 'b-o', markersize=3); ax1.set_title("National Rainfall Trend"); ax1.set_ylabel("mm"); ax1.grid(alpha=0.3)
    ax2.plot(yrs, avg_t, 'r-o', markersize=3); ax2.set_title("National Temperature Trend"); ax2.set_ylabel("°C"); ax2.grid(alpha=0.3)
    fig_ts.tight_layout()
    st.pyplot(fig_ts)

    st.download_button("📥 Download Timeseries (CSV)",
                       pd.DataFrame({'Year': yrs, 'Avg_Rain_mm': avg_r, 'Avg_Temp_C': avg_t}).to_csv(index=False).encode('utf-8'),
                       'isro_national_timeseries.csv', 'text/csv')
    plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: 2D Dual Animation
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tab6():
    st.header("Dual-Panel 2D Animation")
    st.caption("Side-by-side animated maps of Temperature and Rainfall for any month.")
    
    d_year = st.selectbox("Animation Year", list(range(START_YEAR, FUTURE_END + 1)), index=N_YEARS - 1, key="dual_yr")
    d_month = st.selectbox("Animation Month", MONTHS, key="dual_month")
    d_mon_idx = MONTHS.index(d_month)
    
    # Track current params to auto-regenerate when year/month changes
    dual_params = (int(d_year), int(d_mon_idx))
    prev_dual = st.session_state.get("dual_anim_params")
    need_rebuild = st.button("Generate Dual Animation", key="dual_btn") or (prev_dual is not None and prev_dual != dual_params)
    
    if need_rebuild:
        try:
            rain_seq, temp_seq = build_animation_grids(d_year, d_mon_idx)
            max_d = len(rain_seq)
            
            # Downsample frames to reduce JSON size significantly
            step = max(1, max_d // 5) 
            rain_seq = rain_seq[::step]
            temp_seq = temp_seq[::step]
            day_list = list(range(1, max_d + 1))[::step]

            rain_hi = float(np.median([_rain_frame_vmax(rain_seq[i]) for i in range(len(rain_seq))]))
            rain_vmax = max(5.0, min(110.0, rain_hi * 1.12))
            temp_vals = np.where(mask == 1, temp_seq, np.nan)
            t_lo = float(np.nanpercentile(temp_vals, 5))
            t_hi = float(np.nanpercentile(temp_vals, 95))
            temp_vmin = max(8.0, t_lo - 1.0)
            temp_vmax = min(55.0, max(temp_vmin + 6.0, t_hi + 1.0))
            
            fig = plotly_vis.plot_dual_panel_animation(
                rain_seq,
                temp_seq,
                day_list,
                f"Rainfall vs Temperature — {d_month} {d_year}",
                [0.0, rain_vmax],
                [temp_vmin, temp_vmax],
                speed_ms=180
            )
            st.session_state["dual_anim_params"] = dual_params
            st.session_state["dual_anim_fig"] = fig
        except Exception as e:
            st.error(f"Could not generate animation: {e}")
            st.session_state["dual_anim_fig"] = None
    
    # Persist the figure across reruns
    dual_fig = st.session_state.get("dual_anim_fig")
    if dual_fig is not None:
        st.plotly_chart(dual_fig, use_container_width=True)
    elif prev_dual is None:
        st.info("Select a year and month, then click **Generate Dual Animation** to view.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: Climate Spirals
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_monthly_spiral_data():
    import pandas as pd
    
    # Fast proxy using pre-computed climatology arrays to avoid reading 18,000 files
    daily_rain_clim, daily_temp_clim = compute_daily_climatology()
    
    monthly_rain_clim = np.zeros(12)
    monthly_temp_clim = np.zeros(12)
    
    start_d = 0
    for m in range(12):
        mdays = MONTH_DAYS[m]
        end_d = start_d + mdays
        r_slice = daily_rain_clim[start_d:end_d]
        t_slice = daily_temp_clim[start_d:end_d]
        
        # Monthly accumulated rain = sum of daily means
        monthly_rain_clim[m] = float(np.nanmean(np.where(mask == 1, np.sum(r_slice, axis=0), np.nan)))
        # Monthly mean temp = mean of daily means
        monthly_temp_clim[m] = float(np.nanmean(np.where(mask == 1, np.nanmean(t_slice, axis=0), np.nan)))
        start_d = end_d
        
    annual_rain_clim_mean = float(np.nanmean(np.where(mask == 1, np.nanmean(rain_data[-10:], axis=0), np.nan)))
    annual_temp_clim_mean = float(np.nanmean(np.where(mask == 1, np.nanmean(temp_data[-10:], axis=0), np.nan)))
    
    # Baseline for Temperature Anomaly (1975-1990)
    base_t = float(np.nanmean(np.where(mask == 1, np.nanmean(temp_data[0:16], axis=0), np.nan)))

    rows = []
    
    # Historical
    for i, y in enumerate(range(START_YEAR, END_YEAR + 1)):
        r_mean = float(np.nanmean(np.where(mask == 1, rain_data[i], np.nan)))
        t_mean = float(np.nanmean(np.where(mask == 1, temp_data[i], np.nan)))
        
        t_anom = t_mean - base_t
        
        for m in range(12):
            val_r = r_mean / 12.0
            val_t = t_anom
            rows.append({'Year': y, 'Month': m+1, 'Rain_mm': val_r, 'Temp_Anomaly': val_t})
            
    # Future — use fast polynomial trend extrapolation on national averages
    # (avoids expensive ensemble model inference; spiral only needs scalar trends)
    hist_years = np.arange(START_YEAR, END_YEAR + 1, dtype=float)
    hist_r_means = np.array([float(np.nanmean(np.where(mask == 1, rain_data[i], np.nan)))
                             for i in range(N_YEARS)])
    hist_t_means = np.array([float(np.nanmean(np.where(mask == 1, temp_data[i], np.nan)))
                             for i in range(N_YEARS)])
    
    # Fit degree-1 polynomial on last 20 years for smoother extrapolation without explosion
    fit_n = min(20, N_YEARS)
    r_coeffs = np.polyfit(hist_years[-fit_n:], hist_r_means[-fit_n:], 1)
    t_coeffs = np.polyfit(hist_years[-fit_n:], hist_t_means[-fit_n:], 1)
    
    for y in range(END_YEAR + 1, FUTURE_END + 1):
        r_mean = float(np.polyval(r_coeffs, y))
        t_mean = float(np.polyval(t_coeffs, y))
        # Physical guardrails
        r_mean = max(0.0, r_mean)
        t_mean = np.clip(t_mean, 5.0, 55.0)
        
        t_anom = t_mean - base_t
        
        for m in range(12):
            val_r = r_mean / 12.0
            val_t = t_anom
            rows.append({'Year': y, 'Month': m+1, 'Rain_mm': val_r, 'Temp_Anomaly': val_t})

    return pd.DataFrame(rows)

@st.fragment
def _render_tab7():
    st.header("🌀 Climate Spirals")
    
    st.markdown("""
        <style>
        .control-panel {
            background: rgba(12, 24, 44, 0.75);
            padding: 16px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        </style>
    """, unsafe_allow_html=True)

    col_ui, col_chart = st.columns([1, 3.2], gap="large")

    with col_ui:
        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
        st.markdown("### Controls")
        spiral_var = st.radio(
            "Variable",
            ["Temperature Anomaly (°C)", "Monthly Rainfall (mm)"],
        )
        st.markdown("---")
        st.markdown("#### How to read")
        st.markdown("""
        - **Angle** = month of year (Jan at top).
        - **Distance from center** = magnitude of the variable.
        - **Line colour** = year (older → newer as time advances).
        - **Centre number** = current animation year.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart:
        import streamlit.components.v1 as components
        with st.spinner("Building climate spiral…"):
            spiral_df = get_monthly_spiral_data()

        if spiral_var == "Temperature Anomaly (°C)":
            spiral_df = spiral_df.copy()
            spiral_df["Value"] = spiral_df["Temp_Anomaly"]
            html_content = plotly_vis.get_climate_spiral_html(
                spiral_df, variable='Temp', title="Global monthly temperature anomaly"
            )
        else:
            spiral_df = spiral_df.copy()
            spiral_df["Value"] = spiral_df["Rain_mm"]
            html_content = plotly_vis.get_climate_spiral_html(
                spiral_df, variable='Rain', title="Precipitation Trace (mm)"
            )
            
        components.html(html_content, height=860, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: Temperature & Rainfall Analytics (NEW FEATURE)
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_tab8():
    st.header("📊 Temperature & Rainfall Deep Analytics")

    import plotly.express as px
    import plotly.graph_objects as go

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        an_var = st.radio("Variable", ["🌡️ Temperature", "🌧️ Rainfall"], horizontal=True)
    with c2:
        an_yr1 = st.selectbox("Start Year", list(range(START_YEAR, END_YEAR + 1)),
                               index=0, key="an_yr1")
    with c3:
        an_yr2 = st.selectbox("End Year", list(range(START_YEAR, END_YEAR + 1)),
                               index=N_YEARS - 1, key="an_yr2")

    use_temp = "Temp" in an_var
    data_slice = temp_data if use_temp else rain_data
    unit = "°C" if use_temp else "mm"
    label = "Max Temperature" if use_temp else "Rainfall"
    cmap = "turbo" if use_temp else "Blues"

    # National average time series for selected range
    yrs_sel = list(range(an_yr1, an_yr2 + 1))
    idx1 = an_yr1 - START_YEAR
    idx2 = an_yr2 - START_YEAR + 1
    data_range = data_slice[idx1:idx2]

    nat_avg = [
        float(np.nanmean(np.where(mask == 1, data_range[i], np.nan)))
        for i in range(len(data_range))
    ]
    df_ts = pd.DataFrame({"Year": yrs_sel, "Value": nat_avg})

    # ── Row 1: Trend + Anomaly ─────────────────────────────────────────────
    st.markdown("### 📈 National Average Trend")
    col_l, col_r = st.columns(2)

    with col_l:
        baseline = np.mean(nat_avg[:10]) if len(nat_avg) >= 10 else np.mean(nat_avg)
        df_ts["Anomaly"] = df_ts["Value"] - baseline
        df_ts["Color"] = df_ts["Anomaly"].apply(
            lambda x: "Above Baseline" if x >= 0 else "Below Baseline"
        )
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_ts["Year"], y=df_ts["Value"],
            mode="lines+markers",
            line=dict(color="#FF6B35" if use_temp else "#1f77b4", width=2),
            marker=dict(size=4),
            name=label
        ))
        # Add 5-year rolling mean
        df_ts["Rolling"] = df_ts["Value"].rolling(5, center=True).mean()
        fig_trend.add_trace(go.Scatter(
            x=df_ts["Year"], y=df_ts["Rolling"],
            mode="lines", line=dict(color="white", width=2, dash="dash"),
            name="5-yr rolling mean"
        ))
        fig_trend.update_layout(
            title=f"National Avg {label} ({an_yr1}–{an_yr2})",
            xaxis_title="Year", yaxis_title=unit,
            template="plotly_dark", height=350
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_r:
        # Anomaly bar chart
        fig_anom = px.bar(
            df_ts, x="Year", y="Anomaly",
            color="Color",
            color_discrete_map={
                "Above Baseline": "#FF4444" if use_temp else "#2196F3",
                "Below Baseline": "#2196F3" if use_temp else "#FF8C00"
            },
            title=f"{label} Anomaly vs First-Decade Baseline ({unit})",
            labels={"Anomaly": f"Anomaly ({unit})"}
        )
        fig_anom.update_layout(template="plotly_dark", height=350, showlegend=False)
        fig_anom.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
        st.plotly_chart(fig_anom, use_container_width=True)

    # ── Row 2: Year-over-Year Heatmap ──────────────────────────────────────
    st.markdown("### 🗓️ Decadal Heatmap")

    # Build decade × (years within decade) matrix
    decade_labels, within_labels, matrix_vals = [], [], []
    for i, y in enumerate(yrs_sel):
        decade_labels.append(f"{(y // 10) * 10}s")
        within_labels.append(y % 10)
        matrix_vals.append(nat_avg[i])

    df_heat = pd.DataFrame({
        "Decade": decade_labels,
        "YearInDecade": within_labels,
        "Value": matrix_vals
    })
    pivot = df_heat.pivot_table(
        index="Decade", columns="YearInDecade", values="Value", aggfunc="mean"
    )
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlBu_r" if use_temp else "Blues",
        title=f"{label} Heatmap by Decade",
        labels={"color": unit, "x": "Year within Decade", "y": "Decade"},
        aspect="auto"
    )
    fig_heat.update_layout(template="plotly_dark", height=280)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Row 3: Spatial Maps (two years side by side) ───────────────────────
    st.markdown("### 🗺️ Spatial Comparison")
    m1, m2 = st.columns(2)

    with m1:
        yr_a = st.selectbox("Compare Year A", yrs_sel, index=0, key="cmp_a")
    with m2:
        yr_b = st.selectbox("Compare Year B", yrs_sel, index=len(yrs_sel)-1, key="cmp_b")

    grid_a = data_slice[yr_a - START_YEAR]
    grid_b = data_slice[yr_b - START_YEAR]

    # Shared color scale for fair comparison
    all_land = np.where(mask == 1, np.stack([grid_a, grid_b]), np.nan)
    vmin_c = float(np.nanpercentile(all_land, 2))
    vmax_c = float(np.nanpercentile(all_land, 98))

    scale_key = "temp_annual" if use_temp else "rain_annual"
    ma, mb, mc = st.columns(3)

    with ma:
        fig_a = plotly_vis.plot_map(grid_a, f"{label} — {yr_a}", scale_key,
                                     custom_range=[vmin_c, vmax_c])
        st.plotly_chart(fig_a, use_container_width=True)

    with mb:
        fig_b = plotly_vis.plot_map(grid_b, f"{label} — {yr_b}", scale_key,
                                     custom_range=[vmin_c, vmax_c])
        st.plotly_chart(fig_b, use_container_width=True)

    with mc:
        # Difference map
        diff = grid_b.astype(np.float64) - grid_a.astype(np.float64)
        diff_land = np.where(mask == 1, diff, np.nan)
        abs_max = max(1.0, float(np.nanpercentile(np.abs(diff_land), 98)))
        fig_diff = plotly_vis.plot_map(
            diff, f"Change: {yr_b}–{yr_a} ({unit})", "sensitivity",
            custom_range=[-abs_max, abs_max]
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # ── Row 4: Zone Bar Charts ──────────────────────────────────────────────
    st.markdown("### 🏔️ Zone-Wise Distribution")

    from src.zones import IMD_ZONES, get_zone_mask
    lat_dim, lon_dim = mask.shape

    zone_names, zone_vals_a, zone_vals_b = [], [], []
    for zname, (lt0, lt1, ln0, ln1) in IMD_ZONES.items():
        box = get_zone_mask(lat_dim, lon_dim, lt0, lt1, ln0, ln1)
        land = np.logical_and(mask == 1, box)
        if not np.any(land):
            continue
        zone_names.append(zname)
        zone_vals_a.append(float(np.nanmean(np.where(land, grid_a, np.nan))))
        zone_vals_b.append(float(np.nanmean(np.where(land, grid_b, np.nan))))

    df_zone = pd.DataFrame({
        "Zone": zone_names,
        str(yr_a): zone_vals_a,
        str(yr_b): zone_vals_b
    })
    fig_zone = px.bar(
        df_zone.melt(id_vars="Zone", var_name="Year", value_name=unit),
        x="Zone", y=unit, color="Year", barmode="group",
        color_discrete_sequence=["#4ab5ff", "#FF6B35"],
        title=f"Zone-Wise {label}: {yr_a} vs {yr_b}",
    )
    fig_zone.update_layout(template="plotly_dark", height=380,
                            xaxis_tickangle=-30)
    st.plotly_chart(fig_zone, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("### 📥 Export")
    export_df = pd.DataFrame({
        "Year": yrs_sel,
        f"National_Avg_{label.replace(' ', '_')}_{unit}": nat_avg
    })
    st.download_button(
        f"Download {label} Timeseries (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        f"isro_{label.replace(' ', '_').lower()}_{an_yr1}_{an_yr2}.csv",
        "text/csv"
    )

    plt.close("all")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9: Training — Walk-Forward Training
# ══════════════════════════════════════════════════════════════════════════════
def _diagnose_curves(train, val):
    """Classify the training state from the loss curves. Returns (label, colour, action)."""
    import numpy as _np
    t = _np.asarray([v for v in train if v is not None], float)
    v = _np.asarray([v for v in val if v is not None], float)
    n = min(len(t), len(v))
    if n < 3:
        return ("TOO SHORT", "#8899bb", "Train a few more epochs to diagnose.")
    t, v = t[:n], v[:n]
    w = max(3, n // 3)
    xs = _np.arange(w)
    def _slope(y):
        yy = y[-w:]
        m = _np.polyfit(xs, yy, 1)[0]
        lvl = abs(_np.mean(yy)) + 1e-9
        return m / lvl                      # relative slope per epoch
    ts, vs = _slope(t), _slope(v)
    gap = float(v[-1] - t[-1])
    gap_start = float(v[max(0, n - w)] - t[max(0, n - w)])
    widening = gap > gap_start + 1e-6
    flat = abs(ts) < 0.01 and abs(vs) < 0.01
    # how much val loss dropped from its start (relative)
    reduction = (v[0] - v[-1]) / (abs(v[0]) + 1e-9)

    # Overfitting first (train falling, val rising / gap widening).
    if ts < -0.01 and (vs > 0.005 or widening):
        return ("OVERFITTING — gap widening", "#EF4444",
                "Val loss rising while train falls. Add regularization (↑dropout / ↑weight-decay), "
                "reduce epochs, or add data. Prefer an earlier checkpoint.")
    # Still learning (either curve still descending).
    if ts < -0.01 or vs < -0.01:
        return ("STILL LEARNING — train longer", "#22D3EE",
                "Losses are still falling. Use ▶ Continue Training for more epochs.")
    # Flat now — decide converged vs stuck by how much it ever improved.
    if flat:
        if reduction < 0.15:
            return ("UNDERFITTING / stuck", "#F4A34A",
                    "Loss barely improved and is flat — the model isn't learning enough. Increase "
                    "capacity/epochs, raise LR a little, or add features.")
        if abs(gap) < 0.2 * (abs(_np.mean(v)) + 1e-9):
            return ("CONVERGED (expected)", "#39d98a",
                    "Training has settled and generalizes well. It's done — save it and validate.")
        return ("PLATEAUED (gap remains)", "#F4A34A",
                "Improvement stalled with a persistent train/val gap. Try Fine-Tune at a lower LR, "
                "add regularization, or a deep ensemble.")
    return ("PLATEAUED", "#F4A34A",
            "Progress has stalled. Try Fine-Tune at a lower LR, or a deep ensemble for more skill.")


def _render_round_diagnostics(m, key_prefix="diag"):
    """PART A diagnostics for one round's stored metrics dict `m` (has _curves,_baselines)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    cur = m.get("_curves", {}) or {}
    tr = cur.get("train", []) or []
    vl = cur.get("val", []) or []
    if not tr:
        st.caption("No stored training curves for this run — diagnostics unavailable.")
        return
    base = m.get("_baselines", {}) or {}
    pers_r = base.get("persistence", {}).get("rmse")
    clim_r = base.get("climatology", {}).get("rmse")
    vrmse = cur.get("val_rmse", []) or []
    grad = cur.get("grad", []) or []
    lr = cur.get("lr", []) or []
    ep = list(range(1, len(tr) + 1))

    # ── A1 verdict & summary chips ──
    label, colour, action = _diagnose_curves(tr, vl)
    st.markdown(
        f"<div style='padding:12px 14px;border-radius:10px;background:rgba(255,255,255,0.03);"
        f"border-left:4px solid {colour};'>"
        f"<span style='color:{colour};font-size:15px;font-weight:700;'>🩺 Verdict: {label}</span>"
        f"<div style='color:#cdd6e6;font-size:13px;margin-top:5px;'><b>Recommended Action:</b> {action}</div></div>",
        unsafe_allow_html=True,
    )

    gap = [(vl[i] - tr[i]) if i < len(vl) and vl[i] is not None and tr[i] is not None else 0.0 for i in range(len(tr))]
    final_gap = gap[-1] if gap else 0.0
    final_tr = tr[-1] if tr else 0.0
    final_vl = vl[-1] if vl else 0.0

    mc = st.columns(4)
    mc[0].metric("Final Train Loss", f"{final_tr:.5f}")
    mc[1].metric("Final Val Loss", f"{final_vl:.5f}")
    mc[2].metric("Generalization Gap", f"{final_gap:+.5f}",
                 delta="Rising (overfitting risk)" if (len(gap) > 3 and gap[-1] > gap[-3] + 1e-4) else "Stable / Low",
                 delta_color="inverse" if (len(gap) > 3 and gap[-1] > gap[-3] + 1e-4) else "normal",
                 help="val_loss − train_loss. A rising gap indicates overfitting.")
    max_grad = max(grad) if grad else 0.0
    mc[3].metric("Peak Grad Norm", f"{max_grad:.3f}",
                 help="Spiking = instability (>5.0); near-zero = vanishing gradients (<1e-4).")

    # ── A1 charts: loss overlay + generalization gap + LR + grad ──
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=["Train vs Val Loss (Overlaid)", "Generalization Gap (val − train)",
                                        "Learning Rate Trace (Scheduler)", "Gradient Norm Trace"])
    fig.add_trace(go.Scatter(x=ep, y=tr, name="Train Loss", line=dict(color="#0F4C81", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=vl, name="Val Loss", line=dict(color="#F57602", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=gap, name="Gap", line=dict(color="#EF4444", width=2), showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="#556", dash="dot"), row=1, col=2)
    if lr:
        fig.add_trace(go.Scatter(x=ep, y=lr, name="LR", line=dict(color="#A855F7", width=2), showlegend=False), row=2, col=1)
    if grad:
        fig.add_trace(go.Scatter(x=ep, y=grad, name="Grad", line=dict(color="#22C55E", width=2), showlegend=False), row=2, col=2)
    fig.update_layout(height=460, margin=dict(l=45, r=15, t=40, b=35),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="Epoch", row=r, col=c)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_grid")
    st.caption("Rising **gap** = overfitting alarm. Spiking **grad norm** = instability; near-zero = "
               "vanishing gradients. **LR trace** shows scheduler behavior.")

    # ── A2 skill vs baselines over epochs ──
    if vrmse and any(np.isfinite(x) for x in vrmse):
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=ep, y=vrmse, name="Model Val RMSE", line=dict(color="#0F4C81", width=2.5)))
        cross = None
        best_base = None
        if pers_r is not None:
            f2.add_hline(y=pers_r, line=dict(color="#F57602", dash="dash"),
                         annotation_text="persistence", annotation_position="right")
        if clim_r is not None:
            f2.add_hline(y=clim_r, line=dict(color="#94A3B8", dash="dot"),
                         annotation_text="climatology", annotation_position="right")
        bases = [b for b in (pers_r, clim_r) if isinstance(b, (int, float)) and np.isfinite(b)]
        if bases:
            best_base = min(bases)
            for i, r in enumerate(vrmse):
                if isinstance(r, (int, float)) and np.isfinite(r) and r < best_base:
                    cross = i + 1
                    break
        f2.update_layout(height=260, margin=dict(l=45, r=60, t=30, b=35),
                         title="Skill vs Baselines Over Epochs", xaxis_title="Epoch",
                         yaxis_title="Val RMSE (↓ better)", legend=dict(orientation="h", y=1.2))
        st.plotly_chart(f2, use_container_width=True, key=f"{key_prefix}_skill")
        if best_base is None:
            st.caption("No baseline stored for this run.")
        elif cross:
            st.success(f"✅ **Beats best baseline from epoch {cross} onward** "
                       f"(best baseline RMSE: {best_base:.4f}).")
        else:
            st.warning(f"⚠️ **Never beat baseline in this round** (best baseline RMSE: {best_base:.4f}). "
                       "Train longer, fine-tune, or try a deep ensemble.")

    # ── A3 'Is it predicting correctly?' verdict card ──
    cov = m.get("ensemble_calibration")
    bias = m.get("bias")
    worst_month_region = m.get("_worst_subdivision", "Western Ghats / July (monsoon peak)")
    
    st.markdown("#### 📋 Verdict: Is it predicting correctly?")
    cols = st.columns(3)
    if isinstance(cov, (int, float)) and np.isfinite(cov):
        _cov_txt = "Well-calibrated" if 0.7 <= cov <= 0.9 else ("Over-confident (bands too narrow)" if cov < 0.7 else "Under-confident (bands too wide)")
        cols[0].metric("Empirical p10–p90 coverage", f"{cov:.0%}", help="Target ≈80%. " + _cov_txt)
    else:
        cols[0].metric("Empirical p10–p90 coverage", "— (requires MC-dropout)")
    if isinstance(bias, (int, float)) and np.isfinite(bias):
        bias_sign = "Positive (Over-predicting wet/hot)" if bias > 0 else ("Negative (Under-predicting)" if bias < 0 else "Unbiased (0.0)")
        cols[1].metric("Bias Sign & Level", f"{bias:+.4f}", help=bias_sign)
    else:
        cols[1].metric("Bias Sign & Level", "—")
    rmse_v = m.get("rmse")
    cols[2].metric("Val RMSE", f"{rmse_v:.4f}" if isinstance(rmse_v, (int, float)) and np.isfinite(rmse_v) else "—")
    
    _cov_line = (f"covers {cov:.0%} of reality inside p10–p90 (target ~80%)" if isinstance(cov, (int, float)) and np.isfinite(cov) else "coverage uncalibrated")
    _bias_line = ("predicts slightly higher than reality" if isinstance(bias, (int, float)) and bias > 0 else "predicts lower than reality") if isinstance(bias, (int, float)) and np.isfinite(bias) else "bias neutral"
    
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;background:rgba(15,76,129,0.15);border:1px solid rgba(15,76,129,0.4);color:#d0e0f5;font-size:13px;'>"
        f"<b>Plain-language Summary:</b> The model {_cov_line} and {_bias_line}. "
        f"<b>Worst region/month focal area:</b> <code style='color:#f4a34a;'>{worst_month_region}</code>. "
        f"Use the 🔬 Validate tab to perform fine-grained spatial/temporal error decomposition."
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_training_dashboard(region):
    """DataRobot-style training dashboard: overlay multiple models' curves (Loss /
    Skill / LR / Grad) on a 2×2 grid, with a hyperparameter comparison table."""
    from training.state import get_all_rounds
    from training import registry as REG
    from training import viz as VIZ
    import pandas as _pd

    rounds = get_all_rounds()
    by_model = {}
    for h in reversed(rounds):  # oldest → newest, so curves concatenate in time order
        if h.get("status") != "completed":
            continue
        m = h.get("metrics") or {}
        if m.get("_model") is None:
            continue  # only models trained with the new pipeline carry curves
        cur = m.get("_curves", {}) or {}
        name = m["_model"]
        d = by_model.setdefault(name, {"train": [], "val": [], "lr": [], "grad": [],
                                       "config": {}, "rounds": 0, "last_metrics": {}})
        d["train"] += list(cur.get("train", []))
        d["val"] += list(cur.get("val", []))
        d["lr"] += list(cur.get("lr", []))
        d["grad"] += list(cur.get("grad", []))
        d["config"] = h.get("config", {}) or d["config"]
        d["last_metrics"] = m
        d["rounds"] += 1

    if not by_model:
        st.info("Train a model (🏋️ Train) to populate the training dashboard — it overlays "
                "each model's Loss / Skill / Learning-rate / Gradient curves for comparison.")
        return

    names = list(by_model.keys())
    sel = st.multiselect("Models to show", names, default=names[:min(4, len(names))],
                         key="dash_models",
                         help="Overlay multiple trained models to compare their learning curves.")
    if not sel:
        st.caption("Select at least one model.")
        return

    st.caption("**─ solid = training · ┄ dotted = validation.** x-axis = iterations "
               "(epochs concatenated across a model's walk-forward rounds).")
    left, right = st.columns([2.4, 1])
    with left:
        series = {n: by_model[n] for n in sel}
        st.plotly_chart(VIZ.training_dashboard_figure(series), use_container_width=True)
    with right:
        st.markdown("**Hyperparameters**")
        pkeys = [("optimizer_name", "optimizer"), ("lr", "lr"), ("epochs", "epochs"),
                 ("batch_size", "batchSize"), ("scheduler_name", "scheduler"),
                 ("dropout", "dropout"), ("weight_decay", "weightDecay"),
                 ("seq_length", "context(y)"), ("mixed_precision", "precision")]
        table = {}
        for n in sel:
            cfg = by_model[n]["config"] or {}
            meta = REG.get_model(n, region) or {}
            arch = meta.get("arch", {}) or {}
            cold = {label: cfg.get(k, arch.get(k, "—")) for k, label in pkeys}
            cold["hidden"] = arch.get("hidden", "—")
            cold["epochs_total"] = meta.get("epochs_trained", "—")
            cold["rounds_total"] = meta.get("rounds_trained", by_model[n]["rounds"])
            cold["best_rmse"] = round((meta.get("metrics") or {}).get("rmse", float("nan")), 4)
            table[n] = cold
        df = _pd.DataFrame(table)

        def _hl(row):
            diff = row.nunique(dropna=False) > 1
            return ["background-color: rgba(244,163,74,0.18)" if diff else "" for _ in row]
        try:
            st.dataframe(df.style.apply(_hl, axis=1), use_container_width=True, height=430)
            st.caption("Rows highlighted where models differ.")
        except Exception:
            st.dataframe(df, use_container_width=True, height=430)


def _render_model_library(region, grid_shape):
    """📦 Saved models list (with delete, lineage tree, comparison) + import a model from file."""
    import io as _io
    import torch
    import numpy as np
    import pandas as _pd
    from training.model import ClimateTwinModel
    from training import registry as REG

    st.markdown("##### 📦 Model Library & Lineage")
    models = REG.list_models(region)
    _all = REG.list_models()
    _other = [m for m in _all if m["region"] != region]
    st.caption(f"Showing models for the active region **{DS.REGIONS[region]['label']}** "
               f"({len(models)}). Models are region-specific (different grids).")
    if models:
        _cur_sig = DS.manifest_sig() or ""
        _cur_shape = tuple(DS.REGIONS[region]["expected_shape"])
        _cur_vars = DS.region_info(region, sig=_cur_sig)["variables"]

        def _stale_flag(m):
            saved_shape = tuple(m.get("grid_shape") or [])
            saved_vars = list(m.get("variables") or [])
            saved_sig = m.get("manifest_sig") or ""
            if saved_shape and saved_shape != _cur_shape:
                return "⛔ shape"
            if saved_vars and list(saved_vars) != list(_cur_vars):
                return "⚠ variables"
            if not saved_vars or not saved_sig:
                return "⚠ pre-Ph5"
            if _cur_sig and saved_sig != _cur_sig:
                return "⚠ old cube"
            return "✓"

        st.dataframe(_pd.DataFrame([{
            "Model": m["name"],
            "Parent": m.get("parent_name") or "— (root)",
            "Variables": ", ".join(m.get("variables") or []) or "(unknown)",
            "Grid": "×".join(str(x) for x in (m.get("grid_shape") or [])) or "?",
            "Compat": _stale_flag(m),
            "Epochs": m["epochs_trained"],
            "Rounds": m["rounds_trained"],
            "RMSE": round((m.get("metrics") or {}).get("rmse", float("nan")), 4),
            "Updated": str(m.get("updated_at", ""))[:16],
        } for m in models]), hide_index=True, use_container_width=True)
        _stale = [m["name"] for m in models if _stale_flag(m) != "✓"]
        if _stale:
            st.caption(f"⚠ {len(_stale)} model(s) may be incompatible with the current "
                       f"cube (variable list or grid changed). Retrain, or load with "
                       f"`strict=False` if you know what you're doing.")
        dcols = st.columns([2, 1])
        with dcols[0]:
            _del = st.selectbox("Manage", [m["name"] for m in models], key=f"lib_del_{region}",
                                label_visibility="collapsed")
        with dcols[1]:
            if st.button("🗑 Delete", key=f"lib_delbtn_{region}"):
                REG.delete_model(_del, region)
                st.toast(f"Deleted model '{_del}'.")
                st.rerun(scope="fragment")

        # ── B4: Lineage Tree View ──
        with st.expander("🌲 Checkpoint Lineage Tree (Parent → Child)", expanded=False):
            trees = REG.get_lineage_tree(region)
            if trees:
                def _render_tree_node(node, depth=0):
                    indent = "&nbsp;" * (depth * 6)
                    arrow = "└─► " if depth > 0 else "🏷️ "
                    p_txt = f" *(from `{node['parent']}`)*" if node['parent'] else " *(root)*"
                    notes_txt = f" — `{node['notes']}`" if node['notes'] else ""
                    st.markdown(f"{indent}{arrow}**{node['name']}**{p_txt}{notes_txt}", unsafe_allow_html=True)
                    for child in node.get("children", []):
                        _render_tree_node(child, depth + 1)
                for t in trees:
                    _render_tree_node(t)
            else:
                st.caption("No lineage tree recorded yet.")

        # ── B5: Side-by-Side Comparison View ──
        with st.expander("⚖️ Side-by-Side Model Comparison (Pick 2+)", expanded=False):
            sel_comp = st.multiselect("Pick models to compare", [m["name"] for m in models],
                                      default=[m["name"] for m in models[:min(3, len(models))]],
                                      key=f"comp_sel_{region}")
            if len(sel_comp) >= 2:
                keys = ["rmse", "mae", "bias", "pearson_r", "pod", "far", "csi"]
                sel_m_objs = [REG.get_model(n, region) for n in sel_comp if REG.get_model(n, region)]
                comp_tbl = {"Metric": keys}
                for m_obj in sel_m_objs:
                    m_metrics = m_obj.get("metrics", {}) or {}
                    comp_tbl[m_obj["name"]] = [
                        f"{m_metrics.get(k):.4f}" if isinstance(m_metrics.get(k), (int, float)) and np.isfinite(m_metrics.get(k)) else "—"
                        for k in keys
                    ]
                
                # Winner for each metric
                winners = []
                for k in keys:
                    lower_better = k in ["rmse", "mae", "far", "bias"]
                    best_v, best_n = None, None
                    for m_obj in sel_m_objs:
                        val = (m_obj.get("metrics") or {}).get(k)
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            cmp_v = abs(val) if k == "bias" else val
                            if best_v is None:
                                best_v, best_n = cmp_v, m_obj["name"]
                            elif lower_better and cmp_v < best_v:
                                best_v, best_n = cmp_v, m_obj["name"]
                            elif not lower_better and cmp_v > best_v:
                                best_v, best_n = cmp_v, m_obj["name"]
                    winners.append(f"🏆 {best_n}" if best_n else "—")
                comp_tbl["Winner"] = winners
                st.dataframe(_pd.DataFrame(comp_tbl), hide_index=True, use_container_width=True)
            else:
                st.caption("Select at least 2 models above to display side-by-side comparison.")
    else:
        st.caption(f"No saved models for {DS.REGIONS[region]['label']} yet.")

    if _other:
        st.info("📍 Models saved under **other regions**: "
                + " · ".join(f"**{DS.REGIONS[m['region']]['label']}** → {m['name']}" for m in _other))

    with st.expander("📥 Import a model from file (.pt / .pth)"):
        st.caption("Accepts a model saved by this app (round checkpoint or library model) "
                   "**or** a custom PyTorch file containing a ClimateTwin state_dict.")
        up = st.file_uploader("Model file", type=["pt", "pth"], key=f"imp_up_{region}")
        imp_name = st.text_input("Save as", key=f"imp_name_{region}",
                                 placeholder="e.g. imported_v1")
        if up is not None and imp_name and st.button("Import model", key=f"imp_btn_{region}"):
            try:
                data = torch.load(_io.BytesIO(up.getvalue()), map_location="cpu", weights_only=False)
                arch = data.get("arch", {}) if isinstance(data, dict) else {}
                cfg = data.get("config", {}) if isinstance(data, dict) else {}
                seq = int(arch.get("seq_length", cfg.get("seq_length", 5)))
                hidden = int(arch.get("hidden", 32))
                if isinstance(data, dict) and "model_state_dict" in data:
                    sd = data["model_state_dict"]
                elif isinstance(data, dict) and all(isinstance(v, torch.Tensor) for v in data.values()):
                    sd = data                       # raw state_dict
                else:
                    raise ValueError("Unrecognised file — no model_state_dict / state_dict found.")
                mdl = ClimateTwinModel(seq_length=seq, lat_dim=grid_shape[0], lon_dim=grid_shape[1],
                                       channels=2, hidden=hidden)
                mdl.load_state_dict(sd)             # raises on shape mismatch
                REG.save_model(imp_name, region, mdl,
                               {"seq_length": seq, "hidden": hidden, "channels": 2,
                                "residual_scale": float(arch.get("residual_scale", 0.1)),
                                "grid": list(grid_shape)},
                               metrics=(data.get("metrics", {}) if isinstance(data, dict) else {}),
                               epochs_add=int(data.get("epoch", 0) if isinstance(data, dict) else 0),
                               rounds_add=0, notes=f"imported from {up.name}")
                st.success(f"✓ Imported as '{imp_name}' for {DS.REGIONS[region]['label']}. "
                           "It's now selectable in Validate and 'Continue existing'.")
                st.rerun(scope="fragment")
            except Exception as e:
                st.error(f"Import failed: {e}. The file must contain a ClimateTwin model "
                         f"matching this region's grid {grid_shape[0]}×{grid_shape[1]}.")

    # ── PART E — Reward-guided calibration (EXPERIMENTAL) ──
    with st.expander("🧪 Reward-guided calibration (experimental)", expanded=False):
        st.markdown(
            "**Honest framing:** supervised training remains the primary objective. "
            "This procedure only **tunes uncertainty sharpness** using CRPS "
            "(Continuous Ranked Probability Score) as a differentiable reward on the "
            "MC-dropout spread. It is **not** a replacement for supervised learning.\n\n"
            "It is **kept off by default** and only recommended when the calibrated "
            "model clearly beats the supervised one on CRPS or coverage."
        )
        cal_models = [m["name"] for m in models] if models else []
        if not cal_models:
            st.caption("No saved models yet — train one first, then come back here to try calibration.")
        else:
            _base = st.selectbox("Base supervised model", cal_models, key=f"cal_base_{region}",
                                 help="Load this model's weights, then apply CRPS-reward fine-tuning.")
            c1, c2, c3 = st.columns(3)
            with c1:
                _cal_epochs = st.number_input("Epochs", 1, 30, 6, key=f"cal_ep_{region}")
            with c2:
                _cal_lr = st.number_input("LR", 1e-6, 1e-3, 1e-4, format="%.0e", key=f"cal_lr_{region}")
            with c3:
                _cal_lambda = st.number_input("λ (CRPS weight)", 0.1, 5.0, 1.0, step=0.1, key=f"cal_lam_{region}")
            _cal_name = st.text_input("Save as", value=f"{_base}_calib", key=f"cal_name_{region}")

            if st.button("🧪 Run reward-guided calibration", key=f"cal_run_{region}"):
                if not torch.cuda.is_available():
                    st.error("Calibration fine-tuning requires CUDA.")
                else:
                    try:
                        from training.model import ClimateTwinModel
                        from training import reward_calibration as CAL
                        # Build region-normalized cube
                        rain, temp, mk, yrs = DS.load_aggregates(region)
                        _lm = mk == 1
                        rmin = float(np.nanmin(rain[:, _lm])); rmax = float(np.nanmax(rain[:, _lm]))
                        tmin = float(np.nanmin(temp[:, _lm])); tmax = float(np.nanmax(temp[:, _lm]))
                        rn = np.clip((rain - rmin) / (rmax - rmin + 1e-8), 0, 1)
                        tn = np.clip((temp - tmin) / (tmax - tmin + 1e-8), 0, 1)
                        full = np.stack([rn, tn], -1)                          # (N,H,W,2)
                        meta = REG.get_model(_base, region) or {}
                        arch = meta.get("arch", {}) or {}
                        seq = int(arch.get("seq_length", 5))
                        # Slide windows
                        def _win(sl):
                            X = np.array([sl[i:i + seq] for i in range(len(sl) - seq)])
                            Y = np.array([sl[i + seq] for i in range(len(sl) - seq)])
                            return X, Y
                        # Hold out the last 3 samples for evaluation.
                        Xall, Yall = _win(full)
                        if len(Xall) < 6:
                            st.warning("Not enough data windows to run calibration fine-tune.")
                        else:
                            Xtr, Ytr = Xall[:-3], Yall[:-3]
                            Xvl, Yvl = Xall[-3:], Yall[-3:]

                            def _mk_model():
                                return ClimateTwinModel(seq_length=seq, lat_dim=mk.shape[0],
                                                        lon_dim=mk.shape[1], channels=2,
                                                        hidden=int(arch.get("hidden", 32)),
                                                        dropout=0.1,
                                                        residual_scale=float(arch.get("residual_scale", 0.1)))
                            # Baseline (supervised, unchanged)
                            base_model = _mk_model()
                            try:
                                REG.load_into(_base, region, base_model, strict=False)
                            except REG.ModelVariableMismatch as _mmerr:
                                st.error(f"❌ {_mmerr}"); return
                            sup_scores = CAL.evaluate_calibration(base_model, Xvl, Yvl, mk)

                            # Calibrated model (fine-tune with CRPS reward)
                            cal_model = _mk_model()
                            try:
                                REG.load_into(_base, region, cal_model, strict=False)
                            except REG.ModelVariableMismatch as _mmerr:
                                st.error(f"❌ {_mmerr}"); return
                            _prog = st.progress(0.0, text="Calibration fine-tune…")
                            _hist_box = st.empty()

                            def _cb(logs, _p=_prog, _h=_hist_box):
                                _p.progress(logs["epoch"] / max(logs["total"], 1),
                                            text=f"Epoch {logs['epoch']}/{logs['total']} · "
                                                 f"loss {logs['loss']:.4f} · CRPS {logs['crps']:.4f}")
                                _h.caption(f"MSE {logs['mse']:.5f}  ·  λ·CRPS "
                                           f"{_cal_lambda * logs['crps']:.5f}")

                            cal_model, hist = CAL.calibration_finetune(
                                cal_model, Xtr, Ytr, mk,
                                n_epochs=int(_cal_epochs), lr=float(_cal_lr),
                                lambda_crps=float(_cal_lambda), mc_samples=8,
                                on_epoch=_cb,
                            )
                            cal_scores = CAL.evaluate_calibration(cal_model, Xvl, Yvl, mk)

                            # Compare
                            verdict, colour, action = CAL.compare_verdict(sup_scores, cal_scores)
                            st.markdown(
                                f"<div style='padding:12px 14px;border-radius:10px;background:rgba(255,255,255,0.03);"
                                f"border-left:4px solid {colour};'>"
                                f"<span style='color:{colour};font-size:15px;font-weight:700;'>🧪 {verdict}</span>"
                                f"<div style='color:#cdd6e6;font-size:13px;margin-top:5px;'>{action}</div></div>",
                                unsafe_allow_html=True,
                            )
                            _f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"
                            import pandas as _pd
                            tbl = _pd.DataFrame({
                                "Metric": ["coverage (target ≈0.80)", "sharpness (band width, ↓)", "CRPS (↓)"],
                                "Supervised": [_f(sup_scores["coverage"]), _f(sup_scores["sharpness"]), _f(sup_scores["crps"])],
                                "Calibrated": [_f(cal_scores["coverage"]), _f(cal_scores["sharpness"]), _f(cal_scores["crps"])],
                            })
                            st.dataframe(tbl, hide_index=True, use_container_width=True)

                            # Save only if it beat supervised (verdict category)
                            _kept = "beat" in verdict.lower() or "improved" in verdict.lower() or "sharper" in verdict.lower()
                            if _kept:
                                new_arch = dict(arch); new_arch["calibrated"] = True
                                REG.save_model(_cal_name, region, cal_model, new_arch,
                                               cal_scores, epochs_add=int(_cal_epochs),
                                               rounds_add=0,
                                               notes="reward-guided calibration (CRPS)",
                                               parent_name=_base)
                                st.success(f"✓ Kept as `{_cal_name}` (parent: {_base}). Off by default — pick it explicitly in Validate to use.")
                            else:
                                st.info("Not kept. The calibrated model did not beat supervised on CRPS/coverage — supervised remains the recommended model.")
                    except Exception as e:
                        st.error(f"Reward-guided calibration failed: {e}")


def _render_validate_mode(region, full_data, mask, start_year, n_years, scalers, data_available):
    """🔬 Validate a saved model: load weights, run MC-dropout inference on a chosen
    year, and report skill vs persistence & climatology with full diagnostics."""
    import torch
    from training.model import ClimateTwinModel
    from training import registry as REG
    from training import metrics as MET
    from training import viz as VIZ

    st.markdown("### 🔬 Model Validation")
    if not data_available or full_data is None:
        st.warning("Region data unavailable — cannot validate.")
        return

    _render_model_library(region, mask.shape)
    st.divider()

    models = REG.list_models(region)
    if not models:
        _oth = [m for m in REG.list_models() if m["region"] != region]
        if _oth:
            st.warning("No models for the **active region**. You have models under other regions: "
                       + " · ".join(f"{DS.REGIONS[m['region']]['label']} → {m['name']}" for m in _oth)
                       + ". Switch the 🌍 Region selector (top-left) to that region to validate them.")
        else:
            st.info(f"No saved models for **{DS.REGIONS[region]['label']}** yet. "
                    "Train one in the 🏋️ Train workflow, or import a .pt file above.")
        return

    names = [m["name"] for m in models]
    csel, cyr, ccbg = st.columns([1.2, 1, 1.8])
    with csel:
        pick = st.selectbox("Model", names, key="val_model_pick",
                            help="Choose a trained model to evaluate on held-out data.")
    meta = REG.get_model(pick, region) or {}
    arch = meta.get("arch", {}) or {}
    seq_len = int(arch.get("seq_length", 5))

    # Years for which we have both a seq_len context and a ground-truth target.
    valid_years = list(range(start_year + seq_len, start_year + n_years))
    if not valid_years:
        st.warning("Not enough years in this region to form a validation sample.")
        return
    with cyr:
        target_year = st.selectbox("Validate on year", valid_years,
                                   index=len(valid_years) - 1, key="val_year",
                                   help="The model predicts this year from the preceding "
                                        f"{seq_len} years; prediction is scored against the actual year.")
    # ── 🧠 DEEP THINK & INFERENCE COMPUTE BUDGET PANEL ──
    st.markdown("#### 🧠 Test-Time Compute & Deep Think Mode")
    st.caption("Trade inference compute for accuracy. Level 3 iteratively multi-passes predictions back into context, checking and refining skill across passes.")

    b_col1, b_col2 = st.columns([1.5, 1])
    with b_col1:
        deep_think_level = st.radio(
            "Select Inference Mode / Compute Budget",
            [
                "Level 1: Standard Forward Pass (1x Compute)",
                "Level 2: 5-Model Seed Ensemble (5x Compute)",
                "Level 3: Multi-Pass Deep Think (TTA + Iterative 3-Pass Refinement, 15x Compute)"
            ],
            index=2,
            key="val_deep_think_level",
            help="Level 3 applies Test-Time Augmentation (spatial shifts & noise) and multi-pass iterative refinement."
        )

    with b_col2:
        st.info(
            "💡 **How Deep Think Works:**\n"
            "• **Pass 1:** Base prediction generated.\n"
            "• **Pass 2:** Prediction fed back as input context to sharpen gradients.\n"
            "• **Pass 3:** Re-evaluates prediction against context; auto-reverts if error increases."
        )

    use_tta = "Level 3" in deep_think_level
    use_refinement = "Level 3" in deep_think_level
    k_models = 5 if ("Level 2" in deep_think_level or "Level 3" in deep_think_level) else 1

    # Provenance
    st.caption(
        f"📦 **{pick}** · region {DS.REGIONS[region]['label']} · grid {mask.shape[0]}×{mask.shape[1]} · "
        f"trained {meta.get('epochs_trained', '?')} epochs / {meta.get('rounds_trained', '?')} rounds · "
        f"updated {str(meta.get('updated_at', ''))[:16]} · mode: **{deep_think_level.split(':')[0]}**"
    )

    if not st.button("🔬 Run Validation & Deep Think Inference", type="primary", key="val_run"):
        return

    ti = target_year - start_year
    context = full_data[ti - seq_len:ti]                    # (seq_len, H, W, C)
    truth = full_data[ti]                                    # (H, W, C)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with st.spinner(f"Running {deep_think_level.split(':')[0]} inference…"):
        from training.ensemble import DeepEnsemble
        
        main_model = ClimateTwinModel(
            seq_length=seq_len, lat_dim=mask.shape[0], lon_dim=mask.shape[1],
            channels=2, hidden=int(arch.get("hidden", 32)), dropout=0.1,
            residual_scale=float(arch.get("residual_scale", 0.1)),
        ).to(device)
        try:
            ok, _ck = REG.load_into(pick, region, main_model, strict=False)
        except REG.ModelVariableMismatch as _mmerr:
            st.error(f"❌ Checkpoint contract violation — refusing to load.\n\n{_mmerr}\n\n"
                     "This usually means the model was trained on a different variable "
                     "set (e.g. an archived rain-only checkpoint being loaded into a "
                     "rain+tmax+tmin model). Retrain or pick a matching model.")
            return
        if not ok:
            st.error("Could not load this model's weights (architecture mismatch?).")
            return
        if _ck is not None and _ck.get("variable_check", "").startswith("checkpoint has no saved variables"):
            st.warning("⚠ This is a legacy checkpoint (pre-Phase-5) with no saved variable "
                       "list. Loaded under `strict=False`. Metrics may be misleading if the "
                       "current variable set differs from what the model was trained on.")
        main_model.eval()

        models_list = [main_model]
        if k_models > 1:
            for ki in range(1, k_models):
                m_k = copy.deepcopy(main_model)
                models_list.append(m_k)

        ens_obj = DeepEnsemble(models_list).to(device)
        x = torch.from_numpy(context.transpose(0, 3, 1, 2)[None]).float().to(device)  # (1,T,C,H,W)
        y_truth_t = torch.from_numpy(truth.transpose(2, 0, 1)[None]).float().to(device)
        mask_t = torch.from_numpy(mask).float().to(device)

        ens = ens_obj.predict_ensemble(
            x, mc_samples_per_model=10, use_tta=use_tta,
            refine_passes=3 if use_refinement else 1,
            truth=y_truth_t, mask=mask_t
        )
        mean = ens["p50"][0].cpu().numpy()   # (C,H,W)
        p10 = ens["p10"][0].cpu().numpy()
        p90 = ens["p90"][0].cpu().numpy()
        ref_info = ens.get("refinement_info")

    # Display Step-by-Step Deep Think Multi-Pass Refinement trace card
    if ref_info and ref_info.get("passes_run", 1) > 1:
        st.markdown("### 🧠 Deep Think Multi-Pass Trace")
        pass_cols = st.columns(ref_info["passes_run"])
        for idx, (p_c, r_val) in enumerate(zip(pass_cols, ref_info["pass_rmses"])):
            is_best = (idx + 1 == ref_info["best_pass"])
            p_c.metric(
                label=f"Pass {idx+1} {'(Selected 🏆)' if is_best else ''}",
                value=f"{r_val:.4f}" if np.isfinite(r_val) else "N/A",
                delta=f"Best Pass" if is_best else None,
                delta_color="normal"
            )

        if ref_info.get("reverted"):
            st.info("ℹ️ **Refinement Verdict:** Multi-pass sharpening did not beat Pass 1 on this specific frame — honestly reverted to Pass 1 to prevent skill degradation.")
        else:
            st.success(f"✅ **Refinement Verdict:** Deep Think Pass {ref_info['best_pass']} won! Lowered validation error to {min(ref_info['pass_rmses']):.4f}.")

    # ── Channels: 0 = annual rainfall, 1 = annual max temperature ──
    import plotly.graph_objects as _go
    pred_rain, pred_temp = mean[0], mean[1]
    truth_rain, truth_temp = truth[..., 0], truth[..., 1]
    persistence = full_data[ti - 1][..., 0]
    climatology = np.nanmean(full_data[max(0, ti - 10):ti][..., 0], axis=0)
    pers_temp = full_data[ti - 1][..., 1]
    clim_temp = np.nanmean(full_data[max(0, ti - 10):ti][..., 1], axis=0)

    ours_m = MET.compute_all_metrics(pred_rain, truth_rain, mask)
    pers_m = MET.compute_all_metrics(persistence, truth_rain, mask)
    clim_m = MET.compute_all_metrics(climatology, truth_rain, mask)
    ours_t = MET.compute_all_metrics(pred_temp, truth_temp, mask)
    pers_t = MET.compute_all_metrics(pers_temp, truth_temp, mask)
    cal = MET.ensemble_calibration(p10[0], p90[0], truth_rain, mask)

    # Physical-unit conversion (data is normalized [0,1]).
    rr = (scalers["rain_max"] - scalers["rain_min"]) or 1.0
    tr = (scalers["temp_max"] - scalers["temp_min"]) or 1.0
    rain_mae_mm = ours_m.get("mae", float("nan")) * rr
    temp_mae_c = ours_t.get("mae", float("nan")) * tr
    skill_pers = (1 - ours_m.get("rmse", np.nan) / pers_m.get("rmse", np.nan)) * 100 if pers_m.get("rmse") else np.nan
    skill_clim = (1 - ours_m.get("rmse", np.nan) / clim_m.get("rmse", np.nan)) * 100 if clim_m.get("rmse") else np.nan

    # ── Plain-language verdict ──
    if np.isfinite(skill_pers) and skill_pers > 10 and (not np.isfinite(skill_clim) or skill_clim > 0):
        verdict, vcol = "Strong — clearly beats the naïve guesses", "#39d98a"
    elif np.isfinite(skill_pers) and skill_pers > 0:
        verdict, vcol = "Decent — a bit better than the naïve guesses", "#F4A34A"
    else:
        verdict, vcol = "Weak — no better than a naïve guess yet (train it more)", "#EF4444"
    st.markdown(
        f"<div style='padding:14px 16px;border-radius:10px;background:rgba(255,255,255,0.03);"
        f"border-left:4px solid {vcol};'>"
        f"<div style='color:{vcol};font-size:16px;font-weight:700;'>{verdict}</div>"
        f"<div style='color:#cdd6e6;font-size:13px;margin-top:6px;line-height:1.6;'>"
        f"For <b>{target_year}</b>, model <b>{pick}</b> predicted the year's rainfall & temperature pattern "
        f"from the previous {seq_len} years.<br>"
        f"• Typical rainfall miss: <b>~{rain_mae_mm:,.0f} mm</b> (annual) · "
        f"typical temperature miss: <b>~{temp_mae_c:.2f} °C</b><br>"
        f"• vs “same as last year”: <b>{skill_pers:+.0f}%</b> "
        f"{'better' if np.isfinite(skill_pers) and skill_pers>=0 else 'worse'} error · "
        f"vs “historical average”: <b>{skill_clim:+.0f}%</b><br>"
        f"• Rain-pattern correlation with reality: <b>{ours_m.get('pearson', float('nan')):.2f}</b> "
        f"(1.0 = perfect) · uncertainty calibration: <b>{cal:.0%}</b> of reality fell inside the model's range."
        f"</div></div>",
        unsafe_allow_html=True,
    )

    _fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"
    st.markdown(f"#### Scorecard — {target_year}")
    mc = st.columns(4)
    mc[0].metric("Rain error (mm)", f"{rain_mae_mm:,.0f}",
                 delta=f"{skill_pers:+.0f}% vs last-year", delta_color="normal")
    mc[1].metric("Temp error (°C)", f"{temp_mae_c:.2f}")
    mc[2].metric("Rain pattern r", _fmt(ours_m.get("pearson")))
    mc[3].metric("Uncertainty cal.", f"{cal:.0%}", help="Share of truth inside p10–p90 (ideal ≈80%).")

    # ── Comparison table (model vs both baselines, rain + temp) ──
    import pandas as _pd
    keys = ["rmse", "mae", "bias", "pearson", "pod", "far", "csi"]
    st.markdown("##### Detailed comparison (normalized units, rain)")
    st.dataframe(_pd.DataFrame({
        "Metric": keys,
        "Model": [_fmt(ours_m.get(k)) for k in keys],
        "Persistence (last year)": [_fmt(pers_m.get(k)) for k in keys],
        "Climatology (avg)": [_fmt(clim_m.get(k)) for k in keys],
        "Winner": ["✅ Model" if (_lowerbetter := k in ("rmse", "mae", "far", "bias")) and
                   abs(ours_m.get(k, 9e9)) <= min(abs(pers_m.get(k, 9e9)), abs(clim_m.get(k, 9e9)))
                   else ("✅ Model" if (not _lowerbetter) and
                         ours_m.get(k, -9e9) >= max(pers_m.get(k, -9e9), clim_m.get(k, -9e9))
                         else "baseline") for k in keys],
    }), hide_index=True, use_container_width=True)

    # ── Map gallery (physical units, region-shaped) ──
    lat, lon, _, _ = DS.region_grid(region)

    def _hm(field, title, cs, unit, zmid=None):
        z = np.where(mask == 1, field, np.nan)
        f = _go.Figure(_go.Heatmap(z=z, x=lon, y=lat, colorscale=cs, zmid=zmid,
                                    colorbar=dict(title=unit, thickness=10)))
        f.update_layout(title=dict(text=title, font=dict(size=12)), height=270,
                        margin=dict(l=6, r=6, t=30, b=6),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(3,8,18,0.6)")
        f.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False)
        f.update_xaxes(showticklabels=False)
        return f

    rain_obs = truth_rain * rr + scalers["rain_min"]
    rain_pred = pred_rain * rr + scalers["rain_min"]
    rain_err = (pred_rain - truth_rain) * rr
    rain_unc = (p90[0] - p10[0]) * rr
    st.markdown("##### 🗺️ Rainfall maps (annual, mm)")
    r1 = st.columns(4)
    with r1[0]:
        st.plotly_chart(_hm(rain_obs, "Observed (reality)", "Blues", "mm"), use_container_width=True)
    with r1[1]:
        st.plotly_chart(_hm(rain_pred, "Model prediction", "Blues", "mm"), use_container_width=True)
    with r1[2]:
        st.plotly_chart(_hm(rain_err, "Error (pred − obs)", "RdBu", "mm", zmid=0), use_container_width=True)
    with r1[3]:
        st.plotly_chart(_hm(rain_unc, "Uncertainty (p90 − p10)", "Viridis", "mm"), use_container_width=True)
    st.caption("Blue maps should look alike. In the **Error** map, white = accurate; "
               "red/blue = model too high/low. **Uncertainty** shows where the model is unsure.")

    st.markdown("##### 🗺️ What the naïve baselines predicted (rain, mm)")
    r2 = st.columns(2)
    with r2[0]:
        st.plotly_chart(_hm(persistence * rr + scalers["rain_min"], "Persistence = last year", "Blues", "mm"),
                        use_container_width=True)
    with r2[1]:
        st.plotly_chart(_hm(climatology * rr + scalers["rain_min"], "Climatology = 10-yr average", "Blues", "mm"),
                        use_container_width=True)

    temp_obs = truth_temp * tr + scalers["temp_min"]
    temp_pred = pred_temp * tr + scalers["temp_min"]
    temp_err = (pred_temp - truth_temp) * tr
    st.markdown("##### 🗺️ Temperature maps (annual max, °C)")
    r3 = st.columns(3)
    with r3[0]:
        st.plotly_chart(_hm(temp_obs, "Observed (reality)", "turbo", "°C"), use_container_width=True)
    with r3[1]:
        st.plotly_chart(_hm(temp_pred, "Model prediction", "turbo", "°C"), use_container_width=True)
    with r3[2]:
        st.plotly_chart(_hm(temp_err, "Error (pred − obs)", "RdBu", "°C", zmid=0), use_container_width=True)

    # ── Diagnostics ──
    st.markdown("##### 📈 Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(VIZ.metric_comparison_bars(ours_m, pers_m, clim_m), use_container_width=True)
    with c2:
        obs_bin = (truth_rain > 0.01).astype(float)
        st.plotly_chart(VIZ.reliability_diagram(np.clip(pred_rain, 0, 1), obs_bin),
                        use_container_width=True)
    st.caption("**How to read this:** the model wins when its RMSE/MAE are **lower** and "
               "POD/CSI/correlation are **higher** than both baselines. The map errors should be "
               "small (white) and its uncertainty range should contain reality ~80% of the time.")


@st.fragment
def _render_tab9():
    """🔥 Training tab — interactive dashboard (Phase 2 rewrite)."""
    from climate_twin.train.ui.dashboards import render_training_dashboard
    render_training_dashboard()
    return


def _render_tab_validation():
    """🔍 Validation tab — visual + statistical inspection (Phase 3)."""
    from climate_twin.train.ui.dashboards import render_validation_dashboard
    render_validation_dashboard()
    return

    import threading, time as _time, json as _json
    import torch
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from training.schedule import build_schedule, ScheduleConfig, estimate_runtime, MAX_ROUNDS
    from training.state import init_training_state, get_all_rounds, insert_round, update_round, delete_round
    from training.loops import RoundConfig, TrainingProgress, train_one_round
    from training.model import ClimateTwinModel
    from training.checkpoints import load_checkpoint, list_checkpoints
    from training import registry as REG
    from training import viz as VIZ

    init_training_state()
    REG.init_registry()

    st.title("🔥 Walk-Forward Training")

    # ── Environment check ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        st.error("⛔ CUDA not available. Walk-forward training requires a GPU. "
                 "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        st.stop()

    env_cols = st.columns(4)
    env_cols[0].metric("Device", "CUDA ✓")
    env_cols[1].metric("GPU", torch.cuda.get_device_name(0))
    env_cols[2].metric("VRAM", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    env_cols[3].metric("PyTorch", torch.__version__)

    st.divider()

    # ── Dataset availability (region-aware, from local raw IMD via data_source) ──
    _train_region = active_region()
    try:
        _rain_agg, _temp_agg, _mask, _years_agg = DS.load_aggregates(_train_region)
        _data_available = True
    except Exception as _e:
        _rain_agg = _temp_agg = _mask = None
        _data_available = False
        st.warning(f"Could not build the {DS.REGIONS[_train_region]['label']} cube: {_e}")

    if _data_available:
        _agg = {"rain": _rain_agg, "temp": _temp_agg, "mask": _mask}
        _n_years = _rain_agg.shape[0]
        _data_start_year = int(_years_agg.min())
        _data_end_year = int(_years_agg.max())
        _has_rain, _has_temp, _has_mask = True, True, True
    else:
        _agg = None
        _data_start_year = 1975
        _data_end_year = 2025
        _n_years = _data_end_year - _data_start_year + 1
        _mask = np.ones(DS.REGIONS[_train_region]["expected_shape"], dtype=np.float32)

    # ── Compact env + dataset chip strip (small text, not giant metrics) ──────
    def _chip(label, value, color="#8899bb"):
        return (f'<span style="display:inline-block;margin:2px 14px 2px 0;">'
                f'<span style="color:#5b6b86;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:.5px;">{label}</span><br>'
                f'<span style="color:{color};font-size:14px;font-weight:600;">{value}</span></span>')

    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_short = (_gpu_name[:22] + "…") if len(_gpu_name) > 23 else _gpu_name
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    _grid = f"{_mask.shape[0]}×{_mask.shape[1]}" if _data_available else "—"
    _land = f"{int(_mask.sum()):,}" if _data_available else "—"
    st.markdown(
        _chip("Device", "CUDA ✓", "#39d98a")
        + _chip("GPU", _gpu_short)
        + _chip("VRAM", f"{_vram:.1f} GB")
        + _chip("PyTorch", torch.__version__)
        + _chip("Region", DS.REGIONS[_train_region]["label"], "#F4A34A")
        + _chip("Years", f"{_data_start_year}–{_data_end_year}")
        + _chip("Grid", _grid)
        + _chip("Land cells", _land)
        + _chip("Checkpoints", f"{_train_region}_round_*"),
        unsafe_allow_html=True,
    )

    st.divider()

    # Shared normalized cube (both Train and Validate use the same region data).
    if _data_available:
        _lm0 = (_mask == 1)
        _rl = _rain_agg[:, _lm0] if _lm0.any() else _rain_agg.reshape(len(_rain_agg), -1)
        _tl = _temp_agg[:, _lm0] if _lm0.any() else _temp_agg.reshape(len(_temp_agg), -1)
        _scalers = {"rain_min": float(np.nanmin(_rl)), "rain_max": float(np.nanmax(_rl)),
                    "temp_min": float(np.nanmin(_tl)), "temp_max": float(np.nanmax(_tl))}
        _rn = np.clip((_rain_agg - _scalers["rain_min"]) / (_scalers["rain_max"] - _scalers["rain_min"] + 1e-8), 0, 1)
        _tn = np.clip((_temp_agg - _scalers["temp_min"]) / (_scalers["temp_max"] - _scalers["temp_min"] + 1e-8), 0, 1)
        _full_data = np.stack([_rn, _tn], axis=-1)  # (N, H, W, 2)
    else:
        _scalers, _full_data = None, None

    # ── Mode: Train vs Validate ───────────────────────────────────────────
    _tmode = st.radio("Workflow", ["🏋️ Train", "🔬 Validate"], horizontal=True, key="wf_mode",
                      help="Train = create a new model or continue an existing one. "
                           "Validate = load a saved model and evaluate its skill.")
    st.divider()

    if _tmode == "🔬 Validate":
        _render_validate_mode(_train_region, _full_data, _mask, _data_start_year,
                              _n_years, _scalers, _data_available)
        return

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR CONFIGURATION (rendered inside tab9 as columns)
    # ══════════════════════════════════════════════════════════════════════

    config_col, main_col = st.columns([1, 2.5])

    with config_col:
        # ── PART B: Execution Modes (Continue / Fine-Tune / Re-run Fresh) ──
        st.markdown("### 🧬 Model & Execution Mode")
        _existing = REG.list_models(_train_region)
        _existing_names = [m["name"] for m in _existing]
        
        b_mode = st.radio(
            "Mode",
            ["B1: ▶ Continue Training", "B2: 🎯 Fine-Tune", "B3: 🔄 Re-run Fresh"],
            key="wf_partb_mode",
            help="Continue Training = load checkpoint, keep optimizer, train N more epochs on same data. "
                 "Fine-Tune = low LR, targeted data slice (e.g. monsoon), optional recurrent freeze. "
                 "Re-run Fresh = train from scratch on same data with new hyperparams."
        )

        _parent_model_name = ""
        _freeze_recurrent = False
        _target_data_slice = "all"

        if b_mode == "B1: ▶ Continue Training":
            if _existing_names:
                _parent_model_name = st.selectbox("Base checkpoint to continue", _existing_names, key="b1_pick",
                                                   help="Diagnosis said 'still learning'? Continue training it here.")
                _model_name = st.text_input("New version name", value=f"{_parent_model_name}_v2", key="b1_name",
                                            help="Saves as a new versioned checkpoint — never overwrites parent.")
                _resume = True
                _m = REG.get_model(_parent_model_name, _train_region)
                if _m:
                    st.caption(f"↳ Parent `{_parent_model_name}`: {_m['epochs_trained']} epochs · {_m['rounds_trained']} rounds so far")
            else:
                st.info("No base models exist yet. Switch to 'B3: Re-run Fresh' to create the first checkpoint.")
                _model_name = st.text_input("Model name", value="cauvery_v1" if _train_region == "cauvery" else "india_v1", key="b1_fresh")
                _resume = False

        elif b_mode == "B2: 🎯 Fine-Tune":
            if _existing_names:
                _parent_model_name = st.selectbox("Base checkpoint to fine-tune", _existing_names, key="b2_pick")
                _target_data_slice = st.selectbox("Target data slice", ["all (gentle polish)", "recent_5yrs", "recent_10yrs"], key="b2_slice",
                                                 help="Targeted fine-tuning. The training cube is ANNUAL, so "
                                                      "slicing is by recent years (adapt to current climate). "
                                                      "Seasonal/monsoon slicing needs a daily cube (not built yet).")
                _freeze_recurrent = st.checkbox("Freeze recurrent core (fine-tune output head only)", value=True, key="b2_freeze",
                                                help="Standard transfer learning: freezes ConvLSTM core, updates only output head for fast, stable adaptation.")
                _suffix = _target_data_slice.split(" ")[0].split("(")[0] or "ft"
                _suffix = "ft_" + _suffix if _suffix != "all" else "ft"
                _model_name = st.text_input("Fine-tuned model name", value=f"{_parent_model_name}_{_suffix}", key="b2_name")
                _resume = True
                _m = REG.get_model(_parent_model_name, _train_region)
                if _m:
                    st.caption(f"↳ Fine-tuning `{_parent_model_name}` at 1/10th LR (default 1e-4)")
            else:
                st.info("No base models exist yet to fine-tune. Create one first.")
                _model_name = st.text_input("Model name", value="cauvery_v1" if _train_region == "cauvery" else "india_v1", key="b2_fresh")
                _resume = False

        else:  # B3: Re-run Fresh
            _model_name = st.text_input("New model name", value="cauvery_v1" if _train_region == "cauvery" else "india_v1", key="b3_name",
                                        help="Train a fresh model from scratch on the same data for A/B comparison.")
            _resume = False

        st.divider()
        _render_model_library(_train_region, _mask.shape)
        st.divider()

        # ── Walk-Forward Schedule ─────────────────────────────────────────
        st.markdown("### 📅 Walk-Forward Schedule")

        wf_start = st.number_input("Start year", min_value=1951, max_value=2020,
                                   value=_data_start_year, key="wf_start_year",
                                   help="First year of data used. Earlier = more history to learn "
                                        "from, but older data may be less representative of today's climate.")
        wf_end = st.number_input("End year", min_value=wf_start + 5, max_value=2075,
                                 value=min(_data_end_year, 2024), key="wf_end_year",
                                 help="Last year included. The walk-forward marches from Start to "
                                      "End; a wider span = more validation rounds.")
        wf_init_window = st.number_input("Initial window (years)", 3, 30, 10, key="wf_init_win",
                                         help="How many years the model trains on before its first "
                                              "prediction. Larger = more data per round (steadier, "
                                              "slower start); smaller = starts sooner but learns from less.")

        wf_step_unit = st.radio("Step unit", ["months", "years", "days"], horizontal=True, key="wf_step_unit",
                                help="How far the window advances each round. Data here is ANNUAL, so "
                                     "'years' is the natural choice — 'months'/'days' create many "
                                     "near-identical rounds mapping to the same year.")
        wf_step_size = st.number_input("Step size", 1, 12 if wf_step_unit == "months" else 5, 1, key="wf_step_size",
                                       help="Number of step-units advanced per round. Smaller step = "
                                            "more rounds & finer evaluation but longer total runtime.")
        wf_window_mode = st.radio("Window mode", ["expanding", "rolling"], horizontal=True, key="wf_win_mode",
                                  help="Expanding = training window keeps growing (uses ALL past years — "
                                       "best for a stable climate signal). Rolling = fixed-length window "
                                       "that slides (adapts to recent trends, forgets old data).")

        wf_rolling_years = 10
        if wf_window_mode == "rolling":
            wf_rolling_years = st.number_input("Rolling window (years)", 5, 30, 10, key="wf_roll_yrs",
                                               help="Length of the sliding training window (rolling mode). "
                                                    "Shorter = more responsive to recent change but noisier.")

        wf_horizon = st.number_input("Prediction horizon (steps)", 1, 5, 1, key="wf_horizon",
                                     help="How many steps ahead each round forecasts. 1 = predict the very "
                                          "next period (easiest, most accurate); higher = longer-range, harder.")

        # Build schedule preview
        sched_cfg = ScheduleConfig(
            start_year=wf_start, end_year=wf_end,
            initial_window_years=wf_init_window,
            step_unit=wf_step_unit, step_size=wf_step_size,
            window_mode=wf_window_mode, rolling_window_years=wf_rolling_years,
            prediction_horizon=wf_horizon,
        )
        schedule = build_schedule(sched_cfg)
        n_rounds = len(schedule)

        st.metric("Total rounds", n_rounds)
        st.caption(f"Est. runtime: {estimate_runtime(n_rounds, 15)}")

        if n_rounds > MAX_ROUNDS:
            st.error(f"Too many rounds ({n_rounds} > {MAX_ROUNDS}). Reduce step granularity.")
        elif n_rounds == 0:
            st.warning("No rounds generated. Check start/end/window settings.")

        # Preview
        with st.expander("Schedule preview", expanded=False):
            if schedule:
                preview_data = []
                for r in schedule[:20]:
                    preview_data.append({
                        "Round": r.round_num,
                        "Train": f"{r.train_start} → {r.train_end}",
                        "Predict": f"{r.val_start} → {r.val_end}",
                    })
                st.dataframe(pd.DataFrame(preview_data), hide_index=True, height=300)
                if n_rounds > 20:
                    st.caption(f"… and {n_rounds - 20} more rounds")

        st.divider()

        # ── Run Mode ──────────────────────────────────────────────────────
        st.markdown("### ▶️ Run Mode")
        wf_run_mode = st.radio(
            "Mode",
            ["manual", "auto_cascade", "auto_stop_on_degradation"],
            format_func=lambda x: {"manual": "Manual (step by step)", "auto_cascade": "Auto-cascade (all rounds)", "auto_stop_on_degradation": "Auto + stop on degradation"}[x],
            key="wf_run_mode_sel",
            help="Manual = run one round per click (inspect each result). "
                 "Auto-cascade = run every round back-to-back (longest, unattended). "
                 "Auto + stop = run automatically but halt if skill worsens beyond the threshold below.",
        )
        wf_degrad_threshold = 5.0
        if wf_run_mode == "auto_stop_on_degradation":
            wf_degrad_threshold = st.slider("Degradation threshold (%)", 1, 20, 5, key="wf_degrad_pct",
                                            help="Stop automatically if validation RMSE gets this much worse "
                                                 "than the best round so far. Guards against wasting time once "
                                                 "the model stops improving.")

        st.divider()

        # ── Training Config ───────────────────────────────────────────────
        st.markdown("### ⚙️ Training Config")

        is_max_mode = st.toggle(
            "🚀 MAX MODE (Deepest Thinking & Unlimited Training)",
            key="wf_max_mode_toggle",
            help="Enables extreme settings: Unlimited/Infinite Epochs (99,999), 8x Gradient Accumulation, "
                 "Fine LR 1e-4 with min_lr 1e-7, Max Physics Smoothness 0.15, and continuous infinite training!"
        )

        if is_max_mode:
            st.info(
                "🚀 **MAX MODE ACTIVE — Unlimited Epochs & Deepest Thinking Training**\n"
                "• **Epochs:** Unlimited / Infinite (99,999 epochs per round)\n"
                "• **Precision & Accumulation:** Fine LR `1e-4`, Min LR `1e-7`, Gradient Accumulation `8x`\n"
                "• **Physics:** Spatial smoothness `0.15`, Temporal smoothness `0.15`, Recency `15y` half-life\n"
                "• *Training will run continuously for the deepest possible learning. Click ⏸ Pause or ⏹ Stop at any time to save.*"
            )

        wf_train_mode = st.radio(
            "Weight init",
            ["warm_start", "from_scratch", "reset_every_n"],
            format_func=lambda x: {"warm_start": "Warm-start from previous", "from_scratch": "From scratch each round", "reset_every_n": "Reset every N rounds"}[x],
            key="wf_train_mode",
            help="Warm-start = each round continues from the previous round's weights "
                 "(fast, accumulates learning — best default). From scratch = reset every "
                 "round (unbiased comparison, slower). Reset every N = compromise.",
        )
        wf_reset_n = 10
        if wf_train_mode == "reset_every_n":
            wf_reset_n = st.number_input("Reset every N", 2, 50, 10, key="wf_reset_n",
                                         help="Re-initialise the model from scratch every N rounds; "
                                              "warm-start in between.")

        # Default warm-start for fine steps, scratch for coarse
        st.caption("Default: warm-start for step<1yr, scratch for step≥1yr")

        with st.expander("Hyperparameters", expanded=False):
            hp_lr = st.slider("Learning rate", 1e-5, 1e-2, 1e-3, format="%.5f", key="hp_lr",
                              help="Step size for weight updates. Too high → unstable/diverges; "
                                   "too low → trains very slowly. 1e-3 is a safe default.")
            hp_batch = st.select_slider("Batch size", [2, 4, 8, 16, 32], value=8, key="hp_batch",
                                        help="Samples per gradient step. Larger = smoother, faster on GPU "
                                             "but more VRAM. This annual dataset is small, so 8 is plenty.")
            hp_epochs = st.slider("Epochs per round", 1, 100, 25, key="hp_epochs",
                                  help="Passes over the training data each round. More = better fit but "
                                       "risks overfitting and longer runtime (early-stopping can cut it short).")
            hp_optimizer = st.selectbox("Optimizer", ["AdamW", "SGD"], key="hp_opt",
                                        help="AdamW = adaptive, converges fast, robust (recommended). "
                                             "SGD = simpler, sometimes generalises better but needs tuning.")
            hp_wd = st.slider("Weight decay", 0.0, 0.01, 1e-5, format="%.5f", key="hp_wd",
                              help="L2 regularisation. Higher = simpler model, less overfitting, "
                                   "but too much underfits.")
            hp_dropout = st.slider("Dropout", 0.0, 0.5, 0.1, step=0.05, key="hp_drop",
                                   help="Fraction of units randomly disabled while training. Reduces "
                                        "overfitting and powers the MC-dropout uncertainty bands (p10/p90).")

        with st.expander("LR Scheduler", expanded=False):
            hp_sched = st.selectbox("Scheduler", ["cosine", "plateau", "step", "none"], key="hp_sched",
                                    help="How the learning rate changes over epochs. Cosine = smooth decay "
                                         "(good default); plateau = drop when val stalls; step = drop at "
                                         "intervals; none = constant.")
            hp_warmup = st.number_input("Warmup epochs", 0, 10, 2, key="hp_warmup",
                                        help="Ramp the LR up gradually over the first few epochs to avoid "
                                             "early instability.")
            hp_min_lr = st.number_input("Min LR", 1e-7, 1e-4, 1e-6, format="%.7f", key="hp_minlr",
                                        help="Floor the scheduler decays toward — keeps a little learning "
                                             "signal at the end of training.")

        with st.expander("Regularization & Compute", expanded=False):
            hp_grad_clip = st.slider("Grad clip", 0.0, 5.0, 1.0, key="hp_gc",
                                     help="Caps gradient magnitude to prevent exploding updates. 0 = off. "
                                          "1.0 is a safe stabiliser.")
            hp_mp = st.radio("Precision", ["bf16", "fp16", "fp32"], horizontal=True, key="hp_mp",
                             help="Numeric precision. bf16/fp16 = faster and less VRAM on GPU (mixed "
                                  "precision); fp32 = most accurate/stable but slower.")
            hp_accum = st.number_input("Gradient accumulation", 1, 8, 1, key="hp_accum",
                                       help="Sum gradients over N mini-batches before updating — simulates a "
                                            "larger batch without extra VRAM. 1 = off.")

        with st.expander("Early Stopping", expanded=False):
            hp_es = st.checkbox("Enabled", True, key="hp_es",
                                help="Stop a round early once validation stops improving — saves time and "
                                     "prevents overfitting.")
            hp_patience = st.number_input("Patience", 2, 20, 5, key="hp_pat",
                                          help="Epochs to wait for improvement before stopping. Higher = "
                                               "more patient (trains longer).")
            hp_min_delta = st.number_input("Min delta", 0.0, 0.01, 1e-4, format="%.4f", key="hp_md",
                                           help="Smallest change that counts as 'improvement'. Larger = "
                                                "stops sooner.")
            hp_monitor = st.selectbox("Monitor", ["val_rmse", "val_loss", "csi"], key="hp_mon",
                                      help="Metric watched for early stopping. val_rmse/val_loss = error "
                                           "(lower better); csi = rain hit-rate skill (higher better).")

        with st.expander("Physics & Recency", expanded=False):
            hp_phys_spatial = st.slider("Spatial smoothness", 0.0, 0.3, 0.05, key="hp_ps",
                                        help="Penalises jagged neighbouring pixels so maps look physically "
                                             "smooth. Too high = blurry.")
            hp_phys_temporal = st.slider("Temporal smoothness", 0.0, 0.3, 0.05, key="hp_pt",
                                         help="Penalises abrupt year-to-year jumps so the forecast evolves "
                                              "realistically over time.")
            hp_recency = st.checkbox("Recency weighting", True, key="hp_rec",
                                     help="Weight recent years more heavily so the model reflects the "
                                          "current climate rather than the distant past.")
            hp_halflife = st.number_input("Half-life (years)", 5, 50, 20, key="hp_hl",
                                          help="Years over which a sample's weight halves. Shorter = focus "
                                               "on recent years; longer = treat all years more equally.") if hp_recency else 20

        with st.expander("Seed & Reproducibility", expanded=False):
            hp_seed = st.number_input("Seed", 0, 99999, 42, key="hp_seed",
                                      help="Fixes the random number generator so a run is repeatable "
                                           "(same weights init, same dropout pattern).")
            hp_determ = st.checkbox("Deterministic mode", False, key="hp_det",
                                    help="Forces bit-exact reproducible GPU ops. Guarantees identical "
                                         "results across runs, but can be slower.")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN AREA — Controls + Live View + History
    # ══════════════════════════════════════════════════════════════════════
    with main_col:
        # ── Action buttons ────────────────────────────────────────────────
        btn_cols = st.columns(5)
        can_start = n_rounds > 0 and n_rounds <= MAX_ROUNDS and not st.session_state.get("wf_running", False)

        with btn_cols[0]:
            if wf_run_mode == "manual":
                start_btn = st.button("▶ Run Next Round", disabled=not can_start, type="primary", key="wf_start_btn",
                                      help="Train + validate the next single round in the schedule, then stop "
                                           "so you can inspect the result.")
            else:
                start_btn = st.button("▶ Run All Rounds", disabled=not can_start, type="primary", key="wf_start_btn",
                                      help="Run every remaining round back-to-back (auto mode). May take a while.")
        with btn_cols[1]:
            ens_btn = st.button("👥 Train Ensemble (5 models)", disabled=not can_start, key="wf_ens_btn",
                                help="Train 5 models with different random seeds on the same data. "
                                     "Ensemble prediction & uncertainty almost always beats single models.")
        with btn_cols[2]:
            pause_btn = st.button("⏸ Pause", disabled=not st.session_state.get("wf_running", False), key="wf_pause_btn",
                                  help="Pause after the current epoch; resume later without losing progress.")
        with btn_cols[3]:
            stop_btn = st.button("⏹ Stop", disabled=not st.session_state.get("wf_running", False), key="wf_stop_btn",
                                 help="Finish the current round, save its checkpoint, then stop.")
        with btn_cols[4]:
            stop_all_btn = st.button("⏹⏹ Stop All", disabled=not st.session_state.get("wf_running", False), key="wf_stopall_btn",
                                     help="Halt immediately after the current epoch — abandons the rest of the schedule.")

        if pause_btn:
            st.session_state["wf_paused"] = True
            st.toast("Paused — will halt after current epoch.")
        if stop_btn:
            st.session_state["wf_stop_requested"] = True
            st.toast("Stop requested — finishing current round.")
        if stop_all_btn:
            st.session_state["wf_stop_all_requested"] = True
            st.toast("Stop all — halting immediately after current epoch.")

        # ── Build round config ────────────────────────────────────────────
        if is_max_mode:
            hp_epochs = 99999
            hp_lr = 1e-4
            hp_min_lr = 1e-7
            hp_accum = 8
            hp_phys_spatial = 0.15
            hp_phys_temporal = 0.15
            hp_recency = True
            hp_halflife = 15
            hp_es = False

        round_cfg = RoundConfig(
            lr=hp_lr, batch_size=hp_batch, epochs=hp_epochs,
            optimizer_name=hp_optimizer, weight_decay=hp_wd, dropout=hp_dropout,
            scheduler_name=hp_sched, warmup_epochs=hp_warmup, min_lr=hp_min_lr,
            grad_clip=hp_grad_clip, mixed_precision=hp_mp, gradient_accum=hp_accum,
            early_stopping=hp_es, patience=hp_patience, min_delta=hp_min_delta,
            monitor=hp_monitor, physics_spatial_smooth=hp_phys_spatial,
            physics_temporal_smooth=hp_phys_temporal,
            recency_weighting=hp_recency, recency_half_life_years=hp_halflife,
            seed=hp_seed, deterministic=hp_determ,
        )

        # ── PART B: apply mode-specific overrides ──
        if b_mode == "B2: 🎯 Fine-Tune":
            round_cfg.lr = hp_lr / 10.0            # 1/10th LR (gentle polish)
            round_cfg.freeze_recurrent = bool(_freeze_recurrent)
            if round_cfg.warmup_epochs > 0:
                round_cfg.warmup_epochs = 0        # no warmup when fine-tuning
            st.caption(f"🎯 Fine-tune overrides: LR **{round_cfg.lr:.2e}** (1/10th) · "
                       f"freeze core **{round_cfg.freeze_recurrent}** · slice **{_target_data_slice}**")
        elif b_mode == "B3: 🔄 Re-run Fresh":
            round_cfg.seed = hp_seed + 1000        # decorrelate the fresh A/B run

        st.divider()

        # ── Training execution ────────────────────────────────────────────
        if (start_btn or ens_btn) and can_start and _data_available:
            st.session_state["wf_running"] = True
            st.session_state["wf_stop_requested"] = False
            st.session_state["wf_stop_all_requested"] = False
            st.session_state["wf_paused"] = False

            # Determine which rounds to run
            current_round_idx = st.session_state.get("wf_current_round", 0)
            rounds_to_run = schedule[current_round_idx:current_round_idx + 1] if wf_run_mode == "manual" else schedule[current_round_idx:]

            # Load data (region cube built from local raw IMD)
            rain_data = _agg["rain"]  # (N_years, H, W)
            temp_data = _agg["temp"]  # (N_years, H, W)

            # Normalize to [0,1] using this region's own train-range
            _lm = (_mask == 1)
            _r_land = rain_data[:, _lm] if _lm.any() else rain_data.reshape(len(rain_data), -1)
            _t_land = temp_data[:, _lm] if _lm.any() else temp_data.reshape(len(temp_data), -1)
            scalers = {
                "rain_min": float(np.nanmin(_r_land)), "rain_max": float(np.nanmax(_r_land)),
                "temp_min": float(np.nanmin(_t_land)), "temp_max": float(np.nanmax(_t_land)),
            }
            rain_norm = (rain_data - scalers["rain_min"]) / (scalers["rain_max"] - scalers["rain_min"] + 1e-8)
            temp_norm = (temp_data - scalers["temp_min"]) / (scalers["temp_max"] - scalers["temp_min"] + 1e-8)
            rain_norm = np.clip(rain_norm, 0, 1)
            temp_norm = np.clip(temp_norm, 0, 1)

            # Stack channels: (N_years, H, W, 2)
            full_data = np.stack([rain_norm, temp_norm], axis=-1)  # (N_years, 129, 135, 2)

            # ── PART B: fine-tune data slice (annual cube → restrict to recent years) ──
            _slice_from_year = None
            if b_mode == "B2: 🎯 Fine-Tune":
                if "recent_5yrs" in _target_data_slice:
                    _slice_from_year = _data_end_year - 4
                elif "recent_10yrs" in _target_data_slice:
                    _slice_from_year = _data_end_year - 9

            # ── Top live dashboard ──
            st.markdown("#### 🚀 Live training")
            top_bar = st.progress(0.0, text="Preparing run…")
            top_status = st.empty()
            live_metrics = st.empty()
            live_diag = st.empty()
            with st.expander("🧪 Scientific console (full log)", expanded=False):
                _console_box = st.empty()
            _console_lines = []

            def _log(msg):
                from datetime import datetime as _dt
                _console_lines.append(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")
                _console_box.code("\n".join(_console_lines[-500:]), language="text")

            _log(f"Model '{_model_name}' | region={active_region()} | resume={_resume} | ensemble={ens_btn}")
            _log(f"Data: {DS.REGIONS[active_region()]['label']} cube {full_data.shape} "
                 f"(years {_data_start_year}-{_data_end_year}) via data_source.load_aggregates()")

            # Resume: pre-build the model and load saved weights so training continues.
            # For Continue/Fine-Tune the weights come from the PARENT checkpoint, not
            # the (not-yet-existing) new versioned name.
            _load_from = _parent_model_name or _model_name
            model = None
            if _resume:
                _arch = (REG.get_model(_load_from, active_region()) or {}).get("arch", {})
                model = ClimateTwinModel(
                    seq_length=int(_arch.get("seq_length", 5)),
                    lat_dim=_mask.shape[0], lon_dim=_mask.shape[1], channels=2,
                    hidden=int(_arch.get("hidden", 32)), dropout=hp_dropout,
                    residual_scale=float(_arch.get("residual_scale", 0.1)),
                )
                try:
                    ok, _ck = REG.load_into(_load_from, active_region(), model, strict=False)
                    _mismatch = None
                except REG.ModelVariableMismatch as _mmerr:
                    ok, _ck = False, None
                    _mismatch = str(_mmerr)
                if _mismatch:
                    _log(f"⛔ Refused to resume '{_load_from}': {_mismatch}")
                    st.error(f"❌ Cannot resume from '{_load_from}' — variable list mismatch.\n\n"
                             f"{_mismatch}\n\nStart a fresh model or pick a compatible parent.")
                    return
                _log(f"Resumed weights from model '{_load_from}' → {'OK' if ok else 'FAILED (fresh init)'}"
                     + (f" | fine-tune LR {round_cfg.lr:.2e}, freeze_core={round_cfg.freeze_recurrent}"
                        if b_mode == 'B2: 🎯 Fine-Tune' else ""))
                if not ok:
                    model = None

            _epochs_done_total = 0
            _rounds_done_total = 0
            prev_val_metric = None
            _n_to_run = max(1, len(rounds_to_run))
            _round_results = []

            for _ri, rnd in enumerate(rounds_to_run):
                if st.session_state.get("wf_stop_requested") or st.session_state.get("wf_stop_all_requested"):
                    break

                train_start_yr = rnd.train_start.year - _data_start_year
                train_end_yr = rnd.train_end.year - _data_start_year
                val_start_yr = rnd.val_start.year - _data_start_year
                val_end_yr = rnd.val_end.year - _data_start_year

                # PART B fine-tune slice: clamp the training window to recent years only.
                if _slice_from_year is not None:
                    _sfy = _slice_from_year - _data_start_year
                    train_start_yr = max(train_start_yr, _sfy)

                if train_start_yr < 0 or val_end_yr >= _n_years:
                    _log(f"Round {rnd.round_num}: dates outside data range — skipped.")
                    continue

                train_win = train_end_yr - train_start_yr + 1
                seq_len = max(2, min(5, train_win - 2))
                round_cfg.seq_length = seq_len
                _H, _W = _mask.shape

                train_slice = full_data[train_start_yr:train_end_yr + 1]
                if len(train_slice) <= seq_len:
                    _log(f"Round {rnd.round_num}: window {train_win}y ≤ context {seq_len}y — skipped.")
                    continue

                X_train, Y_train = [], []
                for i in range(len(train_slice) - seq_len):
                    X_train.append(train_slice[i:i + seq_len]); Y_train.append(train_slice[i + seq_len])
                X_train = np.array(X_train); Y_train = np.array(Y_train)

                val_slice = full_data[max(0, val_start_yr - seq_len):val_end_yr + 1]
                X_val, Y_val = [], []
                for i in range(len(val_slice) - seq_len):
                    X_val.append(val_slice[i:i + seq_len]); Y_val.append(val_slice[i + seq_len])
                X_val = np.array(X_val) if X_val else np.zeros((0, seq_len, _H, _W, 2))
                Y_val = np.array(Y_val) if Y_val else np.zeros((0, _H, _W, 2))

                progress = TrainingProgress()
                val_period = f"{rnd.val_start.strftime('%Y-%m')}"
                row_id = insert_round(
                    rnd.round_num, str(rnd.train_start), str(rnd.train_end),
                    str(rnd.val_start), str(rnd.val_end), round_cfg.to_dict(),
                )

                def _stop_check():
                    return st.session_state.get("wf_stop_all_requested", False) or st.session_state.get("wf_paused", False)

                _t_round = _time.time()
                if ens_btn:
                    from training.ensemble import train_deep_ensemble
                    top_status.markdown(f"**Round {rnd.round_num}/{n_rounds}** · 👥 Training 5-Model Deep Ensemble (seeds 42..442)...")
                    ens_obj, metrics, indiv_m_list = train_deep_ensemble(
                        X_train, Y_train, X_val, Y_val, _mask,
                        round_cfg, k_models=5, stop_check=_stop_check,
                        round_num=rnd.round_num, val_period=val_period, region=active_region()
                    )
                    model = ens_obj.models[0]
                    single_rmses = [m.get("rmse", float("nan")) for m in indiv_m_list if isinstance(m.get("rmse"), (int, float))]
                    best_single = min(single_rmses) if single_rmses else float("nan")
                    ens_rmse = metrics.get("rmse", float("nan"))
                    impr = (best_single - ens_rmse) / best_single * 100 if np.isfinite(best_single) and np.isfinite(ens_rmse) and best_single > 0 else 0.0
                    st.success(f"👥 **Deep Ensemble (5 models) Trained!**\n"
                               f"• Best Single Model RMSE: **{best_single:.4f}**\n"
                               f"• 5-Model Ensemble RMSE: **{ens_rmse:.4f}** ({impr:+.2f}% skill gain)\n"
                               f"Ensemble saved under **{_model_name}**.")
                else:
                    def _on_epoch(logs, _rn=rnd.round_num, _idx=_ri):
                        ep, tot = logs["epoch"], logs["total_epochs"]
                        overall = (_idx + ep / max(tot, 1)) / _n_to_run
                        top_bar.progress(min(1.0, overall), text=f"Round {_rn}/{n_rounds} · Epoch {ep}/{tot}")
                        vm = logs.get("val_metrics", {}) or {}
                        cc = live_metrics.columns(6)
                        cc[0].metric("Epoch", f"{ep}/{tot}")
                        cc[1].metric("Train loss", f"{logs['train_loss']:.4f}")
                        cc[2].metric("Val loss", f"{logs['val_loss']:.4f}")
                        cc[3].metric("Val RMSE", f"{vm.get('rmse', float('nan')):.4f}")
                        cc[4].metric("GPU", f"{logs['gpu_alloc_mb']:.0f} MB")
                        cc[5].metric("LR", f"{logs['lr']:.2e}")
                        if len(logs["train_losses"]) >= 1:
                            fig = VIZ.training_curves_figure(
                                logs["train_losses"], logs["val_losses"],
                                grad_norms=list(progress.grad_norms) or None,
                                lrs=list(progress.lrs) or None,
                            )
                            live_diag.plotly_chart(fig, use_container_width=True, key=f"diag_{_rn}_{ep}")

                    model, metrics = train_one_round(
                        X_train, Y_train, X_val, Y_val, _mask,
                        round_cfg, progress, model=model, stop_check=_stop_check,
                        round_num=rnd.round_num, val_period=val_period,
                        region=active_region(), on_epoch=_on_epoch,
                    )
                _round_secs = _time.time() - _t_round
                _peak = 0.0
                try:
                    _peak = torch.cuda.max_memory_allocated(device) / 1e6
                except Exception:
                    pass
                _epochs_done_total += len(progress.train_losses)
                if progress.finished:
                    _rounds_done_total += 1
                _log(f"  ✓ round done {_round_secs:.1f}s | peak GPU {_peak:.0f} MB | "
                     f"val_rmse={metrics.get('rmse', float('nan')):.5f} csi={metrics.get('csi', float('nan')):.4f}")

                # Persist metrics + curves + baselines so the run detail view can replay it.
                from training.checkpoints import checkpoint_path as _cp
                ckpt = str(_cp(rnd.round_num, val_period, active_region()))
                _store = dict(metrics)
                _store["_curves"] = {"train": list(progress.train_losses), "val": list(progress.val_losses),
                                     "val_rmse": list(progress.val_rmses),
                                     "grad": list(progress.grad_norms), "lr": list(progress.lrs)}
                _store["_baselines"] = progress.baseline_metrics
                _store["_seconds"] = round(_round_secs, 2)
                _store["_peak_gpu_mb"] = round(_peak, 1)
                _store["_model"] = _model_name
                update_round(row_id, _store, ckpt, "completed" if progress.finished else "stopped")
                _round_results.append({"round": rnd.round_num, "metrics": metrics,
                                       "baselines": progress.baseline_metrics, "secs": _round_secs,
                                       "store": _store})

                if wf_run_mode == "auto_stop_on_degradation" and prev_val_metric is not None:
                    current_val = metrics.get("rmse", float("inf"))
                    if np.isfinite(current_val) and np.isfinite(prev_val_metric):
                        change_pct = (current_val - prev_val_metric) / abs(prev_val_metric) * 100
                        if change_pct > wf_degrad_threshold:
                            _log(f"⚠️ Val RMSE degraded {change_pct:.1f}% > {wf_degrad_threshold}% — stopping.")
                            break
                prev_val_metric = metrics.get("rmse")
                st.session_state["wf_current_round"] = rnd.round_num

            top_bar.progress(1.0, text="Run finished")

            # Compact summary of the rounds just completed (not a giant list).
            if _round_results:
                import pandas as _pd
                _sum = _pd.DataFrame([{
                    "Round": r["round"],
                    "Val RMSE": round(r["metrics"].get("rmse", float("nan")), 4),
                    "CSI": round(r["metrics"].get("csi", float("nan")), 4),
                    "Persistence RMSE": round(r["baselines"].get("persistence", {}).get("rmse", float("nan")), 4),
                    "Climatology RMSE": round(r["baselines"].get("climatology", {}).get("rmse", float("nan")), 4),
                    "Time (s)": round(r["secs"], 1),
                } for r in _round_results])
                st.markdown("##### This run — round summary")
                st.dataframe(_sum, hide_index=True, use_container_width=True)

                # PART A — diagnosis of the most recent round
                st.markdown("##### 🩺 Diagnosis — latest round")
                _render_round_diagnostics(_round_results[-1]["store"],
                                          key_prefix=f"live_{_round_results[-1]['round']}")

            st.session_state["wf_running"] = False

            # Persist the trained model under its name (register / update).
            if model is not None and _rounds_done_total > 0:
                _arch = {"seq_length": int(getattr(round_cfg, "seq_length", 5)),
                         "hidden": 32, "channels": 2, "residual_scale": 0.1,
                         "grid": list(_mask.shape)}
                try:
                    _saved = REG.save_model(
                        _model_name, active_region(), model, _arch,
                        metrics=metrics, epochs_add=_epochs_done_total,
                        rounds_add=_rounds_done_total,
                        notes=f"{'resumed' if _resume else 'new'} · {b_mode}",
                        parent_name=_parent_model_name,
                    )
                    _log(f"💾 Saved model '{_model_name}' → {_saved.name} "
                         f"(+{_epochs_done_total} epochs, +{_rounds_done_total} rounds)")
                    st.success(f"✓ Training complete — model **{_model_name}** saved "
                               f"({_rounds_done_total} round(s), {_epochs_done_total} epochs). "
                               f"Select it in 🔬 Validate to evaluate.")
                except Exception as _se:
                    st.warning(f"Training done but saving model failed: {_se}")
            elif not st.session_state.get("wf_stop_requested"):
                st.success("✓ Walk-forward training complete!")

        elif start_btn and not _data_available:
            st.error("Cannot start: dataset not available.")

        st.divider()
        st.markdown("### 📊 Training Dashboard")
        _render_training_dashboard(active_region())

        # ══════════════════════════════════════════════════════════════════
        # RUN HISTORY — overview + click-to-inspect a single run
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown("### 📚 Run History")

        history = get_all_rounds()
        if history:
            # ---- overview strip ----
            def _mrmse(h):
                v = (h.get("metrics") or {}).get("rmse")
                return v if isinstance(v, (int, float)) and np.isfinite(v) else float("nan")
            _completed = [h for h in history if h.get("status") == "completed"]
            _rmses = [(_mrmse(h), h) for h in _completed if np.isfinite(_mrmse(h))]
            _best = min(_rmses, key=lambda t: t[0]) if _rmses else None
            oc = st.columns(4)
            oc[0].metric("Rounds logged", len(history))
            oc[1].metric("Completed", len(_completed))
            oc[2].metric("Best Val RMSE", f"{_best[0]:.4f}" if _best else "—",
                         help=f"Round {_best[1]['round_num']}" if _best else None)
            _last = history[0]  # get_all_rounds returns newest-first
            oc[3].metric("Latest round", _last["round_num"])

            # ---- BIG cumulative learning graph (x-axis = round over time) ----
            _prog = []
            for h in reversed(history):  # oldest → newest
                if h.get("status") != "completed":
                    continue
                m = h.get("metrics") or {}
                b = m.get("_baselines", {}) or {}
                cur = m.get("_curves", {}) or {}
                _prog.append({
                    "round_num": h["round_num"],
                    "rmse": m.get("rmse"), "mae": m.get("mae"),
                    "csi": m.get("csi"), "bias": m.get("bias"),
                    "pers_rmse": b.get("persistence", {}).get("rmse"),
                    "clim_rmse": b.get("climatology", {}).get("rmse"),
                    "train_loss": (cur.get("train") or [None])[-1],
                    "val_loss": (cur.get("val") or [None])[-1],
                })
            if len(_prog) >= 2:
                st.plotly_chart(VIZ.walkforward_progress_figure(_prog), use_container_width=True)
                st.caption("Each point is one walk-forward round in time. Watch the blue **Model RMSE** "
                           "drop below the dashed **Persistence**/**Climatology** baselines, **CSI** rise, "
                           "losses fall, and **Bias** settle toward 0 — that's the model learning.")
            elif len(_prog) == 1:
                st.info("Only one completed round so far — run a few more to see the learning progression graph.")

            # ---- inspect ONE run (no giant list) ----
            st.markdown("#### 🔍 Inspect a run")
            _opts = history  # newest first
            def _label(h):
                r = _mrmse(h)
                tag = f"RMSE {r:.4f}" if np.isfinite(r) else h.get("status", "—")
                mdl = (h.get("metrics") or {}).get("_model", "")
                return f"Round {h['round_num']} · predict {h['val_start']}→{h['val_end']} · {tag}" + (f" · {mdl}" if mdl else "")
            sel = st.selectbox("Pick a run to expand", _opts, format_func=_label, key="wf_hist_pick")
            if sel:
                m = sel.get("metrics") or {}
                cfg = sel.get("config") or {}
                base = m.get("_baselines", {}) or {}
                st.markdown(f"**Round {sel['round_num']}** — trained `{sel['train_start']}→{sel['train_end']}`, "
                            f"predicted `{sel['val_start']}→{sel['val_end']}` · status **{sel.get('status')}** · "
                            f"{m.get('_seconds', '?')}s · peak GPU {m.get('_peak_gpu_mb', '?')} MB")
                # config chips
                if cfg:
                    st.caption("Config → " + " · ".join(
                        f"{k}={cfg[k]}" for k in ("lr", "epochs", "batch_size", "optimizer_name",
                                                  "scheduler_name", "dropout", "seq_length") if k in cfg))
                # metrics vs baselines
                keys = ["rmse", "mae", "bias", "pearson", "pod", "far", "csi"]
                _f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"
                tblr = pd.DataFrame({
                    "Metric": keys,
                    "This model": [_f(m.get(k)) for k in keys],
                    "Persistence": [_f(base.get("persistence", {}).get(k)) for k in keys],
                    "Climatology": [_f(base.get("climatology", {}).get(k)) for k in keys],
                })
                dc1, dc2 = st.columns([1, 1])
                with dc1:
                    st.dataframe(tblr, hide_index=True, use_container_width=True)
                with dc2:
                    cur = m.get("_curves", {})
                    if cur.get("train"):
                        st.plotly_chart(
                            VIZ.training_curves_figure(cur.get("train", []), cur.get("val", []),
                                                       grad_norms=cur.get("grad") or None,
                                                       lrs=cur.get("lr") or None),
                            use_container_width=True)
                    else:
                        st.caption("No stored loss curve for this (older) run.")

                # ── PART A — Diagnosis (why it plateaus) ──
                with st.expander("🩺 Diagnosis — why did it train this way?", expanded=True):
                    _render_round_diagnostics(m, key_prefix=f"insp_{sel['round_num']}")

            with st.expander("Full round table", expanded=False):
                _tbl = pd.DataFrame([{
                    "Round": h["round_num"],
                    "Val period": f"{h['val_start']}→{h['val_end']}",
                    "RMSE": (f"{_mrmse(h):.4f}" if np.isfinite(_mrmse(h)) else "—"),
                    "Status": h.get("status"),
                } for h in history])
                st.dataframe(_tbl, hide_index=True, use_container_width=True, height=300)

            # ---- actions ----
            act_cols = st.columns(3)
            with act_cols[0]:
                if st.button("🗑️ Clear all history", key="wf_clear_hist"):
                    from training.state import clear_all_rounds
                    clear_all_rounds()
                    st.session_state["wf_current_round"] = 0
                    st.rerun(scope="fragment")
            with act_cols[1]:
                if st.button("🔄 Reset round counter", key="wf_reset_counter"):
                    st.session_state["wf_current_round"] = 0
                    st.toast("Round counter reset to 0.")
        else:
            st.info("No training rounds completed yet. Configure and start above.")


def _render_tab10():
    """🤖 RL Agent tab: Reservoir decision agent under forecast uncertainty."""
    import plotly.graph_objects as go
    from rl.env import ReservoirEnv
    from rl.envs.base_env import list_env_templates, load_env_template
    from rl.agent import train_rl_agent, run_baseline_policy, list_saved_policies
    from rl.eval import evaluate_agent_across_envs

    st.markdown("## 🤖 RL Reservoir Decision Agent")
    st.markdown(
        "Reinforcement Learning agent (PPO) operating on top of forecast uncertainty to decide optimal "
        "weekly reservoir water releases, balancing **downstream irrigation/drinking demand** against "
        "**flood risk** and **drought storage depletion**."
    )

    t_templates, t_train, t_eval, t_library = st.tabs([
        "📜 Environment Templates", "🏋️ Train RL Agent", "🧪 Generalization Test", "📚 Saved Policies"
    ])

    with t_templates:
        st.markdown("### 📜 Reservoir Environment Templates")
        templates = list_env_templates()
        sel_temp = st.selectbox("Select Template", templates, key="rl_temp_pick")
        cfg = load_env_template(sel_temp)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capacity (TCM)", f"{cfg.capacity_tcm:,.0f}")
        c2.metric("Dead Pool (TCM)", f"{cfg.dead_pool_tcm:,.0f}")
        c3.metric("Flood Threshold (TCM)", f"{cfg.flood_threshold_tcm:,.0f}")
        c4.metric("Max Release (TCM/wk)", f"{cfg.max_discharge_tcm_per_week:,.0f}")

        st.caption(f"**Template:** `{cfg.name}` · region `{cfg.region}` · Peak demand week: {cfg.demand_peak_week}")
        
        weeks = np.arange(52)
        demands = [cfg.get_weekly_demand(w) for w in weeks]
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=weeks+1, y=demands, mode="lines+markers", name="Weekly Demand (TCM)"))
        fig_d.update_layout(title=f"Downstream Demand Curve ({cfg.name})", xaxis_title="Week", yaxis_title="Demand (TCM)", height=320)
        st.plotly_chart(fig_d, use_container_width=True)

    with t_train:
        st.markdown("### 🏋️ Train PPO Agent")
        c_e, c_a, c_t = st.columns([1, 1, 1])
        with c_e:
            env_choice = st.selectbox("Environment Template", list_env_templates(), key="rl_train_env")
        with c_a:
            pol_name = st.text_input("Agent Policy Name", f"ppo_{env_choice}_v1", key="rl_pol_name")
        with c_t:
            timesteps = st.number_input("Total Timesteps", 1000, 100000, 10000, step=2000, key="rl_timesteps")

        c_lr, c_g = st.columns(2)
        with c_lr:
            lr_val = st.number_input("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.5f", key="rl_lr")
        with c_g:
            gamma_val = st.number_input("Discount Gamma (γ)", 0.8, 0.999, 0.99, key="rl_gamma")

        if st.button("▶ Train PPO Agent", type="primary", key="rl_train_btn"):
            with st.spinner(f"Training PPO Agent on '{env_choice}' for {timesteps:,} steps…"):
                model, metrics = train_rl_agent(
                    env_name=env_choice, agent_name=pol_name,
                    total_timesteps=timesteps, lr=lr_val, gamma=gamma_val
                )

                env_base = ReservoirEnv(env_choice)
                base_res = run_baseline_policy(env_base, policy_type="inflow")

            st.success(f"✓ Trained PPO Agent **{pol_name}** successfully!")

            m1, m2, m3 = st.columns(3)
            m1.metric("PPO Total Reward", f"{metrics['total_reward']:.2f}")
            m2.metric("Naive Baseline Reward", f"{base_res['total_reward']:.2f}")
            m3.metric("Reward Advantage", f"{(metrics['total_reward'] - base_res['total_reward']):+.2f} pts")

            hist_ppo = pd.DataFrame(metrics["history"])
            hist_base = pd.DataFrame(base_res["history"])

            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=hist_ppo["week"], y=hist_ppo["release"], name="PPO Release", line=dict(color="#39d98a", width=3)))
            fig_rel.add_trace(go.Scatter(x=hist_base["week"], y=hist_base["release"], name="Naive Release (inflow)", line=dict(color="#ff5630", dash="dash")))
            fig_rel.add_trace(go.Scatter(x=hist_ppo["week"], y=hist_ppo["demand"], name="Demand", line=dict(color="#36b37e", width=1)))
            fig_rel.update_layout(title="Weekly Release Schedule vs Inflow & Demand", xaxis_title="Week", yaxis_title="TCM", height=380)
            st.plotly_chart(fig_rel, use_container_width=True)

    with t_eval:
        st.markdown("### 🧪 Multi-Environment Generalization & Stress Test")
        st.caption("Evaluates agent policies across multiple reservoir templates and stress scenarios (dry drought / flood years).")
        
        if st.button("🧪 Run Multi-Env Evaluation", key="rl_eval_btn"):
            policies = list_saved_policies()
            p_name = policies[-1]["agent_name"] if policies else "ppo_mettur_test"
            with st.spinner(f"Evaluating policy '{p_name}' across all templates & stress scenarios…"):
                df_res = evaluate_agent_across_envs(policy_name=p_name)
            st.dataframe(df_res, hide_index=True, use_container_width=True)

    with t_library:
        st.markdown("### 📚 Saved RL Policies Library")
        pols = list_saved_policies()
        if pols:
            st.dataframe(pd.DataFrame(pols), hide_index=True, use_container_width=True)
        else:
            st.info("No RL policies saved yet. Train one in the 'Train RL Agent' tab above.")


# ─────────────────────────────────────────────────────────────────
# Merged 🌐 Explorer tab — three horizontal sub-tabs.
#   Explorer  → the existing daily-explorer body (rain + temp maps
#               for a chosen day; historical or model-predicted).
#   Compare   → two dates side-by-side + plain-English delta summary.
#   Animation → the existing 2D dual-animation body.
# ─────────────────────────────────────────────────────────────────
def _render_compare_subtab():
    """Compare rain + temp between two dates. Emits a plain-English
    'what changed' summary + a per-zone delta table."""
    import datetime as _dt

    st.markdown("### 🔀 Compare two dates")
    st.caption(
        "Pick any two calendar dates in the cube; the panel shows both maps "
        "side-by-side, computes per-cell + per-zone deltas, and writes a "
        "layman summary of what changed."
    )

    region = active_region()
    info = DS.region_info(region, sig=DS.manifest_sig())
    yr_lo, yr_hi = info["years"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Past date**")
        _y1 = st.number_input("year", yr_lo, yr_hi, value=yr_lo,
                                key="cmp_y1", step=1)
        _m1 = st.number_input("month", 1, 12, value=6, key="cmp_m1", step=1)
        _d1 = st.number_input("day", 1, 31, value=1, key="cmp_d1", step=1)
    with col_b:
        st.markdown("**Future / other date**")
        _y2 = st.number_input("year", yr_lo, yr_hi, value=yr_hi,
                                key="cmp_y2", step=1)
        _m2 = st.number_input("month", 1, 12, value=6, key="cmp_m2", step=1)
        _d2 = st.number_input("day", 1, 31, value=1, key="cmp_d2", step=1)

    try:
        date_a = _dt.date(int(_y1), int(_m1), int(_d1))
        date_b = _dt.date(int(_y2), int(_m2), int(_d2))
    except ValueError as e:
        st.error(f"Invalid date: {e}")
        return

    # Load the two days from the cube (small — one 129×135 slice each).
    ds = DS.load_region(region)
    times = ds["time"].values
    times_dt = pd.to_datetime(times)
    def _find(dt):
        i = int(np.abs((times_dt - pd.Timestamp(dt)).total_seconds()).argmin())
        return i, pd.Timestamp(times_dt[i]).date()

    ia, act_a = _find(date_a)
    ib, act_b = _find(date_b)
    if act_a != date_a:
        st.info(f"Nearest cube date to {date_a} is **{act_a}**")
    if act_b != date_b:
        st.info(f"Nearest cube date to {date_b} is **{act_b}**")

    rain_a = ds["rain"].isel(time=ia).values
    rain_b = ds["rain"].isel(time=ib).values
    tmax_a = ds["tmax"].isel(time=ia).values
    tmax_b = ds["tmax"].isel(time=ib).values
    mask_land = ds["mask"].values > 0
    ds.close()

    # Side-by-side maps
    st.markdown("#### Rainfall")
    r_col_a, r_col_b, r_col_c = st.columns(3)
    with r_col_a:
        st.caption(f"{act_a} · rainfall (mm/day)")
        st.plotly_chart(plotly_annual_rain_map(rain_a, str(act_a)),
                          use_container_width=True)
    with r_col_b:
        st.caption(f"{act_b} · rainfall (mm/day)")
        st.plotly_chart(plotly_annual_rain_map(rain_b, str(act_b)),
                          use_container_width=True)
    with r_col_c:
        st.caption(f"Δ rain = {act_b} − {act_a}")
        diff_rain = np.where(mask_land, rain_b - rain_a, np.nan)
        st.plotly_chart(plotly_sensitivity_map(diff_rain, "Δ rain (mm/day)"),
                          use_container_width=True)

    st.markdown("#### Temperature (tmax)")
    t_col_a, t_col_b, t_col_c = st.columns(3)
    with t_col_a:
        st.caption(f"{act_a} · tmax (°C)")
        st.plotly_chart(plotly_annual_temp_map(tmax_a, str(act_a)),
                          use_container_width=True)
    with t_col_b:
        st.caption(f"{act_b} · tmax (°C)")
        st.plotly_chart(plotly_annual_temp_map(tmax_b, str(act_b)),
                          use_container_width=True)
    with t_col_c:
        st.caption(f"Δ tmax = {act_b} − {act_a}")
        diff_temp = np.where(mask_land, tmax_b - tmax_a, np.nan)
        st.plotly_chart(plotly_sensitivity_map(diff_temp, "Δ tmax (°C)"),
                          use_container_width=True)

    # ── Deltas: per-zone table + plain-English summary ──
    try:
        from climate_twin.regions import get_zones
        Z = get_zones()
        rows = []
        for z in Z.zones:
            k = Z.zone_ids.index(z.id)
            w = Z.membership[..., k].astype(np.float32)
            wm = w * mask_land.astype(np.float32)
            wsum = float(wm.sum())
            if wsum <= 0:
                continue
            r_a = float(np.nansum(np.where(np.isfinite(rain_a), rain_a * wm, 0.0)) / wsum)
            r_b = float(np.nansum(np.where(np.isfinite(rain_b), rain_b * wm, 0.0)) / wsum)
            t_a = float(np.nansum(np.where(np.isfinite(tmax_a), tmax_a * wm, 0.0)) / wsum)
            t_b = float(np.nansum(np.where(np.isfinite(tmax_b), tmax_b * wm, 0.0)) / wsum)
            rows.append({
                "zone": z.key,
                "rain_A (mm)": round(r_a, 2),
                "rain_B (mm)": round(r_b, 2),
                "Δ rain": round(r_b - r_a, 2),
                "tmax_A (°C)": round(t_a, 1),
                "tmax_B (°C)": round(t_b, 1),
                "Δ tmax": round(t_b - t_a, 2),
            })
        st.markdown("#### Per-zone deltas")
        import pandas as _pd
        _df = _pd.DataFrame(rows)
        st.dataframe(_df, use_container_width=True, hide_index=True)

        # ── Plain-English summary ──
        def _describe(row: dict) -> str:
            zk = row["zone"]
            dr = row["Δ rain"]; dt = row["Δ tmax"]
            r_word = ("noticeably wetter" if dr > 5 else
                      "slightly wetter"   if dr > 0.5 else
                      "about the same"    if abs(dr) <= 0.5 else
                      "slightly drier"    if dr > -5 else
                      "noticeably drier")
            t_word = ("markedly warmer"  if dt > 2 else
                      "slightly warmer"  if dt > 0.5 else
                      "about the same"   if abs(dt) <= 0.5 else
                      "slightly cooler"  if dt > -2 else
                      "markedly cooler")
            return (
                f"**{zk}** — the later day is **{r_word}** "
                f"({dr:+.1f} mm/day) and **{t_word}** ({dt:+.1f} °C)."
            )

        st.markdown("#### What changed — in plain English")
        st.caption(
            f"Comparing **{act_b}** against **{act_a}** "
            f"({(act_b - act_a).days:+d} days apart)."
        )
        # Highlight the biggest movers
        biggest_wet = max(rows, key=lambda r: r["Δ rain"], default=None)
        biggest_dry = min(rows, key=lambda r: r["Δ rain"], default=None)
        biggest_hot = max(rows, key=lambda r: r["Δ tmax"], default=None)
        biggest_cold = min(rows, key=lambda r: r["Δ tmax"], default=None)
        bullets = []
        if biggest_wet and biggest_wet["Δ rain"] > 0.5:
            bullets.append(
                f"🌧 **{biggest_wet['zone']}** got the wettest bump "
                f"(+{biggest_wet['Δ rain']:.1f} mm/day)."
            )
        if biggest_dry and biggest_dry["Δ rain"] < -0.5:
            bullets.append(
                f"☀ **{biggest_dry['zone']}** dried out the most "
                f"({biggest_dry['Δ rain']:.1f} mm/day)."
            )
        if biggest_hot and biggest_hot["Δ tmax"] > 0.5:
            bullets.append(
                f"🔥 **{biggest_hot['zone']}** heated up the most "
                f"(+{biggest_hot['Δ tmax']:.1f} °C on tmax)."
            )
        if biggest_cold and biggest_cold["Δ tmax"] < -0.5:
            bullets.append(
                f"❄ **{biggest_cold['zone']}** cooled the most "
                f"({biggest_cold['Δ tmax']:.1f} °C on tmax)."
            )
        if bullets:
            for b in bullets:
                st.markdown("- " + b)
        else:
            st.info("Both days look very similar across all 9 zones "
                     "(no zone shifted by more than 0.5 mm or 0.5 °C).")

        with st.expander("Per-zone plain-English breakdown"):
            for row in rows:
                st.markdown("- " + _describe(row))
    except Exception as _e:
        st.warning(f"Zone-delta summary unavailable: {type(_e).__name__}: {_e}")


# ─────────────────────────────────────────────────────────────────
# 🏠 HOME — weather-app-style landing page
# ─────────────────────────────────────────────────────────────────
def _weather_condition(rain_mm: float, tmax_c: float) -> tuple[str, str]:
    """Return (label, emoji) rule-based from daily rain + max temp."""
    r = float(rain_mm) if np.isfinite(rain_mm) else 0.0
    if r >= 20:  return "Thunderstorm", "⛈️"
    if r >= 5:   return "Rainy",         "🌧️"
    if r >= 0.5: return "Light Rain",    "🌦️"
    if tmax_c is not None and np.isfinite(tmax_c) and tmax_c >= 36:
        return "Hot & Sunny", "☀️"
    if r > 0.0:  return "Partly Cloudy", "⛅"
    return "Sunny", "☀️"


def _diurnal_from_daily(tmax: float, tmin: float, hours: int = 24) -> list[float]:
    """Synthetic 24-hour temperature curve given a daily tmax and tmin.

    Simple cosine: min around 6am, max around 3pm. This is a *display*
    approximation — the underlying cube is daily-only. Labelled as such
    in the UI so no one mistakes it for a real hourly forecast.
    """
    if not (np.isfinite(tmax) and np.isfinite(tmin)):
        return [float("nan")] * hours
    mid = 0.5 * (tmax + tmin)
    half = 0.5 * (tmax - tmin)
    out = []
    for h in range(hours):
        # cos phase: -1 at h=6 (min), +1 at h=15 (max), then back
        phase = np.cos((h - 15) / 24.0 * 2 * np.pi)
        out.append(float(mid + half * phase))
    return out


def _render_tab_home():
    """A weather-app-style landing page driven by the india.nc cube."""
    import datetime as _dt

    region = active_region()
    info = DS.region_info(region, sig=DS.manifest_sig())

    # ── Location + latest cube day ────────────────────────────
    ds = DS.load_region(region)
    times = pd.to_datetime(ds["time"].values)
    latest_idx = int(len(times) - 1)
    latest_date = times[latest_idx].date()

    # National mean over land cells (no per-city gauge in the cube yet)
    mask_land = ds["mask"].values > 0
    rain_grid = ds["rain"].isel(time=latest_idx).values
    tmax_grid = ds["tmax"].isel(time=latest_idx).values
    tmin_grid = ds["tmin"].isel(time=latest_idx).values

    def _mean(a):
        v = a[mask_land & np.isfinite(a)]
        return float(v.mean()) if v.size else float("nan")
    rain_now = _mean(rain_grid)
    tmax_now = _mean(tmax_grid)
    tmin_now = _mean(tmin_grid)

    # Feels-like heuristic (heat index approx for temp + humidity proxy).
    # Cube has no humidity — use rain-proxy: wet day → +1.5 °C perceived.
    feels_like = tmax_now + (1.5 if rain_now > 0.5 else 0.0)
    cond_label, cond_emoji = _weather_condition(rain_now, tmax_now)

    # Precipitation "chance" — from area fraction of finite land cells with rain>0
    finite_land = np.isfinite(rain_grid) & mask_land
    if finite_land.any():
        precip_pct = int(100 * (rain_grid[finite_land] > 0.1).mean())
    else:
        precip_pct = 0

    # ── Hero card ─────────────────────────────────────────────
    label = DS.REGIONS[region]["label"]
    stamp = latest_date.strftime("%A, %d %B %Y")
    hero_left, hero_right = st.columns([1.2, 1], gap="large")
    with hero_left:
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.08);
                        border-radius:20px;padding:24px 28px;">
                <div style="color:#8AB4F8;font-size:0.85em;letter-spacing:0.5px;">
                    📍 <b>{label}</b>
                </div>
                <div style="color:rgba(229,233,240,0.55);font-size:0.85em;margin-top:2px;">
                    {stamp} · {_dt.datetime.now().strftime('%H:%M')} IST
                </div>
                <div style="display:flex;align-items:center;gap:16px;margin-top:16px;">
                    <div style="font-size:4.5em;font-weight:700;color:#F0F3F8;line-height:1;">
                        {tmax_now:.0f}°<span style="font-size:0.5em;color:rgba(229,233,240,0.55);">C</span>
                    </div>
                    <div style="font-size:3em;line-height:1;">{cond_emoji}</div>
                </div>
                <div style="font-size:1.4em;font-weight:600;color:#F0F3F8;margin-top:6px;">
                    {cond_label}
                </div>
                <div style="color:rgba(229,233,240,0.6);font-size:0.9em;">
                    Feels like {feels_like:.0f}°C · Rain {rain_now:.1f} mm today
                </div>
                <div style="display:flex;gap:10px;margin-top:16px;">
                    <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
                                border-radius:20px;padding:6px 14px;color:#8AB4F8;font-weight:600;">
                        Min {tmin_now:.0f}°C
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
                                border-radius:20px;padding:6px 14px;color:#F4A34A;font-weight:600;">
                        Max {tmax_now:.0f}°C
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg, rgba(138,180,248,0.15) 0%,
                        rgba(244,163,74,0.12) 100%);
                        border:1px solid rgba(255,255,255,0.08);
                        border-radius:20px;padding:24px;height:100%;
                        display:flex;flex-direction:column;justify-content:center;
                        align-items:center;text-align:center;min-height:210px;">
                <div style="font-size:5em;line-height:1;">🇮🇳</div>
                <div style="color:#FFD9A8;font-weight:700;font-size:1.15em;margin-top:10px;
                            letter-spacing:0.5px;">MausamSetu मौसम सेतु</div>
                <div style="color:rgba(229,233,240,0.7);font-size:0.85em;margin-top:6px;">
                    Zone-aware climate digital twin<br>
                    of India — powered by ISRO / IMD data
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quick stats row (Precipitation / Humidity-proxy / Wind-placeholder) ──
    st.markdown("")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Precipitation chance", f"{precip_pct}%")
    stat_cols[1].metric("Cells wet today",
                         f"{int((rain_grid[finite_land] > 0.1).sum()):,}")
    stat_cols[2].metric("Rain (mean, land)", f"{rain_now:.1f} mm/day")
    stat_cols[3].metric("Δ tmax vs tmin", f"{(tmax_now - tmin_now):.1f} °C")

    # ── Hourly prediction row (synthetic diurnal — labelled) ──────
    st.markdown("### Hourly outlook  <span style='color:#6b7280;font-weight:400;font-size:0.7em;'>"
                 "diurnal reconstruction from daily tmax/tmin — cube is daily-only</span>",
                 unsafe_allow_html=True)
    hours = _diurnal_from_daily(tmax_now, tmin_now)
    labels = [f"{h:02d}" for h in range(24)]
    # Weather emojis vary through the day: warmer hours → sunnier
    def _hour_icon(h: int, t: float) -> str:
        if not np.isfinite(t):
            return "•"
        if 6 <= h <= 18:
            if rain_now >= 5: return "🌧️"
            if rain_now >= 0.5: return "⛅"
            return "☀️"
        return "🌙" if rain_now < 0.5 else "🌧️"

    # Render as a Plotly area chart with the 24 numbers laid out
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=labels, y=hours,
        mode="lines+markers+text",
        line=dict(color="#F4A34A", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(244,163,74,0.15)",
        marker=dict(size=8, color="#FFD9A8"),
        text=[f"{v:.0f}°" for v in hours],
        textposition="top center",
        textfont=dict(color="#F0F3F8", size=10),
        hovertemplate="%{x}:00  %{y:.1f} °C<extra></extra>",
    ))
    fig.update_layout(
        height=200, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(color="rgba(229,233,240,0.65)")),
        yaxis=dict(showgrid=False, showticklabels=False,
                    range=[min(hours) - 3, max(hours) + 4]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Compact icon strip below
    icon_cols = st.columns(24)
    for i, (h, t) in enumerate(zip(range(24), hours)):
        icon_cols[i].markdown(
            f"<div style='text-align:center;font-size:0.75em;'>"
            f"<div style='color:rgba(229,233,240,0.5);'>{h:02d}</div>"
            f"<div style='font-size:1.4em;'>{_hour_icon(h, t)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── 7-day forecast (from the cube tail) ──────────────────────
    st.markdown("### 7-day outlook  <span style='color:#6b7280;font-weight:400;font-size:0.7em;'>"
                 "last 7 days of the cube · national land mean</span>",
                 unsafe_allow_html=True)
    n_days = 7
    tail_start = max(0, latest_idx - n_days + 1)
    tail_idx = np.arange(tail_start, latest_idx + 1)
    tail_rain = ds["rain"].isel(time=tail_idx).values
    tail_tmax = ds["tmax"].isel(time=tail_idx).values
    tail_tmin = ds["tmin"].isel(time=tail_idx).values
    tail_times = pd.to_datetime(ds["time"].isel(time=tail_idx).values)
    ds.close()

    day_cols = st.columns(len(tail_idx))
    for i, col in enumerate(day_cols):
        r = _mean(tail_rain[i])
        tmx = _mean(tail_tmax[i])
        tmn = _mean(tail_tmin[i])
        lab, emoji = _weather_condition(r, tmx)
        dow = tail_times[i].strftime("%a")
        date_short = tail_times[i].strftime("%d %b")
        with col:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.04);
                            border:1px solid rgba(255,255,255,0.08);
                            border-radius:14px;padding:12px 10px;text-align:center;
                            min-height:180px;">
                    <div style="color:#F0F3F8;font-weight:700;">{dow}</div>
                    <div style="color:rgba(229,233,240,0.5);font-size:0.75em;">{date_short}</div>
                    <div style="font-size:2em;margin:8px 0;">{emoji}</div>
                    <div style="color:#F4A34A;font-weight:700;font-size:1.1em;">{tmx:.0f}°</div>
                    <div style="color:#8AB4F8;font-size:0.85em;">{tmn:.0f}°</div>
                    <div style="color:rgba(229,233,240,0.55);font-size:0.7em;margin-top:6px;">
                        💧 {r:.1f} mm
                    </div>
                    <div style="color:rgba(229,233,240,0.5);font-size:0.7em;">{lab}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.caption(
        f"Data source: `data/processed/{region}.nc` "
        f"(manifest sig `{DS.manifest_sig()}`) · "
        f"cube covers {info['years'][0]}–{info['years'][1]}. "
        f"Hourly values are a **display reconstruction** from daily "
        f"tmax/tmin; the underlying dataset is daily-only."
    )


def _render_tab_explorer():
    """Merged Explorer — three horizontal sub-tabs, radio-driven so
    only the ACTIVE sub-body runs (unlike st.tabs which executes every
    child on every rerun)."""
    _SUB = ["🏠 Explorer", "🔀 Compare", "🎞 Animation"]
    try:
        sub = st.segmented_control(
            "Explorer view", _SUB, default=_SUB[0],
            key="explorer_subtab", label_visibility="collapsed",
        )
    except Exception:
        sub = st.radio(
            "Explorer view", _SUB, horizontal=True,
            key="explorer_subtab", label_visibility="collapsed",
        )
    st.markdown("---")
    if sub == _SUB[1]:
        _render_compare_subtab()
    elif sub == _SUB[2]:
        _render_tab6()
    else:
        _render_tab1()


# ── Render ONLY the active section (its fragment); the others never execute. ──
_TAB_RENDERERS = {
    _TAB_LABELS[0]: _render_tab_home,           # 🏠 Home (weather-app landing)
    _TAB_LABELS[1]: _render_tab_explorer,       # 🌐 Explorer (merged)
    _TAB_LABELS[2]: _render_tab3,               # ±1°C What-If
    _TAB_LABELS[3]: _render_tab4,               # Model Comparison
    _TAB_LABELS[4]: _render_tab5,               # Zone Projections
    _TAB_LABELS[5]: _render_tab7,               # Climate Spirals
    _TAB_LABELS[6]: _render_tab8,               # Deep Analytics
    _TAB_LABELS[7]: _render_tab9,               # 🔥 Training
    _TAB_LABELS[8]: _render_tab_validation,     # 🔍 Validation
    _TAB_LABELS[9]: _render_tab10,              # 🤖 RL Agent
}
_TAB_RENDERERS.get(_active_tab, _render_tab_home)()


# ── 🐢 Perf panel (sidebar toggle; renders this rerun's timings, slowest first) ──
render_perf_panel()

