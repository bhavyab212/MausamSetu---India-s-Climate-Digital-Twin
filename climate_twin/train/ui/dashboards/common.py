"""
train.ui.dashboards.common — shared helpers for the two dashboards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from climate_twin.runtime import (
    DeviceInfo,
    RuntimeCudaRequired,
    TrainingExecutor,
    describe_gpu,
    gpu_utilization_snapshot,
)

DEVICE_BANNER_CSS = """
<style>
.mausam-banner {
  padding: 10px 14px; border-radius: 6px; border: 1px solid;
  font-family: monospace; font-size: 0.9em; margin-bottom: 10px;
}
.mausam-banner-ok    { background:#0f2a1d; border-color:#28a745; color:#c9f7d5; }
.mausam-banner-warn  { background:#332506; border-color:#f4a34a; color:#ffe8be; }
.mausam-banner-err   { background:#3a0d0d; border-color:#e63946; color:#ffd8d8; }
.mausam-kv { display:inline-block; margin-right:16px; }
.mausam-kv b { color:#F4A34A; }
</style>
"""


# ---------------------------------------------------------------------------
# Executor session-state helpers
# ---------------------------------------------------------------------------
def get_or_create_executor(key: str = "climate_twin_executor") -> TrainingExecutor:
    """Return the singleton training executor for this Streamlit session."""
    if key not in st.session_state:
        st.session_state[key] = TrainingExecutor()
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Device banner (GPU / CPU refusal)
# ---------------------------------------------------------------------------
def render_device_banner() -> DeviceInfo:
    """Header banner: green (GPU present), red (CPU-only) + explanation."""
    st.markdown(DEVICE_BANNER_CSS, unsafe_allow_html=True)
    gpu = describe_gpu()
    if gpu.available:
        snap = gpu_utilization_snapshot()
        util = f"{snap['util_percent']:.0f}%" if snap.get('util_percent') is not None else "—"
        mem = ""
        if snap.get('memory_used_mb') is not None and snap.get('memory_total_mb'):
            mem = (f"{snap['memory_used_mb'] / 1024:.1f} / "
                   f"{snap['memory_total_mb'] / 1024:.1f} GB")
        temp = f"{snap['temperature_c']:.0f}°C" if snap.get('temperature_c') is not None else ""
        html = (
            f'<div class="mausam-banner mausam-banner-ok">'
            f'<span class="mausam-kv"><b>device</b> cuda:{gpu.device_index}</span>'
            f'<span class="mausam-kv"><b>gpu</b> {gpu.name}</span>'
            f'<span class="mausam-kv"><b>util</b> {util}</span>'
            f'<span class="mausam-kv"><b>mem</b> {mem}</span>'
            f'<span class="mausam-kv"><b>temp</b> {temp}</span>'
            f'<span class="mausam-kv"><b>torch</b> {gpu.torch_version} '
            f'(cuda {gpu.torch_cuda_version})</span>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="mausam-banner mausam-banner-err">'
            '<b>⛔ CUDA not available.</b> Training and inference are disabled. '
            'Install a CUDA-enabled PyTorch build (e.g. torch 2.5.1+cu121) and '
            'verify with <code>python -c "import torch; print(torch.cuda.is_available())"</code>.'
            '</div>',
            unsafe_allow_html=True,
        )
    return gpu


# ---------------------------------------------------------------------------
# Config picker + editor
# ---------------------------------------------------------------------------
def config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config"


def experiments_dir() -> Path:
    return config_dir() / "experiments"


def list_experiments() -> list[Path]:
    return sorted(experiments_dir().glob("*.yaml"))


def render_config_editor(key: str = "cfg_edit"):
    """Config dropdown + inline editor + validation. Returns (cfg, path or None)
    when a valid config is loaded, else (None, None)."""
    from climate_twin.train.config import load_config

    yamls = list_experiments()
    if not yamls:
        st.warning("No experiment yamls under `climate_twin/train/config/experiments/`.")
        return None, None

    names = [p.name for p in yamls]
    default_ix = names.index("base.yaml") if "base.yaml" in names else 0
    pick = st.selectbox("Experiment yaml", names, index=default_ix, key=f"{key}_pick")
    path = experiments_dir() / pick

    with st.expander("⚙ Edit YAML", expanded=False):
        raw = path.read_text(encoding="utf-8")
        edited = st.text_area(
            "YAML source", raw, height=340, key=f"{key}_yaml",
            help="Any change is validated before running. Save-as writes a new file.",
        )
        col_a, col_b, col_c = st.columns([1, 1, 3])
        with col_a:
            if st.button("✔ Validate", key=f"{key}_valid"):
                try:
                    # Write to a temp file first so extends: resolves
                    tmp = path.with_suffix(".yaml.__tmp__")
                    tmp.write_text(edited, encoding="utf-8")
                    load_config(tmp)
                    tmp.unlink(missing_ok=True)
                    st.success("Config is valid.")
                except Exception as e:
                    st.error(f"❌ {type(e).__name__}: {e}")
        with col_b:
            if st.button("💾 Save changes", key=f"{key}_save"):
                path.write_text(edited, encoding="utf-8")
                st.success(f"Saved to {path.name}")
                st.rerun(scope="fragment")
        with col_c:
            new_name = st.text_input("Save as new (name)", key=f"{key}_saveas",
                                      placeholder="my_experiment")
            if st.button("💾 Save as new", key=f"{key}_saveas_btn") and new_name:
                new_path = experiments_dir() / f"{new_name.strip()}.yaml"
                new_path.write_text(edited, encoding="utf-8")
                st.success(f"Saved to {new_path.name}")
                st.rerun(scope="fragment")

    # Attempt to load (from the on-disk file, not the edited buffer)
    try:
        cfg = load_config(path)
        return cfg, path
    except Exception as e:
        st.error(f"❌ config invalid: {type(e).__name__}: {e}")
        return None, path


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def fmt_hms(seconds: float | int | None) -> str:
    if seconds is None or seconds != seconds:      # NaN
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def fmt_num(v: Any, decimals: int = 4, dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
        if f != f:                                    # NaN
            return dash
        return f"{f:.{decimals}f}"
    except Exception:
        return str(v) if isinstance(v, str) else dash
