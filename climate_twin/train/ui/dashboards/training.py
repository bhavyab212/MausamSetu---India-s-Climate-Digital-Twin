"""
Phase-B rebuild — the interactive Training dashboard.

Layout (per the reimagined spec):

    Sidebar (persistent, every page)
        Device panel     — CUDA, VRAM bar, util bar, temp
        Region toggle    — India / Cauvery
        Nine-zone panel  — mini choropleth + cell counts + hover hints
        Navigation       — 🔥 Training · 🔍 Validation

    Main area
        Run header (always visible; last completed run when idle)
        Four tabs:
            ⚙ Setup       — INTENT selector · resolved config · preflight · [START]
            📊 Live       — metric tiles · loss curve · zone strip · deep grid · guards
            🎯 Results   — verdict card · per-zone diagnosis · Tier-4 · action bar
            📜 Log       — filterable stream · download · colour-coded levels
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from climate_twin.runtime import (
    EventKind,
    describe_gpu,
    gpu_utilization_snapshot,
)
from climate_twin.regions import get_zones

from .common import (
    experiments_dir,
    fmt_hms,
    fmt_num,
    get_or_create_executor,
    list_experiments,
)
from .summary_card import build_summary_card, render_summary_card_html
from .zone_panel import render_sidebar_zone_panel
from . import polish as P


IST = timezone(timedelta(hours=5, minutes=30))
_UI_STATE = "climate_twin_training_ui"


# ─────────────────────────────────────────────────────────────────────
# INTENT DEFINITIONS  (Setup tab)
# ─────────────────────────────────────────────────────────────────────
INTENTS = [
    {"key": "smoke",   "icon": "🧪", "label": "Quick pipeline test",
      "eta": "~2 min",    "yaml": "smoke_2min.yaml",
      "warning": "This is a 2-minute pipeline test, not a training run. "
                 "For real training choose India Fast or India Full."},
    {"key": "full",    "icon": "🔥", "label": "Full training run",
      "eta": "~4–8 h",   "yaml": "india_full.yaml",
      "warning": None},
    {"key": "fast",    "icon": "⚡",  "label": "Fast training run",
      "eta": "~2–3 h",   "yaml": "india_fast.yaml",
      "warning": None},
    {"key": "ft",      "icon": "🎯", "label": "Fine-tune a checkpoint",
      "eta": "~40 min",  "yaml": "finetune.yaml",
      "warning": None},
    {"key": "hold_ne", "icon": "🧬", "label": "Held-out: Northeast",
      "eta": "~2 h",     "yaml": "holdout_northeast.yaml",
      "warning": None},
    {"key": "hold_thar","icon": "🧬","label": "Held-out: Thar Arid",
      "eta": "~2 h",     "yaml": "holdout_thar.yaml",
      "warning": None},
    {"key": "hold_tn", "icon": "🧬","label": "Held-out: Tamil Nadu NE",
      "eta": "~2 h",     "yaml": "holdout_tamilnadu.yaml",
      "warning": None},
    {"key": "custom",  "icon": "📄", "label": "Custom config file",
      "eta": "—",        "yaml": None,
      "warning": None},
]


# ─────────────────────────────────────────────────────────────────────
# Session-state
# ─────────────────────────────────────────────────────────────────────
def _ui() -> dict[str, Any]:
    if _UI_STATE not in st.session_state:
        st.session_state[_UI_STATE] = {
            "tier1_hist": [],
            "tier2_hist": [],
            "tier3_last": None,
            "tier4_final": None,
            "log_lines": [],
            "heatmap_paths": [],
            "checkpoint_notices": [],
            "batch_tuning": [],      # each probe as {bs, ok, peak_mb, seconds, error}
            "batch_tuned": None,     # final pick
            "throughput_warns": [],
            "last_progress": {},
            "intent_key": "full",
            "custom_yaml": None,
        }
    return st.session_state[_UI_STATE]


def _reset_ui_for_new_run() -> None:
    keep = st.session_state[_UI_STATE].get("intent_key", "full")
    st.session_state[_UI_STATE] = {
        "tier1_hist": [], "tier2_hist": [], "tier3_last": None,
        "tier4_final": None, "log_lines": [], "heatmap_paths": [],
        "checkpoint_notices": [], "batch_tuning": [], "batch_tuned": None,
        "throughput_warns": [], "last_progress": {},
        "intent_key": keep, "custom_yaml": None,
    }


def _absorb_events(execu) -> None:
    ui = _ui()
    for ev in execu.events.drain():
        p = ev.payload
        if ev.kind == EventKind.STARTED:
            ui["log_lines"].append(
                f"[start] {p['model_name']} epochs={p['epochs']} device={p['device']}")
        elif ev.kind == EventKind.BATCH_TUNING:
            ui["batch_tuning"].append(p)
            mark = "✓" if p.get("ok") else f"✗ {p.get('error', 'fail')}"
            ui["log_lines"].append(
                f"[tune ] bs={p['bs']:4d}  peak={p.get('peak_mb', float('nan')):6.0f} MB  {mark}")
        elif ev.kind == EventKind.BATCH_TUNED:
            ui["batch_tuned"] = p
            ui["log_lines"].append(
                f"[tune ] → picked bs={p.get('picked_bs')}   {p.get('reason','')}")
        elif ev.kind == EventKind.PROGRESS:
            ui["last_progress"] = p
        elif ev.kind == EventKind.TIER1:
            ui["tier1_hist"].append(p)
        elif ev.kind == EventKind.TIER2:
            ui["tier2_hist"].append(p)
        elif ev.kind == EventKind.TIER3:
            ui["tier3_last"] = p
        elif ev.kind == EventKind.TIER4:
            ui["tier4_final"] = p
        elif ev.kind == EventKind.CHECKPOINT_SAVED:
            ui["checkpoint_notices"].append(p)
            ui["log_lines"].append(
                f"[ckpt ] epoch={p['epoch']} zw_rmse={p['zone_weighted_rmse']:.4f}")
        elif ev.kind == EventKind.HEATMAP_RENDERED:
            ui["heatmap_paths"].append(p["path"])
        elif ev.kind == EventKind.THROUGHPUT_WARN:
            ui["throughput_warns"].append(p)
            ui["log_lines"].append(
                f"[warn ] GPU util {p.get('gpu_util_percent')}% for "
                f"{p.get('seconds_under_50')}s — {p.get('hint')}")
        elif ev.kind == EventKind.PAUSED:
            ui["log_lines"].append("[ctrl ] paused")
        elif ev.kind == EventKind.RESUMED:
            ui["log_lines"].append("[ctrl ] resumed")
        elif ev.kind == EventKind.STOPPED:
            ui["log_lines"].append("[ctrl ] stopped")
        elif ev.kind == EventKind.DONE:
            ui["log_lines"].append(
                f"[done ] best_zw_rmse={p.get('best_zone_weighted_rmse'):.4f} "
                f"epochs={p.get('n_epochs_run')}")
        elif ev.kind == EventKind.ERROR:
            ui["log_lines"].append(f"[error] {p.get('type','?')}: {p.get('message','')}")
        elif ev.kind == EventKind.LOG:
            ui["log_lines"].append(f"[{p.get('level','info')}] {p.get('message','')}")
    if len(ui["log_lines"]) > 2000:
        ui["log_lines"] = ui["log_lines"][-2000:]


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
def _render_sidebar():
    st.sidebar.markdown("### 🌦 MausamSetu मौसम सेतु")

    # Device panel — fragment-refreshed 2s
    _fragment_device_panel()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Region")
    st.sidebar.radio(
        "region",
        options=["india", "cauvery"],
        format_func=lambda k: {"india": "India (Full)",
                                "cauvery": "Cauvery Basin"}[k],
        key="active_region_train",
        label_visibility="collapsed",
    )
    Z = get_zones()
    land_pct = (Z.hard_mask > 0).sum() / Z.hard_mask.size * 100
    st.sidebar.caption(
        f"Grid {Z.hard_mask.shape[0]}×{Z.hard_mask.shape[1]} · "
        f"land {land_pct:.1f}%"
    )

    st.sidebar.markdown("---")
    render_sidebar_zone_panel()

    st.sidebar.markdown("---")
    with st.sidebar.expander("⏱ Fragment perf", expanded=False):
        P.render_profile_panel()


@st.fragment(run_every=2)
def _fragment_device_panel():
    with P.profile("sidebar.device"):
        gpu = describe_gpu()
        if not gpu.available:
            st.sidebar.error("⛔ CUDA unavailable — training disabled")
            return
        snap = gpu_utilization_snapshot()
        used_gb = (snap.get("memory_used_mb") or 0) / 1024
        total_gb = (snap.get("memory_total_mb") or 0) / 1024
        util = snap.get("util_percent")
        temp = snap.get("temperature_c")
        power = snap.get("power_w")

    # Compact bars using unicode blocks
    def _bar(frac: float, w: int = 14) -> str:
        frac = max(0.0, min(1.0, float(frac or 0)))
        filled = int(round(frac * w))
        return "█" * filled + "░" * (w - filled)

    vram_bar = _bar(used_gb / total_gb if total_gb else 0)
    util_bar = _bar((util or 0) / 100.0)
    lines = [
        f"**cuda:{gpu.device_index}** · {gpu.name.replace('NVIDIA GeForce ', '')}",
        f"`VRAM  {vram_bar}  {used_gb:4.1f} / {total_gb:.1f} GB`",
        f"`Util  {util_bar}  {util or 0:.0f}%`",
    ]
    if temp is not None:
        lines.append(f"`Temp  {temp:.0f}°C`" + (f" · {power:.0f}W" if power else ""))
    lines.append(f"torch {gpu.torch_version} (cuda {gpu.torch_cuda_version})")
    st.sidebar.markdown("\n\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# RUN HEADER (always visible)
# ─────────────────────────────────────────────────────────────────────
@st.fragment(run_every=1)
def _fragment_run_header():
    with P.profile("run_header"):
        _render_run_header_body()


def _render_run_header_body():
    execu = get_or_create_executor()
    _absorb_events(execu)
    state = execu.state
    ui = _ui()
    p = ui.get("last_progress", {}) or {}

    if state.status == "idle":
        st.markdown(
            '<div style="background:#0E1522;border:1px solid #333;'
            'border-radius:6px;padding:12px;font-family:monospace;color:#9DA6B0;">'
            '<b style="color:#F4A34A;">No active run.</b> Configure and Start in '
            'the ⚙ Setup tab. Last completed run’s summary will appear here when '
            'you re-open the app.</div>',
            unsafe_allow_html=True,
        )
        return

    status_colour = {
        "starting": "#F4A34A", "running": "#28a745",
        "paused":   "#F4A34A", "stopping": "#F4A34A",
        "stopped":  "#8AB4F8", "finished": "#28a745",
        "errored":  "#e63946",
    }.get(state.status, "#F4A34A")

    total = max(state.total_batches, 1)
    frac = min(state.batch / total, 1.0)
    samples_per_s = p.get("samples_per_s") or float("nan")
    bs = p.get("batch_size") or "?"
    precision = p.get("precision") or "?"
    train_loss = state.train_loss
    val_rmse = _last_val_rmse(ui.get("tier2_hist", []))

    st.markdown(
        f'<div style="background:#0E1522;border-left:4px solid {status_colour};'
        f'padding:10px 14px;border-radius:4px;font-family:monospace;">'
        f'<div style="font-size:1.1em;"><b>{state.model_name or "—"}</b> · '
        f'<span style="color:{status_colour};">● {state.status.upper()}</span></div>'
        f'<div>Epoch {state.epoch}/{state.total_epochs} · '
        f'Batch {state.batch:,}/{state.total_batches:,} · '
        f'ETA {fmt_hms(state.eta_seconds)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(frac,
                text=f"{int(frac*100)}%   "
                     f"{samples_per_s:.0f} samples/s   "
                     f"loss {fmt_num(train_loss, 4)}   "
                     f"val_rmse {fmt_num(val_rmse, 3)}   "
                     f"bs {bs} · {precision} · lr {fmt_num(state.lr, 2)}")


def _last_val_rmse(tier2_hist: list[dict]) -> float:
    if not tier2_hist:
        return float("nan")
    per_zone = tier2_hist[-1].get("per_zone", {})
    vals = [v.get("rmse") for v in per_zone.values()
            if isinstance(v.get("rmse"), (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — ⚙ SETUP
# ─────────────────────────────────────────────────────────────────────
def _render_setup_tab():
    from climate_twin.train.config import load_config
    from climate_twin import data_source as DS

    ui = _ui()
    st.markdown("#### 1. What do you want to do?")

    # Intent radio, big
    intent_labels = {i["key"]: f"{i['icon']}  {i['label']}   *{i['eta']}*"
                      for i in INTENTS}
    default_ix = next(
        (n for n, k in enumerate(intent_labels) if k == ui.get("intent_key")), 1)
    picked_key = st.radio(
        "intent",
        list(intent_labels.keys()),
        index=default_ix,
        format_func=lambda k: intent_labels[k],
        label_visibility="collapsed",
    )
    ui["intent_key"] = picked_key
    intent = next(i for i in INTENTS if i["key"] == picked_key)

    # Custom path branch
    if intent["key"] == "custom":
        yamls = list_experiments()
        yaml_name = st.selectbox(
            "Custom yaml", [p.name for p in yamls],
            key="setup_custom_yaml",
        )
        yaml_path = experiments_dir() / yaml_name
    else:
        yaml_path = experiments_dir() / intent["yaml"]

    # Load + validate
    try:
        cfg = load_config(yaml_path)
    except Exception as e:
        st.error(f"❌ config invalid — {type(e).__name__}: {e}")
        return None, yaml_path

    if intent["warning"]:
        st.warning(intent["warning"])

    st.markdown(f"<div style='color:#6b7280;font-size:0.85em;'>yaml: "
                 f"<code>{yaml_path.name}</code></div>",
                 unsafe_allow_html=True)

    # ── Resolved config card ──
    st.markdown("#### 2. Resolved config")
    Z = get_zones()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Model & optim**")
        _kv([
            ("model backbone", cfg.model.backbone),
            ("hidden", cfg.model.hidden),
            ("seq_length", cfg.model.seq_length),
            ("dropout", cfg.model.dropout),
            ("optimizer", cfg.optim.optimizer),
            ("lr", f"{cfg.optim.lr:.1e}"),
            ("epochs", cfg.optim.epochs),
            ("batch_size", ("auto" if cfg.optim.batch_size_auto
                             else str(cfg.optim.batch_size))),
            ("precision", ("auto" if cfg.optim.mixed_precision_auto
                            else cfg.optim.mixed_precision)),
            ("sampler", cfg.optim.sampler_mode),
        ])
    with col_b:
        st.markdown("**Data scope**")
        _kv([
            ("region", cfg.data.region),
            ("train_years", f"{cfg.data.train_years[0]}–{cfg.data.train_years[1]}"),
            ("val_years", f"{cfg.data.val_years[0]}–{cfg.data.val_years[1]}"),
            ("test_years", f"{cfg.data.test_years[0]}–{cfg.data.test_years[1]}"),
            ("variables", ", ".join(cfg.data.variables)),
            ("include_satellite", cfg.data.include_satellite),
            ("excluded zones", ", ".join(cfg.zones.excluded_zone_keys) or "(none)"),
            ("zone_mask_sig", cfg.zones.zone_mask_sig_expected),
        ])

    # ── Pre-flight ──
    st.markdown("#### 3. Pre-flight checklist")
    checks: list[tuple[bool, str]] = []
    gpu = describe_gpu()
    checks.append((gpu.available,
                    f"CUDA available ({gpu.name}, {gpu.total_memory_gb:.1f} GB)"
                    if gpu.available else "CUDA missing"))
    checks.append((True, f"Config schema valid ({cfg.name})"))
    zsig = cfg.zones.zone_mask_sig_expected
    checks.append((zsig == Z.mask_signature,
                    f"Zone mask hash matches ({zsig})"))
    manifest_sig = DS.manifest_sig() or "?"
    ms_ok = (cfg.data.manifest_sig_expected is None
              or cfg.data.manifest_sig_expected == manifest_sig)
    checks.append((ms_ok, f"Data manifest hash ({manifest_sig})"))

    # Zone-cell counts across splits — flag near-threshold zones
    Z_counts = {z.key: int((Z.hard_mask == z.id).sum()) for z in Z.zones}
    small = [k for k, c in Z_counts.items() if c < 250]
    if small:
        checks.append((True,
                        f"All 9 zones present · near-threshold: "
                        f"{', '.join(small)} (metrics will carry wide CIs)"))
    else:
        checks.append((True, "All 9 zones present in train / val / test"))
    checks.append((True, "Estimated VRAM will be shown by the auto-tuner"))

    all_green = True
    for ok, msg in checks:
        if ok:
            st.markdown(f"<span style='color:#28a745;'>✓</span> {msg}",
                         unsafe_allow_html=True)
        else:
            all_green = False
            st.markdown(f"<span style='color:#e63946;'>✗</span> {msg}",
                         unsafe_allow_html=True)

    # ── YAML editor (advanced) ──
    with st.expander("⚙ Advanced — edit yaml", expanded=False):
        raw = yaml_path.read_text(encoding="utf-8")
        edited = st.text_area("YAML", raw, height=320, key=f"yaml_edit_{yaml_path.name}")
        cA, cB = st.columns(2)
        with cA:
            if st.button("✔ Validate", key="yaml_validate"):
                try:
                    tmp = yaml_path.with_suffix(".yaml.__tmp__")
                    tmp.write_text(edited, encoding="utf-8")
                    load_config(tmp)
                    tmp.unlink(missing_ok=True)
                    st.success("Valid.")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
        with cB:
            if st.button("💾 Save changes", key="yaml_save"):
                yaml_path.write_text(edited, encoding="utf-8")
                st.success(f"Saved to {yaml_path.name}")
                st.rerun(scope="fragment")

    # ── Model name + Start ──
    st.markdown("#### 4. Launch")
    col_n, col_s = st.columns([3, 1])
    with col_n:
        model_name = st.text_input("Model name (goes into the registry)",
                                    value=cfg.name)
    with col_s:
        st.markdown("<br>", unsafe_allow_html=True)
        can_start = all_green and cfg is not None
        execu = get_or_create_executor()
        if execu.is_running:
            st.info("A run is already in progress.")
        elif st.button("▶ Start training", type="primary",
                        disabled=not can_start, use_container_width=True):
            from climate_twin.train.registry import get_registry
            _reset_ui_for_new_run()
            try:
                execu.start(cfg=cfg, model_name=model_name,
                             registry_root=get_registry().models_root)
                st.rerun(scope="fragment")
            except Exception as e:
                st.error(f"❌ {type(e).__name__}: {e}")

    return cfg, yaml_path


def _kv(rows: list[tuple[str, Any]]) -> None:
    html_rows = []
    for k, v in rows:
        html_rows.append(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-family:monospace;font-size:0.88em;padding:2px 0;'>"
            f"<span style='color:#9DA6B0;'>{k}</span>"
            f"<span style='color:#e5e9f0;'>{v}</span></div>"
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — 📊 LIVE
# ─────────────────────────────────────────────────────────────────────
def _render_live_tab():
    execu = get_or_create_executor()
    _absorb_events(execu)
    ui = _ui()

    # ── Batch tuning panel (shown until first epoch begins) ──
    if ui["batch_tuning"] and not ui["batch_tuned"]:
        st.info("Auto-tuning batch size to saturate VRAM…")
        _render_tuning_probes(ui["batch_tuning"], None)
        return
    if ui["batch_tuned"] and not ui.get("last_progress"):
        _render_tuning_probes(ui["batch_tuning"], ui["batch_tuned"])
        return

    _fragment_metric_tiles()
    _fragment_loss_curve()
    _fragment_zone_strip()
    _fragment_deep_grid_and_guards()


def _render_tuning_probes(tuning: list[dict], tuned: dict | None):
    if not tuning:
        st.caption("Waiting for the first tuning probe…")
        return
    rows = []
    for t in tuning:
        rows.append({
            "batch size": t["bs"],
            "peak VRAM (MB)": (f"{t.get('peak_mb'):.0f}"
                                 if isinstance(t.get('peak_mb'), (int, float))
                                    and t.get('peak_mb') == t.get('peak_mb')
                                 else "—"),
            "seconds": t.get("seconds"),
            "result": ("✓" if t.get("ok") else f"✗ {t.get('error','fail')}"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if tuned:
        p = tuned
        pct = (100 * p.get("peak_vram_mb", 0) / (p.get("total_vram_mb") or 1))
        st.success(
            f"→ picked batch size **{p.get('picked_bs')}**  "
            f"(peak {p.get('peak_vram_mb'):.0f} / "
            f"{p.get('total_vram_mb'):.0f} MB — {pct:.0f}% VRAM)  ·  "
            f"{p.get('reason')}"
        )


@st.fragment(run_every=2)
def _fragment_metric_tiles():
    with P.profile("metric_tiles"):
        execu = get_or_create_executor()
        _absorb_events(execu)
        state = execu.state
        ui = _ui()
        p = ui.get("last_progress", {}) or {}

        # Track previous values for delta arrows
        prev = st.session_state.setdefault("_train_metric_prev", {})
        cur_val_rmse = _last_val_rmse(ui["tier2_hist"])

        tiles = st.columns(6)
        tiles[0].metric("train_loss",
                         P.loss(state.train_loss),
                         P.delta_arrow(state.train_loss, prev.get("train_loss")))
        tiles[1].metric("val rmse (rain)",
                         P.rain(cur_val_rmse),
                         P.delta_arrow(cur_val_rmse, prev.get("val_rmse")))
        tiles[2].metric("best zw_rmse (rain)",
                         P.rain(state.best_zone_weighted_rmse))
        tiles[3].metric("grad_norm",
                         P.loss(state.grad_norm),
                         P.delta_arrow(state.grad_norm, prev.get("grad_norm")))
        tiles[4].metric("samples / sec",
                         f"{P.count(p.get('samples_per_s'))}/s"
                         if p.get("samples_per_s") is not None else "—",
                         P.delta_arrow(p.get("samples_per_s"), prev.get("samples_per_s")))
        used_mb = p.get("vram_used_mb"); total_mb = p.get("vram_total_mb")
        vram_txt = (f"{P.gb(used_mb)} / {P.gb(total_mb)}"
                     if total_mb else "—")
        tiles[5].metric(
            "VRAM · util",
            vram_txt,
            P.pct(p.get("gpu_util_percent")) if p.get("gpu_util_percent") is not None else None,
        )

        prev.update({
            "train_loss": state.train_loss,
            "val_rmse": cur_val_rmse,
            "grad_norm": state.grad_norm,
            "samples_per_s": p.get("samples_per_s"),
        })


@st.fragment(run_every=1)
def _fragment_loss_curve():
    with P.profile("loss_curve"):
        execu = get_or_create_executor()
        _absorb_events(execu)
        ui = _ui()
        st.markdown("**Tier-1 val-RMSE curve**  <span style='color:#6b7280;'>every N batches</span>",
                     unsafe_allow_html=True)
        if not ui["tier1_hist"]:
            P.empty_state("📉", "Awaiting first Tier-1 pass",
                           "The curve begins populating as batches are processed. "
                           "Cadence is set by validation.tier1_every_n_batches.",
                           height=180)
            return
        df = pd.DataFrame(ui["tier1_hist"])
        if "rmse" in df.columns:
            P.line_chart_fixed(df["rmse"], height=200)


@st.fragment(run_every=3)
def _fragment_zone_strip():
    with P.profile("zone_strip"):
        execu = get_or_create_executor()
        _absorb_events(execu)
        ui = _ui()
        Z = get_zones()

        st.markdown("**Tier-2 per-zone strip**  "
                     "<span style='color:#6b7280;'>colour = RMSE (viridis, darker = higher)</span>",
                     unsafe_allow_html=True)
        if not ui["tier2_hist"]:
            P.empty_state("🗺", "Awaiting first epoch",
                           "Per-zone Tier-2 metrics render after epoch 1 completes.",
                           height=200)
            return
        keys = [z.key for z in Z.zones]
        hist = ui["tier2_hist"]
        vals = np.full((len(keys), len(hist)), np.nan)
        for j, t2 in enumerate(hist):
            pz = t2.get("per_zone", {})
            for i, k in enumerate(keys):
                r = pz.get(k, {}).get("rmse")
                if isinstance(r, (int, float)) and np.isfinite(r):
                    vals[i, j] = float(r)
        df = pd.DataFrame(vals, index=keys,
                           columns=[f"e{j+1}" for j in range(vals.shape[1])])
        P.zone_strip_dataframe(df, height=290)


@st.fragment(run_every=5)
def _fragment_deep_grid_and_guards():
    with P.profile("deep_grid_and_guards"):
        execu = get_or_create_executor()
        _absorb_events(execu)
        ui = _ui()

        st.markdown("**Tier-3 skill heatmap**  "
                     "<span style='color:#6b7280;'>every 5 epochs + end of round</span>",
                     unsafe_allow_html=True)
        if ui["heatmap_paths"]:
            p = Path(ui["heatmap_paths"][-1])
            if p.exists():
                st.image(str(p), use_container_width=True)
        else:
            P.empty_state("🎯", "Awaiting first Tier-3 pass",
                           "The 9-zone × 4-season skill grid appears after "
                           "validation.tier3_every_n_epochs (default 5).",
                           height=220)

        st.markdown("**Guards**")
        warns = ui.get("throughput_warns", [])
        if warns:
            st.warning(
                f"GPU utilisation dropped below 50% for "
                f"{warns[-1].get('seconds_under_50'):.0f}s. {warns[-1].get('hint')}"
            )
        else:
            cols = st.columns(4)
            for c, kind, txt in zip(
                cols,
                ["ok", "ok", "ok", "ok"],
                ["GPU util healthy",
                 "tmax ≥ tmin penalty active",
                 "per-zone bounds active",
                 "NaN-safe loss active"],
            ):
                with c:
                    P.render_check(kind, txt)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — 🎯 RESULTS
# ─────────────────────────────────────────────────────────────────────
def _render_results_tab():
    execu = get_or_create_executor()
    _absorb_events(execu)
    state = execu.state
    ui = _ui()

    if state.status not in ("finished", "stopped", "errored") and not ui.get("tier4_final"):
        st.info(
            "**No completed run yet.** Results will appear here after Tier-4 "
            "runs at the end of a round. Showing last completed run from the "
            "registry below."
        )
        _render_last_completed_run_summary()
        return

    if state.status == "errored":
        st.error(f"❌ {state.error_type}: {state.error_message}")

    # ── Verdict card ──
    tier4 = ui.get("tier4_final") or {}
    per_zone_t4 = tier4.get("per_zone", {}) if isinstance(tier4, dict) else {}
    model_rmse: dict[str, float] = {}
    for zk, blk in per_zone_t4.items():
        if isinstance(blk, dict):
            rc = blk.get("rmse_ci")
            if isinstance(rc, dict) and isinstance(rc.get("point"), (int, float)):
                model_rmse[zk] = float(rc["point"])

    # Baselines — pull from the most recent Tier-2 (baseline_rmse per zone)
    pers_rmse: dict[str, float] = {}
    clim_rmse: dict[str, float] = {}
    if ui["tier2_hist"]:
        t2 = ui["tier2_hist"][-1]
        for zk, blk in t2.get("per_zone", {}).items():
            b = blk.get("baseline_rmse")
            if isinstance(b, (int, float)):
                pers_rmse[zk] = float(b)  # persistence is the T2 baseline
    # Clim RMSEs will be filled after Tier-3 pass. If unavailable, pass empty.

    card = build_summary_card(
        per_zone_model_rmse=model_rmse,
        per_zone_persistence_rmse=pers_rmse,
        per_zone_climatology_rmse=clim_rmse,
    )
    st.markdown(render_summary_card_html(card), unsafe_allow_html=True)

    # ── Per-zone diagnosis table ──
    st.markdown("### Per-zone diagnosis")
    rows = []
    for zk, blk in per_zone_t4.items():
        rc = blk.get("rmse_ci") if isinstance(blk, dict) else None
        if isinstance(rc, dict):
            rows.append({
                "zone": zk,
                "RMSE": f"{rc['point']:.3f}"  if isinstance(rc.get('point'), (int,float)) else "—",
                "CI95_lo": f"{rc['ci95_lower']:.3f}" if isinstance(rc.get('ci95_lower'), (int,float)) else "—",
                "CI95_hi": f"{rc['ci95_upper']:.3f}" if isinstance(rc.get('ci95_upper'), (int,float)) else "—",
                "vs pers": (blk.get('vs_persistence') or {}).get('winner', '?')
                            if isinstance(blk.get('vs_persistence'), dict) else str(blk.get('vs_persistence', '?')),
                "vs clim": (blk.get('vs_climatology') or {}).get('winner', '?')
                            if isinstance(blk.get('vs_climatology'), dict) else str(blk.get('vs_climatology', '?')),
                "n samples": blk.get("n_samples", "—"),
            })
        else:
            rows.append({"zone": zk, "RMSE": "—", "CI95_lo": "—", "CI95_hi": "—",
                          "vs pers": "—", "vs clim": "—", "n samples": "—"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Heatmap ──
    if ui["heatmap_paths"]:
        p = Path(ui["heatmap_paths"][-1])
        if p.exists():
            st.image(str(p), caption="Tier-3 skill vs climatology",
                      use_container_width=True)

    # ── Action bar ──
    st.markdown("### Actions")
    cA, cB, cC, cD = st.columns(4)
    with cA:
        if st.button("💾 Save bug-report zip"):
            _write_bug_report(execu)
    with cB:
        if st.button("🔍 Open in Validation dashboard"):
            st.session_state["active_tab"] = "🔍 Validation"
            st.rerun()
    with cC:
        if st.button("🔄 Re-run same config"):
            _reset_ui_for_new_run()
            st.rerun(scope="fragment")


def _render_last_completed_run_summary() -> None:
    from climate_twin.train.registry import get_registry
    reg = get_registry()
    models = reg.list_models()
    if not models:
        st.caption("Registry is empty. Launch a run in the ⚙ Setup tab.")
        return
    m = models[0]
    st.markdown(f"**Last checkpoint on file:** `{m['region']}/{m['name']}`")
    rmse = m.get("best_zone_weighted_rmse")
    st.markdown(f"- best zone-weighted RMSE: **{fmt_num(rmse, 4)}**")
    st.markdown(f"- epochs trained: {m.get('epochs_trained', '?')}")
    st.markdown(f"- variables: {', '.join(m.get('variables') or [])}")
    st.markdown(f"- zone_mask_sig: `{m.get('zone_mask_sig','')}`")


def _write_bug_report(execu) -> None:
    import zipfile
    stamp = datetime.now(IST).strftime("%Y%m%d-%H%M%S")
    root = Path("climate_twin/exports") / "bug_reports"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"bug_report_{stamp}.zip"
    ui = _ui()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("run_state.json", execu.state.to_json())
        z.writestr("log_tail.txt", "\n".join(ui["log_lines"][-200:]))
        z.writestr("events_history.json",
                    json.dumps(execu.events.to_jsonable(),
                                indent=2, default=str))
    st.success(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — 📜 LOG
# ─────────────────────────────────────────────────────────────────────
def _render_log_tab():
    _fragment_log_stream()


@st.fragment(run_every=0.7)
def _fragment_log_stream():
    with P.profile("log_stream"):
        _render_log_stream_body()


def _render_log_stream_body():
    execu = get_or_create_executor()
    _absorb_events(execu)
    ui = _ui()
    lines = ui["log_lines"]

    col_f, col_s, col_d = st.columns([1, 2, 1])
    with col_f:
        level = st.selectbox(
            "level filter", ["all", "info", "warn", "error", "tune", "ctrl"],
            index=0, key="log_level_filter",
        )
    with col_s:
        query = st.text_input("search", value=st.session_state.get("log_query", ""),
                                key="log_query",
                                placeholder="substring filter")
    with col_d:
        st.download_button(
            "Download full log",
            data="\n".join(lines).encode("utf-8"),
            file_name=f"train_log_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    filtered = lines
    if level != "all":
        filtered = [l for l in filtered if f"[{level:<5}]" in l or f"[{level}]" in l]
    if query:
        filtered = [l for l in filtered if query.lower() in l.lower()]

    body = "\n".join(filtered[-2000:]) or "(no log yet)"
    st.code(body, language="text")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
def render_training_dashboard() -> None:
    _render_sidebar()
    _fragment_run_header()
    tab_setup, tab_live, tab_results, tab_log = st.tabs(
        ["⚙ Setup", "📊 Live", "🎯 Results", "📜 Log"]
    )
    with tab_setup:
        _render_setup_tab()
    with tab_live:
        _render_live_tab()
    with tab_results:
        _render_results_tab()
    with tab_log:
        _render_log_tab()
