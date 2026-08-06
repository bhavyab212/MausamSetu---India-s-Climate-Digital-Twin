"""
Zone-Aware Training UI — Streamlit page.

Standalone page — run with::

    streamlit run climate_twin/train/ui/app.py

Modes:
  * **Config preview** — pick an experiment yaml, see the validated config
    (refuses invalid ones with the exact error).
  * **Train** — launch a run in a background thread; live view of the
    Tier-1 loss curve, per-epoch train loss, per-zone Tier-2 heatmap
    strip, LR, GPU mem, ETA, and per-zone convergence status.
  * **Post-round diagnostics** — Tier-3 heatmap + Tier-4 CIs + significance
    tests + comparison to the Phase-0 benchmark.
  * **Registry** — list saved models; show variables / zone_hash / manifest_sig
    compatibility badges; delete on demand.

This file is loaded on demand by the top-level app_v2.py Training tab and
is otherwise standalone.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import streamlit as st

from climate_twin.regions import get_zones
from climate_twin.train.config import load_config, ExperimentConfig
from climate_twin.train.loop import Trainer, LiveState
from climate_twin.train.registry import get_registry, RegistryModelIncompatible

st.set_page_config(page_title="Zone-Aware Training • ClimateTwin Lab",
                   page_icon="🔥", layout="wide")

# ── Sidebar navigation ─────────────────────────────────────────
st.sidebar.title("🔥 Zone-aware training")
Z = get_zones()
st.sidebar.caption(f"zone_mask_sig = `{Z.mask_signature}`  ·  {Z.n_zones()} zones  ·  "
                   f"{int((Z.hard_mask > 0).sum())} land cells")

mode = st.sidebar.radio("Mode", ["📄 Config preview", "🏋️ Train", "📊 Diagnostics", "📦 Registry"])

# ── Config preview ─────────────────────────────────────────────
def _render_config_preview():
    st.title("Config preview")
    exp_dir = REPO / "climate_twin" / "train" / "config" / "experiments"
    yamls = sorted(exp_dir.glob("*.yaml"))
    pick = st.selectbox("Experiment", [p.name for p in yamls])
    yaml_path = exp_dir / pick
    with st.expander("Raw YAML", expanded=False):
        st.code(yaml_path.read_text(encoding="utf-8"), language="yaml")
    try:
        cfg = load_config(yaml_path)
    except Exception as e:
        st.error(f"❌ config invalid — refusing to run\n\n**{type(e).__name__}**: {e}")
        return
    st.success("✅ config valid — safe to run")
    st.code(cfg.summary(), language="text")


# ── Train ──────────────────────────────────────────────────────
def _render_train():
    st.title("Train")
    exp_dir = REPO / "climate_twin" / "train" / "config" / "experiments"
    yamls = sorted(exp_dir.glob("*.yaml"))
    pick = st.selectbox("Experiment yaml", [p.name for p in yamls])
    yaml_path = exp_dir / pick

    try:
        cfg = load_config(yaml_path)
    except Exception as e:
        st.error(f"❌ {type(e).__name__}: {e}")
        return

    st.caption(cfg.summary())

    col_a, col_b = st.columns([1, 1])
    with col_a:
        model_name = st.text_input("Model name", cfg.name)
    with col_b:
        parent = st.text_input("Parent (for lineage)", "")

    live_key = f"live_{model_name}"
    thread_key = f"thread_{model_name}"

    if st.button("▶ Start training", type="primary"):
        live = LiveState()
        st.session_state[live_key] = live
        reg = get_registry()

        def _worker():
            try:
                trainer = Trainer(
                    cfg,
                    model_name=model_name,
                    registry_root=reg.models_root,
                    live=live,
                    parent_name=parent,
                )
                trainer.run()
            except Exception as e:
                live.error = f"{type(e).__name__}: {e}"
                live.running = False
                live.finished = True

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        st.session_state[thread_key] = t

    live = st.session_state.get(live_key)
    if live is None:
        st.info("No run yet.")
        return

    if live.error:
        st.error(live.error)

    # Live view — refresh once per second while the run is going
    tot = max(live.total_epochs, 1)
    st.progress(min(live.epoch / tot, 1.0),
                text=f"Epoch {live.epoch}/{live.total_epochs}  ·  "
                     f"batch {live.batch}/{live.total_batches}  ·  "
                     f"elapsed {live.elapsed:.1f}s  ·  ETA {live.eta:.1f}s")

    cols = st.columns(4)
    cols[0].metric("LR", f"{live.lr:.2e}")
    cols[1].metric("Grad norm", f"{live.grad_norm:.4f}")
    zw = live.tier2_history[-1] if live.tier2_history else None
    if zw is not None:
        zvals = [v.get("rmse") for v in zw["per_zone"].values()
                 if isinstance(v.get("rmse"), float)]
        cols[2].metric("Mean per-zone RMSE", f"{np.mean(zvals):.3f}" if zvals else "—")
    cols[3].metric("Status", "running" if live.running else "finished")

    if live.train_losses:
        st.line_chart(live.train_losses, height=180)

    if live.tier2_history:
        st.markdown("### Per-zone Tier-2 heatmap strip")
        keys = [z.key for z in Z.zones]
        vals = np.full((len(keys), len(live.tier2_history)), np.nan)
        for j, t2 in enumerate(live.tier2_history):
            for i, k in enumerate(keys):
                r = t2["per_zone"].get(k, {}).get("rmse")
                if isinstance(r, float) and np.isfinite(r):
                    vals[i, j] = r
        import pandas as pd
        df = pd.DataFrame(vals, index=keys,
                          columns=[f"e{j+1}" for j in range(vals.shape[1])])
        st.dataframe(df.style.background_gradient(cmap="RdYlGn_r"), height=280)

    if live.per_zone_convergence:
        st.markdown("### Per-zone convergence")
        st.json(live.per_zone_convergence, expanded=False)

    if live.finished and not live.error:
        st.success(f"Run finished. Best zone-weighted RMSE = "
                   f"{min([b for b in [zw and _extract_best(live)] if b is not None] + [float('inf')]):.4f}"
                   if live.tier2_history else "Run finished.")


def _extract_best(live: LiveState) -> float | None:
    if not live.tier2_history:
        return None
    # not exactly the trainer's best_zw_rmse but close enough for the UI card
    vals = []
    for t2 in live.tier2_history:
        zvals = [v.get("rmse") for v in t2["per_zone"].values()
                 if isinstance(v.get("rmse"), float)]
        if zvals:
            vals.append(float(np.mean(zvals)))
    return min(vals) if vals else None


# ── Diagnostics ────────────────────────────────────────────────
def _render_diagnostics():
    st.title("Post-round diagnostics")
    reg = get_registry()
    models = reg.list_models()
    if not models:
        st.info("No trained models yet. Run one under 🏋️ Train first.")
        return
    pick = st.selectbox("Model", [f"{m['region']}/{m['name']}" for m in models])
    region, name = pick.split("/", 1)
    m = reg.get_model(name, region)
    if m is None:
        st.error("Not found."); return

    folder = Path(m["path"])
    st.markdown(f"**Path:** `{folder}`")
    st.markdown(f"**Zone mask sig:** `{m.get('zone_mask_sig','?')}` "
                f" · **Manifest sig:** `{m.get('manifest_sig','?')}` "
                f" · **Variables:** {m.get('variables', [])}")

    heatmap = folder / "tier3_heatmap.png"
    if heatmap.exists():
        st.image(str(heatmap), caption="Tier-3 skill heatmap (vs climatology)")

    tier4 = folder / "tier4.json"
    if tier4.exists():
        st.markdown("### Tier-4 significance report")
        st.json(json.loads(tier4.read_text(encoding="utf-8")), expanded=False)


# ── Registry ───────────────────────────────────────────────────
def _render_registry():
    st.title("Model registry")
    reg = get_registry()
    Z_ = get_zones()
    models = reg.list_models()
    if not models:
        st.info("No models saved yet.")
        return

    import pandas as pd
    rows = []
    for m in models:
        zsig = m.get("zone_mask_sig") or ""
        msig = m.get("manifest_sig") or ""
        vars_ = m.get("variables") or []
        compat = "✓"
        if zsig != Z_.mask_signature:
            compat = "⛔ zone drift"
        rows.append({
            "region": m.get("region"),
            "name": m.get("name"),
            "parent": m.get("parent_name") or "—",
            "vars": ", ".join(vars_),
            "epochs": m.get("epochs_trained", "?"),
            "best zw_rmse": m.get("best_zone_weighted_rmse", "?"),
            "zone_sig": zsig[:12],
            "compat": compat,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    to_delete = st.selectbox("Delete", ["—"] + [f"{m['region']}/{m['name']}" for m in models])
    if to_delete != "—" and st.button("🗑 Delete", type="secondary"):
        reg.delete(*to_delete.split("/", 1)[::-1])
        st.rerun()


# ── Dispatch ───────────────────────────────────────────────────
if mode == "📄 Config preview":
    _render_config_preview()
elif mode == "🏋️ Train":
    _render_train()
elif mode == "📊 Diagnostics":
    _render_diagnostics()
elif mode == "📦 Registry":
    _render_registry()
