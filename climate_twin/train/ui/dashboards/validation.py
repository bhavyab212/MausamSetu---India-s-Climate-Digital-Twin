"""
Phase 3 — Validation dashboard.

Tabs:
    A. Select what to inspect (checkpoint(s), val year, ensemble size)
    B. Summary card (deterministic STRONG/MIXED/WEAK)
    C. Visual views
    D. Statistical views
    E. Compare mode
    F. Export

All inference runs on GPU. Metric aggregation, plotting, and PDF export
stay on CPU. Results are cached by (checkpoint_name, region, val_year,
ensemble_size).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from climate_twin.regions import get_zones
from climate_twin import data_source as DS
from climate_twin.train.config.schema import ExperimentConfig
from climate_twin.train.data import PerZoneZScore, DailyWindowDataset, default_collate
from climate_twin.train.eval.metrics import (
    INSUFFICIENT,
    per_zone_metrics,
    weighted_rmse,
    seasons_of,
)
from climate_twin.train.eval.baselines import (
    persistence_prediction,
    climatology_prediction,
)
from climate_twin.train.model import build_model
from climate_twin.train.registry import (
    RegistryModelIncompatible,
    get_registry,
)
from .common import render_device_banner
from .summary_card import build_summary_card, render_summary_card_html


IMD_THRESHOLDS = {
    "moderate": 15.6,
    "heavy": 64.5,
    "very_heavy": 115.6,
    "extremely_heavy": 204.5,
}


# ---------------------------------------------------------------------------
# Inference (GPU) + baselines (CPU)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running ensemble inference…")
def _run_inference(checkpoint_name: str, region: str, val_year: int,
                    ensemble_size: int) -> dict[str, Any]:
    """Return dict with pred (T,H,W), truth, persistence, climatology,
    months, doys — all as arrays. Cached by args."""
    import torch
    from climate_twin.runtime import ensure_cuda
    ensure_cuda(context="ensemble inference")

    reg = get_registry()
    m = reg.get_model(checkpoint_name, region)
    if m is None:
        raise FileNotFoundError(f"{region}/{checkpoint_name}")

    cfg = ExperimentConfig(**m["config"])
    device = "cuda"
    Z = get_zones()
    model = build_model(cfg, zones=Z).to(device)
    reg.load_into(checkpoint_name, region, model,
                   expected_variables=list(cfg.data.variables),
                   expected_zone_mask_sig=Z.mask_signature, strict=True)
    model.eval()

    # Build dataset for the requested year
    tf = PerZoneZScore(tuple(cfg.data.variables))
    ds = DailyWindowDataset(
        region=region,
        variables=tuple(cfg.data.variables),
        seq_length=cfg.model.seq_length,
        year_range=(val_year, val_year),
        transform=tf,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg.optim.batch_size, shuffle=False,
                        collate_fn=default_collate, num_workers=0,
                        pin_memory=True)
    zone_map = torch.from_numpy(Z.membership).permute(2, 0, 1).unsqueeze(0).to(device)

    all_preds: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    months, doys = [], []
    # Enable dropout for MC sampling when ensemble_size > 1
    with torch.no_grad():
        for batch in loader:
            x = torch.nan_to_num(batch["x"], nan=0.0).to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            zm = zone_map.expand(x.shape[0], -1, -1, -1).contiguous()

            if ensemble_size <= 1:
                model.eval()
                out = model(x, zm)
                p_amount = torch.nn.functional.softplus(out["rain"]["amount"]).squeeze(1)
                p_occ = torch.sigmoid(out["rain"]["logit_occurrence"]).squeeze(1)
                pred = p_occ * p_amount
                all_preds.append(pred.cpu().numpy())
            else:
                model.train()   # enable dropout
                samples = []
                for _ in range(int(ensemble_size)):
                    out = model(x, zm)
                    p_amount = torch.nn.functional.softplus(out["rain"]["amount"]).squeeze(1)
                    p_occ = torch.sigmoid(out["rain"]["logit_occurrence"]).squeeze(1)
                    samples.append((p_occ * p_amount).cpu().numpy())
                all_preds.append(np.stack(samples, axis=0).mean(axis=0))
                model.eval()

            all_truth.append(y[:, 0].cpu().numpy())
            months.extend(batch["month"])
            doys.extend(batch["doy"])
    ds.close()

    pred = np.concatenate(all_preds, axis=0)
    truth = np.concatenate(all_truth, axis=0)

    # Baselines aligned with pred/truth — lazy per-slice reads to avoid the
    # 1.78 GB full-cube materialisation on a 75-year cube.
    ds_full = DS.load_region(region)
    years = np.asarray(ds_full["time.year"].values)
    val_sel = years == val_year
    target_idx = np.flatnonzero(val_sel)
    rain_var = ds_full["rain"]
    truth_full = rain_var.isel(time=target_idx).values
    prev = rain_var.isel(time=target_idx - 1).values
    train_sel = ((years >= cfg.data.train_years[0]) & (years <= cfg.data.train_years[1]))
    train_doys = np.asarray(ds_full["time.dayofyear"].values)[train_sel]
    train_mask_idx = np.flatnonzero(train_sel)
    # Build DOY → cube-time-index buckets once
    _doy_buckets: dict[int, list[int]] = {}
    for _pos, _d in enumerate(train_doys):
        _doy_buckets.setdefault(int(_d), []).append(int(train_mask_idx[_pos]))
    H, W = truth_full.shape[-2:]
    clim366 = np.full((366, H, W), np.nan, dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        for d in range(1, 367):
            _idxs = _doy_buckets.get(d)
            if _idxs:
                slab = rain_var.isel(time=np.asarray(_idxs)).values
                clim366[d - 1] = np.nanmean(slab, axis=0)
    val_doys = np.asarray(ds_full["time.dayofyear"].values)[target_idx]
    climatology = clim366[val_doys - 1]
    persistence = prev
    ds_full.close()

    return {
        "pred": pred,
        "truth": truth,
        "persistence": persistence,
        "climatology": climatology,
        "months": np.array(months, dtype=np.int32),
        "doys": np.array(doys, dtype=np.int32),
        "checkpoint": f"{region}/{checkpoint_name}",
        "zone_mask_sig": Z.mask_signature,
        "manifest_sig": DS.manifest_sig(),
        "val_year": int(val_year),
        "cfg": cfg.model_dump(),
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def _per_zone_rmse(pred, truth, Z, mask) -> dict[str, float]:
    out = {}
    for k, z in enumerate(Z.zones):
        w = Z.membership[..., k].astype(np.float32)
        r = weighted_rmse(pred, truth, w, mask, min_cells=5)
        if isinstance(r, (int, float)):
            out[z.key] = float(r)
    return out


def _render_summary_card(res: dict) -> None:
    Z = get_zones()
    mask = (Z.hard_mask > 0).astype(np.float32)
    model_r = _per_zone_rmse(res["pred"], res["truth"], Z, mask)
    pers_r = _per_zone_rmse(res["persistence"], res["truth"], Z, mask)
    clim_r = _per_zone_rmse(res["climatology"], res["truth"], Z, mask)

    # Simple ensemble-calibration proxy: n/a for single-sample; approx via
    # softplus spread. Treated as unknown for the smoke checkpoints here.
    card = build_summary_card(
        per_zone_model_rmse=model_r,
        per_zone_persistence_rmse=pers_r,
        per_zone_climatology_rmse=clim_r,
        calibration=None,
        physics_violation_pct=None,
    )
    st.markdown(render_summary_card_html(card), unsafe_allow_html=True)
    return card


def _render_skill_map(res: dict, metric: str = "rmse") -> None:
    Z = get_zones()
    mask = (Z.hard_mask > 0).astype(np.float32)
    # Compute per-zone metric for the model, then paint each hard-zone cell
    values = np.full(Z.hard_mask.shape, np.nan, dtype=np.float32)
    per_zone = _per_zone_rmse(res["pred"], res["truth"], Z, mask)
    for z in Z.zones:
        v = per_zone.get(z.key)
        if isinstance(v, (int, float)):
            values[Z.hard_mask == z.id] = v

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("#050912")
    ax.set_facecolor("#0E1522")
    im = ax.imshow(values, cmap="RdYlGn_r", origin="lower")
    ax.set_title(f"Skill map — per-zone {metric.upper()} (mm/day)",
                  color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors="white")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _render_error_map(res: dict) -> None:
    Z = get_zones()
    mask = (Z.hard_mask > 0).astype(np.float32)
    err = np.nanmean(res["pred"] - res["truth"], axis=0)
    err[mask == 0] = np.nan
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("#050912")
    ax.set_facecolor("#0E1522")
    im = ax.imshow(err, cmap="RdBu_r", origin="lower",
                    vmin=-np.nanpercentile(np.abs(err), 98) if np.isfinite(err).any() else -5,
                    vmax=np.nanpercentile(np.abs(err), 98) if np.isfinite(err).any() else 5)
    ax.set_title("Mean error map (pred − obs, mm/day)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors="white")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _render_sample_gallery(res: dict, n: int = 4) -> None:
    """Pick n days from the val window (wettest, driest, most variance) and
    render [obs][p50][error] rows."""
    Z = get_zones()
    mask = (Z.hard_mask > 0)
    per_day_rain = np.nansum(np.where(mask[None], res["truth"], 0.0), axis=(1, 2))
    order_wet = np.argsort(-per_day_rain)[:n]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 2.5 * n))
    fig.patch.set_facecolor("#050912")
    if n == 1:
        axes = axes.reshape(1, 3)
    for i, idx in enumerate(order_wet):
        obs = np.where(mask, res["truth"][idx], np.nan)
        pred = np.where(mask, res["pred"][idx], np.nan)
        err = np.where(mask, res["pred"][idx] - res["truth"][idx], np.nan)
        vmax = float(np.nanpercentile(obs, 99)) if np.isfinite(obs).any() else 20.0
        for j, (arr, cmap, title, vmin, vmax_) in enumerate([
            (obs,  "Blues",   f"obs (day {idx})",  0, vmax),
            (pred, "Blues",   f"p50",                0, vmax),
            (err,  "RdBu_r",  f"error",           -vmax, vmax),
        ]):
            ax = axes[i, j]
            ax.set_facecolor("#0E1522")
            ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax_)
            ax.set_title(title, color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=7)
            for s in ax.spines.values():
                s.set_color("white")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _render_deep_table(res: dict) -> None:
    Z = get_zones()
    mask = (Z.hard_mask > 0).astype(np.float32)
    months = res["months"]

    # Load percentile thresholds from Phase-1 stats.
    # From validation.py: parents[0]=dashboards, [1]=ui, [2]=train, [3]=climate_twin
    stats_path = Path(__file__).resolve().parents[3] / "regions" / "zone_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    zp = {
        zkey: {
            "p90": float(b["percentiles_rain_mm_day"]["p90"]),
            "p95": float(b["percentiles_rain_mm_day"]["p95"]),
            "p99": float(b["percentiles_rain_mm_day"]["p99"]),
        }
        for zkey, b in stats["zones"].items()
    }
    result = per_zone_metrics(
        res["pred"], res["truth"], months, Z, mask,
        IMD_THRESHOLDS, zp, min_cells=5,
    )
    rows = []
    for zk, zblk in result.items():
        for season, blk in zblk.items():
            row = {"zone": zk, "season": season}
            for m_k in ("rmse", "mae", "bias"):
                v = blk.get(m_k)
                row[m_k] = round(float(v), 3) if isinstance(v, (int, float)) else "—"
            rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=360)
    return df


def _render_compare_view(res_list: list[dict]) -> None:
    if len(res_list) < 2:
        st.caption("Select ≥ 2 checkpoints to enable Compare mode.")
        return
    Z = get_zones()
    mask = (Z.hard_mask > 0).astype(np.float32)
    rows = []
    for res in res_list:
        pz = _per_zone_rmse(res["pred"], res["truth"], Z, mask)
        for zk, v in pz.items():
            rows.append({"zone": zk, "checkpoint": res["checkpoint"], "rmse": round(v, 3)})
    df = pd.DataFrame(rows)
    st.dataframe(df.pivot(index="zone", columns="checkpoint", values="rmse"),
                  use_container_width=True)

    # winner-per-zone summary
    piv = df.pivot(index="zone", columns="checkpoint", values="rmse")
    winners = piv.idxmin(axis=1).to_frame("winner (lowest RMSE)")
    st.dataframe(winners, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render_validation_dashboard() -> None:
    render_device_banner()
    st.title("🔍 Validation")
    st.caption("Read-only. GPU used only for ensemble inference.")

    reg = get_registry()
    Z = get_zones()

    st.markdown("### A. Select what to inspect")

    models = reg.list_models()
    if not models:
        st.info("No trained models in the registry yet. Run one from the 🔥 Training tab.")
        return

    # Only show compatible checkpoints unless the user chooses otherwise
    compat_only = st.checkbox("Show compatible checkpoints only "
                                "(zone_mask_sig matches)", value=True)
    if compat_only:
        models = [m for m in models if m.get("zone_mask_sig") == Z.mask_signature]

    labels = [f"{m['region']}/{m['name']}"
              + (" ⛔" if m.get("zone_mask_sig") != Z.mask_signature else "")
              for m in models]
    picks = st.multiselect(
        "Checkpoint(s)",
        options=list(range(len(models))),
        format_func=lambda i: labels[i],
        default=[0] if models else [],
    )
    if not picks:
        st.info("Pick at least one checkpoint.")
        return

    val_year = st.number_input("Validation year", min_value=1951, max_value=2100,
                                value=2023, step=1)
    ensemble_size = st.select_slider("Ensemble size (MC dropout)",
                                       options=[1, 5, 20, 50], value=1)

    if st.button("▶ Run Validation", type="primary"):
        st.session_state["_val_results"] = []
        for i in picks:
            m = models[i]
            try:
                res = _run_inference(m["name"], m["region"], int(val_year), int(ensemble_size))
                st.session_state["_val_results"].append(res)
            except RegistryModelIncompatible as e:
                st.error(f"⛔ {m['name']}: {e}")
            except Exception as e:
                st.error(f"❌ {m['name']}: {type(e).__name__}: {e}")

    results = st.session_state.get("_val_results", [])
    if not results:
        st.caption("Press Run Validation to populate the panels below.")
        return

    st.markdown("### B. Summary card")
    for res in results:
        st.markdown(f"**{res['checkpoint']}** — val year {res['val_year']}")
        _render_summary_card(res)
        st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### C. Visual views")
    tabs = st.tabs(["Skill Map", "Error Map", "Sample Gallery"])
    with tabs[0]:
        for res in results:
            st.caption(res["checkpoint"])
            _render_skill_map(res)
    with tabs[1]:
        for res in results:
            st.caption(res["checkpoint"])
            _render_error_map(res)
    with tabs[2]:
        for res in results:
            st.caption(res["checkpoint"])
            _render_sample_gallery(res, n=4)

    st.markdown("### D. Statistical views")
    tabs = st.tabs(["Deep Table (Tier-3)", "Skill vs Baselines"])
    with tabs[0]:
        for res in results:
            st.caption(res["checkpoint"])
            _render_deep_table(res)
    with tabs[1]:
        Z = get_zones()
        mask = (Z.hard_mask > 0).astype(np.float32)
        rows = []
        for res in results:
            model_r = _per_zone_rmse(res["pred"], res["truth"], Z, mask)
            pers_r = _per_zone_rmse(res["persistence"], res["truth"], Z, mask)
            clim_r = _per_zone_rmse(res["climatology"], res["truth"], Z, mask)
            for zk in model_r:
                rows.append({
                    "checkpoint": res["checkpoint"],
                    "zone": zk,
                    "model": round(model_r[zk], 3),
                    "persistence": round(pers_r.get(zk, float("nan")), 3),
                    "climatology": round(clim_r.get(zk, float("nan")), 3),
                    "beats_pers": model_r[zk] < pers_r.get(zk, float("inf")),
                    "beats_clim": model_r[zk] < clim_r.get(zk, float("inf")),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### E. Compare mode")
    _render_compare_view(results)

    st.markdown("### F. Export")
    if st.button("📁 Save all figures to climate_twin/exports/YYYY-MM-DD/"):
        from datetime import datetime, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        stamp = datetime.now(IST).strftime("%Y-%m-%d_%H%M%S")
        out = Path("climate_twin/exports") / stamp
        out.mkdir(parents=True, exist_ok=True)
        for res in results:
            name = res["checkpoint"].replace("/", "_")
            (out / f"{name}_summary.json").write_text(
                json.dumps({"checkpoint": res["checkpoint"],
                            "val_year": res["val_year"],
                            "zone_mask_sig": res["zone_mask_sig"],
                            "manifest_sig": res["manifest_sig"]},
                            indent=2))
        st.success(f"Wrote {out}")
