"""
phase5_benchmark_compare.py — Phase 5c honest comparison.

Load a trained model, evaluate on the 2024-2025 TEST years, and report
per-zone RMSE side-by-side against ``_phase0/benchmark_persistence_climatology.json``.

Writes ``_phase0/phase5/benchmark_compare.json`` with per-zone win/loss
against the best of (persistence, climatology). Also renders a bar-style
delta chart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from climate_twin.regions import get_zones
from climate_twin import data_source as DS
from climate_twin.train.config import load_config
from climate_twin.train.model import build_model
from climate_twin.train.registry import get_registry
from climate_twin.train.data import PerZoneZScore, DailyWindowDataset, default_collate
from torch.utils.data import DataLoader

OUT_DIR = REPO / "climate_twin" / "_phase0" / "phase5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _predict_on_years(model, cfg, year_range, device):
    tf = PerZoneZScore(tuple(cfg.data.variables))
    ds = DailyWindowDataset(
        region=cfg.data.region,
        variables=tuple(cfg.data.variables),
        seq_length=cfg.model.seq_length,
        year_range=year_range,
        transform=tf,
    )
    Z = get_zones()
    zone_map = (
        torch.from_numpy(Z.membership).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=default_collate, num_workers=0)
    preds, truths, iso = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = torch.nan_to_num(batch["x"], nan=0.0).to(device)
            y = batch["y"].to(device)
            zm = zone_map.expand(x.shape[0], -1, -1, -1).contiguous()
            out = model(x, zm)
            p_amount = torch.nn.functional.softplus(out["rain"]["amount"]).squeeze(1)
            p_occ = torch.sigmoid(out["rain"]["logit_occurrence"]).squeeze(1)
            preds.append((p_occ * p_amount).detach().cpu().numpy())
            truths.append(y[:, 0].detach().cpu().numpy())
            iso.extend(batch["iso_date"])
    ds.close()
    return np.concatenate(preds), np.concatenate(truths), iso


def _weighted_rmse_per_zone(pred, truth, zones):
    memb = zones.membership.astype(np.float32)
    hard = zones.hard_mask
    mask = (hard > 0).astype(np.float32)
    out = {}
    for k, zone in enumerate(zones.zones):
        w = memb[..., k]
        finite = np.isfinite(pred) & np.isfinite(truth) & (mask[None] > 0)
        w_full = np.broadcast_to(w[None, :, :], pred.shape)
        num = float(np.where(finite, (pred - truth) ** 2 * w_full, 0.0).sum())
        den = float(np.where(finite, w_full, 0.0).sum())
        out[zone.key] = float(np.sqrt(num / den)) if den > 0 else float("nan")
    return out


def main():
    reg = get_registry()
    model_name = sys.argv[1] if len(sys.argv) > 1 else "ph4_smoke"
    region = "india"
    m = reg.get_model(model_name, region)
    if m is None:
        print(f"model {region}/{model_name} not in registry")
        return 1

    cfg_dict = m["config"]
    from climate_twin.train.config.schema import ExperimentConfig
    cfg = ExperimentConfig(**cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z = get_zones()
    model = build_model(cfg, zones=Z).to(device)
    reg.load_into(
        model_name, region, model,
        expected_variables=list(cfg.data.variables),
        expected_zone_mask_sig=Z.mask_signature,
        strict=True,
    )
    print(f"[bench] loaded {region}/{model_name}")

    # Load Phase-0 baseline
    bench = json.loads(
        (REPO / "climate_twin/_phase0/benchmark_persistence_climatology.json").read_text()
    )
    bench_train = tuple(bench["train_years"])
    bench_test = tuple(bench["test_years"])
    print(f"[bench] Phase-0 train_years={bench_train}  test_years={bench_test}")
    print(f"[bench] model train_years={cfg.data.train_years} — {'matches' if tuple(cfg.data.train_years)==bench_train else 'DIFFERS (note this honestly)'}")

    # Predict on test years
    pred, truth, iso = _predict_on_years(model, cfg, bench_test, device)
    print(f"[bench] pred shape={pred.shape}  test range {iso[0]}..{iso[-1]}")

    model_rmse = _weighted_rmse_per_zone(pred, truth, Z)

    # Build per-zone comparison
    comparison = {
        "model": f"{region}/{model_name}",
        "model_train_years": list(cfg.data.train_years),
        "phase0_baseline_train_years": list(bench_train),
        "test_years": list(bench_test),
        "zone_mask_sig": Z.mask_signature,
        "note": (
            f"Model train_years {tuple(cfg.data.train_years)} differ from Phase-0's "
            f"{bench_train} — the Phase-0 climatology used 72 years of stats. "
            "Any model-vs-clim comparison is honest only insofar as we acknowledge "
            "this data-quantity gap."
            if tuple(cfg.data.train_years) != bench_train
            else "Model and Phase-0 baseline trained on the same year window; apples-to-apples."
        ),
        "per_zone_test_rmse_rain_mm_day": {},
        "hard_wins_vs_best_baseline": 0,
        "total_zones_evaluated": 0,
    }
    for zk, r_model in model_rmse.items():
        b = bench["zones"].get(zk, {}).get("metrics", {}).get("rain", {})
        pers_rmse = b.get("persistence", {}).get("rmse", float("nan"))
        clim_rmse = b.get("climatology", {}).get("rmse", float("nan"))
        best_baseline = None
        if isinstance(pers_rmse, (int, float)) and isinstance(clim_rmse, (int, float)):
            best_baseline = min(pers_rmse, clim_rmse)
            beat = float(r_model) < float(best_baseline)
            comparison["hard_wins_vs_best_baseline"] += int(beat)
        comparison["total_zones_evaluated"] += 1
        comparison["per_zone_test_rmse_rain_mm_day"][zk] = {
            "model": r_model,
            "persistence": pers_rmse,
            "climatology": clim_rmse,
            "best_baseline": best_baseline,
            "delta_vs_best": (r_model - best_baseline) if isinstance(best_baseline, (int, float)) else None,
            "beats_best_baseline": (r_model < best_baseline) if isinstance(best_baseline, (int, float)) else None,
        }

    (OUT_DIR / f"benchmark_compare_{model_name}.json").write_text(
        json.dumps(comparison, indent=2))

    print()
    print(f"── benchmark comparison ({model_name} vs Phase-0 baselines on 2024-2025) ──")
    print(f"  {'zone':22s}  model    pers     clim   best_base  Δ_vs_best  win?")
    for zk, row in comparison["per_zone_test_rmse_rain_mm_day"].items():
        m_s = f"{row['model']:6.3f}" if isinstance(row['model'], float) else "   —"
        p_s = f"{row['persistence']:6.3f}" if isinstance(row['persistence'], float) else "   —"
        c_s = f"{row['climatology']:6.3f}" if isinstance(row['climatology'], float) else "   —"
        b_s = f"{row['best_baseline']:6.3f}" if isinstance(row['best_baseline'], float) else "   —"
        d_s = f"{row['delta_vs_best']:+6.3f}" if isinstance(row['delta_vs_best'], float) else "   —"
        w_s = "✓ WIN" if row['beats_best_baseline'] else "✗ loss" if row['beats_best_baseline'] is False else "  ?"
        print(f"  {zk:22s}  {m_s}  {p_s}  {c_s}   {b_s}   {d_s}   {w_s}")
    print()
    print(f"  wins: {comparison['hard_wins_vs_best_baseline']}/{comparison['total_zones_evaluated']}")


if __name__ == "__main__":
    sys.exit(main())
