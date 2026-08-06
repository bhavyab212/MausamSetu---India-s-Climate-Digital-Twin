"""
phase5_cauvery_consistency.py — Phase 5b cross-region consistency.

Two independent evaluations of the same trained checkpoint:
  1. Load ph4_smoke, evaluate against the INDIA cube on 2023 val-year,
     select cells belonging to the Cauvery basin, compute RMSE.
  2. Load ph4_smoke, evaluate against the CAUVERY cube on 2023 val-year
     (same days, same variables), compute RMSE on the basin cells.

Because build_cube guarantees Cauvery ⊂ India (Phase 4 tests), both
predictions should produce IDENTICAL RMSE on the basin. Report max
per-cell disagreement.

The consistency test is not a claim about model skill — it's a claim
that the SAME model applied to the two cubes yields the same answer on
the basin. Any drift is a bug.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from climate_twin.regions import get_zones
from climate_twin import data_source as DS
from climate_twin.train.config import load_config
from climate_twin.train.model import build_model
from climate_twin.train.registry import get_registry
from climate_twin.train.data import PerZoneZScore, DailyWindowDataset, default_collate
from torch.utils.data import DataLoader

OUT = REPO / "climate_twin" / "_phase0" / "phase5" / "cauvery_consistency.json"


def _predict_over_val(model, region: str, cfg, tf, device):
    ds = DailyWindowDataset(
        region=region,
        variables=tuple(cfg.data.variables),
        seq_length=cfg.model.seq_length,
        year_range=tuple(cfg.data.val_years),
        transform=tf,
    )
    Z = get_zones()
    zone_map = (
        torch.from_numpy(Z.membership).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=default_collate, num_workers=0)
    preds = []
    dates = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = torch.nan_to_num(batch["x"], nan=0.0).to(device)
            zm = zone_map.expand(x.shape[0], -1, -1, -1).contiguous()
            out = model(x, zm)
            p_amount = torch.nn.functional.softplus(out["rain"]["amount"]).squeeze(1)
            p_occ = torch.sigmoid(out["rain"]["logit_occurrence"]).squeeze(1)
            preds.append((p_occ * p_amount).detach().cpu().numpy())
            dates.extend(batch["iso_date"])
    ds.close()
    return np.concatenate(preds, axis=0), dates


def main():
    reg = get_registry()
    meta = reg.get_model("ph4_smoke", "india")
    if meta is None:
        print("ph4_smoke not in registry — nothing to check")
        return 1

    # Load the trained model
    yaml_path = REPO / "climate_twin/train/config/experiments/ph4_smoke.yaml"
    cfg = load_config(yaml_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = PerZoneZScore(tuple(cfg.data.variables))
    model = build_model(cfg).to(device)
    reg.load_into(
        "ph4_smoke", "india", model,
        expected_variables=list(cfg.data.variables),
        expected_zone_mask_sig=get_zones().mask_signature,
        strict=True,
    )

    # Evaluate over 2023 twice: once against india cube, once against cauvery
    pred_india, dates_i = _predict_over_val(model, "india", cfg, tf, device)
    pred_cauvery, dates_c = _predict_over_val(model, "cauvery", cfg, tf, device)

    # The two cubes have identical time axes on the master grid; the inputs
    # differ only in NaN masking outside the Cauvery basin. So the model's
    # OUTPUTS may differ off-basin, but should match ON the basin cells.
    ds_c = DS.load_region("cauvery")
    cauvery_mask = ds_c["mask"].values.astype(bool)   # (H, W), True in basin
    ds_c.close()

    # Compute both predictions on the basin cells only
    p_i = pred_india[:, cauvery_mask]      # (T, N_basin)
    p_c = pred_cauvery[:, cauvery_mask]
    diff = p_i - p_c
    max_abs = float(np.nanmax(np.abs(diff)))
    mean_abs = float(np.nanmean(np.abs(diff)))
    rms = float(np.sqrt(np.nanmean(diff ** 2)))
    p95_abs = float(np.nanpercentile(np.abs(diff), 95))

    # Ground truth on the same 2023 basin cells
    ds_gt = DS.load_region("india")
    years = np.asarray(ds_gt["time.year"].values)
    val_idx = np.where(years == cfg.data.val_years[0])[0]
    truth = ds_gt["rain"].values[val_idx][:, cauvery_mask]
    ds_gt.close()

    finite = np.isfinite(truth)
    rmse_india = float(np.sqrt(np.mean((p_i - truth) ** 2, where=finite)))
    rmse_cauvery = float(np.sqrt(np.mean((p_c - truth) ** 2, where=finite)))

    result = {
        "zone_mask_sig": get_zones().mask_signature,
        "manifest_sig": DS.manifest_sig(),
        "model": "india/ph4_smoke",
        "n_basin_cells": int(cauvery_mask.sum()),
        "n_val_days": int(pred_india.shape[0]),
        "pred_disagreement_on_basin": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rms": rms,
            "p95_abs": p95_abs,
        },
        "rmse_on_basin_vs_truth": {
            "predict_from_india_cube": rmse_india,
            "predict_from_cauvery_cube": rmse_cauvery,
            "abs_diff": abs(rmse_india - rmse_cauvery),
        },
    }

    print(json.dumps(result, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
