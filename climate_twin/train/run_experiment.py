"""
run_experiment.py — CLI runner. Given an experiment yaml, run the trainer
end-to-end and print a summary.

Usage:
    python -m climate_twin.train.run_experiment climate_twin/train/config/experiments/ph4_smoke.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from climate_twin.train.config import load_config
from climate_twin.train.loop import Trainer, LiveState
from climate_twin.train.registry import get_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="path to an experiment yaml")
    ap.add_argument("--name", type=str, default=None,
                    help="override the experiment.name (used as the model folder)")
    ap.add_argument("--parent", type=str, default="",
                    help="parent model name (for lineage)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(cfg.summary())
    print()

    reg = get_registry()
    live = LiveState()

    def on_epoch(info):
        print(f"  epoch {info['epoch']:3d}  train_loss={info['train_loss']:.4f}  "
              f"zw_rmse={info['zone_weighted_rmse']:.4f}  best={info['best_zw_rmse']:.4f}  "
              f"lr={info['lr']:.2e}  elapsed={info['elapsed']:.1f}s")

    trainer = Trainer(
        cfg,
        model_name=args.name or cfg.name,
        registry_root=reg.models_root,
        live=live,
        on_epoch=on_epoch,
        parent_name=args.parent,
    )
    out = trainer.run()

    print()
    print("── run complete ─────────────────────────────")
    print(f"  weights           = {out.best_state_dict_path}")
    print(f"  meta              = {out.meta_json_path}")
    print(f"  tier4             = {out.tier4_json_path}")
    print(f"  heatmap           = {out.heatmap_path}")
    print(f"  best zw_rmse      = {out.best_zone_weighted_rmse:.4f}")
    print(f"  epochs run        = {out.n_epochs_run}")
    print()
    print("  per-zone final:")
    for zk, v in out.per_zone_final.items():
        rc = v.get("rmse_ci")

        def _winner(x):
            if isinstance(x, dict):
                return x.get("winner", "?")
            if isinstance(x, str):
                return x
            return "?"

        if isinstance(rc, dict):
            print(f"    {zk:22s}  rmse={rc['point']:.3f} "
                  f"[{rc['ci95_lower']:.3f},{rc['ci95_upper']:.3f}]"
                  f"  vs_pers={_winner(v.get('vs_persistence'))}"
                  f"  vs_clim={_winner(v.get('vs_climatology'))}")
        else:
            print(f"    {zk:22s}  {rc}")


if __name__ == "__main__":
    main()
