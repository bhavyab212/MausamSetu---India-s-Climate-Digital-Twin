"""
phase5_holdout_runner.py — run 3 held-out-zone experiments sequentially.

Writes a summary JSON that compares each held-out zone's val-year skill
to the same zone's skill under the full-training ph4_smoke checkpoint.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from climate_twin.train.config import load_config
from climate_twin.train.loop import Trainer, LiveState
from climate_twin.train.registry import get_registry

EXPERIMENTS = ["ph5_hold_tnne", "ph5_hold_northeast", "ph5_hold_thar"]
CONFIG_DIR = REPO / "climate_twin" / "train" / "config" / "experiments"
OUT_DIR = REPO / "climate_twin" / "_phase0" / "phase5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run(name: str) -> dict:
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    print()
    print(f"══ {name} ══════════════════════════")
    print(cfg.summary())
    reg = get_registry()
    live = LiveState()

    def on_epoch(info):
        print(f"  epoch {info['epoch']:3d}  train_loss={info['train_loss']:.4f}  "
              f"zw_rmse={info['zone_weighted_rmse']:.4f}  "
              f"best={info['best_zw_rmse']:.4f}  elapsed={info['elapsed']:.1f}s")

    trainer = Trainer(cfg, model_name=name, registry_root=reg.models_root,
                       live=live, on_epoch=on_epoch)
    out = trainer.run()
    return {
        "name": name,
        "excluded": list(cfg.zones.excluded_zone_keys),
        "best_zone_weighted_rmse": out.best_zone_weighted_rmse,
        "epochs_run": out.n_epochs_run,
        "heatmap": str(out.heatmap_path) if out.heatmap_path else None,
        "per_zone_final": out.per_zone_final,
    }


def main():
    t0 = time.perf_counter()
    results = {}
    for name in EXPERIMENTS:
        try:
            results[name] = _run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
    (OUT_DIR / "holdout_results.json").write_text(json.dumps(results, indent=2, default=str))
    print()
    print(f"── total runtime = {time.perf_counter() - t0:.1f}s ──")
    print(f"summary written to {OUT_DIR / 'holdout_results.json'}")


if __name__ == "__main__":
    main()
