"""
phaseA_measure.py — launch india_fast, capture Part-A numbers, stop.

Prints:
    picked batch size
    precision (auto-selected)
    VRAM used / total after batch tune
    GPU utilisation over the first ~90s of training
    samples/sec averaged over the same window
    time per epoch (measured from PROGRESS events)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from climate_twin.runtime import (
    TrainingExecutor,
    EventKind,
    describe_gpu,
    gpu_utilization_snapshot,
)
from climate_twin.train.config import load_config
from climate_twin.train.registry import get_registry


def main() -> int:
    print("── Part-A measurement run — india_fast ──")
    gpu = describe_gpu()
    print(f"GPU: {gpu.summary()}")

    cfg_path = REPO / "climate_twin/train/config/experiments/india_fast.yaml"
    cfg = load_config(cfg_path)
    print(f"config train_years={cfg.data.train_years}  epochs={cfg.optim.epochs}  "
          f"batch_size_auto={cfg.optim.batch_size_auto}  "
          f"mixed_precision_auto={cfg.optim.mixed_precision_auto}")

    execu = TrainingExecutor()
    reg = get_registry()
    execu.start(cfg=cfg, model_name="partA_measure",
                 registry_root=reg.models_root)

    # Watch tuning events specifically first
    print()
    print("── batch-size tuning probes ──")
    tuned = False
    prev_progress = 0
    t_start_watch = time.time()
    picked_bs = None
    picked_precision = None
    first_epoch_dt = None
    samples_seen: list[float] = []
    util_seen: list[float] = []
    vram_used: list[float] = []
    vram_total_mb = None
    epoch_start_ts: dict[int, float] = {}
    epoch_end_ts: dict[int, float] = {}

    # Stop after ~4 minutes or after epoch 2 completes (whichever first)
    MAX_WATCH_SECS = 240
    STOP_AFTER_EPOCH = 2

    while time.time() - t_start_watch < MAX_WATCH_SECS:
        for ev in execu.events.drain():
            if ev.kind == EventKind.BATCH_TUNING:
                p = ev.payload
                mark = "✓" if p.get("ok") else f"✗ {p.get('error', 'fail')}"
                print(f"  probe bs={p['bs']:4d}  peak={p.get('peak_mb', float('nan')):6.1f} MB  "
                      f"{p.get('seconds', 0):.2f}s  {mark}")
            elif ev.kind == EventKind.BATCH_TUNED:
                p = ev.payload
                picked_bs = p.get("picked_bs")
                vram_total_mb = p.get("total_vram_mb")
                print()
                print(f"  → picked batch size = {picked_bs}")
                print(f"  → reason = {p.get('reason')}")
                print(f"  → peak VRAM at last successful bs = "
                      f"{p.get('peak_vram_mb', float('nan')):.1f} / "
                      f"{p.get('total_vram_mb', float('nan')):.1f} MB")
                tuned = True
            elif ev.kind == EventKind.LOG and "precision" in ev.payload.get("message", "").lower():
                msg = ev.payload["message"]
                if msg.startswith("precision → "):
                    picked_precision = msg.split("→ ")[1].split(" ")[0]
                    print(f"  → precision = {picked_precision}")
            elif ev.kind == EventKind.PROGRESS:
                p = ev.payload
                if isinstance(p.get("samples_per_s"), (int, float)) and p["samples_per_s"] > 0:
                    samples_seen.append(float(p["samples_per_s"]))
                if isinstance(p.get("gpu_util_percent"), (int, float)):
                    util_seen.append(float(p["gpu_util_percent"]))
                if isinstance(p.get("vram_used_mb"), (int, float)):
                    vram_used.append(float(p["vram_used_mb"]))
                # Track epoch boundaries
                ep = p.get("epoch")
                if ep and ep not in epoch_start_ts:
                    epoch_start_ts[ep] = time.time()
                if ep and ep - 1 in epoch_start_ts and (ep - 1) not in epoch_end_ts:
                    epoch_end_ts[ep - 1] = time.time()
            elif ev.kind == EventKind.TIER2:
                ep = ev.payload.get("epoch")
                if ep and ep not in epoch_end_ts:
                    epoch_end_ts[ep] = time.time()
                    dt = epoch_end_ts[ep] - epoch_start_ts.get(ep, time.time())
                    print(f"  epoch {ep} done in {dt:.1f}s")
                if ep and ep >= STOP_AFTER_EPOCH:
                    print(f"  reached epoch {ep} — stopping")
                    execu.stop()
            elif ev.kind == EventKind.STOPPED or ev.kind == EventKind.DONE:
                break
            elif ev.kind == EventKind.ERROR:
                print(f"  ✗ ERROR {ev.payload.get('type')}: {ev.payload.get('message')}")
                if ev.payload.get("traceback"):
                    print(ev.payload["traceback"])
        if not execu.is_running:
            break
        time.sleep(0.5)

    execu.stop()
    execu.join(timeout=60)

    print()
    print("── Part-A summary ──")
    print(f"  picked batch size  = {picked_bs}")
    print(f"  precision           = {picked_precision}")
    if vram_used:
        print(f"  VRAM used (peak)    = {max(vram_used):.0f} / {vram_total_mb:.0f} MB  "
              f"({100*max(vram_used)/vram_total_mb:.0f}%)")
    if util_seen:
        import statistics
        print(f"  GPU util            = median {statistics.median(util_seen):.0f}%  "
              f"max {max(util_seen):.0f}%  n={len(util_seen)}")
    if samples_seen:
        import statistics
        print(f"  samples/sec         = median {statistics.median(samples_seen):.1f}  "
              f"max {max(samples_seen):.1f}  n={len(samples_seen)}")
    completed = [(ep, epoch_end_ts[ep] - epoch_start_ts[ep])
                 for ep in sorted(epoch_end_ts) if ep in epoch_start_ts]
    if completed:
        print("  epoch times:")
        for ep, dt in completed:
            print(f"    epoch {ep}: {dt:.1f}s")

    # Dump events + state for the STOP report
    out = REPO / "climate_twin/_phase0/partA"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps(execu.events.to_jsonable(),
                                                  indent=2, default=str))
    (out / "state.json").write_text(execu.state.to_json())
    print(f"\n  event log: {out / 'events.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
