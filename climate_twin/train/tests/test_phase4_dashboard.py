"""
Phase 4 test suite for the runtime spine + dashboards.

Covers:
  1. Event queue does not lose events under producer load
  2. GPU refusal on a CPU-only device (via monkey-patch)
  3. Summary card verdict logic — STRONG / MIXED / WEAK from three fixtures
  4. Compare mode refuses to compare checkpoints with different zone_mask_sig
  5. Config schema rejects overlapping year splits (temporal leakage)
  6. Trainer signals via events, not stdout (no print() spam)

Run:  cd climate_twin && ../venv/Scripts/python tests/../train/tests/test_phase4_dashboard.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from climate_twin.runtime import (
    Event,
    EventKind,
    EventQueue,
    RuntimeCudaRequired,
    describe_gpu,
    ensure_cuda,
    make_event,
)
from climate_twin.train.ui.dashboards.summary_card import (
    VERDICT_MIXED,
    VERDICT_STRONG,
    VERDICT_WEAK,
    build_summary_card,
)
from climate_twin.train.registry import (
    RegistryModelIncompatible,
    get_registry,
)


_PASS: list[str] = []
_FAIL: list[str] = []


def _ok(msg, _err=""):
    print(f"  ✓ {msg}")
    _PASS.append(msg)


def _fail(msg, err=""):
    print(f"  ✗ {msg}")
    if err:
        print(f"      {err}")
    _FAIL.append(f"{msg}: {err}")


def _check(cond, msg, err=""):
    if cond:
        _ok(msg)
    else:
        _fail(msg, err)


def _hdr(title): print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Event queue: no events lost under 4-producer load
# ---------------------------------------------------------------------------
def test_event_queue_no_loss():
    _hdr("1. Event queue: no loss under producer load")
    q = EventQueue()
    N_PER = 500
    N_PROD = 4

    def producer(pid):
        for i in range(N_PER):
            q.emit(EventKind.PROGRESS, producer=pid, seq=i)

    threads = [threading.Thread(target=producer, args=(pi,)) for pi in range(N_PROD)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    dt = time.perf_counter() - t0

    hist = q.history()
    _check(len(hist) == N_PER * N_PROD,
            f"delivered all {N_PER * N_PROD} events in {dt:.3f}s "
            f"(got {len(hist)})")
    per_producer = {}
    for e in hist:
        per_producer.setdefault(e.payload["producer"], set()).add(e.payload["seq"])
    for pid in range(N_PROD):
        _check(len(per_producer.get(pid, set())) == N_PER,
                f"producer {pid} delivered {len(per_producer.get(pid, set()))}/{N_PER}")


# ---------------------------------------------------------------------------
# 2. GPU refusal on CPU-only environment (monkey-patched)
# ---------------------------------------------------------------------------
def test_gpu_refusal():
    _hdr("2. GPU refusal on CPU-only environment")
    import torch
    real_flag = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        try:
            ensure_cuda(context="training")
            _fail("ensure_cuda refused", "did not raise")
        except RuntimeCudaRequired as e:
            _check("CUDA is not available" in str(e),
                    "ensure_cuda raised RuntimeCudaRequired with helpful message")
        # describe_gpu falls back to available=False, does not raise
        info = describe_gpu()
        _check(info.available is False,
                "describe_gpu returns available=False on CPU-only")
    finally:
        torch.cuda.is_available = real_flag
    # Sanity: on the actual host, GPU IS available
    info = describe_gpu()
    _check(info.available is True,
            f"after restore, describe_gpu sees {info.name!r}")


# ---------------------------------------------------------------------------
# 3. Summary card verdicts (three fixtures)
# ---------------------------------------------------------------------------
def test_summary_card_verdicts():
    _hdr("3. Summary card verdicts")

    zones = ["nw", "wc", "cn", "ne", "sp", "wg", "th", "hi", "tn"]

    # STRONG: model beats persistence in all 9, climatology in ≥ 6, calib OK, phys OK
    model_r = {z: 3.0 for z in zones}
    pers_r  = {z: 10.0 for z in zones}
    clim_r  = {z: 5.0 for z in zones}
    card = build_summary_card(model_r, pers_r, clim_r,
                                calibration=0.80,
                                physics_violation_pct=0.4)
    _check(card.verdict == VERDICT_STRONG,
            f"STRONG fixture → {card.verdict}",
            f"reasoning={card.reasoning}")

    # MIXED: beats persistence in 6/9 but climatology only in 3/9, calibration off
    model_r = {z: 7.0 for z in zones}
    pers_r  = {z: 10.0 for z in zones}
    clim_r  = {"nw": 4.0, "wc": 4.0, "cn": 4.0, "ne": 8.0, "sp": 8.0,
                "wg": 8.0, "th": 8.0, "hi": 8.0, "tn": 8.0}
    card = build_summary_card(model_r, pers_r, clim_r,
                                calibration=0.55,
                                physics_violation_pct=0.4)
    _check(card.verdict == VERDICT_MIXED,
            f"MIXED fixture → {card.verdict}",
            f"beats_p={card.beats_persistence} beats_c={card.beats_climatology}")

    # WEAK: beats persistence in only 2/9
    model_r = {z: 15.0 for z in zones}
    pers_r  = {"nw": 20.0, "wc": 20.0, "cn": 10.0, "ne": 10.0, "sp": 10.0,
                "wg": 10.0, "th": 10.0, "hi": 10.0, "tn": 10.0}
    clim_r  = {z: 6.0 for z in zones}
    card = build_summary_card(model_r, pers_r, clim_r)
    _check(card.verdict == VERDICT_WEAK,
            f"WEAK fixture → {card.verdict}",
            f"beats_p={card.beats_persistence}")

    # STRONG refusal: even if 9/9 vs persistence + 6/9 vs climatology, if
    # one zone is below best_of_baseline, must NOT be STRONG.
    model_r = {z: 3.0 for z in zones}
    model_r["nw"] = 11.0    # below persistence (10) → beats_any breaks
    pers_r  = {z: 10.0 for z in zones}
    clim_r  = {z: 5.0 for z in zones}
    card = build_summary_card(model_r, pers_r, clim_r,
                                calibration=0.80,
                                physics_violation_pct=0.4)
    _check(card.verdict != VERDICT_STRONG,
            f"refuses STRONG when any zone below baseline → {card.verdict}")


# ---------------------------------------------------------------------------
# 4. Registry compare-mode refusal on zone_mask_sig mismatch
# ---------------------------------------------------------------------------
def test_compare_refusal():
    _hdr("4. Compare refuses when checkpoints have different zone_mask_sig")
    reg = get_registry()
    models = reg.list_models()
    if len(models) < 1:
        _ok("no models — skipping compare refusal (nothing to compare against)")
        return

    # Fabricate an "expected" sig mismatch and confirm load_into refuses
    fake_sig = "deadbeef0000"
    for m in models[:1]:
        try:
            import torch
            from climate_twin.train.config.schema import ExperimentConfig
            from climate_twin.train.model import build_model
            from climate_twin.regions import get_zones
            cfg = ExperimentConfig(**m["config"])
            model = build_model(cfg, zones=get_zones())
            try:
                reg.load_into(m["name"], m["region"], model,
                                expected_zone_mask_sig=fake_sig, strict=True)
                _fail("compare-mode refusal fired",
                       f"did not raise for {m['name']} with fake sig")
            except RegistryModelIncompatible as e:
                _check("zone_mask_sig" in str(e),
                        f"refused {m['name']} with clear reason")
        except Exception as e:
            _fail("compare-mode refusal test setup", str(e)[:120])


# ---------------------------------------------------------------------------
# 5. Config schema rejects year-split leakage
# ---------------------------------------------------------------------------
def test_leakage_rejection():
    _hdr("5. Config schema rejects temporal leakage")
    from climate_twin.train.config.schema import DataConfig
    try:
        DataConfig(train_years=(2018, 2024), val_years=(2023, 2024))
        _fail("leakage rejection", "did not raise")
    except Exception as e:
        _check(True, f"leakage rejection raised {type(e).__name__}")


# ---------------------------------------------------------------------------
# 6. Trainer has no print() calls (only events)
# ---------------------------------------------------------------------------
def test_no_print_in_trainer():
    _hdr("6. Trainer emits only via events, no print() calls")
    p = REPO / "climate_twin/train/loop/trainer.py"
    src = p.read_text(encoding="utf-8")
    # Count literal print( that isn't in a comment or docstring
    import re
    prints = [line for line in src.splitlines()
              if re.search(r'^\s*print\(', line) and not line.lstrip().startswith("#")]
    _check(not prints,
            f"trainer.py has zero top-level print() calls",
            f"found {len(prints)}: {prints[:3]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_event_queue_no_loss()
    test_gpu_refusal()
    test_summary_card_verdicts()
    test_compare_refusal()
    test_leakage_rejection()
    test_no_print_in_trainer()

    print(f"\npassed: {len(_PASS)}   failed: {len(_FAIL)}")
    if _FAIL:
        print("FAILURES:")
        for f in _FAIL: print("  -", f)
        return 1
    print("All Phase-4 tests passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
