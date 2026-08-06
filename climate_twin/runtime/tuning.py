"""
runtime.tuning — auto-tune batch size and mixed-precision to saturate the GPU.

Two helpers:

``choose_precision(policy)``
    Return one of "bf16", "fp16", "fp32".
    * "auto"    → bf16 on Ampere (SM 8.0) or newer; fp16 on older CUDA;
                  fp32 only when the caller forces it.
    * anything else → returned verbatim.

``tune_batch_size(build_fn, start=8, cap=256, safety=0.85, emit=None)``
    Doubles the batch size on each successful (forward + backward + step)
    probe. On the first ``torch.cuda.OutOfMemoryError``, backs off to
    ``floor(safety × last_success)``, capped at ``cap``.

    ``build_fn(bs)`` is a caller-provided closure that must:
      * build the model on the current CUDA device
      * synthesise a single batch of size ``bs`` on the same device
      * run one forward, one loss, one backward, one optimizer.step()
      * return the peak VRAM used (in MB, ``torch.cuda.max_memory_allocated``)
    The closure is responsible for cleaning up (call
    ``torch.cuda.empty_cache()`` at the end). We call it once per probe.

    Returns ``TuneResult(batch_size, precision, peak_vram_mb, tried,
    picked_reason, safety, cap)``.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class TuneResult:
    batch_size: int
    precision: str
    peak_vram_mb: float
    total_vram_mb: float
    tried: list[dict[str, Any]]            # [{bs, ok, peak_mb, seconds, error?}]
    picked_reason: str
    safety: float
    cap: int


# ---------------------------------------------------------------------------
# Precision auto-select
# ---------------------------------------------------------------------------
def choose_precision(policy: str) -> tuple[str, str]:
    """Return ``(precision, reason)``.

    - ``bf16`` on any GPU with SM ≥ 8.0 (Ampere onward: RTX 30xx, 40xx,
      A100, H100). bf16 has fp32 exponent range so it is drift-safe for
      long training. Numerically indistinguishable from bf16 tensor cores.
    - ``fp16`` on Volta / Turing (SM 7.x). Needs GradScaler.
    - ``fp32`` when the caller explicitly forces it (policy != "auto") or
      when CUDA reports SM < 7 (very rare on modern PyTorch installs).
    """
    if policy != "auto":
        return policy, f"forced by config (mixed_precision={policy!r})"
    if not torch.cuda.is_available():
        return "fp32", "no CUDA — fp32 (only used by tests)"
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return "bf16", f"CUDA SM {major}.{minor} ≥ 8.0 → bf16 (Ampere+ tensor cores)"
    if major == 7:
        return "fp16", f"CUDA SM {major}.{minor} — fp16 with GradScaler"
    return "fp32", f"CUDA SM {major}.{minor} < 7.0 — fp32 fallback"


# ---------------------------------------------------------------------------
# Batch-size tuner
# ---------------------------------------------------------------------------
def tune_batch_size(
    build_fn: Callable[[int], float],
    start: int = 8,
    cap: int = 256,
    safety: float = 0.85,
    emit: Callable[[dict], None] | None = None,
) -> TuneResult:
    """Probe doubling batch sizes until CUDA OOM; return the safety-scaled
    winner. Any Python exception other than OOM aborts the tune.

    ``emit(dict)`` fires once per probe with ``{bs, ok, peak_mb, seconds,
    error}`` so the UI can render live tuning progress.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("tune_batch_size called without CUDA")

    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    # Windows WDDM lets torch spill into shared system memory when a probe
    # exceeds physical VRAM — the probe "succeeds" but the real training
    # loop's pin_memory pathway (which is hard-VRAM-only) will OOM. Treat
    # anything above 95% of physical VRAM as effective OOM.
    VRAM_HARD_CAP_MB = total_mb * 0.95
    tried: list[dict[str, Any]] = []
    last_ok_bs: int | None = None
    last_ok_peak: float = 0.0
    bs = int(start)
    while bs <= cap:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            peak_mb = float(build_fn(bs))
            dt = time.perf_counter() - t0
            # Reject probes that succeeded only by spilling into shared memory.
            if peak_mb > VRAM_HARD_CAP_MB:
                entry = {"bs": bs, "ok": False, "peak_mb": peak_mb,
                          "seconds": round(dt, 2),
                          "error": f"peak {peak_mb:.0f} MB > "
                                    f"{VRAM_HARD_CAP_MB:.0f} MB VRAM cap "
                                    f"(shared-memory spill would OOM in real run)"}
                tried.append(entry)
                if emit is not None:
                    try: emit(entry)
                    except Exception: pass
                torch.cuda.empty_cache()
                gc.collect()
                break
            entry = {"bs": bs, "ok": True, "peak_mb": peak_mb,
                      "seconds": round(dt, 2), "error": None}
            tried.append(entry)
            if emit is not None:
                try: emit(entry)
                except Exception: pass
            last_ok_bs = bs
            last_ok_peak = peak_mb
            bs *= 2
        except torch.cuda.OutOfMemoryError as e:
            dt = time.perf_counter() - t0
            entry = {"bs": bs, "ok": False, "peak_mb": float("nan"),
                      "seconds": round(dt, 2), "error": "OOM"}
            tried.append(entry)
            if emit is not None:
                try: emit(entry)
                except Exception: pass
            torch.cuda.empty_cache()
            gc.collect()
            break
        except RuntimeError as e:
            # Some builds surface OOM as generic RuntimeError with the
            # substring "out of memory". Handle that too.
            msg = str(e).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                dt = time.perf_counter() - t0
                entry = {"bs": bs, "ok": False, "peak_mb": float("nan"),
                          "seconds": round(dt, 2), "error": "OOM (runtime)"}
                tried.append(entry)
                if emit is not None:
                    try: emit(entry)
                    except Exception: pass
                torch.cuda.empty_cache()
                gc.collect()
                break
            raise

    if last_ok_bs is None:
        # Even bs=start OOMed. Return the failed report; caller must decide.
        return TuneResult(batch_size=0, precision="",
                           peak_vram_mb=float("nan"), total_vram_mb=total_mb,
                           tried=tried,
                           picked_reason=f"OOM even at batch size {start}",
                           safety=safety, cap=cap)

    picked = min(max(int(last_ok_bs * safety), 1), cap)
    reason = (f"largest OK bs={last_ok_bs} (peak {last_ok_peak:.0f}/{total_mb:.0f} MB) "
              f"× safety {safety:.2f} → {picked}"
              + (f", capped at {cap}" if last_ok_bs * safety > cap else ""))
    return TuneResult(batch_size=picked, precision="", peak_vram_mb=last_ok_peak,
                       total_vram_mb=total_mb, tried=tried,
                       picked_reason=reason, safety=safety, cap=cap)
