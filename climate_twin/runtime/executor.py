"""
runtime.executor — background training thread manager.

Owns the daemon thread that runs ``Trainer.run()``. Exposes:

    executor.start(cfg, model_name, ...)   spawn the thread
    executor.pause()                       cooperative pause between batches
    executor.resume()                      release the pause flag
    executor.stop()                        cooperative stop
    executor.state                         thread-safe RunState snapshot
    executor.events                        EventQueue (drain from any thread)
    executor.is_running                    liveness probe
    executor.join(timeout)                 wait for the thread

Contract:
  * ``start()`` refuses if a run is already alive.
  * Every state mutation goes through ``self._state_lock``.
  * Trainer instantiation happens INSIDE the thread so PyTorch's CUDA
    context is initialised on the worker, not on the Streamlit main
    thread (avoids "cuda handle from wrong process" issues under
    Streamlit rerun cycles).
  * Errors captured on the state + emitted as ``EventKind.ERROR``.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .device import RuntimeCudaRequired, describe_gpu, ensure_cuda
from .event_queue import EventKind, EventQueue
from .run_state import RunState, RunStatus, hash_config


class TrainingExecutor:
    """Thread-safe orchestrator for a single training run."""

    def __init__(self, state_dump_path: Path | None = None):
        self._thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._pause_event.set()                # "set" == not paused; clear() to pause
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._state = RunState()
        self._events = EventQueue()
        self._state_dump_path = state_dump_path

    # ── state accessors ─────────────────────────────────────────
    @property
    def state(self) -> RunState:
        """Return an atomic snapshot of the current run state."""
        with self._state_lock:
            return RunState(**{k: v for k, v in self._state.__dict__.items()})

    @property
    def events(self) -> EventQueue:
        return self._events

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._state.status in (
                RunStatus.STARTING.value,
                RunStatus.RUNNING.value,
                RunStatus.PAUSED.value,
                RunStatus.STOPPING.value,
            )

    def _update(self, **kwargs) -> None:
        with self._state_lock:
            for k, v in kwargs.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)
            if self._state_dump_path is not None:
                try:
                    self._state.dump(self._state_dump_path)
                except Exception:
                    pass

    # ── control primitives ──────────────────────────────────────
    def pause(self) -> None:
        """Ask the trainer to pause at the next batch boundary."""
        self._pause_event.clear()
        self._events.emit(EventKind.PAUSED)
        self._update(status=RunStatus.PAUSED.value, message="paused")

    def resume(self) -> None:
        """Release a paused trainer."""
        self._pause_event.set()
        self._events.emit(EventKind.RESUMED)
        self._update(status=RunStatus.RUNNING.value, message="resumed")

    def stop(self) -> None:
        """Ask the trainer to stop cleanly at the next batch boundary."""
        self._stop_event.set()
        # If paused, unblock so the trainer can honour the stop
        self._pause_event.set()
        self._update(status=RunStatus.STOPPING.value, message="stop requested")

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def wait_if_paused(self) -> None:
        """Block the trainer thread while paused. Returns immediately when
        resumed. Used at the batch boundary."""
        self._pause_event.wait()

    # ── start ───────────────────────────────────────────────────
    def start(
        self,
        cfg,
        model_name: str,
        registry_root: Path,
        parent_name: str = "",
    ) -> threading.Thread:
        """Launch the training thread. Refuses if one is already alive
        or if CUDA is unavailable."""
        if self.is_running or (self._thread is not None and self._thread.is_alive()):
            raise RuntimeError("training thread is already running")

        # Fail fast on the calling thread so the UI can render a red banner.
        # This is the "refuse to start on CPU" gate.
        gpu = ensure_cuda(context="training")

        self._pause_event.set()
        self._stop_event.clear()
        cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
        self._update(
            status=RunStatus.STARTING.value,
            model_name=model_name,
            region=cfg.data.region,
            device=gpu.summary(),
            total_epochs=cfg.optim.epochs,
            config_hash=hash_config(cfg_dict),
            zone_mask_sig=str(cfg.zones.zone_mask_sig_expected),
            manifest_sig=str(cfg.data.manifest_sig_expected or ""),
            start_time_epoch=time.time(),
            error_type="",
            error_message="",
            message="starting",
        )

        def _worker():
            try:
                # Late import so the runtime package can be imported without
                # loading torch/xarray until a run actually starts.
                from climate_twin.train.loop.trainer import Trainer

                self._update(status=RunStatus.RUNNING.value, message="running")
                self._events.emit(
                    EventKind.STARTED,
                    model_name=model_name,
                    region=cfg.data.region,
                    epochs=cfg.optim.epochs,
                    device=gpu.summary(),
                    config_hash=self._state.config_hash,
                )

                trainer = Trainer(
                    cfg,
                    model_name=model_name,
                    registry_root=registry_root,
                    parent_name=parent_name,
                    executor=self,           # trainer emits via self.events
                )
                outputs = trainer.run()

                if self.is_stop_requested():
                    self._update(status=RunStatus.STOPPED.value,
                                  message="stopped by user")
                    self._events.emit(EventKind.STOPPED)
                else:
                    self._update(status=RunStatus.FINISHED.value,
                                  message="training complete")
                    self._events.emit(
                        EventKind.DONE,
                        best_zone_weighted_rmse=outputs.best_zone_weighted_rmse,
                        best_state_dict_path=str(outputs.best_state_dict_path),
                        heatmap_path=(str(outputs.heatmap_path)
                                       if outputs.heatmap_path else None),
                        n_epochs_run=outputs.n_epochs_run,
                    )

            except Exception as e:
                tb = traceback.format_exc(limit=8)
                self._update(
                    status=RunStatus.ERRORED.value,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    message=f"error: {type(e).__name__}",
                )
                self._events.emit(EventKind.ERROR, type=type(e).__name__,
                                   message=str(e), traceback=tb)

        self._thread = threading.Thread(target=_worker, name="climate-twin-train",
                                         daemon=True)
        self._thread.start()
        return self._thread

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker to finish. Returns True if it terminated."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()
