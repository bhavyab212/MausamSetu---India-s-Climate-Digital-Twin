"""
runtime.event_queue — thread-safe event bus for the dashboard.

Contract:
    * Producer (training thread) calls ``queue.put(event)`` from any thread.
    * Consumer (Streamlit fragment or Python test) calls
      ``queue.drain()`` to pull all pending events without blocking.
    * Every event carries: kind, timestamp (IST-naive UTC seconds), and
      a JSON-serialisable payload.
    * Zero events are lost while a consumer is not yet listening — the
      backing ``collections.deque`` grows unbounded until drained.
      The trainer only emits ~O(epochs × 5) heavy events per run; this
      is safe for any realistic training length.

Event kinds:
    STARTED            once, when the executor thread enters run()
    PROGRESS           every batch — small payload (batch, loss, lr, grad_norm, gpu%)
    TIER1              every N batches — global RMSE (cheap)
    TIER2              every epoch — per-zone RMSE + skill
    TIER3              every K epochs — full 9×4 cross-product
    TIER4              once at end — bootstrap CIs + Wilcoxon
    CHECKPOINT_SAVED   whenever a best-of checkpoint is written
    LOG                a plain-text line for the scrolling log
    HEATMAP_RENDERED   path to a freshly written PNG
    PAUSED / RESUMED   flag toggles
    STOPPED            cooperative stop honoured
    DONE               training completed normally (payload = RunOutputs dict)
    ERROR              exception raised (payload = {"type", "message"})

The producer NEVER prints. The consumer decides how to render.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    STARTED = "started"
    BATCH_TUNING = "batch_tuning"          # each probe fires one of these
    BATCH_TUNED = "batch_tuned"            # once, final pick
    PROGRESS = "progress"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    TIER4 = "tier4"
    CHECKPOINT_SAVED = "checkpoint_saved"
    LOG = "log"
    HEATMAP_RENDERED = "heatmap_rendered"
    PAUSED = "paused"
    RESUMED = "resumed"
    STOPPED = "stopped"
    THROUGHPUT_WARN = "throughput_warn"    # <50% GPU util for 30+ s
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """One event on the bus. Immutable so consumers can retain history safely."""
    kind: EventKind
    ts: float                        # seconds since epoch (UTC)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "ts": self.ts, "payload": self.payload}


def make_event(kind: EventKind, **payload: Any) -> Event:
    """Convenience constructor: ``make_event(EventKind.PROGRESS, batch=1, loss=0.4)``."""
    return Event(kind=kind, ts=time.time(), payload=payload)


class EventQueue:
    """Thread-safe unbounded event bus. Multiple producers, single consumer.

    Not a ``queue.Queue`` because we want ``drain_all()`` (pull everything
    pending in one non-blocking call) rather than one-at-a-time ``get()``.
    Streamlit consumers iterate on each rerun; a queue would force
    per-event calls with lock contention. A deque + Lock is simpler and
    faster for this pattern.
    """

    def __init__(self):
        self._q: deque[Event] = deque()
        self._lock = threading.Lock()
        self._history: list[Event] = []   # every event ever seen, for post-run review

    def put(self, event: Event) -> None:
        with self._lock:
            self._q.append(event)
            self._history.append(event)

    def emit(self, kind: EventKind, **payload: Any) -> None:
        """Convenience method used from the training thread."""
        self.put(make_event(kind, **payload))

    def drain(self) -> list[Event]:
        """Pull every pending event, non-blocking. Returns [] when empty."""
        with self._lock:
            if not self._q:
                return []
            out = list(self._q)
            self._q.clear()
            return out

    def peek(self) -> Event | None:
        """Return the newest event without removing anything. Debug aid."""
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self, kind: EventKind | None = None) -> list[Event]:
        """Return the entire recorded history, optionally filtered."""
        with self._lock:
            snap = list(self._history)
        if kind is None:
            return snap
        return [e for e in snap if e.kind == kind]

    def count(self, kind: EventKind | None = None) -> int:
        return len(self.history(kind))

    def to_jsonable(self) -> list[dict[str, Any]]:
        """Serialise the full history for on-disk logging."""
        return [e.to_dict() for e in self.history()]
