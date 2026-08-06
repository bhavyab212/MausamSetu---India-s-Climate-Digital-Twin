"""
climate_twin.runtime — the execution spine.

Everything the dashboard needs to run training as a Python function
(never a subprocess) lives here:

    device.py       — CUDA detect + refuse-on-CPU policy
    event_queue.py  — thread-safe event bus (progress, tier1..4, done, error)
    run_state.py    — pickle-serializable run state
    executor.py     — background training thread manager
"""
from .device import (
    CUDA_UNAVAILABLE_MESSAGE,
    DeviceInfo,
    RuntimeCudaRequired,
    describe_gpu,
    ensure_cuda,
    gpu_utilization_snapshot,
)
from .event_queue import (
    Event,
    EventKind,
    EventQueue,
    make_event,
)
from .run_state import RunState, RunStatus
from .executor import TrainingExecutor
from .tuning import TuneResult, choose_precision, tune_batch_size

__all__ = [
    "CUDA_UNAVAILABLE_MESSAGE",
    "DeviceInfo",
    "RuntimeCudaRequired",
    "describe_gpu",
    "ensure_cuda",
    "gpu_utilization_snapshot",
    "Event",
    "EventKind",
    "EventQueue",
    "make_event",
    "RunState",
    "RunStatus",
    "TrainingExecutor",
    "TuneResult",
    "choose_precision",
    "tune_batch_size",
]
