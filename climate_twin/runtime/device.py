"""
runtime.device — CUDA-required policy + GPU utilization snapshot.

The dashboard refuses to train without CUDA. There is no silent fallback.
Inference and validation follow the same rule when the user explicitly
selects an ensemble run; small helper routines (metric aggregation,
plotting, PDF export) stay on CPU.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import torch


class RuntimeCudaRequired(RuntimeError):
    """Raised when training or inference is invoked without a CUDA GPU."""


CUDA_UNAVAILABLE_MESSAGE = (
    "CUDA is not available on this machine. Training and inference in "
    "ClimateTwin Lab require a CUDA-capable GPU with a matching PyTorch "
    "build. This is intentional: silent CPU fallback would report metrics "
    "at ~50× the real GPU rate and hide performance regressions.\n\n"
    "Fix by installing PyTorch with CUDA (e.g. torch 2.5.1+cu121) and "
    "verifying with:\n"
    "    python -c \"import torch; print(torch.cuda.is_available())\"\n"
)


@dataclass(frozen=True)
class DeviceInfo:
    """One-shot description of the CUDA device the trainer will use."""
    available: bool
    device_index: int
    name: str
    total_memory_gb: float
    driver_version: str
    torch_version: str
    torch_cuda_version: str | None

    def summary(self) -> str:
        if not self.available:
            return "no CUDA device"
        return (
            f"cuda:{self.device_index}  {self.name}  "
            f"{self.total_memory_gb:.1f} GB  "
            f"driver {self.driver_version}  "
            f"torch {self.torch_version}"
            + (f" (cuda {self.torch_cuda_version})" if self.torch_cuda_version else "")
        )


def describe_gpu(device_index: int = 0) -> DeviceInfo:
    """Return a snapshot of the primary GPU. Never raises — returns
    ``DeviceInfo(available=False)`` when CUDA is unavailable so the UI
    can render the banner."""
    if not torch.cuda.is_available():
        return DeviceInfo(
            available=False,
            device_index=-1,
            name="",
            total_memory_gb=0.0,
            driver_version="",
            torch_version=torch.__version__,
            torch_cuda_version=None,
        )
    props = torch.cuda.get_device_properties(device_index)
    driver = ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=3,
            ).strip().splitlines()
            driver = out[0] if out else ""
        except Exception:
            driver = ""
    return DeviceInfo(
        available=True,
        device_index=device_index,
        name=props.name,
        total_memory_gb=props.total_memory / (1024 ** 3),
        driver_version=driver,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
    )


def ensure_cuda(context: str = "training") -> DeviceInfo:
    """Refuse to proceed unless a CUDA GPU is available.

    Args:
        context: user-facing string used in the raised message, e.g.
            "training" or "ensemble inference".

    Raises:
        RuntimeCudaRequired: on any CPU-only environment.

    Returns:
        DeviceInfo describing the GPU that will be used.
    """
    if not torch.cuda.is_available():
        raise RuntimeCudaRequired(
            f"{context.capitalize()} requires CUDA.\n\n{CUDA_UNAVAILABLE_MESSAGE}"
        )
    return describe_gpu(0)


def gpu_utilization_snapshot(device_index: int = 0) -> dict[str, Any]:
    """Return a lightweight snapshot of GPU load for the live view.

    Never blocks the training thread. Falls back to torch-internal counters
    if nvidia-smi is not on PATH.
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "util_percent": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "memory_reserved_mb": None,
            "temperature_c": None,
            "power_w": None,
        }
    # torch-internal (always available). Prefer *peak* allocation over the
    # instantaneous value — the instantaneous number drops to near-zero
    # between batches (activations freed at optimizer.step) and hides the
    # real footprint. Peak-since-last-reset is what the user cares about.
    alloc_mb = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 2)

    util = None
    temp = None
    power = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--id={device_index}",
                 "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=1.0,
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 3:
                util = float(parts[0]) if parts[0].replace(".", "", 1).isdigit() else None
                temp = float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else None
                power = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else None
        except Exception:
            pass

    return {
        "available": True,
        "util_percent": util,
        "memory_used_mb": round(alloc_mb, 1),
        "memory_reserved_mb": round(reserved_mb, 1),
        "memory_total_mb": round(total_mb, 1),
        "temperature_c": temp,
        "power_w": power,
    }
