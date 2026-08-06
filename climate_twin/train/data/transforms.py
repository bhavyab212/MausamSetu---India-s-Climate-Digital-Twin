"""
train.data.transforms — per-zone z-score normalisation (train-only stats).

The frozen ``regions/zone_stats.json`` contains per-zone (mean, std) fitted
on train years 1951-2022 only. This module produces:

* an ``(H, W)`` per-cell mean tensor  = Σ_k membership[k] · zone_k.mean
* an ``(H, W)`` per-cell std tensor   = Σ_k membership[k] · zone_k.std

Both derived from soft membership so boundary cells get a smoothly
interpolated normalisation — this is what makes "the atmosphere has no
hard boundary" work at the input-normalisation layer.

Contract:
    normalise(x_physical)  = (x - mean) / std
    denormalise(x_z)       =  x * std + mean

NaN targets stay NaN (never zero-filled). Cells outside any zone (mask==0)
get mean=0, std=1 so they contribute no signal but don't produce NaN.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from climate_twin.regions import get_zones


HERE_STATS = Path(__file__).resolve().parents[2] / "regions" / "zone_stats.json"


class PerZoneZScore:
    """Load per-zone (mean, std) for a set of variables and produce ``(C, H, W)``
    per-cell mean and std tensors ready to broadcast against a batch.

    Example
    -------
    >>> Z = get_zones()
    >>> tf = PerZoneZScore(("rain", "tmax", "tmin"), zones=Z)
    >>> x_z = tf.normalise(x_phys)             # x_phys: (B, T, C, H, W)
    >>> x_back = tf.denormalise(x_z)
    """

    def __init__(self, variables: tuple[str, ...], zones=None, stats_path: Path | None = None):
        self.variables = tuple(variables)
        self.zones = zones if zones is not None else get_zones()
        stats_path = stats_path or HERE_STATS
        if not stats_path.exists():
            raise FileNotFoundError(
                f"zone_stats.json missing at {stats_path}. Run Phase 1: "
                "`python -m climate_twin.regions.fit_zone_stats`."
            )
        self._stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.zone_stats_sig = self._stats.get("zone_mask_sig")
        if self.zone_stats_sig != self.zones.mask_signature:
            raise RuntimeError(
                f"zone_stats.json was fit against mask {self.zone_stats_sig!r} "
                f"but on-disk zone registry is {self.zones.mask_signature!r}. "
                "Refit stats with `python -m climate_twin.regions.fit_zone_stats`."
            )
        self._mean, self._std = self._build_tensors()

    def _build_tensors(self):
        Z = self.zones
        H, W = Z.hard_mask.shape
        C = len(self.variables)
        memb = Z.membership.astype(np.float32)           # (H, W, K)
        K = memb.shape[-1]

        # (C, K) per-variable per-zone stats
        mean_zk = np.zeros((C, K), dtype=np.float32)
        std_zk = np.ones((C, K), dtype=np.float32)
        for ci, var in enumerate(self.variables):
            for k, zone in enumerate(Z.zones):
                sn = self._stats["zones"][zone.key]["norm"].get(var)
                if sn is None:
                    continue
                if sn["std"] is None or not np.isfinite(float(sn["std"])) or float(sn["std"]) <= 0:
                    continue
                mean_zk[ci, k] = float(sn["mean"])
                std_zk[ci, k] = float(sn["std"])

        # Broadcast to (C, H, W): weight per-zone stats by soft membership
        mean_hw = np.einsum("ck,hwk->chw", mean_zk, memb).astype(np.float32)
        std_hw = np.einsum("ck,hwk->chw", std_zk, memb).astype(np.float32)

        # Outside-any-zone cells: mean=0, std=1 so normalise is identity
        out_of_zone = (Z.hard_mask == 0)
        for ci in range(C):
            mean_hw[ci][out_of_zone] = 0.0
            std_hw[ci][out_of_zone] = 1.0

        # Clamp std to a floor so per-cell division never explodes
        std_hw = np.maximum(std_hw, 1e-3)
        return mean_hw, std_hw

    @property
    def mean_np(self) -> np.ndarray:
        """``(C, H, W)`` per-cell mean, ready to broadcast."""
        return self._mean

    @property
    def std_np(self) -> np.ndarray:
        return self._std

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """x: ``(..., C, H, W)`` physical → z-score. NaNs preserved."""
        return (x - self._mean) / self._std

    def denormalise(self, x_z: np.ndarray) -> np.ndarray:
        return x_z * self._std + self._mean

    def denormalise_channel(self, x_ci: np.ndarray, ci: int) -> np.ndarray:
        """Denormalise a single-channel prediction. ``x_ci`` shape: ``(..., H, W)``."""
        return x_ci * self._std[ci] + self._mean[ci]
