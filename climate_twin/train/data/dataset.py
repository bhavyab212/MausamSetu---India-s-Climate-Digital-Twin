"""
train.data.dataset — daily windowed dataset for one region.

For each ``i`` in the requested year-split (train / val / test), yields:

    input   : (T, C, H, W) normalised gauge sequence — days [i, i+T)
    target  : (C, H, W)     — day [i+T]
    doy     : int            day-of-year of the target (1..366)
    month   : int            month of the target
    time    : ISO date string of the target (for logging)

Sub-arrays are torch.Tensor at ``__getitem__`` time; the underlying storage
stays numpy on the cube's memory map. Every window's target is guaranteed
to fall inside the split; the input can start inside or slightly before
the split boundary (we allow the first `T-1` context days to come from
the previous year — that is not leakage, just prior context).

Contract:
  * Uses the frozen zone mask + soft membership.
  * Applies per-zone z-score normalisation on the fly.
  * Never zero-fills NaN — the loss handles missing cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from climate_twin.regions import ZoneRegistry, get_zones
from climate_twin import data_source as DS

from .transforms import PerZoneZScore


@dataclass
class WindowMeta:
    idx_target: int          # cube time index of the target day
    doy: int
    month: int
    year: int
    iso_date: str


class DailyWindowDataset(Dataset):
    """Slice a ``(T, C, H, W)`` window ending at each day in ``year_range``.

    Args:
        region: "india" or "cauvery"
        variables: ordered tuple of channel names
        seq_length: number of input days
        year_range: inclusive (start, end) — the target day must fall in this range
        zones: ZoneRegistry (defaults to get_zones())
        transform: an optional PerZoneZScore (created here if None)
        include_satellite: reserved for Phase 5; kept in cube read path

    The dataset holds a reference to the opened xarray Dataset for its
    lifetime — cheap because xarray uses memory-mapped netCDF. Close on
    garbage-collect via `__del__`.
    """

    def __init__(
        self,
        region: str,
        variables: tuple[str, ...],
        seq_length: int,
        year_range: tuple[int, int],
        zones: ZoneRegistry | None = None,
        transform: PerZoneZScore | None = None,
        include_satellite: bool = False,
    ):
        self.region = region
        self.variables = tuple(variables)
        self.seq_length = int(seq_length)
        self.year_range = tuple(year_range)
        self.zones = zones if zones is not None else get_zones()
        self.transform = transform or PerZoneZScore(self.variables, zones=self.zones)
        self.include_satellite = bool(include_satellite)

        # Build metadata once from a temporary handle; drop it before we ever
        # get pickled to a DataLoader worker. Every __getitem__ (including the
        # first in each worker) will lazily open its own handle.
        ds = DS.load_region(region)
        years = np.asarray(ds["time.year"].values)
        months = np.asarray(ds["time.month"].values)
        doys = np.asarray(ds["time.dayofyear"].values)
        times = np.asarray(ds["time"].values)
        self._H = int(ds.sizes["lat"])
        self._W = int(ds.sizes["lon"])
        ds.close()

        in_range = (years >= self.year_range[0]) & (years <= self.year_range[1])
        target_idx = np.where(in_range)[0]
        target_idx = target_idx[target_idx >= self.seq_length]
        self._target_idx = target_idx
        self._doy = doys
        self._months = months
        self._years = years
        self._times = times

        # xarray handles below are OPENED LAZILY per instance / per worker.
        # Setting them to None here means the __getstate__ / __setstate__
        # cycle done by torch's DataLoader spawn on Windows is safe.
        self._ds = None
        self._var_arrays = None

    # ── worker-safe handle management ────────────────────────
    def _ensure_ds(self):
        if self._ds is None:
            self._ds = DS.load_region(self.region)
            self._var_arrays = {v: self._ds[v] for v in self.variables}

    def __getstate__(self):
        """Drop the open netCDF handle before pickling to a worker."""
        state = self.__dict__.copy()
        state["_ds"] = None
        state["_var_arrays"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ds = None
        self._var_arrays = None

    def __len__(self) -> int:
        return int(self._target_idx.size)

    def __getitem__(self, i: int) -> dict:
        self._ensure_ds()
        ti = int(self._target_idx[i])
        start = ti - self.seq_length
        stop = ti + 1                     # inclusive target

        # Read one contiguous slice from the cube (T+1 days, C vars)
        C = len(self.variables)
        T = self.seq_length
        H = self._H
        W = self._W
        buf = np.zeros((T + 1, C, H, W), dtype=np.float32)
        for ci, v in enumerate(self.variables):
            slab = self._var_arrays[v].isel(time=slice(start, stop)).values  # (T+1, H, W)
            buf[:, ci] = slab.astype(np.float32)

        # Normalise (train-years-only stats fitted per zone)
        z = self.transform.normalise(buf)   # (T+1, C, H, W)
        x = torch.from_numpy(z[:T].copy())        # (T, C, H, W)
        y = torch.from_numpy(buf[T].copy())       # (C, H, W)  — target stays in physical units
        # ^ targets stay physical because the loss uses physical thresholds
        # (rain occurrence at 0.1 mm) — z-scoring targets would require
        # threshold rescaling and is not worth the complication.
        return {
            "x": x,
            "y": y,
            "doy": int(self._doy[ti]),
            "month": int(self._months[ti]),
            "year": int(self._years[ti]),
            "iso_date": str(self._times[ti])[:10],
            "target_idx": ti,
        }

    def close(self):
        try:
            self._ds.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


def default_collate(batch: list[dict]) -> dict:
    """Simple collate that stacks tensors and gathers meta as lists."""
    out = {}
    out["x"] = torch.stack([b["x"] for b in batch], dim=0)
    out["y"] = torch.stack([b["y"] for b in batch], dim=0)
    for k in ("doy", "month", "year", "target_idx", "iso_date"):
        out[k] = [b[k] for b in batch]
    return out
