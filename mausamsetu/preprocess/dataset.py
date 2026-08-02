"""
dataset.py
===========
PyTorch Dataset wrapper for the Cauvery processed NetCDF.

WHAT IT DOES
------------
Reads cauvery.nc and generates (INPUT, TARGET) pairs by SLIDING WINDOW:

  Example i:
     INPUT  = days [i, i+1, ..., i+5]     → shape (6, 19, 17, 5)
     TARGET = days [i+6, i+7, ..., i+12]  → shape (7, 19, 17, 3)

  5 input channels: rain_anom, tmax_anom, tmin_anom, insat_lst, insat_rain
  3 output vars:    rain_anom, tmax_anom, tmin_anom

We use ANOMALIES (deviation from climatology), not raw values, because:
  - the AI focuses on the "interesting deviation from normal"
  - reduces the size of numbers it has to learn
  - post-processing adds climatology back

TRAIN / VAL / TEST split is by YEAR (not random) — this is critical for
honest evaluation. Random splitting would leak future info into training.
"""
from __future__ import annotations
import json
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

from mausamsetu import config


# ============================================================================
# CHANNEL DEFINITIONS
# ============================================================================
INPUT_VARS  = ["rain_anom", "tmax_anom", "tmin_anom", "insat_lst", "insat_rain"]
OUTPUT_VARS = ["rain_anom", "tmax_anom", "tmin_anom"]


# ============================================================================
# CORE DATASET
# ============================================================================
class CauveryWindowDataset(Dataset):
    """
    A PyTorch Dataset that yields (input, target) tensor pairs.

    Parameters
    ----------
    split : "train" | "val" | "test"
    input_days : int, default = 6
    forecast_days : int, default = 7
    normalize : bool, default True — standardise to z-scores using train stats
    """

    def __init__(
        self,
        split: str = "train",
        input_days: int = config.INPUT_DAYS,
        forecast_days: int = config.FORECAST_DAYS,
        normalize: bool = True,
    ):
        assert split in {"train", "val", "test"}, f"Unknown split: {split}"
        self.split = split
        self.input_days = input_days
        self.forecast_days = forecast_days
        self.window = input_days + forecast_days
        self.normalize = normalize

        # --- Load processed dataset ---
        if not config.CAUVERY_NC.exists():
            raise FileNotFoundError(
                f"{config.CAUVERY_NC} not found. Run: "
                f"python -m mausamsetu.preprocess.build_dataset"
            )
        ds = xr.open_dataset(config.CAUVERY_NC)

        # --- Filter by year ---
        years_map = {
            "train": config.TRAIN_YEARS,
            "val":   config.VAL_YEARS,
            "test":  config.TEST_YEARS,
        }
        year_mask = ds.time.dt.year.isin(years_map[split])
        ds = ds.sel(time=year_mask)

        # --- Extract channel arrays as float32 numpy ---
        # Shape: (T, H, W)
        self.time_axis = ds.time.values
        self.lat = ds.lat.values.astype(np.float32)
        self.lon = ds.lon.values.astype(np.float32)

        # Stack into (T, H, W, C)
        input_stack = np.stack(
            [np.nan_to_num(ds[v].values.astype(np.float32), nan=0.0) for v in INPUT_VARS],
            axis=-1,
        )
        target_stack = np.stack(
            [np.nan_to_num(ds[v].values.astype(np.float32), nan=0.0) for v in OUTPUT_VARS],
            axis=-1,
        )

        # --- Load normalization stats (train-only, from attrs) ---
        stats_json = ds.attrs.get("norm_stats_json", "{}")
        self.norm_stats = json.loads(stats_json)

        if normalize:
            input_stack = self._normalize(input_stack, INPUT_VARS)
            # Targets: normalize the anomaly channels too (matches network output)
            target_stack = self._normalize(target_stack, OUTPUT_VARS)

        self.inputs  = input_stack   # (T, H, W, C_in)
        self.targets = target_stack  # (T, H, W, C_out)

        # --- Compute valid window start indices ---
        T = self.inputs.shape[0]
        self.n_windows = T - self.window + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"Not enough days ({T}) for a window of size {self.window} in split={split}."
            )
        ds.close()

    # ------------------------------------------------------------------
    def _normalize(self, arr: np.ndarray, var_names: list[str]) -> np.ndarray:
        """Apply per-channel z-score using saved training stats."""
        out = arr.copy()
        for c, v in enumerate(var_names):
            stats = self.norm_stats.get(v, {"mean": 0.0, "std": 1.0})
            out[..., c] = (arr[..., c] - stats["mean"]) / (stats["std"] + 1e-8)
        return out

    def denormalize(self, arr: np.ndarray, var_names: list[str] = None) -> np.ndarray:
        """Inverse of _normalize — needed after model prediction."""
        var_names = var_names or OUTPUT_VARS
        out = arr.copy()
        for c, v in enumerate(var_names):
            stats = self.norm_stats.get(v, {"mean": 0.0, "std": 1.0})
            out[..., c] = arr[..., c] * (stats["std"] + 1e-8) + stats["mean"]
        return out

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one (input, target) example.

        input:  torch.Tensor of shape (input_days, C_in, H, W)  — PyTorch uses (T, C, H, W)
        target: torch.Tensor of shape (forecast_days, C_out, H, W)
        """
        assert 0 <= idx < self.n_windows
        # Slice the window
        x = self.inputs[idx : idx + self.input_days]                        # (T_in, H, W, C_in)
        y = self.targets[idx + self.input_days : idx + self.window]         # (T_out, H, W, C_out)

        # Rearrange to PyTorch's (T, C, H, W) convention
        # numpy: (T, H, W, C) → transpose to (T, C, H, W)
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.transpose(y, (0, 3, 1, 2))

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ============================================================================
# CONVENIENCE FACTORY
# ============================================================================
def make_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,   # 0 for Windows; increase on Linux
) -> dict:
    """
    Build train/val/test DataLoaders in one call.

    Returns
    -------
    dict with keys 'train', 'val', 'test' → DataLoader instances
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CauveryWindowDataset(split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
        print(f"  {split}: {len(ds)} windows, batch_size={batch_size}")
    return loaders


# ============================================================================
# CLI SANITY CHECK
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CauveryWindowDataset sanity check")
    print("=" * 60)

    for split in ("train", "val", "test"):
        ds = CauveryWindowDataset(split=split)
        x, y = ds[0]
        print(f"\n{split.upper():5s} | windows={len(ds):4d} | "
              f"input={tuple(x.shape)} | target={tuple(y.shape)}")
        print(f"        input dtype={x.dtype}, target dtype={y.dtype}")
        print(f"        input stats:  mean={x.mean():.3f}, std={x.std():.3f}")
        print(f"        target stats: mean={y.mean():.3f}, std={y.std():.3f}")

    print("\nBuilding DataLoaders...")
    loaders = make_dataloaders()
    print("\n✓ DataLoaders ready")

    # Peek one batch
    batch_x, batch_y = next(iter(loaders["train"]))
    print(f"\nOne batch from train: X {tuple(batch_x.shape)} | Y {tuple(batch_y.shape)}")
    print("  (batch, time, channels, height, width)")
