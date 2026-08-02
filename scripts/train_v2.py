"""MausamSetu v2 Training Script — self-contained, GPU-ready.

Fixes vs the original train.py:
  1. No sigmoid gate on rain output (uses MausamSetuForecasterV2)
  2. 3 input channels (no fake INSAT)
  3. Mask-aware MSE (NaN cells excluded from loss, not set to 0)
  4. Occurrence BCE in PHYSICAL space (threshold = 0.1 mm/day actual rain)
  5. Harmonic climatology dataset (cauvery_v2.nc)

Usage (Kaggle GPU):
    python scripts/train_v2.py --data data/processed/cauvery_v2.nc --output checkpoints/forecaster_v2.pt --epochs 50

Local CPU smoke test:
    python scripts/train_v2.py --data data/processed/cauvery_v2.nc --output checkpoints/forecaster_v2.pt --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config (self-contained, no dependency on mausamsetu.config for portability)
# ---------------------------------------------------------------------------

INPUT_DAYS = 6
FORECAST_DAYS = 7
INPUT_CHANNELS = 3  # rain_anom, tmax_anom, tmin_anom
OUTPUT_CHANNELS = 3
HIDDEN_CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
DROPOUT = 0.2

RAIN_OCC_THRESHOLD_MM = 0.1  # physical rainfall threshold for occurrence

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CauveryV2Dataset(Dataset):
    """PyTorch dataset for cauvery_v2.nc — 3 input channels, mask-aware."""

    def __init__(self, nc_path: str | Path, split: str = "train", normalize: bool = True):
        import xarray as xr

        ds = xr.open_dataset(nc_path).load()
        self.norm_stats = json.loads(ds.attrs["norm_stats_json"])

        # Year-based splits
        years_map = {
            "train": [2020, 2021],
            "val": [2022],
            "test": [2023],
        }
        years = years_map[split]
        mask = np.isin(ds.time.dt.year.values, years)
        ds_split = ds.isel(time=mask)

        # Stack input/target arrays: (time, lat, lon) per variable
        self.input_days = INPUT_DAYS
        self.forecast_days = FORECAST_DAYS
        input_vars = ["rain_anom", "tmax_anom", "tmin_anom"]
        output_vars = ["rain_anom", "tmax_anom", "tmin_anom"]

        # Build (T, C, H, W) tensors
        self.inputs = np.stack(
            [ds_split[v].values.astype(np.float32) for v in input_vars], axis=1
        )  # (T, 3, H, W)
        self.targets = np.stack(
            [ds_split[v].values.astype(np.float32) for v in output_vars], axis=1
        )  # (T, 3, H, W)

        # For physical-space occurrence: keep raw rain
        self.rain_physical = ds_split["rain"].values.astype(np.float32)  # (T, H, W)

        # NaN mask (True = valid): same across all channels for a given cell
        self.valid_mask = np.isfinite(self.inputs[:, 0])  # (T, H, W)

        # Normalize (z-score) — NaN stays NaN, will be masked in loss
        if normalize:
            for c, v in enumerate(input_vars):
                m, s = self.norm_stats[v]["mean"], self.norm_stats[v]["std"]
                self.inputs[:, c] = (self.inputs[:, c] - m) / s
            for c, v in enumerate(output_vars):
                m, s = self.norm_stats[v]["mean"], self.norm_stats[v]["std"]
                self.targets[:, c] = (self.targets[:, c] - m) / s

        # Replace NaN with 0 AFTER normalization (so masked = mean in z-space)
        self.inputs = np.nan_to_num(self.inputs, nan=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0)
        self.rain_physical = np.nan_to_num(self.rain_physical, nan=0.0)
        self.valid_mask = self.valid_mask.astype(np.float32)  # 1=valid, 0=masked

        self.n_days = self.inputs.shape[0]
        self.window = INPUT_DAYS + FORECAST_DAYS
        self.n_windows = self.n_days - self.window + 1
        self.time_values = ds_split.time.values

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # Input: (INPUT_DAYS, C, H, W)
        x = self.inputs[idx: idx + self.input_days]
        # Target: (FORECAST_DAYS, C, H, W)
        t0 = idx + self.input_days
        y = self.targets[t0: t0 + self.forecast_days]
        # Mask for the target window
        m = self.valid_mask[t0: t0 + self.forecast_days]  # (FORECAST_DAYS, H, W)
        # Physical rain for occurrence target
        rain_phys = self.rain_physical[t0: t0 + self.forecast_days]  # (FORECAST_DAYS, H, W)

        return (
            torch.from_numpy(x),       # (T_in, C, H, W)
            torch.from_numpy(y),       # (T_out, C, H, W)
            torch.from_numpy(m),       # (T_out, H, W)
            torch.from_numpy(rain_phys),  # (T_out, H, W)
        )


# ---------------------------------------------------------------------------
# Model (inline for Kaggle portability — same as lab/core/model_v2.py)
# ---------------------------------------------------------------------------

def _make_model(device: str) -> nn.Module:
    """Import and instantiate the V2 forecaster."""
    import sys
    # Add project root for imports
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mausamsetu.lab.core.model_v2 import MausamSetuForecasterV2
    model = MausamSetuForecasterV2(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        forecast_days=FORECAST_DAYS,
        dropout=DROPOUT,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss excluding masked (out-of-basin) cells.

    pred, target: (B, T, C, H, W)
    mask: (B, T, H, W) — 1=valid, 0=masked
    """
    mask_expanded = mask.unsqueeze(2)  # (B, T, 1, H, W)
    diff_sq = (pred - target) ** 2
    masked_diff = diff_sq * mask_expanded
    n_valid = mask_expanded.sum() * pred.shape[2]  # count across channels
    return masked_diff.sum() / (n_valid + 1e-8)


def physics_penalty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize negative rain and tmin > tmax, only on valid cells.

    pred: (B, T, 3, H, W) — channels [rain, tmax, tmin] in z-space
    """
    rain = pred[:, :, 0]
    tmax = pred[:, :, 1]
    tmin = pred[:, :, 2]

    # Negative rain penalty (with dead band at -1 to avoid suppressing small negatives)
    neg_rain = F.relu(-rain - 1.0) * mask
    # tmin > tmax penalty
    tmin_violation = F.relu(tmin - tmax + 0.1) * mask

    n_valid = mask.sum() + 1e-8
    return (neg_rain.sum() + tmin_violation.sum()) / n_valid


def occurrence_bce(
    occ_logits: torch.Tensor,
    rain_physical: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = RAIN_OCC_THRESHOLD_MM,
) -> torch.Tensor:
    """BCE loss on rain occurrence in PHYSICAL space.

    occ_logits: (B, T, 1, H, W) — raw logits from rain_occ_head
    rain_physical: (B, T, H, W) — actual rainfall in mm/day
    mask: (B, T, H, W) — 1=valid
    """
    target = (rain_physical > threshold).float()  # physical threshold
    logits = occ_logits.squeeze(2)  # (B, T, H, W)

    # Compute per-element BCE
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    masked_bce = bce * mask
    return masked_bce.sum() / (mask.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    print(f"Loading dataset: {args.data}")
    train_ds = CauveryV2Dataset(args.data, split="train")
    val_ds = CauveryV2Dataset(args.data, split="val")
    print(f"Train: {len(train_ds)} windows · Val: {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = _make_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Weights
    mse_w = args.mse_w
    occ_w = args.occ_w
    phys_w = args.phys_w

    # Training state
    best_val = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_pred_std": [], "lr": []}

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    print(f"Loss weights: MSE={mse_w} · OCC_BCE={occ_w} · Physics={phys_w}")
    print(f"{'='*70}")

    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bce = 0.0
        n_batches = 0

        for x, y, mask, rain_phys in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            rain_phys = rain_phys.to(device)

            optimizer.zero_grad()

            pred, occ_logits = model(x, return_occ=True)

            loss_mse = masked_mse(pred, y, mask)
            loss_occ = occurrence_bce(occ_logits, rain_phys, mask)
            loss_phys = physics_penalty(pred, mask)

            loss = mse_w * loss_mse + occ_w * loss_occ + phys_w * loss_phys
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_bce += loss_occ.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_pred_std = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y, mask, rain_phys in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                rain_phys = rain_phys.to(device)

                pred, occ_logits = model(x, return_occ=True)
                v_mse = masked_mse(pred, y, mask)
                v_occ = occurrence_bce(occ_logits, rain_phys, mask)
                v_phys = physics_penalty(pred, mask)
                v_total = mse_w * v_mse + occ_w * v_occ + phys_w * v_phys

                val_loss += v_total.item() * x.size(0)
                val_pred_std += pred[:, :, 0].std().item() * x.size(0)
                val_n += x.size(0)

        avg_val = val_loss / max(val_n, 1)
        avg_pred_std = val_pred_std / max(val_n, 1)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_pred_std"].append(avg_pred_std)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"train={avg_train:.4f} mse={epoch_mse/n_batches:.4f} bce={epoch_bce/n_batches:.4f} | "
            f"val={avg_val:.4f} pred_std={avg_pred_std:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"ETA {eta/60:.1f}m"
        )

        # Save best
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": best_val,
                "epoch": epoch + 1,
                "train_losses": history["train_loss"],
                "val_losses": history["val_loss"],
                "val_pred_stds": history["val_pred_std"],
                "config": {
                    "in_channels": INPUT_CHANNELS,
                    "out_channels": OUTPUT_CHANNELS,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "forecast_days": FORECAST_DAYS,
                    "dropout": DROPOUT,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "mse_w": mse_w,
                    "occ_w": occ_w,
                    "phys_w": phys_w,
                    "weight_decay": args.weight_decay,
                },
            }, args.output)
            print(f"  ✓ Saved best checkpoint (val={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={args.patience})")
                break

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Training complete: {epoch+1} epochs in {total_time/60:.1f} min")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Final pred anomaly std: {history['val_pred_std'][-1]:.4f} (target ≈1.0)")
    print(f"Checkpoint: {args.output}")

    # Save history
    history_path = Path(args.output).with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"History: {history_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MausamSetu v2 forecaster")
    parser.add_argument("--data", type=str, default="data/processed/cauvery_v2.nc")
    parser.add_argument("--output", type=str, default="checkpoints/forecaster_v2.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mse-w", type=float, default=1.0)
    parser.add_argument("--occ-w", type=float, default=0.3)
    parser.add_argument("--phys-w", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(args)
