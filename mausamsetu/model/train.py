"""
train.py
=========
Training loop for the MausamSetu forecaster.

WHAT THIS DOES
--------------
1. Loads train + val DataLoaders (from cauvery.nc)
2. Instantiates the ConvLSTM forecaster + physics-informed loss
3. Trains for N epochs, tracking train/val loss
4. Uses EARLY STOPPING (halt if val_loss doesn't improve for N epochs)
5. Saves the BEST checkpoint (by val_loss) to checkpoints/forecaster_best.pt

HONEST TRAINING PRACTICES:
  - Split by YEAR (no data leakage from test → train)
  - Validation set (2022) used ONLY for early stopping / hyperparameters
  - Test set (2023-2024) NEVER touched until final evaluation

RUN
---
    python -m mausamsetu.model.train
    # or with a smaller run for laptop CPU:
    python -m mausamsetu.model.train --epochs 5 --batch-size 4
"""
from __future__ import annotations
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mausamsetu import config
from mausamsetu.preprocess.dataset import CauveryWindowDataset, make_dataloaders
from mausamsetu.model.forecaster import MausamSetuForecaster
from mausamsetu.model.loss import PhysicsInformedLoss


# ============================================================================
# HELPERS
# ============================================================================
def make_rain_binary(y: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Convert normalized rain targets to binary occurrence.
    Threshold is applied in NORMALIZED anomaly space (any positive anomaly).
    In practice, ≈ 0 covers "did it rain today?".
    """
    rain = y[:, :, 0:1]  # (B, T, 1, H, W)
    return (rain > threshold).float()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = {"total": 0.0, "mse": 0.0, "occ": 0.0, "physics": 0.0, "smoothness": 0.0}
    n_batches = 0
    for x, y in tqdm(loader, desc="train", leave=False, ncols=80):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred, rain_prob = model(x, return_occ=True)
        rain_bin = make_rain_binary(y)

        losses = loss_fn(y_pred, y, rain_prob=rain_prob, rain_true_binary=rain_bin)
        losses["total"].backward()

        # Gradient clipping to stabilise training on limited data
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running = {"total": 0.0, "mse": 0.0, "occ": 0.0, "physics": 0.0, "smoothness": 0.0}
    n_batches = 0
    for x, y in tqdm(loader, desc="val", leave=False, ncols=80):
        x, y = x.to(device), y.to(device)
        y_pred, rain_prob = model(x, return_occ=True)
        rain_bin = make_rain_binary(y)
        losses = loss_fn(y_pred, y, rain_prob=rain_prob, rain_true_binary=rain_bin)
        for k in running:
            running[k] += losses[k].item()
        n_batches += 1
    return {k: v / n_batches for k, v in running.items()}


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    early_stop_patience: int = config.EARLY_STOP,
    device: str = None,
) -> Path:
    """Run the training loop. Returns path to best checkpoint."""

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"MAUSAMSETU FORECASTER TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"Early stopping patience: {early_stop_patience}")
    print()

    config.ensure_dirs()

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    print("Loading data...")
    loaders = make_dataloaders(batch_size=batch_size, num_workers=0)
    print()

    # ------------------------------------------------------------
    # Model + Loss + Optimizer
    # ------------------------------------------------------------
    model = MausamSetuForecaster().to(device)
    loss_fn = PhysicsInformedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    best_val = float("inf")
    epochs_no_improve = 0
    ckpt_path = config.CHECKPOINT_DIR / "forecaster_best.pt"
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        train_losses = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val_losses   = validate(model, loaders["val"], loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t_epoch
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_losses['total']:.4f} "
            f"(mse={train_losses['mse']:.3f} occ={train_losses['occ']:.3f}) | "
            f"val_loss={val_losses['total']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.5f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train": train_losses,
            "val": val_losses,
        })

        # ---- Checkpoint if best ----
        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": best_val,
                "epoch": epoch,
                "config_snapshot": {
                    "in_channels": config.INPUT_CHANNELS,
                    "out_channels": config.OUTPUT_VARS,
                    "hidden_channels": config.HIDDEN_CHANNELS,
                    "forecast_days": config.FORECAST_DAYS,
                    "dropout": config.DROPOUT,
                },
            }, ckpt_path)
            print(f"    ✓ Saved best checkpoint → {ckpt_path.name} (val={best_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(f"✓ Training complete in {total_time/60:.1f} min")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Best checkpoint: {ckpt_path}")
    print("=" * 70)

    return ckpt_path


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stop_patience=args.patience,
        device=args.device,
    )
