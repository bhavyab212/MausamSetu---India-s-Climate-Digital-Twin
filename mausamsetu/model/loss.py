"""
loss.py
========
Physics-Informed Loss for the MausamSetu forecaster.

TOTAL LOSS = MSE + BCE(rain occurrence) + Physics penalty + Spatial smoothness

WHY EACH TERM?
--------------
1. MSE loss on the full prediction: standard "get the values right"
2. BCE loss on rain occurrence: forces the model to distinguish
   "rain vs no-rain" days (handles zero-inflation)
3. Physics penalty:
   - punish negative rainfall (impossible)
   - punish Tmin > Tmax (impossible)
4. Spatial smoothness: encourages meteorologically-realistic maps
   (rainfall doesn't jump from 0 to 100mm across one pixel)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from mausamsetu import config


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss for the forecaster.
    - Predictions are in NORMALIZED anomaly space.
    - Targets are also in NORMALIZED anomaly space.
    """

    def __init__(
        self,
        mse_weight: float = config.LOSS_MSE_W,
        occ_weight: float = config.LOSS_OCC_W,
        physics_weight: float = config.LOSS_PHYSICS_W,
        smoothness_weight: float = 0.05,
    ):
        super().__init__()
        self.mse_w = mse_weight
        self.occ_w = occ_weight
        self.phys_w = physics_weight
        self.smooth_w = smoothness_weight

    def forward(
        self,
        y_pred: torch.Tensor,        # (B, T, 3, H, W) — predicted [rain, tmax, tmin] anomalies
        y_true: torch.Tensor,        # (B, T, 3, H, W) — ground truth
        rain_prob: torch.Tensor = None,   # (B, T, 1, H, W) — occurrence prob (optional)
        rain_true_binary: torch.Tensor = None,  # (B, T, 1, H, W) — 1 if rain>threshold
    ) -> dict:
        """
        Returns
        -------
        dict with keys: 'total', 'mse', 'occ', 'physics', 'smoothness'
        """
        losses = {}

        # ---- MSE on all three channels ----
        mse = F.mse_loss(y_pred, y_true)
        losses["mse"] = mse

        # ---- Rain-occurrence BCE (if given) ----
        if rain_prob is not None and rain_true_binary is not None:
            occ = F.binary_cross_entropy(
                rain_prob.clamp(1e-6, 1 - 1e-6),
                rain_true_binary,
            )
        else:
            occ = torch.tensor(0.0, device=y_pred.device)
        losses["occ"] = occ

        # ---- Physics penalties ----
        # 1. Rain shouldn't be very negative (once denormalized)
        #    In normalized space, penalize predictions much less than 0
        rain_pred = y_pred[:, :, 0]                          # (B, T, H, W)
        neg_rain_penalty = torch.relu(-rain_pred - 1.0).mean()

        # 2. Tmax should be >= Tmin (in anomaly space they can be close;
        #    we don't have raw values here, so we use a soft margin)
        tmax_pred = y_pred[:, :, 1]
        tmin_pred = y_pred[:, :, 2]
        temp_inversion = torch.relu(tmin_pred - tmax_pred - 0.1).mean()

        physics = neg_rain_penalty + temp_inversion
        losses["physics"] = physics

        # ---- Spatial smoothness (total variation on rain field) ----
        rain = y_pred[:, :, 0]                                # (B, T, H, W)
        tv_lat = (rain[:, :, 1:, :] - rain[:, :, :-1, :]).abs().mean()
        tv_lon = (rain[:, :, :, 1:] - rain[:, :, :, :-1]).abs().mean()
        smoothness = tv_lat + tv_lon
        losses["smoothness"] = smoothness

        # ---- Total ----
        total = (
            self.mse_w * mse
            + self.occ_w * occ
            + self.phys_w * physics
            + self.smooth_w * smoothness
        )
        losses["total"] = total
        return losses
