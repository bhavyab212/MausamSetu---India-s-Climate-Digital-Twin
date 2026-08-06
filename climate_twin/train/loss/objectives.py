"""
train.loss.objectives — NaN-safe zone-weighted objectives.

All losses accept a per-cell weight tensor ``w (H, W)`` computed by
``train.data.sampler`` from the zone registry, and a ground-truth tensor
that may contain NaN. NaN targets are EXCLUDED from the denominator, not
zero-filled. Every objective is safe under bf16.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _reduce(loss: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``loss`` over land ∩ finite cells only.

    loss/mask: (B, H, W).  weight: (H, W)."""
    w = weight.view(1, *weight.shape) * mask.to(loss.dtype)
    denom = w.sum() + 1e-8
    return (loss * w).sum() / denom


def hurdle_rain_loss(
    logit_occ: torch.Tensor,       # (B, 1, H, W)
    amount: torch.Tensor,           # (B, 1, H, W) — softplus-shaped in mm
    target: torch.Tensor,           # (B, 1, H, W) — mm/day (may contain NaN)
    weight: torch.Tensor,           # (H, W)
    threshold_mm: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-part rain loss.

    Returns ``(occurrence_bce, wet_amount_mse)`` both already zone-weighted
    scalars. Combine externally with configurable weights (default 1:1)."""
    finite = torch.isfinite(target)
    valid = finite.squeeze(1)                              # (B, H, W)
    target_safe = torch.where(finite, target, torch.zeros_like(target))

    is_wet = (target_safe > threshold_mm).float()
    # Occurrence BCE
    bce = F.binary_cross_entropy_with_logits(
        logit_occ, is_wet, reduction="none"
    ).squeeze(1)                                            # (B, H, W)
    occ = _reduce(bce, valid, weight)

    # Wet-amount log-space Huber (mm scale; softplus keeps predictions ≥ 0)
    amt_pred = F.softplus(amount).squeeze(1)                # (B, H, W)
    amt_true = target_safe.squeeze(1)
    wet_mask = valid & (amt_true > threshold_mm)
    diff = amt_pred - amt_true
    huber = torch.where(diff.abs() < 1.0,
                        0.5 * diff.pow(2),
                        diff.abs() - 0.5)
    amt = _reduce(huber, wet_mask, weight)
    return occ, amt


def huber_loss(
    pred: torch.Tensor,             # (B, 1, H, W)
    target: torch.Tensor,           # (B, 1, H, W)
    weight: torch.Tensor,           # (H, W)
    delta: float = 1.0,
) -> torch.Tensor:
    """NaN-safe Huber for tmax / tmin (in physical units, °C)."""
    finite = torch.isfinite(target)
    valid = finite.squeeze(1)
    target_safe = torch.where(finite, target, torch.zeros_like(target))
    diff = (pred - target_safe).squeeze(1)
    d = torch.tensor(delta, device=pred.device, dtype=pred.dtype)
    huber = torch.where(diff.abs() < d,
                        0.5 * diff.pow(2),
                        d * (diff.abs() - 0.5 * d))
    return _reduce(huber, valid, weight)


def total_train_loss(
    outputs: dict,
    targets: dict[str, torch.Tensor],
    weight: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite training loss.

    ``outputs`` is the dict returned by :class:`ZoneAwareModel.forward`;
    ``targets`` is a dict of ``(B, 1, H, W)`` per variable. Physics
    penalties are applied by ``train.loop.trainer`` on top of what this
    returns.
    """
    parts: dict[str, torch.Tensor] = {}

    if "rain" in outputs:
        occ, amt = hurdle_rain_loss(
            outputs["rain"]["logit_occurrence"],
            outputs["rain"]["amount"],
            targets["rain"],
            weight=weight,
            threshold_mm=cfg.loss.rain_occurrence_threshold_mm,
        )
        parts["rain_occ"] = occ
        parts["rain_amt"] = amt

    for v in ("tmax", "tmin"):
        if v in outputs and v in targets:
            parts[v] = huber_loss(outputs[v], targets[v], weight=weight, delta=2.0)

    loss = sum(parts.values())
    return loss, {k: float(v.detach().item()) for k, v in parts.items()}
