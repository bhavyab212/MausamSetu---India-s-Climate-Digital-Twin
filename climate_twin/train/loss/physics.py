"""
train.loss.physics — per-zone physics penalties.

* ``tmax_ge_tmin_penalty``   soft hinge (already in Phase 5e; kept here)
* ``spatial_smoothness``     TV-L2 on the prediction, land-mask weighted
* ``temporal_smoothness``    day-to-day L2 on rolled prediction sequence
* ``zone_physics_bounds_penalty``  hinge outside per-zone (rain_max, tmax/tmin range)
"""
from __future__ import annotations

import torch

from climate_twin.regions import ZoneRegistry


def _weighted_mean(x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x, mask: (B, H, W) or (B, 1, H, W).  weight: (H, W)."""
    if x.dim() == 4:
        x = x.squeeze(1)
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    w = weight.view(1, *weight.shape) * mask.to(x.dtype)
    return (x * w).sum() / (w.sum() + 1e-8)


def tmax_ge_tmin_penalty(
    pred_tmax: torch.Tensor,        # (B, 1, H, W)
    pred_tmin: torch.Tensor,
    weight: torch.Tensor,           # (H, W)
    mask: torch.Tensor,             # (H, W)  land = 1
) -> torch.Tensor:
    """Hinge on tmin > tmax on land. Zero when the ordering is satisfied."""
    diff = (pred_tmin - pred_tmax).squeeze(1)                  # (B, H, W)
    viol = torch.clamp(diff, min=0.0)
    m = mask.view(1, *mask.shape)
    return _weighted_mean(viol, m.expand_as(viol), weight)


def spatial_smoothness(
    pred: torch.Tensor,              # (B, 1, H, W)
    weight: torch.Tensor,            # (H, W)
    mask: torch.Tensor,              # (H, W)
) -> torch.Tensor:
    """Land-mask weighted TV-L2 penalty on a single prediction map."""
    p = pred.squeeze(1)                                          # (B, H, W)
    dx = (p[:, :, 1:] - p[:, :, :-1]).pow(2)
    dy = (p[:, 1:, :] - p[:, :-1, :]).pow(2)
    mx = weight[:, 1:] * weight[:, :-1] * mask[:, 1:] * mask[:, :-1]
    my = weight[1:, :] * weight[:-1, :] * mask[1:, :] * mask[:-1, :]
    num = (dx * mx.unsqueeze(0)).sum() + (dy * my.unsqueeze(0)).sum()
    den = mx.sum() + my.sum() + 1e-8
    return num / den


def temporal_smoothness(
    pred_prev: torch.Tensor,         # (B, 1, H, W)  — previous day pred
    pred_curr: torch.Tensor,
    weight: torch.Tensor,            # (H, W)
    mask: torch.Tensor,              # (H, W)
) -> torch.Tensor:
    """Day-to-day L2 smoothness. Applied when the trainer has two adjacent
    predictions in memory."""
    diff = (pred_curr - pred_prev).squeeze(1)
    return _weighted_mean(diff.pow(2), mask.view(1, *mask.shape).expand_as(diff), weight)


def zone_physics_bounds_penalty(
    outputs: dict,
    zones: ZoneRegistry,
    weight: torch.Tensor,            # (H, W)
    mask: torch.Tensor,              # (H, W)
    variables: tuple[str, ...],
) -> torch.Tensor:
    """Per-zone hinge outside plausible physical range.

    For each zone × variable, penalize predictions outside
    ``physics_bounds[zone][var]`` from the frozen ``india_zones.yaml``. The
    penalty is masked so a cell contributes only via its zone(s).
    """
    device = weight.device
    total = torch.tensor(0.0, device=device, dtype=weight.dtype)
    denom = torch.tensor(0.0, device=device, dtype=weight.dtype)

    for zone in zones.zones:
        # Per-zone weight mask (soft membership)
        k = zones.zone_ids.index(zone.id)
        zw = torch.from_numpy(zones.membership[..., k]).to(device=device, dtype=weight.dtype)
        zone_mask = zw * mask

        for var in variables:
            if var == "rain":
                pred = outputs["rain"]["amount"]        # (B, 1, H, W)
                # softplus is applied in the loss, but before that, penalise raw
                # predictions above rain_max
                if zone.rain_max_mm_day is not None:
                    over = torch.clamp(pred.squeeze(1) - zone.rain_max_mm_day, min=0.0)
                    total = total + (over * zone_mask.unsqueeze(0)).sum()
                    denom = denom + zone_mask.sum() * over.shape[0]
            elif var == "tmax" and zone.tmax_c is not None:
                lo, hi = zone.tmax_c
                pred = outputs["tmax"].squeeze(1)
                below = torch.clamp(lo - pred, min=0.0)
                above = torch.clamp(pred - hi, min=0.0)
                total = total + ((below + above) * zone_mask.unsqueeze(0)).sum()
                denom = denom + zone_mask.sum() * pred.shape[0]
            elif var == "tmin" and zone.tmin_c is not None:
                lo, hi = zone.tmin_c
                pred = outputs["tmin"].squeeze(1)
                below = torch.clamp(lo - pred, min=0.0)
                above = torch.clamp(pred - hi, min=0.0)
                total = total + ((below + above) * zone_mask.unsqueeze(0)).sum()
                denom = denom + zone_mask.sum() * pred.shape[0]

    return total / (denom + 1e-8)
