"""
reward_calibration.py — PART E (experimental).

Reward-guided calibration: fine-tune an already-supervised checkpoint using CRPS
(Continuous Ranked Probability Score) as a reward signal to sharpen the
uncertainty bands. CRPS is a proper scoring rule that rewards predictions that
are BOTH confident AND correct, and penalises over- and under-confidence.

HONEST framing:
  - This is NOT a replacement for supervised training.
  - Supervised training remains the primary objective.
  - This only *tunes uncertainty sharpness* on top of a supervised model.
  - Off by default unless it demonstrably beats the supervised baseline on
    calibration (p10-p90 coverage nearer 80%) AND sharpness (band width).

Method — reward-weighted MC-dropout fine-tuning:
  Standard supervised loss + λ * CRPS(mean, std, truth) on the MC-dropout
  ensemble spread. Because CRPS is differentiable in (mean, std), we can
  backprop through it — technically a maximum-likelihood/scoring-rule
  fine-tune, which is the honest, well-founded version of "RL on the
  forecaster". We label it as such in the UI.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable

from .model import ClimateTwinModel


# ─────────────────────────── metrics (differentiable + numpy) ─────────────

def crps_gaussian_torch(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                        mask: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable CRPS assuming a Gaussian predictive distribution.

    CRPS(N(mu,sigma), y) = sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) )
    where z = (y - mu) / sigma. Lower is better; 0 = perfect + zero uncertainty.
    """
    sigma = torch.clamp(sigma, min=1e-4)
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    phi = torch.exp(normal.log_prob(z))
    Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    if mask is not None:
        m = mask.view(1, 1, *mask.shape[-2:])
        crps = crps * m
        denom = mask.sum() * mu.shape[0] * mu.shape[1] + 1e-8
        return crps.sum() / denom
    return crps.mean()


def coverage_p10_p90(p10: np.ndarray, p90: np.ndarray, truth: np.ndarray,
                     mask: np.ndarray) -> float:
    """Fraction of land cells where truth lies inside [p10, p90] (target ≈ 0.80)."""
    m = mask == 1
    if not m.any():
        return float("nan")
    inside = (truth >= p10) & (truth <= p90)
    return float(inside[m].mean())


def sharpness_band(p10: np.ndarray, p90: np.ndarray, mask: np.ndarray) -> float:
    """Mean band width p90 - p10 over land. Smaller = sharper (given calibration)."""
    m = mask == 1
    if not m.any():
        return float("nan")
    return float(np.nanmean((p90 - p10)[m]))


def crps_np(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray,
            truth: np.ndarray, mask: np.ndarray) -> float:
    """Empirical CRPS using a Gaussian fit (sigma ≈ (p90 - p10) / 2.5631) — good enough
    for a comparison scoreboard."""
    m = mask == 1
    if not m.any():
        return float("nan")
    sigma = (p90 - p10) / 2.5631
    sigma = np.clip(sigma, 1e-4, None)
    z = (truth - p50) / sigma
    from math import pi as _pi
    from scipy.stats import norm as _norm
    val = sigma * (z * (2 * _norm.cdf(z) - 1) + 2 * _norm.pdf(z) - 1 / np.sqrt(_pi))
    return float(val[m].mean())


# ─────────────────────────── fine-tune loop ───────────────────────────────

def calibration_finetune(
    model: ClimateTwinModel,
    train_x: np.ndarray, train_y: np.ndarray,
    mask: np.ndarray,
    n_epochs: int = 10,
    lr: float = 1e-4,
    lambda_crps: float = 1.0,
    mc_samples: int = 8,
    on_epoch: Callable[[dict], None] | None = None,
    device: str = "cuda",
) -> tuple[ClimateTwinModel, list[dict]]:
    """Reward-guided calibration fine-tune.

    train_x: (N, T, H, W, C) normalized [0,1]. train_y: (N, H, W, C).
    Loss = supervised_MSE(mean, y) + λ * CRPS_gaussian(mean, sigma_MC, y).
    Sigma_MC is estimated by keeping dropout ACTIVE and running `mc_samples`
    forward passes per step (small — GPU-friendly). The MSE anchor keeps the
    mean supervised; CRPS only reshapes the spread.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Reward-guided calibration requires CUDA.")
    model.to(device); model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mask_t = torch.from_numpy(mask).float().to(device)

    Xt = torch.from_numpy(train_x.transpose(0, 1, 4, 2, 3)).float()
    Yt = torch.from_numpy(train_y.transpose(0, 3, 1, 2)).float()

    history = []
    for e in range(n_epochs):
        idx = torch.randperm(len(Xt))
        losses, mses, crpses = [], [], []
        for i in idx:
            xb = Xt[i:i + 1].to(device)
            yb = Yt[i:i + 1].to(device)

            # Keep dropout on for MC spread estimate.
            for module in model.modules():
                if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                    module.train()
            samples = torch.stack([model(xb) for _ in range(mc_samples)], dim=0)  # (K,1,C,H,W)
            mu = samples.mean(0)
            sigma = samples.std(0) + 1e-4

            mse = F.mse_loss(mu * mask_t.view(1, 1, *mask.shape),
                             yb * mask_t.view(1, 1, *mask.shape))
            crps = crps_gaussian_torch(mu, sigma, yb, mask_t)
            loss = mse + lambda_crps * crps

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item()); mses.append(mse.item()); crpses.append(crps.item())

        logs = {"epoch": e + 1, "total": n_epochs,
                "loss": float(np.mean(losses)),
                "mse": float(np.mean(mses)),
                "crps": float(np.mean(crpses))}
        history.append(logs)
        if on_epoch:
            on_epoch(logs)
    return model, history


# ─────────────────────────── evaluation helper ────────────────────────────

def evaluate_calibration(model: ClimateTwinModel, val_x: np.ndarray, val_y: np.ndarray,
                         mask: np.ndarray, mc_samples: int = 20, device: str = "cuda") -> dict:
    """Return calibration + sharpness + CRPS on val, using MC-dropout p10/p50/p90."""
    model.to(device); model.eval()
    x = torch.from_numpy(val_x.transpose(0, 1, 4, 2, 3)).float().to(device)
    ens = model.predict_ensemble(x, n_samples=mc_samples)
    p10, p50, p90 = (ens[k].cpu().numpy() for k in ("p10", "p50", "p90"))
    truth = val_y.transpose(0, 3, 1, 2)                 # (N,C,H,W)
    covs, sharps, crpss = [], [], []
    for i in range(p10.shape[0]):
        c = coverage_p10_p90(p10[i, 0], p90[i, 0], truth[i, 0], mask)
        s = sharpness_band(p10[i, 0], p90[i, 0], mask)
        k = crps_np(p10[i, 0], p50[i, 0], p90[i, 0], truth[i, 0], mask)
        if np.isfinite(c): covs.append(c)
        if np.isfinite(s): sharps.append(s)
        if np.isfinite(k): crpss.append(k)
    return {"coverage": float(np.mean(covs)) if covs else float("nan"),
            "sharpness": float(np.mean(sharps)) if sharps else float("nan"),
            "crps": float(np.mean(crpss)) if crpss else float("nan"),
            "N": int(p10.shape[0])}


def compare_verdict(supervised: dict, calibrated: dict) -> tuple[str, str, str]:
    """Return (verdict, colour, action). Honest — off by default unless a clear win."""
    def _closer_to_80(x):
        return abs((x or float("nan")) - 0.80) if isinstance(x, (int, float)) and np.isfinite(x) else float("inf")
    d_cov = _closer_to_80(supervised.get("coverage")) - _closer_to_80(calibrated.get("coverage"))
    s_sup, s_cal = supervised.get("sharpness"), calibrated.get("sharpness")
    sharper = (isinstance(s_sup, (int, float)) and isinstance(s_cal, (int, float))
               and s_cal < s_sup and _closer_to_80(calibrated.get("coverage")) <= 0.05)
    k_sup, k_cal = supervised.get("crps"), calibrated.get("crps")
    crps_wins = isinstance(k_sup, (int, float)) and isinstance(k_cal, (int, float)) and k_cal < k_sup

    if d_cov > 0.02 and crps_wins:
        return ("Calibration improved — use the calibrated model", "#39d98a",
                "Coverage is closer to 80% AND CRPS is lower. Recommended to keep.")
    if sharper and crps_wins:
        return ("Sharper without losing calibration — modest win", "#22D3EE",
                "Bands are tighter and CRPS improved. Keep for uncertainty-critical uses.")
    if not crps_wins:
        return ("No improvement — keep the supervised model", "#EF4444",
                "Reward-guided calibration did not beat supervised. Disabled by default.")
    return ("Marginal / mixed — supervised is fine", "#F4A34A",
            "No clear win. Prefer the supervised model unless you specifically need sharper bands.")
