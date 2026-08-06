"""PyTorch Deep Ensemble, Test-Time Augmentation (TTA), and Iterative Refinement.

Part C capability for climate_twin training system:
  - Deep Ensemble: Train K models with different seeds, average p50, combine inter-model + intra-model MC variance.
  - TTA: Spatial shift & perturbation averaging at inference.
  - Iterative Refinement: Multi-pass self-feedback sharpening pass (capped at 3).
  - Compute Budget: 1 model / 5-model ensemble / 5-model + TTA + Refinement ("Deep Think").
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loops import RoundConfig, TrainingProgress, train_one_round
from .model import ClimateTwinModel
from .metrics import compute_all_metrics, masked_rmse


def apply_tta_single(model: ClimateTwinModel, x: torch.Tensor, mc_samples: int = 10) -> torch.Tensor:
    """Test-Time Augmentation (TTA) for a single model or forward call.

    Applies small label-preserving spatial shifts and noise perturbations,
    predicts each, un-transforms spatial shifts, and averages.
    Input: x (B, T, C, H, W). Output: (B, C, H, W) prediction tensor.
    """
    model.eval()
    with torch.no_grad():
        # 1. Original prediction
        p_orig = model(x)
        preds = [p_orig]

        # 2. Horizontal shift +1
        x_shift_r = torch.roll(x, shifts=1, dims=-1)
        p_shift_r = model(x_shift_r)
        preds.append(torch.roll(p_shift_r, shifts=-1, dims=-1))

        # 3. Vertical shift +1
        x_shift_d = torch.roll(x, shifts=1, dims=-2)
        p_shift_d = model(x_shift_d)
        preds.append(torch.roll(p_shift_d, shifts=-1, dims=-2))

        # 4. Tiny noise perturbation
        noise = torch.randn_like(x) * 0.005
        x_noisy = torch.clamp(x + noise, 0.0, 1.0)
        p_noisy = model(x_noisy)
        preds.append(p_noisy)

        # Average over variants
        stacked = torch.stack(preds, dim=0)  # (4, B, C, H, W)
        return torch.mean(stacked, dim=0)


def apply_iterative_refinement(
    model_func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    max_passes: int = 3,
    truth: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Multi-pass iterative refinement ("Deep Think Pass").

    Feeds the model's own p50 prediction back as context for a sharpening pass.
    Capped at `max_passes` (default 3). Reports skill at each pass. Reverts if no gain.
    """
    B, T, C, H, W = x.shape
    curr_x = x.clone()

    pass_preds = []
    pass_rmses = []

    for p in range(max_passes):
        pred = model_func(curr_x)  # (B, C, H, W)
        pass_preds.append(pred)

        if truth is not None and mask is not None:
            r = masked_rmse(pred.cpu().numpy()[0, 0], truth.cpu().numpy()[0, 0], mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask)
            pass_rmses.append(r)
        else:
            pass_rmses.append(float("nan"))

        # Update context for next pass: append pred frame, drop oldest
        new_frame = pred.unsqueeze(1)  # (B, 1, C, H, W)
        curr_x = torch.cat([curr_x[:, 1:], new_frame], dim=1)

    # Check if refinement helped or hurt
    best_pass_idx = 0
    if truth is not None and len(pass_rmses) > 1 and not np.isnan(pass_rmses[0]):
        valid_rmses = [r for r in pass_rmses if not np.isnan(r)]
        if valid_rmses:
            best_r = min(valid_rmses)
            best_pass_idx = pass_rmses.index(best_r)

    final_pred = pass_preds[best_pass_idx]
    refinement_info = {
        "passes_run": max_passes,
        "best_pass": best_pass_idx + 1,
        "pass_rmses": pass_rmses,
        "reverted": best_pass_idx == 0 and max_passes > 1,
    }

    return final_pred, refinement_info


class DeepEnsemble(nn.Module):
    """Deep Ensemble of K ClimateTwinModels trained with different seeds."""

    def __init__(self, models: list[ClimateTwinModel], seeds: list[int] | None = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.seeds = seeds or list(range(len(models)))

    def predict_ensemble(
        self,
        x: torch.Tensor,
        mc_samples_per_model: int = 10,
        use_tta: bool = False,
        refine_passes: int = 1,
        truth: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Runs inference across all K models + optional TTA + optional Iterative Refinement.

        Combines inter-model variance (epistemic) + intra-model MC-dropout variance (aleatoric).
        Returns dict with keys: p10, p50, p90, mean, std, epistemic_std, aleatoric_std, refinement_info.
        """
        device = next(self.parameters()).device
        x = x.to(device)

        def _forward_single_model(m: ClimateTwinModel, in_x: torch.Tensor) -> torch.Tensor:
            if use_tta:
                return apply_tta_single(m, in_x, mc_samples=mc_samples_per_model)
            return m(in_x)

        # Gather predictions from each of the K models
        model_p50s = []
        model_stds = []

        for m in self.models:
            if mc_samples_per_model > 1:
                res = m.predict_ensemble(x, n_samples=mc_samples_per_model)
                model_p50s.append(res["p50"])
                model_stds.append(res["std"])
            else:
                p = _forward_single_model(m, x)
                model_p50s.append(p)
                model_stds.append(torch.zeros_like(p))

        stacked_p50s = torch.stack(model_p50s, dim=0)  # (K, B, C, H, W)
        stacked_stds = torch.stack(model_stds, dim=0)  # (K, B, C, H, W)

        # Average prediction across models
        ens_p50 = torch.mean(stacked_p50s, dim=0)       # (B, C, H, W)
        epistemic_var = torch.var(stacked_p50s, dim=0)   # (B, C, H, W) inter-model spread
        aleatoric_var = torch.mean(stacked_stds ** 2, dim=0) # intra-model MC dropout variance
        total_std = torch.sqrt(epistemic_var + aleatoric_var + 1e-8)

        # Apply iterative refinement if requested
        refinement_info = None
        if refine_passes > 1:
            def _ens_func(in_x):
                preds = [_forward_single_model(m, in_x) for m in self.models]
                return torch.mean(torch.stack(preds, dim=0), dim=0)

            ens_p50, refinement_info = apply_iterative_refinement(
                _ens_func, x, max_passes=refine_passes, truth=truth, mask=mask
            )

        # Quantile approximations using total_std
        p10 = torch.clamp(ens_p50 - 1.28 * total_std, 0.0, 1.0)
        p90 = torch.clamp(ens_p50 + 1.28 * total_std, 0.0, 1.0)

        return {
            "p50": ens_p50,
            "mean": ens_p50,
            "p10": p10,
            "p90": p90,
            "std": total_std,
            "epistemic_std": torch.sqrt(epistemic_var + 1e-8),
            "aleatoric_std": torch.sqrt(aleatoric_var + 1e-8),
            "refinement_info": refinement_info,
            "k_models": len(self.models),
        }


def train_deep_ensemble(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    val_data: np.ndarray,
    val_targets: np.ndarray,
    mask: np.ndarray,
    base_cfg: RoundConfig,
    k_models: int = 5,
    stop_check: Callable[[], bool] | None = None,
    round_num: int = 1,
    val_period: str = "",
    region: str = "india",
    on_progress: Callable[[int, int, dict], None] | None = None,
) -> tuple[DeepEnsemble, dict[str, Any], list[dict[str, Any]]]:
    """Train K models with different random seeds on the same data.

    Returns:
        (DeepEnsemble, ensemble_metrics, individual_models_metrics_list)
    """
    models = []
    indiv_metrics = []

    for i in range(k_models):
        if stop_check and stop_check():
            break

        seed = base_cfg.seed + i * 100
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed

        progress = TrainingProgress()
        model, metrics = train_one_round(
            train_data, train_targets, val_data, val_targets, mask,
            cfg, progress, model=None, stop_check=stop_check,
            round_num=round_num, val_period=val_period, region=region,
        )
        models.append(model)
        indiv_metrics.append(metrics)

        if on_progress:
            on_progress(i + 1, k_models, metrics)

    ens = DeepEnsemble(models)

    # Compute overall Ensemble metrics on validation set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_val = torch.from_numpy(val_data.transpose(0, 1, 4, 2, 3)).float().to(device)
    Y_val = torch.from_numpy(val_targets.transpose(0, 3, 1, 2)).float().to(device)

    ens.eval()
    ens_metrics = {}
    if len(X_val) > 0:
        res = ens.predict_ensemble(X_val, mc_samples_per_model=10)
        p50 = res["p50"].cpu().numpy()
        truth = Y_val.cpu().numpy()

        sample_m = []
        for i in range(min(len(p50), 50)):
            for c in range(p50.shape[1]):
                m = compute_all_metrics(p50[i, c], truth[i, c], mask)
                sample_m.append(m)

        if sample_m:
            for key in sample_m[0]:
                vals = [sm[key] for sm in sample_m if not np.isnan(sm[key])]
                ens_metrics[key] = float(np.mean(vals)) if vals else float("nan")

    return ens, ens_metrics, indiv_metrics
