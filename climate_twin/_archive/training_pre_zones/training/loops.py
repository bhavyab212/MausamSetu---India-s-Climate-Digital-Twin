"""Training loop: train_one_round() + validate().

Handles masked loss, warm-starting, early stopping, and live metric reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .baselines import compute_baseline_metrics
from .checkpoints import save_checkpoint
from .metrics import compute_all_metrics, ensemble_calibration
from .model import ClimateTwinModel


@dataclass
class RoundConfig:
    """All hyperparameters for one training round."""
    lr: float = 1e-3
    batch_size: int = 8
    epochs: int = 25
    weight_decay: float = 1e-5
    dropout: float = 0.1
    optimizer_name: str = "AdamW"  # AdamW, SGD, Lion
    scheduler_name: str = "cosine"  # cosine, plateau, step, none
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    mixed_precision: str = "bf16"  # bf16, fp16, fp32
    gradient_accum: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-4
    monitor: str = "val_rmse"
    physics_water_balance: float = 0.10
    physics_spatial_smooth: float = 0.05
    physics_temporal_smooth: float = 0.05
    physics_tmax_tmin: float = 0.02      # Phase 5e: soft hinge tmax ≥ tmin
    recency_weighting: bool = True
    recency_half_life_years: int = 20
    seed: int = 42
    deterministic: bool = False
    seq_length: int = 30
    hidden: int = 32
    residual_scale: float = 0.1
    mc_samples: int = 20
    freeze_recurrent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingProgress:
    """Mutable progress state for the UI to poll."""
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_rmses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    elapsed: float = 0.0
    eta: float = 0.0
    running: bool = False
    finished: bool = False
    error: str | None = None
    best_val: float = float("inf")
    patience_counter: int = 0


def _build_optimizer(model: nn.Module, cfg: RoundConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer_name == "SGD":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    elif cfg.optimizer_name == "Lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        except ImportError:
            pass
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer, cfg: RoundConfig, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    if cfg.scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.min_lr)
    elif cfg.scheduler_name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=cfg.min_lr)
    elif cfg.scheduler_name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)
    return None


def _masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NaN-safe masked Huber loss.

    pred/target: (B, C, H, W). mask: (H, W) region land mask.

    Post-Phase-4 the target may carry NaN where a variable was unavailable
    (e.g. INSAT LST outside 2020-2021). We form a per-cell "valid" mask =
    ``land_mask AND target_is_finite`` and average only over those cells.
    Cells whose target is NaN contribute zero and don't count toward the
    denominator — so missing data is masked, never zero-filled.
    """
    m = mask.view(1, 1, *mask.shape[-2:])                # (1,1,H,W)
    finite = torch.isfinite(target)                       # (B,C,H,W)
    valid = finite & (m > 0)                              # bool

    # Zero-out NaN targets so multiplication produces finite numbers.
    tgt_safe = torch.where(finite, target, torch.zeros_like(target))
    diff = pred - tgt_safe
    abs_diff = torch.abs(diff)
    delta = 0.08
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    huber = huber * valid.to(huber.dtype)
    denom = valid.to(huber.dtype).sum() + 1e-8
    return huber.sum() / denom


def _spatial_smoothness(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TV-L2 spatial smoothness penalty (land-mask weighted)."""
    m = mask.view(1, 1, *mask.shape[-2:])
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) ** 2
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) ** 2
    mx = m[:, :, :, 1:] * m[:, :, :, :-1]
    my = m[:, :, 1:, :] * m[:, :, :-1, :]
    num = (dx * mx).sum() + (dy * my).sum()
    denom = mx.sum() + my.sum() + 1e-8
    return num / denom


def _tmax_ge_tmin_penalty(
    pred: torch.Tensor,
    mask: torch.Tensor,
    variables: list[str] | None,
) -> torch.Tensor:
    """Hinge penalty enforcing tmax ≥ tmin on land cells.

    Requires ``variables`` to expose both channel names. Returns 0 when the
    round doesn't include both variables (e.g. rain-only mini-run).
    """
    if not variables:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    try:
        i_max = variables.index("tmax")
        i_min = variables.index("tmin")
    except ValueError:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    tmax = pred[:, i_max]                                  # (B, H, W)
    tmin = pred[:, i_min]
    viol = torch.clamp(tmin - tmax, min=0.0)               # >0 when tmin>tmax
    m = mask.view(1, *mask.shape[-2:])
    viol = viol * m
    denom = m.sum() * pred.shape[0] + 1e-8
    return viol.sum() / denom


def train_one_round(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    val_data: np.ndarray,
    val_targets: np.ndarray,
    mask: np.ndarray,
    cfg: RoundConfig,
    progress: TrainingProgress,
    model: ClimateTwinModel | None = None,
    stop_check: Callable[[], bool] | None = None,
    round_num: int = 1,
    val_period: str = "",
    region: str = "india",
    on_epoch: Callable[[dict], None] | None = None,
    variables: list[str] | None = None,
    manifest_sig: str | None = None,
) -> tuple[ClimateTwinModel, dict[str, Any]]:
    """Train one walk-forward round.

    Args:
        train_data: (N_train, seq_length, H, W, C) normalized [0,1]
        train_targets: (N_train, H, W, C) — next-step ground truth
        val_data: (N_val, seq_length, H, W, C)
        val_targets: (N_val, H, W, C)
        mask: (H, W) — 1=valid land, 0=ocean/outside
        cfg: hyperparameters
        progress: mutable UI state
        model: warm-start from existing model, or None for fresh
        stop_check: callable that returns True when user wants to stop
        round_num: for checkpoint naming
        val_period: for checkpoint naming
        variables: ordered list of channel names (Phase 5d contract). Recorded
            in the checkpoint so ``load_checkpoint`` can refuse a mismatched
            load. Length must equal the last dim of ``train_data``.
        manifest_sig: signature of the cube these tensors came from — stored
            alongside so a stale-cube-vs-code drift is auditable.

    Returns:
        (trained_model, final_metrics_dict)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        progress.error = "CUDA not available. Training requires GPU."
        return model, {}

    torch.manual_seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data shapes: input is (N, T, H, W, C) → need (N, T, C, H, W) for PyTorch
    H, W = mask.shape
    C = train_data.shape[-1]

    # Variable-list ↔ channel-count consistency (Phase 5d contract).
    if variables is not None and len(variables) != C:
        progress.error = (
            f"variables list length {len(variables)} != channel count {C} "
            f"(train_data channels={C}, variables={variables})"
        )
        return model, {}

    X_train = torch.from_numpy(train_data.transpose(0, 1, 4, 2, 3)).float()
    Y_train = torch.from_numpy(train_targets.transpose(0, 3, 1, 2)).float()
    X_val = torch.from_numpy(val_data.transpose(0, 1, 4, 2, 3)).float()
    Y_val = torch.from_numpy(val_targets.transpose(0, 3, 1, 2)).float()
    mask_t = torch.from_numpy(mask).float().to(device)

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    # Annual data yields few samples per round — never drop the only batch, and
    # never use a batch larger than the dataset (which would make 0 batches and
    # silently train nothing). Force num_workers=0 (Windows/Streamlit-safe).
    eff_bs = max(1, min(cfg.batch_size, len(train_ds)))
    val_bs = max(1, min(cfg.batch_size, len(val_ds))) if len(val_ds) > 0 else 1
    train_loader = DataLoader(train_ds, batch_size=eff_bs, shuffle=True, drop_last=False,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False,
                            num_workers=0, pin_memory=(device == "cuda"))

    # Model
    if model is None:
        model = ClimateTwinModel(
            seq_length=cfg.seq_length, lat_dim=H, lon_dim=W,
            channels=C, hidden=cfg.hidden, dropout=cfg.dropout,
            residual_scale=cfg.residual_scale,
        )
    model = model.to(device)

    if cfg.freeze_recurrent:
        for name, param in model.named_parameters():
            if "cell" in name or "bn" in name:
                param.requires_grad = False

    model.train()

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))

    # Mixed precision
    use_amp = cfg.mixed_precision in ("bf16", "fp16") and device == "cuda"
    amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

    progress.total_epochs = cfg.epochs
    progress.total_batches = len(train_loader) * cfg.epochs
    progress.running = True
    t_start = time.time()

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        if stop_check and stop_check():
            break

        model.train()
        epoch_loss = 0.0
        epoch_grad = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            if stop_check and stop_check():
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(batch_x)
                loss = _masked_loss(pred, batch_y, mask_t)

                if cfg.physics_spatial_smooth > 0:
                    loss = loss + cfg.physics_spatial_smooth * _spatial_smoothness(pred, mask_t)

                # tmax ≥ tmin physics penalty (only active when both variables present)
                if cfg.physics_tmax_tmin > 0 and variables:
                    loss = loss + cfg.physics_tmax_tmin * _tmax_ge_tmin_penalty(
                        pred, mask_t, variables
                    )

            scaler.scale(loss).backward()

            if (n_batches + 1) % cfg.gradient_accum == 0:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    epoch_grad += grad_norm.item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler and cfg.scheduler_name != "plateau":
                    # Skip warmup scheduling
                    if epoch >= cfg.warmup_epochs:
                        scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            progress.batch += 1

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_truths = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    vp = model(vx)
                    vl = _masked_loss(vp, vy, mask_t)
                val_loss += vl.item() * vx.size(0)
                val_preds.append(vp.cpu().numpy())
                val_truths.append(vy.cpu().numpy())

        val_loss /= max(len(val_ds), 1)
        val_preds = np.concatenate(val_preds, axis=0)  # (N, C, H, W)
        val_truths = np.concatenate(val_truths, axis=0)

        # Compute skill metrics (average over validation samples)
        mask_np = mask
        sample_metrics = []
        for i in range(min(len(val_preds), 50)):  # cap at 50 for speed
            for c in range(C):
                m = compute_all_metrics(val_preds[i, c], val_truths[i, c], mask_np)
                sample_metrics.append(m)
        avg_metrics = {}
        if sample_metrics:
            for key in sample_metrics[0]:
                vals = [sm[key] for sm in sample_metrics if not np.isnan(sm[key])]
                avg_metrics[key] = float(np.mean(vals)) if vals else float("nan")

        # Update progress
        avg_train = epoch_loss / max(n_batches, 1)
        avg_grad = epoch_grad / max(n_batches // cfg.gradient_accum, 1)
        progress.epoch = epoch + 1
        progress.train_loss = avg_train
        progress.val_loss = val_loss
        progress.val_metrics = avg_metrics
        progress.train_losses.append(avg_train)
        progress.val_losses.append(val_loss)
        progress.val_rmses.append(float(avg_metrics.get("rmse", float("nan"))))
        progress.grad_norms.append(avg_grad)
        progress.lrs.append(optimizer.param_groups[0]["lr"])
        progress.elapsed = time.time() - t_start
        progress.eta = progress.elapsed / (epoch + 1) * (cfg.epochs - epoch - 1)

        # Live callback for the UI (GPU memory, throughput, losses per epoch)
        if on_epoch is not None:
            try:
                gpu_alloc = torch.cuda.memory_allocated(device) / 1e6 if device == "cuda" else 0.0
                gpu_reserved = torch.cuda.memory_reserved(device) / 1e6 if device == "cuda" else 0.0
                samples_per_s = (len(train_ds) * (epoch + 1)) / max(progress.elapsed, 1e-6)
                on_epoch({
                    "epoch": epoch + 1,
                    "total_epochs": cfg.epochs,
                    "train_loss": avg_train,
                    "val_loss": val_loss,
                    "val_metrics": avg_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": avg_grad,
                    "elapsed": progress.elapsed,
                    "eta": progress.eta,
                    "gpu_alloc_mb": gpu_alloc,
                    "gpu_reserved_mb": gpu_reserved,
                    "samples_per_s": samples_per_s,
                    "train_losses": list(progress.train_losses),
                    "val_losses": list(progress.val_losses),
                })
            except Exception:
                pass

        # Plateau scheduler
        if scheduler and cfg.scheduler_name == "plateau":
            scheduler.step(val_loss)

        # Early stopping
        monitor_val = avg_metrics.get(cfg.monitor, val_loss)
        if monitor_val < best_val - cfg.min_delta:
            best_val = monitor_val
            patience_counter = 0
            progress.best_val = best_val
            # Save best checkpoint (Phase 5d contract: variable list + sig + grid)
            ckpt_path = save_checkpoint(
                model, optimizer, round_num, val_period,
                cfg.to_dict(), avg_metrics, epoch + 1, region=region,
                variables=variables, manifest_sig=manifest_sig,
                grid_shape=(H, W),
            )
        else:
            patience_counter += 1
            progress.patience_counter = patience_counter
            if cfg.early_stopping and patience_counter >= cfg.patience:
                break

    progress.running = False
    progress.finished = True

    # Final ensemble prediction for calibration
    model.eval()
    if len(X_val) > 0:
        sample_x = X_val[:min(10, len(X_val))].to(device)
        ens = model.predict_ensemble(sample_x, n_samples=cfg.mc_samples)
        p10 = ens["p10"].cpu().numpy()   # (B, C, H, W)
        p90 = ens["p90"].cpu().numpy()
        truth_sample = Y_val[:min(10, len(Y_val))].numpy()  # (B, C, H, W)
        # ensemble_calibration works on 2-D (H, W) fields — average over the
        # validation samples of channel 0.
        cals = []
        for i in range(p10.shape[0]):
            c = ensemble_calibration(p10[i, 0], p90[i, 0], truth_sample[i, 0], mask_np)
            if not np.isnan(c):
                cals.append(c)
        avg_metrics["ensemble_calibration"] = float(np.mean(cals)) if cals else float("nan")

    # Compute baseline metrics
    if len(val_data) > 0:
        last_obs = train_data[-1, -1]  # (H, W, C) — last frame of training
        baseline_m = compute_baseline_metrics(
            val_targets[0], last_obs, train_targets, mask_np,
        )
        progress.baseline_metrics = baseline_m

    return model, avg_metrics
