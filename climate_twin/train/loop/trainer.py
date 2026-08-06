"""
train.loop.trainer — the zone-aware training loop.

Composition:
    * Reads an :class:`ExperimentConfig` produced by ``train.config.load_config``
    * Builds train / val datasets from the frozen cube
    * Builds the FiLM-conditioned model from Phase 2d
    * Applies the NaN-safe zone-weighted loss + per-zone physics penalties
    * Fires Tier 1 every ``validation.tier1_every_n_batches``
    * Fires Tier 2 every ``validation.tier2_every_n_epochs``
    * Fires Tier 3 every ``validation.tier3_every_n_epochs``
    * Fires Tier 4 at the end of the whole round
    * Honours per-zone early stopping (``worst_zone`` default)
    * Emits per-epoch ``on_epoch`` callback so the UI can render live
    * Persists best checkpoint via the registry

The trainer is a plain object with a ``.run()`` method; it does not spawn
threads. The Streamlit UI drives it inside a background thread and reads
:class:`LiveState` for the live view.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from climate_twin import data_source as DS
from climate_twin.regions import ZoneRegistry, get_zones

from ..config.schema import ExperimentConfig
from ..data import (
    DailyWindowDataset,
    PerZoneZScore,
    ZoneStratifiedWeights,
    default_collate,
    log_batch_composition,
)
from ..eval import (
    persistence_prediction,
    climatology_prediction,
    per_zone_convergence_state,
    run_tier1,
    run_tier2,
    run_tier3,
    run_tier4,
    render_tier3_heatmap,
    INSUFFICIENT,
)
from ..loss import (
    tmax_ge_tmin_penalty,
    total_train_loss,
    zone_physics_bounds_penalty,
)
from ..model import build_model


# ---------------------------------------------------------------------------
# LiveState — mutable, thread-safe-ish snapshot for the UI
# ---------------------------------------------------------------------------
@dataclass
class LiveState:
    """Everything the UI's live view wants to display."""
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_losses: list[float] = field(default_factory=list)
    tier1_history: list[dict[str, Any]] = field(default_factory=list)
    tier2_history: list[dict[str, Any]] = field(default_factory=list)
    tier3_last: dict[str, Any] | None = None
    tier4_final: dict[str, Any] | None = None
    per_zone_convergence: dict[str, Any] | None = None
    lr: float = 0.0
    grad_norm: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0
    running: bool = False
    finished: bool = False
    stop_requested: bool = False
    error: str | None = None
    message: str = ""
    heatmap_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RunOutputs — everything the caller receives after `.run()` completes
# ---------------------------------------------------------------------------
@dataclass
class RunOutputs:
    best_state_dict_path: Path
    meta_json_path: Path
    tier4_json_path: Path
    heatmap_path: Path | None
    best_zone_weighted_rmse: float
    n_epochs_run: int
    per_zone_final: dict[str, Any]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Zone-aware trainer."""

    IMD_ABSOLUTE_THRESHOLDS = {
        "moderate": 15.6,
        "heavy": 64.5,
        "very_heavy": 115.6,
        "extremely_heavy": 204.5,
    }

    def __init__(
        self,
        cfg: ExperimentConfig,
        model_name: str,
        registry_root: Path,
        zones: ZoneRegistry | None = None,
        device: str | None = None,
        live: LiveState | None = None,
        on_epoch: Callable[[dict], None] | None = None,
        parent_name: str = "",
        executor=None,
    ):
        self.cfg = cfg
        self.model_name = model_name
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.zones = zones or get_zones()

        # ── CUDA-required policy ────────────────────────────────
        # Silent CPU fallback is forbidden: it would report metrics at
        # ~50× the real GPU rate and hide performance regressions.
        # We do not raise the runtime.device exception here to avoid a
        # circular import; instead we do the check inline with the same
        # user-facing message.
        if device is None:
            if not torch.cuda.is_available():
                from climate_twin.runtime.device import (
                    CUDA_UNAVAILABLE_MESSAGE,
                    RuntimeCudaRequired,
                )
                raise RuntimeCudaRequired(
                    "Trainer requires CUDA.\n\n" + CUDA_UNAVAILABLE_MESSAGE
                )
            self.device = "cuda"
        else:
            self.device = device

        self.live = live or LiveState()
        self.on_epoch = on_epoch
        self.parent_name = parent_name
        self.executor = executor  # runtime.executor.TrainingExecutor | None

        # Deterministic RNG setup
        import os
        import random
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        if cfg.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Per-zone percentile thresholds from Phase 1 stats
        stats_path = Path(__file__).resolve().parents[2] / "regions" / "zone_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.zone_percentile_thresholds = {
            zkey: {
                "p90": float(b["percentiles_rain_mm_day"]["p90"]),
                "p95": float(b["percentiles_rain_mm_day"]["p95"]),
                "p99": float(b["percentiles_rain_mm_day"]["p99"]),
            }
            for zkey, b in stats["zones"].items()
        }

    # ------------------------------------------------------------------
    def _make_datasets(self):
        tf = PerZoneZScore(tuple(self.cfg.data.variables), zones=self.zones)
        train = DailyWindowDataset(
            region=self.cfg.data.region,
            variables=tuple(self.cfg.data.variables),
            seq_length=self.cfg.model.seq_length,
            year_range=tuple(self.cfg.data.train_years),
            zones=self.zones,
            transform=tf,
        )
        val = DailyWindowDataset(
            region=self.cfg.data.region,
            variables=tuple(self.cfg.data.variables),
            seq_length=self.cfg.model.seq_length,
            year_range=tuple(self.cfg.data.val_years),
            zones=self.zones,
            transform=tf,
        )
        return train, val, tf

    # ------------------------------------------------------------------
    def _val_predict(self, model, val_ds, tf, batch_size=4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model over the val set. Return (pred_phys, truth_phys, months)
        each shaped (T_val, H, W) for rain channel 0."""
        model.eval()
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=default_collate, num_workers=0)
        zone_map = (
            torch.from_numpy(self.zones.membership).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        preds = []
        truths = []
        months = []
        C = len(self.cfg.data.variables)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                # Input is z-score already; the target y is physical
                # Normalise input; the dataset already did that.
                zm = zone_map.expand(x.shape[0], -1, -1, -1).contiguous()
                out = model(x, zm)
                # Rain channel prediction (physical) — rain uses softplus(amount) * sigmoid(logit)
                if "rain" in out:
                    p_amount = torch.nn.functional.softplus(out["rain"]["amount"]).squeeze(1)
                    p_occ = torch.sigmoid(out["rain"]["logit_occurrence"]).squeeze(1)
                    pred_rain = p_occ * p_amount
                    truth_rain = y[:, 0]                        # rain is channel 0
                    preds.append(pred_rain.detach().cpu().numpy())
                    truths.append(truth_rain.detach().cpu().numpy())
                    months.extend(batch["month"])
        pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, self.zones.hard_mask.shape[0], self.zones.hard_mask.shape[1]))
        truth = np.concatenate(truths, axis=0) if truths else np.zeros_like(pred)
        return pred, truth, np.array(months, dtype=np.int32)

    # ------------------------------------------------------------------
    def _build_climatology_for_val(self, val_ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build baseline persistence + climatology aligned with val_ds targets.

        Returns (persistence, climatology, mask_of_valid_targets) each ``(T_val, H, W)``.
        Uses the cube directly to have the exact same days as ``_val_predict``.
        """
        ds = DS.load_region(self.cfg.data.region)
        years = np.asarray(ds["time.year"].values)
        doys = np.asarray(ds["time.dayofyear"].values)
        # Match the dataset's target indexing: every day in val_years for
        # which we have at least seq_length days of context. Because the
        # cube starts in 1951 and val_years >= 2018, every val day has
        # ≥ seq_length context — so no [seq_length:] slice is needed here.
        val_sel = (years >= self.cfg.data.val_years[0]) & (years <= self.cfg.data.val_years[1])
        target_idx = np.flatnonzero(val_sel)
        # Persistence = truth[t-1] — read only the days we need
        rain_var = ds["rain"]
        truth = rain_var.isel(time=target_idx).values
        prev = rain_var.isel(time=target_idx - 1).values
        persistence = prev

        # Climatology = per-DOY mean over train years, computed one DOY at a
        # time so we never materialise the full 27k-day rain tensor.
        train_sel = (years >= self.cfg.data.train_years[0]) & (years <= self.cfg.data.train_years[1])
        train_doys = doys[train_sel]
        train_idx_by_doy: dict[int, np.ndarray] = {}
        train_mask = np.flatnonzero(train_sel)
        for _pos, _d in enumerate(train_doys):
            train_idx_by_doy.setdefault(int(_d), []).append(int(train_mask[_pos]))
        H, W = truth.shape[-2:]
        clim366 = np.full((366, H, W), np.nan, dtype=np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            for d in range(1, 367):
                idx = train_idx_by_doy.get(d)
                if not idx:
                    continue
                slab = rain_var.isel(time=np.asarray(idx)).values
                clim366[d - 1] = np.nanmean(slab, axis=0)
        val_doys = doys[target_idx]
        climatology = clim366[val_doys - 1]
        ds.close()
        return persistence, climatology

    # ------------------------------------------------------------------
    def _save_checkpoint(self, model, meta: dict) -> tuple[Path, Path]:
        folder = self.registry_root / meta["region"] / meta["name"]
        folder.mkdir(parents=True, exist_ok=True)
        w = folder / "weights.pt"
        torch.save({"model_state_dict": model.state_dict(),
                    **meta}, w)
        meta_path = folder / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return w, meta_path

    # ------------------------------------------------------------------
    def _emit(self, kind, **payload) -> None:
        """Send an event to the executor's queue (no-op when running standalone)."""
        if self.executor is not None:
            try:
                self.executor.events.emit(kind, **payload)
            except Exception:
                # An event-queue failure must never take down training.
                pass

    def _wait_if_paused(self) -> None:
        """Block if the executor requested a pause. Called between batches."""
        if self.executor is not None:
            self.executor.wait_if_paused()

    def _stop_requested(self) -> bool:
        if self.executor is not None and self.executor.is_stop_requested():
            return True
        return bool(self.live.stop_requested)

    def _executor_state_update(self, **kwargs) -> None:
        """Push a state snapshot to the executor (for the RunState mirror)."""
        if self.executor is not None:
            try:
                self.executor._update(**kwargs)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def run(self) -> RunOutputs:
        cfg = self.cfg
        Z = self.zones
        live = self.live
        live.total_epochs = cfg.optim.epochs
        live.running = True
        t_start = time.perf_counter()

        # Late import to avoid circular imports at module load
        from climate_twin.runtime import (
            EventKind, choose_precision, tune_batch_size,
        )

        # ── Data
        train_ds, val_ds, tf = self._make_datasets()

        # ── Zone weights + masks + zone map (honours Phase 5a held-out zones)
        zw = ZoneStratifiedWeights(
            Z,
            mode=cfg.optim.sampler_mode,
            device=self.device,
            excluded_zone_keys=tuple(cfg.zones.excluded_zone_keys),
        )
        live.message = log_batch_composition(zw, n_batches=5)
        weight = zw.tensor
        mask = torch.from_numpy((Z.hard_mask > 0).astype(np.float32)).to(self.device)
        zone_map = (
            torch.from_numpy(Z.membership).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        # ── Precision auto-select ──
        precision_policy = "auto" if getattr(cfg.optim, "mixed_precision_auto", False) \
                                    else cfg.optim.mixed_precision
        chosen_precision, precision_reason = choose_precision(precision_policy)
        self._emit(EventKind.LOG, level="info",
                    message=f"precision → {chosen_precision}  ({precision_reason})")

        # ── Batch-size auto-tune ──
        chosen_bs = int(cfg.optim.batch_size)
        tune_report: dict[str, Any] = {}
        if getattr(cfg.optim, "batch_size_auto", False) and self.device == "cuda":
            self._emit(EventKind.LOG, level="info",
                        message="tuning batch size to saturate VRAM…")

            def _probe(bs: int) -> float:
                """Build model, run one fwd+bwd+step at batch size `bs`,
                return peak-VRAM in MB. Called by tune_batch_size."""
                probe_model = build_model(cfg, zones=Z).to(self.device)
                probe_model.train()
                probe_opt = torch.optim.AdamW(probe_model.parameters(),
                                                lr=cfg.optim.lr)
                # Synthesise a batch of shape (bs, T, C, H, W)
                T = cfg.model.seq_length
                C = len(cfg.data.variables)
                H, W = Z.hard_mask.shape
                x = torch.randn(bs, T, C, H, W, device=self.device)
                y = torch.randn(bs, C, H, W, device=self.device)
                zm = zone_map.expand(bs, -1, -1, -1).contiguous()
                torch.cuda.reset_peak_memory_stats()
                amp_dtype = ({"bf16": torch.bfloat16,
                               "fp16": torch.float16,
                               "fp32": torch.float32}[chosen_precision])
                autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                                  if chosen_precision in ("bf16", "fp16")
                                  else torch.autocast(device_type="cuda", enabled=False))
                with autocast_ctx:
                    out = probe_model(x, zm)
                    targets = {v: y[:, i:i + 1] for i, v in enumerate(cfg.data.variables)}
                    loss, _ = total_train_loss(out, targets, weight, cfg)
                probe_opt.zero_grad(set_to_none=True)
                loss.backward()
                probe_opt.step()
                peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                del probe_model, probe_opt, x, y, zm, out, loss, targets
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                return peak

            def _emit_probe(entry):
                self._emit(EventKind.BATCH_TUNING, **entry)

            try:
                res = tune_batch_size(_probe, start=8, cap=256, safety=0.85,
                                        emit=_emit_probe)
                chosen_bs = max(1, int(res.batch_size))
                tune_report = {
                    "picked_bs": chosen_bs, "peak_vram_mb": res.peak_vram_mb,
                    "total_vram_mb": res.total_vram_mb, "tried": res.tried,
                    "reason": res.picked_reason,
                }
                self._emit(EventKind.BATCH_TUNED, **tune_report)
                self._emit(EventKind.LOG, level="info",
                            message=f"batch size → {chosen_bs}  ({res.picked_reason})")
            except Exception as _e:
                self._emit(EventKind.LOG, level="warn",
                            message=f"batch tuner failed ({type(_e).__name__}): "
                                    f"{_e}. Falling back to bs={cfg.optim.batch_size}.")
                chosen_bs = int(cfg.optim.batch_size)

        # ── Train loader (with chosen bs + Windows-safe worker settings)
        _dl_workers = int(getattr(cfg.optim, "num_workers", 0) or 0)
        _dl_pin = bool(getattr(cfg.optim, "pin_memory", False)) and self.device == "cuda"
        _dl_pf = int(getattr(cfg.optim, "prefetch_factor", 4) or 4) if _dl_workers > 0 else None
        _dl_persistent = (bool(getattr(cfg.optim, "persistent_workers", True))
                          and _dl_workers > 0)
        _loader_kwargs = dict(
            batch_size=chosen_bs, shuffle=True, collate_fn=default_collate,
            num_workers=_dl_workers, pin_memory=_dl_pin, drop_last=False,
        )
        if _dl_workers > 0:
            _loader_kwargs["prefetch_factor"] = _dl_pf
            _loader_kwargs["persistent_workers"] = _dl_persistent
        train_loader = DataLoader(train_ds, **_loader_kwargs)
        live.total_batches = len(train_loader) * cfg.optim.epochs
        self._executor_state_update(
            total_epochs=cfg.optim.epochs,
            total_batches=live.total_batches,
            device=(self.executor.state.device if self.executor else "cuda"),
        )

        # ── Model (final one used for training)
        model = build_model(cfg, zones=Z).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                                 weight_decay=cfg.optim.weight_decay)

        # ── Precision context factory (used inside the training loop)
        _amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[chosen_precision]
        _use_amp = chosen_precision in ("bf16", "fp16")
        _grad_scaler = (torch.amp.GradScaler("cuda", enabled=(chosen_precision == "fp16"))
                        if _use_amp else None)

        def _autocast():
            if _use_amp:
                return torch.autocast(device_type="cuda", dtype=_amp_dtype)
            # Return a no-op context manager
            import contextlib
            return contextlib.nullcontext()
        if cfg.optim.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, cfg.optim.epochs - cfg.optim.warmup_epochs),
                eta_min=cfg.optim.min_lr,
            )
        else:
            sched = None

        # ── Baselines (persistence + climatology) for skill reporting
        persistence, climatology = self._build_climatology_for_val(val_ds)

        # Per-zone baseline RMSE for Tier 2 skill readout — recompute once
        baseline_rmses = {}
        mask_np = (Z.hard_mask > 0).astype(np.float32)
        # Use TRUTH aligned with val_ds targets — same slicing as _build_climatology_for_val
        ds_ref = DS.load_region(cfg.data.region)
        years = np.asarray(ds_ref["time.year"].values)
        val_sel = (years >= cfg.data.val_years[0]) & (years <= cfg.data.val_years[1])
        # Match the DailyWindowDataset indexing exactly: every val day for
        # which seq_length context days exist in the cube. The cube begins
        # in 1951 so every 2023 target has ≥ seq_length context — no [seq_length:] slice.
        target_idx = np.flatnonzero(val_sel)
        truth_full = ds_ref["rain"].isel(time=target_idx).values
        ds_ref.close()
        months_full = np.asarray(np.array([DS.load_region.__wrapped__ if False else None]))  # noop
        # simpler: just get months from val_ds
        val_months = np.array([val_ds[i]["month"] for i in range(len(val_ds))], dtype=np.int32) if False else None

        # Precompute persistence baseline RMSE per zone (for Tier2 skill)
        for k, zone in enumerate(Z.zones):
            w = Z.membership[..., k].astype(np.float32)
            finite = np.isfinite(persistence) & np.isfinite(truth_full) & (mask_np[None] > 0)
            num = float(np.where(finite, (persistence - truth_full) ** 2 * w[None], 0.0).sum())
            den = float(np.where(finite, w[None], 0.0).sum())
            baseline_rmses[zone.key] = float(np.sqrt(num / den)) if den > 0 else float("nan")

        best_wrmse = float("inf")
        best_epoch = 0
        n_epochs_run = 0

        # Throughput monitor state (Part A5)
        _throughput_last_check = time.perf_counter()
        _throughput_last_batch = 0
        _under_util_streak = 0.0

        try:
            for epoch in range(cfg.optim.epochs):
                if self._stop_requested():
                    live.message = f"stop requested at epoch {epoch}"
                    self._emit(EventKind.LOG, level="info",
                                message=f"stop requested at epoch {epoch}")
                    break
                model.train()
                epoch_losses: list[float] = []
                grad_norm_sum = 0.0
                n_batches = 0
                for batch in train_loader:
                    # ── Pause / stop cooperative check (batch boundary) ──
                    self._wait_if_paused()
                    if self._stop_requested():
                        live.message = "stop requested mid-epoch"
                        self._emit(EventKind.LOG, level="info",
                                    message="stop requested mid-epoch")
                        break

                    x = batch["x"].to(self.device, non_blocking=True)
                    y = batch["y"].to(self.device, non_blocking=True)
                    zm = zone_map.expand(x.shape[0], -1, -1, -1).contiguous()

                    # NaN-safe input: ocean cells + gauge-interpolation gaps
                    # arrive as NaN. The zone weight tensor already zeros
                    # those cells out of the loss; a plain nan_to_num keeps
                    # the conv arithmetic finite without changing what the
                    # trainer considers valid.
                    x = torch.nan_to_num(x, nan=0.0)
                    with _autocast():
                        out = model(x, zm)
                        # Targets are physical rain (mm), tmax/tmin (°C)
                        targets = {v: y[:, i:i + 1] for i, v in enumerate(cfg.data.variables)}
                        loss, parts = total_train_loss(out, targets, weight, cfg)

                        # tmax≥tmin physics
                        if cfg.loss.physics_tmax_ge_tmin > 0 and "tmax" in out and "tmin" in out:
                            loss = loss + cfg.loss.physics_tmax_ge_tmin * tmax_ge_tmin_penalty(
                                out["tmax"], out["tmin"], weight, mask,
                            )

                        # Per-zone physics bounds (soft hinge on outside-plausible)
                        if cfg.loss.physics_bounds_penalty > 0:
                            loss = loss + cfg.loss.physics_bounds_penalty * zone_physics_bounds_penalty(
                                out, Z, weight, mask, tuple(cfg.data.variables),
                            )

                    opt.zero_grad(set_to_none=True)
                    if _grad_scaler is not None and _grad_scaler.is_enabled():
                        _grad_scaler.scale(loss).backward()
                        _grad_scaler.unscale_(opt)
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                        _grad_scaler.step(opt)
                        _grad_scaler.update()
                    else:
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                        opt.step()

                    epoch_losses.append(float(loss.detach().item()))
                    live.train_losses.append(epoch_losses[-1])
                    grad_norm_sum += float(gn.detach().item())
                    n_batches += 1
                    live.batch += 1
                    live.grad_norm = grad_norm_sum / max(n_batches, 1)
                    live.lr = opt.param_groups[0]["lr"]
                    live.elapsed = time.perf_counter() - t_start

                    # ── Throughput + util sample (every ~1 s wall) ──
                    _now = time.perf_counter()
                    samples_per_s = float("nan")
                    gpu_util = None
                    vram_used_mb = None
                    vram_total_mb = None
                    if _now - _throughput_last_check >= 1.0:
                        dt = _now - _throughput_last_check
                        n_new = live.batch - _throughput_last_batch
                        samples_per_s = (n_new * chosen_bs) / max(dt, 1e-6)
                        _throughput_last_check = _now
                        _throughput_last_batch = live.batch
                        try:
                            from climate_twin.runtime import gpu_utilization_snapshot as _gpu_snap
                            snap = _gpu_snap()
                            gpu_util = snap.get("util_percent")
                            vram_used_mb = snap.get("memory_used_mb")
                            vram_total_mb = snap.get("memory_total_mb")
                        except Exception:
                            pass
                        # <50% util warn (30 s sustained)
                        if isinstance(gpu_util, (int, float)) and gpu_util < 50:
                            _under_util_streak += dt
                            if _under_util_streak >= 30.0:
                                self._emit(
                                    EventKind.THROUGHPUT_WARN,
                                    gpu_util_percent=gpu_util,
                                    seconds_under_50=_under_util_streak,
                                    hint=("Increase optim.num_workers, "
                                          "prefetch_factor, or batch_size."),
                                )
                                _under_util_streak = 0.0
                        else:
                            _under_util_streak = 0.0

                    # Emit per-batch progress (bounded — payload is tiny)
                    self._emit(
                        EventKind.PROGRESS,
                        epoch=epoch + 1, batch=live.batch, total_batches=live.total_batches,
                        train_loss=epoch_losses[-1], lr=live.lr, grad_norm=live.grad_norm,
                        elapsed=live.elapsed,
                        samples_per_s=samples_per_s,
                        batch_size=chosen_bs, precision=chosen_precision,
                        gpu_util_percent=gpu_util,
                        vram_used_mb=vram_used_mb, vram_total_mb=vram_total_mb,
                    )
                    self._executor_state_update(
                        epoch=epoch + 1, batch=live.batch,
                        train_loss=epoch_losses[-1], lr=live.lr,
                        grad_norm=live.grad_norm, elapsed_seconds=live.elapsed,
                    )

                    # Tier 1 every N batches (cheap)
                    if n_batches % cfg.validation.tier1_every_n_batches == 0:
                        pred_v, truth_v, _ = self._val_predict(model, val_ds, tf, batch_size=cfg.optim.batch_size)
                        if pred_v.size > 0:
                            t1 = run_tier1(
                                pred_v, truth_v, np.ones_like(mask_np), mask_np,
                                min_cells=cfg.validation.min_cells_for_metric,
                            )
                            live.tier1_history.append(t1)
                            self._emit(EventKind.TIER1, **t1)
                        model.train()

                # After the batch loop, if a stop was requested mid-epoch, break out.
                if self._stop_requested():
                    break

                # ── end-of-epoch validation
                pred_v, truth_v, months_v = self._val_predict(model, val_ds, tf, batch_size=cfg.optim.batch_size)
                if pred_v.size == 0:
                    live.message = "empty validation set"
                    break

                # Tier 2 every epoch
                zw_rmse = float("nan")
                if (epoch + 1) % cfg.validation.tier2_every_n_epochs == 0:
                    t2 = run_tier2(pred_v, truth_v, Z, mask_np,
                                    baseline_rmses=baseline_rmses,
                                    min_cells=cfg.validation.min_cells_for_metric)
                    live.tier2_history.append(t2)
                    self._emit(EventKind.TIER2,
                                epoch=epoch + 1,
                                runtime_s=t2.get("runtime_s"),
                                per_zone=t2["per_zone"])

                    # zone-weighted RMSE for early stopping / best-of
                    weights_arr = np.array([
                        zw.per_zone_effective_weight()[z.key] for z in Z.zones
                    ], dtype=np.float64)
                    weights_arr = weights_arr / weights_arr.sum() if weights_arr.sum() > 0 else weights_arr
                    zw_rmse = 0.0
                    for zi, zone in enumerate(Z.zones):
                        r = t2["per_zone"].get(zone.key, {}).get("rmse", None)
                        if isinstance(r, float) and np.isfinite(r):
                            zw_rmse += weights_arr[zi] * r
                    if zw_rmse < best_wrmse - cfg.early_stopping.min_delta:
                        best_wrmse = zw_rmse
                        best_epoch = epoch + 1
                        # Save checkpoint
                        w_path, meta_path = self._save_checkpoint(model, {
                            "name": self.model_name, "region": cfg.data.region,
                            "parent_name": self.parent_name,
                            "config": cfg.model_dump(),
                            "zone_mask_sig": Z.mask_signature,
                            "zone_stats_sig": tf.zone_stats_sig,
                            "manifest_sig": DS.manifest_sig(),
                            "variables": list(cfg.data.variables),
                            "epoch": epoch + 1,
                            "zone_weighted_rmse": float(zw_rmse),
                            "tier2": t2,
                        })
                        self._emit(EventKind.CHECKPOINT_SAVED,
                                    epoch=epoch + 1,
                                    zone_weighted_rmse=float(zw_rmse),
                                    weights_path=str(w_path),
                                    meta_path=str(meta_path))
                        self._executor_state_update(
                            best_zone_weighted_rmse=float(zw_rmse),
                            best_epoch=epoch + 1,
                        )

                    # Per-zone convergence check
                    live.per_zone_convergence = per_zone_convergence_state(
                        Z, live.tier2_history,
                        monitor_key="rmse",
                        patience=cfg.early_stopping.patience,
                        min_delta=cfg.early_stopping.min_delta,
                        policy=cfg.early_stopping.policy,
                    )

                # Tier 3 every K epochs (expensive)
                if (epoch + 1) % cfg.validation.tier3_every_n_epochs == 0:
                    live.tier3_last = run_tier3(
                        pred_v, truth_v, months_v, Z, mask_np,
                        thresholds_absolute=self.IMD_ABSOLUTE_THRESHOLDS,
                        thresholds_zone_percentile=self.zone_percentile_thresholds,
                        min_cells=cfg.validation.min_cells_for_metric,
                    )
                    self._emit(EventKind.TIER3,
                                epoch=epoch + 1,
                                runtime_s=live.tier3_last.get("runtime_s"))

                if sched is not None and epoch >= cfg.optim.warmup_epochs:
                    sched.step()

                live.epoch = epoch + 1
                live.elapsed = time.perf_counter() - t_start
                live.eta = live.elapsed / (epoch + 1) * (cfg.optim.epochs - epoch - 1)
                n_epochs_run += 1

                if self.on_epoch is not None:
                    try:
                        self.on_epoch({
                            "epoch": epoch + 1,
                            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                            "zone_weighted_rmse": float(zw_rmse) if 'zw_rmse' in locals() else float("nan"),
                            "best_zw_rmse": best_wrmse,
                            "lr": live.lr,
                            "elapsed": live.elapsed,
                        })
                    except Exception:
                        pass

                if live.per_zone_convergence and live.per_zone_convergence["should_stop"]:
                    live.message = f"early stop: {live.per_zone_convergence['reason']}"
                    break

            # ── End of round: Tier 4 (bootstrap + Wilcoxon) + heatmap
            pred_v, truth_v, months_v = self._val_predict(model, val_ds, tf, batch_size=cfg.optim.batch_size)
            live.tier4_final = run_tier4(
                pred_v, truth_v, persistence, climatology, Z, mask_np,
                bootstrap_samples=cfg.validation.bootstrap_samples,
                seed=cfg.seed,
            )
            self._emit(EventKind.TIER4,
                        runtime_s=live.tier4_final.get("runtime_s"),
                        n_zones=len((live.tier4_final or {}).get("per_zone", {})))

            heatmap_path = None
            if live.tier3_last is not None:
                # Compute climatology's own cross-product so heatmap shows skill
                from ..eval.baselines import per_zone_baseline_skill
                base_cross = per_zone_baseline_skill(
                    truth_v, persistence, climatology, months_v, Z, mask_np,
                    self.IMD_ABSOLUTE_THRESHOLDS, self.zone_percentile_thresholds,
                    min_cells=cfg.validation.min_cells_for_metric,
                )
                heatmap_path = self.registry_root / cfg.data.region / self.model_name / "tier3_heatmap.png"
                render_tier3_heatmap(
                    live.tier3_last,
                    {"climatology": {"per_zone_season": base_cross["climatology"]}},
                    Z, out_path=heatmap_path,
                    title=f"{self.model_name} — Tier-3 skill vs climatology",
                    metric_key="rmse", baseline_source="climatology",
                )
                live.heatmap_paths.append(str(heatmap_path))
                self._emit(EventKind.HEATMAP_RENDERED,
                            path=str(heatmap_path),
                            metric="rmse", baseline="climatology")

            # Persist final tier 4 alongside the model
            folder = self.registry_root / cfg.data.region / self.model_name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "tier4.json").write_text(
                json.dumps(live.tier4_final, indent=2, default=str)
            )

            # Meta refresh with final numbers
            meta = {
                "name": self.model_name, "region": cfg.data.region,
                "parent_name": self.parent_name,
                "config": cfg.model_dump(),
                "zone_mask_sig": Z.mask_signature,
                "zone_stats_sig": tf.zone_stats_sig,
                "manifest_sig": DS.manifest_sig(),
                "variables": list(cfg.data.variables),
                "epochs_trained": n_epochs_run,
                "best_epoch": best_epoch,
                "best_zone_weighted_rmse": best_wrmse,
                "final_tier4_summary": {
                    zk: {
                        "rmse_ci": v.get("rmse_ci") if isinstance(v, dict) else v,
                        "vs_pers_winner": (
                            v.get("vs_persistence").get("winner")
                            if isinstance(v, dict) and isinstance(v.get("vs_persistence"), dict)
                            else v.get("vs_persistence") if isinstance(v, dict) else None
                        ),
                        "vs_clim_winner": (
                            v.get("vs_climatology").get("winner")
                            if isinstance(v, dict) and isinstance(v.get("vs_climatology"), dict)
                            else v.get("vs_climatology") if isinstance(v, dict) else None
                        ),
                    }
                    for zk, v in (live.tier4_final or {}).get("per_zone", {}).items()
                },
            }
            (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

            live.finished = True
            live.running = False
            live.message = "training complete"
            return RunOutputs(
                best_state_dict_path=folder / "weights.pt",
                meta_json_path=folder / "meta.json",
                tier4_json_path=folder / "tier4.json",
                heatmap_path=heatmap_path,
                best_zone_weighted_rmse=best_wrmse,
                n_epochs_run=n_epochs_run,
                per_zone_final=(live.tier4_final or {}).get("per_zone", {}),
            )

        except Exception as e:
            live.error = f"{type(e).__name__}: {e}"
            live.finished = True
            live.running = False
            raise
        finally:
            train_ds.close()
            val_ds.close()
