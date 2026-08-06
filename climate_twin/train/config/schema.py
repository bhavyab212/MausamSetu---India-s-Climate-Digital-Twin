"""
train.config.schema — Pydantic v2 validated experiment config.

Every training run is fully specified by one YAML file that maps to
:class:`ExperimentConfig`. Reject-unknown, type-safe, no scattered defaults
elsewhere in the codebase.

Layered defaults:
    1. Field defaults on the Pydantic models (this file)
    2. ``config/base.yaml`` overrides those defaults
    3. ``config/experiments/<name>.yaml`` may override any field with an
       explicit ``extends: <base_yaml_name>`` line at the top.

Load contract (``load_config``):
    * Reads the YAML.
    * If it declares ``extends:``, first loads that base and deep-merges.
    * Validates the merged dict against :class:`ExperimentConfig`.
    * Cross-checks that the referenced zone signature matches
      ``climate_twin.regions.zone_mask.sha256`` — refuses to run against a
      mask that has drifted since the config was written.
    * Returns a frozen ``ExperimentConfig``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


_STRICT = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class DataConfig(BaseModel):
    model_config = _STRICT

    region: Literal["india", "cauvery"] = "india"
    train_years: tuple[int, int] = (1951, 2022)
    val_years: tuple[int, int] = (2023, 2023)
    test_years: tuple[int, int] = (2024, 2025)
    variables: tuple[str, ...] = ("rain", "tmax", "tmin")
    include_satellite: bool = False
    manifest_sig_expected: str | None = None      # optional pin

    @model_validator(mode="after")
    def _year_order(self):
        for name, rng in [("train_years", self.train_years),
                          ("val_years", self.val_years),
                          ("test_years", self.test_years)]:
            if rng[0] > rng[1]:
                raise ValueError(f"{name}: start ({rng[0]}) > end ({rng[1]})")
        if self.train_years[1] >= self.val_years[0]:
            raise ValueError(
                f"train_years {self.train_years} overlaps val_years {self.val_years} "
                "— temporal leakage forbidden"
            )
        if self.val_years[1] >= self.test_years[0]:
            raise ValueError(
                f"val_years {self.val_years} overlaps test_years {self.test_years} "
                "— temporal leakage forbidden"
            )
        return self


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------
class ZonesConfig(BaseModel):
    model_config = _STRICT

    membership: Literal["soft", "hard"] = "soft"
    zone_mask_sig_expected: str        # 12-hex, checked at load
    excluded_zone_keys: tuple[str, ...] = ()   # for Phase 5 held-out-zone tests

    @model_validator(mode="after")
    def _sig_shape(self):
        if len(self.zone_mask_sig_expected) != 12:
            raise ValueError(
                f"zone_mask_sig_expected must be 12 hex chars, got {self.zone_mask_sig_expected!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModelConfig(BaseModel):
    model_config = _STRICT

    backbone: Literal["convlstm"] = "convlstm"
    hidden: int = Field(48, ge=8, le=256)
    seq_length: int = Field(14, ge=2, le=60)
    kernel_size: int = 3
    dropout: float = Field(0.2, ge=0.0, le=0.9)
    residual_scale: float = Field(0.1, ge=0.0, le=1.0)
    film_conditioning: bool = True             # FiLM zone conditioning
    zone_conditioned_heads: bool = True
    satellite_null_dropout: float = Field(0.5, ge=0.0, le=1.0)
    satellite_channels: tuple[str, ...] = ("insat_lst",)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class LossConfig(BaseModel):
    model_config = _STRICT

    objective: Literal["hurdle", "huber", "mse"] = "hurdle"
    rain_occurrence_threshold_mm: float = Field(0.1, ge=0.0)
    zone_weighting: Literal["inverse_variance", "uniform", "area_weighted"] = "inverse_variance"
    physics_tmax_ge_tmin: float = Field(0.02, ge=0.0)
    physics_spatial_smooth: float = Field(0.02, ge=0.0)
    physics_temporal_smooth: float = Field(0.02, ge=0.0)
    physics_bounds_penalty: float = Field(0.01, ge=0.0)   # per-zone bounds


# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------
class OptimConfig(BaseModel):
    model_config = _STRICT

    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    scheduler: Literal["cosine", "plateau", "step", "none"] = "cosine"
    warmup_epochs: int = Field(2, ge=0)
    min_lr: float = Field(1e-6, ge=0.0)
    grad_clip: float = Field(1.0, ge=0.0)
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(4, ge=1, le=256)
    gradient_accum: int = Field(1, ge=1)
    mixed_precision: Literal["fp32", "bf16", "fp16"] = "bf16"

    # Stratified sampler mode — controls per-zone batch composition
    sampler_mode: Literal["proportional", "balanced", "inverse_frequency"] = "inverse_frequency"

    # CPU-side DataLoader knobs (I/O runs on CPU; math on GPU).
    # Post-Windows-safe rewrite: DailyWindowDataset opens its own xarray
    # handle in worker_init_fn, so num_workers > 0 is safe on spawn.
    num_workers: int = Field(4, ge=0, le=16)
    pin_memory: bool = True
    prefetch_factor: int = Field(4, ge=1, le=16)
    persistent_workers: bool = True

    # Auto-tuning flags (Part A of the reimagined training dashboard).
    #   batch_size_auto=True  → trainer probes 8→16→32→…, picks 85% of
    #                            largest OK size, capped at 256.
    #   mixed_precision_auto → bf16 on Ampere+ (SM 8.0+), fp16 older,
    #                          fp32 only if user forces via mixed_precision.
    batch_size_auto: bool = False
    mixed_precision_auto: bool = False

    # Purely descriptive — surfaced by the UI intent selector; never
    # affects execution.
    intent: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationConfig(BaseModel):
    model_config = _STRICT

    tier1_every_n_batches: int = Field(50, ge=1)
    tier2_every_n_epochs: int = Field(1, ge=1)
    tier3_every_n_epochs: int = Field(5, ge=1)
    tier4_end_of_round: bool = True
    thresholds_source: Literal["imd_absolute", "zone_percentile", "both"] = "both"
    mc_samples: int = Field(20, ge=1, le=100)
    bootstrap_samples: int = Field(200, ge=10, le=5000)
    min_cells_for_metric: int = Field(5, ge=1)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStoppingConfig(BaseModel):
    model_config = _STRICT

    policy: Literal["global", "weighted", "worst_zone"] = "worst_zone"
    monitor: Literal["val_rmse", "val_loss", "val_csi", "zone_weighted_rmse"] = "zone_weighted_rmse"
    patience: int = Field(8, ge=1)
    min_delta: float = Field(1e-4, ge=0.0)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""
    model_config = _STRICT

    name: str
    notes: str = ""
    seed: int = 42
    deterministic: bool = True

    data: DataConfig
    zones: ZonesConfig
    model: ModelConfig
    loss: LossConfig
    optim: OptimConfig
    validation: ValidationConfig
    early_stopping: EarlyStoppingConfig

    def summary(self) -> str:
        lines = [
            f"experiment      = {self.name}",
            f"seed            = {self.seed}  deterministic={self.deterministic}",
            f"region          = {self.data.region}",
            f"years           = train {self.data.train_years}  "
            f"val {self.data.val_years}  test {self.data.test_years}",
            f"variables       = {list(self.data.variables)}  "
            f"satellite={self.data.include_satellite}",
            f"zone_mask_sig   = {self.zones.zone_mask_sig_expected}  "
            f"membership={self.zones.membership}",
            f"model           = {self.model.backbone}  hidden={self.model.hidden}  "
            f"seq={self.model.seq_length}  film={self.model.film_conditioning}",
            f"loss            = {self.loss.objective}  "
            f"zone_weighting={self.loss.zone_weighting}  "
            f"phys(tmax≥tmin={self.loss.physics_tmax_ge_tmin}, "
            f"smooth={self.loss.physics_spatial_smooth})",
            f"optim           = {self.optim.optimizer}  lr={self.optim.lr}  "
            f"epochs={self.optim.epochs}  bs={self.optim.batch_size}  "
            f"amp={self.optim.mixed_precision}  sampler={self.optim.sampler_mode}",
            f"validation      = tier1/{self.validation.tier1_every_n_batches}b "
            f"tier2/{self.validation.tier2_every_n_epochs}e "
            f"tier3/{self.validation.tier3_every_n_epochs}e "
            f"thresholds={self.validation.thresholds_source}",
            f"early_stopping  = {self.early_stopping.policy}  "
            f"monitor={self.early_stopping.monitor}  "
            f"patience={self.early_stopping.patience}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader with `extends:` + zone-signature verification
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: override wins for scalars, dicts merge, lists replace."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_extends(path: Path, seen: set[Path] | None = None) -> dict:
    """Recursively resolve `extends:` chains and return the merged dict.

    A yaml may `extends: <name>.yaml` — the loader walks the chain in
    parent-first order, deep-merging each override on top of its base.
    Cycles are detected via a `seen` set and raise `ValueError`.
    """
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"config extends cycle: {path}")
    seen = seen | {path}

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "extends" in doc:
        base_name = doc.pop("extends")
        candidates = [
            path.parent / base_name,
            path.parent.parent / base_name,
            Path(__file__).resolve().parent / base_name,
        ]
        base_path = next((c for c in candidates if c.exists()), None)
        if base_path is None:
            raise FileNotFoundError(
                f"config extends '{base_name}' not found "
                f"(checked {[str(c) for c in candidates]})"
            )
        base_doc = _resolve_extends(base_path, seen)
        doc = _deep_merge(base_doc, doc)
    return doc


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment yaml. Verifies the zone signature at load time.

    ``extends:`` chains resolve recursively — a yaml may extend another
    yaml that itself extends base.yaml. Base defaults are loaded first
    and deep-merged with each override.
    """
    path = Path(path).resolve()
    doc = _resolve_extends(path)
    cfg = ExperimentConfig(**doc)

    # Zone signature check — refuse to run against a drifted mask
    from climate_twin.regions import get_zones
    Z = get_zones()
    if cfg.zones.zone_mask_sig_expected != Z.mask_signature:
        raise ValueError(
            f"config zone_mask_sig_expected = {cfg.zones.zone_mask_sig_expected!r} "
            f"but on-disk zone registry signature is {Z.mask_signature!r}. "
            f"Either rebuild the mask or update the config; refusing to run against "
            f"a drifted zoning."
        )
    # Optional manifest.sig pin
    if cfg.data.manifest_sig_expected is not None:
        from climate_twin import data_source as DS
        got = DS.manifest_sig()
        if got != cfg.data.manifest_sig_expected:
            raise ValueError(
                f"config data.manifest_sig_expected = {cfg.data.manifest_sig_expected!r} "
                f"but data/processed/manifest.sig is {got!r}."
            )
    return cfg
