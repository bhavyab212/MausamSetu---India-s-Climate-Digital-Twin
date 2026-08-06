"""
train.data.sampler — zone-stratified per-cell weighting.

Each training sample IS the full India map (129×135), so every batch already
contains cells from every zone. The problem the plan calls "gradient
domination" happens because the wet zones have ~4× more cells than the arid
ones and ~30× larger σ — their contribution to the loss dwarfs everything
else and the model learns to fit them at the expense of everyone else.

The fix is not sample selection (there is nothing to select — every batch
IS the map) but per-cell WEIGHTING inside the loss. This module produces a
frozen ``(H, W)`` weight tensor with one of three policies:

    proportional      — every cell contributes equally within its zone
                        (weight ∝ 1). Zone with more cells contributes more.
    balanced          — every ZONE contributes equally to the aggregate loss
                        regardless of cell count.
    inverse_frequency — weight ∝ 1 / cell_count_of_that_zone.  Under-
                        represented zones get more weight, so their gradient
                        signal is not drowned out.  Default per plan.

The tensor is a linear combination of the SOFT membership so boundary cells
contribute smoothly to both adjacent zones' effective weights. The plan
also asked for logging the realised composition of the first N batches —
that lives in ``log_batch_composition`` and is called once at training
start.

There is no torch.utils.data.Sampler here. Sampler-of-indices doesn't
apply: every sample is the whole India tensor. The stratification is a
per-cell weight, and the DataLoader is a plain sequential loader over the
time axis (implemented in ``train.data.dataset``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from climate_twin.regions import ZoneRegistry


SamplerMode = Literal["proportional", "balanced", "inverse_frequency"]


def build_zone_weight_map(
    zones: ZoneRegistry,
    mode: SamplerMode = "inverse_frequency",
    clip_ratio: tuple[float, float] | None = None,
    excluded_zone_keys: tuple[str, ...] = (),
) -> np.ndarray:
    """Return an ``(H, W)`` float32 per-cell loss weight tensor.

    Every unassigned cell (hard mask == 0) gets weight 0. Boundary cells
    receive a weighted average of their zones' per-zone weights, blended
    by soft membership.

    ``excluded_zone_keys`` — Phase 5a held-out-zone contract. Zones listed
    here have their base weight forced to 0, so cells in the interior of
    those zones contribute nothing to the loss. Boundary cells that share
    membership with a NON-excluded zone still contribute partially (via
    the retained neighbour weight). The model's FiLM conditioning still
    sees the full soft-membership vector — this is the point of the
    held-out semantics: the model has structural knowledge of the zoning
    but no gradient signal from the excluded regime.

    The mean per-cell weight over VALID (non-excluded interior) cells is
    normalised to 1.0 so the absolute magnitude of the loss stays
    comparable across policies.
    """
    K = zones.n_zones()
    memb = zones.membership          # (H, W, K) float32
    hard = zones.hard_mask           # (H, W) int8

    # Per-zone base weight
    counts = np.array(
        [int((hard == z.id).sum()) for z in zones.zones], dtype=np.float64
    )
    counts = np.maximum(counts, 1.0)

    if mode == "proportional":
        per_zone = np.ones(K, dtype=np.float64)
    elif mode == "balanced":
        # Weight so each zone's total contribution = 1
        per_zone = 1.0 / counts
    elif mode == "inverse_frequency":
        # weight ∝ 1 / freq;  keeps ratios but re-anchored so mean=1 after
        per_zone = counts.mean() / counts
    else:
        raise ValueError(f"unknown sampler mode {mode!r}")

    # Optional clip
    if clip_ratio is not None:
        lo, hi = clip_ratio
        per_zone = np.clip(per_zone, lo, hi)

    # Held-out zone: zero its base weight BEFORE soft blending so a
    # boundary cell with (say) 60% held-out membership + 40% adjacent zone
    # membership contributes ≈ 40% of a normal cell's loss, not zero.
    excluded_set = set(excluded_zone_keys or ())
    for k, zone in enumerate(zones.zones):
        if zone.key in excluded_set:
            per_zone[k] = 0.0

    # Blend via soft membership → (H, W) weights
    w = (memb * per_zone[None, None, :]).sum(axis=-1)      # (H, W)
    w = np.where(hard > 0, w, 0.0)

    # Normalise so mean over CONTRIBUTING cells == 1.0
    contrib = (hard > 0) & (w > 0)
    m = float(w[contrib].mean()) if contrib.any() else 1.0
    if m > 0:
        w = w / m
    return w.astype(np.float32)


class ZoneStratifiedWeights:
    """Cached wrapper that exposes the weight tensor as a torch.Tensor on the
    training device. Cheap enough to construct once at the top of training
    and reuse for every batch.

    ``excluded_zone_keys`` — Phase 5a held-out-zone semantics. See
    :func:`build_zone_weight_map`.
    """

    def __init__(
        self,
        zones: ZoneRegistry,
        mode: SamplerMode = "inverse_frequency",
        device: str | torch.device = "cpu",
        clip_ratio: tuple[float, float] | None = None,
        excluded_zone_keys: tuple[str, ...] = (),
    ):
        self.mode = mode
        self.zones = zones
        self.excluded_zone_keys = tuple(excluded_zone_keys or ())
        self._np = build_zone_weight_map(
            zones, mode=mode, clip_ratio=clip_ratio,
            excluded_zone_keys=self.excluded_zone_keys,
        )
        self._t = torch.from_numpy(self._np).to(device)

    @property
    def tensor(self) -> torch.Tensor:
        """``(H, W)`` float32 weight map, ready to broadcast with a loss."""
        return self._t

    @property
    def numpy(self) -> np.ndarray:
        return self._np

    def per_zone_effective_weight(self) -> dict[str, float]:
        """Return {zone_key: sum of weights over that zone's hard cells}.

        Useful for the sampler-composition log: the sum tells you how much
        loss magnitude that zone contributes on a given batch (all zones
        appear in every batch, so this is a static property of the mask).
        """
        out: dict[str, float] = {}
        for z in self.zones.zones:
            m = self.zones.hard_mask == z.id
            out[z.key] = float(self._np[m].sum())
        return out


def log_batch_composition(
    zw: ZoneStratifiedWeights,
    n_batches: int = 5,
    logger=None,
) -> str:
    """Return a human-readable multi-line summary of the sampler's
    realised composition.

    Because every batch contains the entire India map, "composition" is a
    STATIC property: per-zone effective weight × cells. We report it once,
    labelled as covering the first ``n_batches`` (identical each time).
    """
    per_zone_w = zw.per_zone_effective_weight()
    total = sum(per_zone_w.values())
    lines = [
        f"── stratified sampler composition (mode={zw.mode}) ──",
        f"first {n_batches} batches (identical: full-map samples):",
        f"  {'zone':22s}  cells  eff.weight  %-of-loss",
    ]
    for z in zw.zones.zones:
        cells = int((zw.zones.hard_mask == z.id).sum())
        w = per_zone_w[z.key]
        pct = 100.0 * w / total if total > 0 else 0.0
        lines.append(f"  {z.key:22s}  {cells:5d}   {w:9.2f}   {pct:5.1f}%")
    lines.append(f"  {'TOTAL':22s}  {int((zw.zones.hard_mask > 0).sum()):5d}   "
                 f"{total:9.2f}   100.0%")
    text = "\n".join(lines)
    if logger is not None:
        logger(text)
    return text
