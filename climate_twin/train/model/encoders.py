"""
train.model.encoders — gauge + satellite fusion with learned NULL embedding.

The rebuild plan's Phase 2e says satellite fusion must integrate now, not
be bolted on later. Concretely:

* **Gauge encoder** takes rain + tmax + tmin (always available across the
  full 72-year training window) and lifts them to a common latent.

* **Satellite encoder** takes a small, curated subset of INSAT channels
  (currently only LST; the mechanism generalises when more products are
  added) with **heavy dropout** (≥0.4) and a per-cell embedding of the
  same latent width.

* **Learned NULL embedding** — when satellite is absent (any date outside
  the ~1.5 year INSAT coverage), the satellite branch outputs the NULL
  embedding INSTEAD of a zero-fill. This is a learnable parameter, so the
  model can converge on "what to fall back to" rather than being told
  "sat=0" and having to learn that arbitrary substitution.

* **NULL dropout during training** — even when real satellite exists, we
  randomly substitute NULL with probability ``null_dropout``. This keeps
  the NULL path trained and prevents the model from becoming dependent on
  a channel it cannot count on.

The satellite guard (Phase 2e final rule) belongs in the loop, not here:
after any satellite-phase training the loop compares RMSE on pre-satellite
dates against the gauge-only checkpoint and refuses to declare success if
it regressed. This module just exposes the mechanism.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GaugeEncoder(nn.Module):
    """Lift (B, T, C_gauge, H, W) gauge channels to (B, T, hidden, H, W)."""

    def __init__(self, in_channels: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=pad)
        nn.init.xavier_uniform_(self.conv.weight, gain=0.5)
        nn.init.zeros_(self.conv.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = seq.shape
        x = seq.reshape(B * T, C, H, W)
        x = self.conv(x)
        return x.reshape(B, T, -1, H, W)


class SatelliteEncoder(nn.Module):
    """Lift satellite channels + a learned NULL embedding to (B, T, hidden, H, W).

    ``forward`` accepts either the tensor (with a per-time-step ``valid``
    mask of shape (B, T)) OR ``None`` when the whole window has no sat
    data — in which case the NULL embedding is broadcast for every step.
    ``null_dropout`` randomly substitutes NULL for available steps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        kernel_size: int = 3,
        dropout: float = 0.4,
        null_dropout: float = 0.5,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=pad)
        nn.init.xavier_uniform_(self.conv.weight, gain=0.5)
        nn.init.zeros_(self.conv.bias)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.null_embed = nn.Parameter(torch.zeros(1, hidden, 1, 1))
        nn.init.normal_(self.null_embed, std=0.02)
        self.null_dropout = null_dropout
        self.hidden = hidden

    def _null(self, B: int, T: int, H: int, W: int, device, dtype) -> torch.Tensor:
        # broadcast the learned scalar embedding to (B, T, hidden, H, W)
        emb = self.null_embed.to(device=device, dtype=dtype)
        return emb.expand(B, T, self.hidden, H, W).clone()

    def forward(
        self,
        seq: torch.Tensor | None,        # (B, T, C_sat, H, W) or None
        valid: torch.Tensor | None,      # (B, T) bool — per-step availability
        gauge_reference: torch.Tensor,   # to steal (B, T, H, W) shape when seq is None
    ) -> torch.Tensor:
        Bg, Tg, _, Hg, Wg = gauge_reference.shape
        if seq is None:
            return self._null(Bg, Tg, Hg, Wg, gauge_reference.device, gauge_reference.dtype)

        B, T, C, H, W = seq.shape
        assert (B, T, H, W) == (Bg, Tg, Hg, Wg), "sat/gauge shape mismatch"

        # NaN-safe: replace NaN in sat channels with 0 before conv (they'll
        # be replaced by NULL via `valid` mask anyway).
        seq_clean = torch.nan_to_num(seq, nan=0.0)
        x = seq_clean.reshape(B * T, C, H, W)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.reshape(B, T, -1, H, W)

        # Apply valid mask per (B, T) — NULL replaces invalid steps
        if valid is None:
            return x
        # NULL dropout during training: even valid steps get replaced with
        # probability null_dropout to keep the NULL path trained.
        effective_valid = valid.clone()
        if self.training and self.null_dropout > 0:
            drop = torch.rand_like(effective_valid, dtype=torch.float32) < self.null_dropout
            effective_valid = effective_valid & (~drop)

        null_full = self._null(B, T, H, W, seq.device, seq.dtype)
        mask = effective_valid.view(B, T, 1, 1, 1).to(dtype=x.dtype)
        return mask * x + (1.0 - mask) * null_full
