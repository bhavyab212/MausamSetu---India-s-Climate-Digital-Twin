"""
train.model.heads — output heads.

* HurdleHead: two-part output for rainfall.
    logit_occurrence → BCE loss on ``target > 0.1 mm``.
    amount           → regression loss on the wet-cell subset.
  At inference the expectation is ``sigmoid(logit) * softplus(amount)``.
  This preserves the "no rain today" zero peak that heavy-tailed rainfall
  distributions have.

* Regression head for tmax/tmin.

* ZoneConditionedHead: wraps any head with an extra FiLM modulation so the
  latent → output mapping can vary by regime (e.g. tmax range shifts
  systematically between Himalayan and Thar).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .film import FiLM2d


class HurdleHead(nn.Module):
    """(B, C, H, W) hidden → occurrence-logit + amount for rain."""

    def __init__(self, hidden: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.occurrence = nn.Conv2d(hidden, 1, kernel_size=kernel_size, padding=pad)
        self.amount = nn.Conv2d(hidden, 1, kernel_size=kernel_size, padding=pad)
        nn.init.xavier_uniform_(self.occurrence.weight, gain=0.5)
        nn.init.zeros_(self.occurrence.bias)
        nn.init.xavier_uniform_(self.amount.weight, gain=0.5)
        nn.init.zeros_(self.amount.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.occurrence(h), self.amount(h)


class RegressionHead(nn.Module):
    """(B, hidden, H, W) → (B, 1, H, W)."""

    def __init__(self, hidden: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(hidden, 1, kernel_size=kernel_size, padding=pad)
        nn.init.xavier_uniform_(self.conv.weight, gain=0.5)
        nn.init.zeros_(self.conv.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.conv(h)


class ZoneConditionedHead(nn.Module):
    """Wrap a head module with a FiLM modulation on the latent before the
    conv. Cheap parameter overhead, but lets the latent→output map differ
    per regime."""

    def __init__(self, hidden: int, zone_channels: int, inner: nn.Module):
        super().__init__()
        self.film = FiLM2d(hidden, zone_channels)
        self.inner = inner

    def forward(self, h: torch.Tensor, z: torch.Tensor):
        return self.inner(self.film(h, z))
