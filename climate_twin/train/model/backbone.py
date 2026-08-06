"""
train.model.backbone — ConvLSTM backbone with per-step FiLM zone conditioning.

Two ConvLSTM cells stacked, each followed by a BatchNorm2d and a FiLM2d
modulation. The FiLM conditioning uses the same static per-cell zone
membership at every time step — the zoning of a cell doesn't change over
time, but every conv feature it produces IS modulated by which zone(s)
it belongs to. This is what makes the shared-backbone-with-conditioning
architecture actually behave as a mixture-of-regimes at feature level.

Every activation stays finite under bf16 by using tanh gates and small
init.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM2d


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015)."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )
        # Small orthogonal-ish init on the recurrent path keeps bf16 stable
        nn.init.xavier_uniform_(self.conv.weight, gain=0.5)
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        x: torch.Tensor,                 # (B, C, H, W)
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class ConvLSTMBackbone(nn.Module):
    """Two ConvLSTM cells + BatchNorm + optional FiLM at every layer."""

    def __init__(
        self,
        input_channels: int,
        hidden: int,
        zone_channels: int,
        kernel_size: int = 3,
        film: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cell1 = ConvLSTMCell(input_channels, hidden, kernel_size)
        self.cell2 = ConvLSTMCell(hidden, hidden, kernel_size)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.film = film
        if film:
            self.film1 = FiLM2d(hidden, zone_channels)
            self.film2 = FiLM2d(hidden, zone_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.hidden = hidden

    def forward(
        self,
        seq: torch.Tensor,               # (B, T, C, H, W)
        zone_map: torch.Tensor,          # (B, K, H, W) — static soft membership
    ) -> torch.Tensor:
        B, T, C, H, W = seq.shape
        s1 = None
        s2 = None
        last_h2: torch.Tensor | None = None
        for t in range(T):
            x = seq[:, t]
            h1, s1 = self.cell1(x, s1)
            h1 = self.bn1(h1)
            if self.film:
                h1 = self.film1(h1, zone_map)
            h1 = self.dropout(h1)
            h2, s2 = self.cell2(h1, s2)
            h2 = self.bn2(h2)
            if self.film:
                h2 = self.film2(h2, zone_map)
            h2 = self.dropout(h2)
            last_h2 = h2
        assert last_h2 is not None
        return last_h2                    # (B, hidden, H, W)
