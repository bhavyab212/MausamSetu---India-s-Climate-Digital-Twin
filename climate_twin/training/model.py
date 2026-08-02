"""PyTorch ConvLSTM model mirroring the cloned TF architecture.

Architecture (from src/model.py):
  - Input: (B, seq_length, lat, lon, channels) → we use (B, T, C, H, W) PyTorch convention
  - ConvLSTM(32) → BN → ConvLSTM(32) → BN → Conv2D(channels, tanh) → residual + persistence
  - Output clipped to [0,1] normalized range

Key design: learns a residual from persistence, scaled by 0.1 initially.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=True,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        x: (B, C_in, H, W)
        state: (h, c) each (B, C_hid, H, W) or None
        """
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class ClimateTwinModel(nn.Module):
    """PyTorch port of the cloned DigitalTwinModel.

    Residual ConvLSTM: learns delta from persistence, clipped to [0,1].
    Input: (B, T, C, H, W) in normalized [0,1] space
    Output: (B, C, H, W) — prediction for the next time step
    """

    def __init__(
        self,
        seq_length: int = 30,
        lat_dim: int = 129,
        lon_dim: int = 135,
        channels: int = 2,
        hidden: int = 32,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.channels = channels
        self.residual_scale = residual_scale

        # Two-layer ConvLSTM encoder
        self.cell1 = ConvLSTMCell(channels, hidden)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.cell2 = ConvLSTMCell(hidden, hidden)
        self.bn2 = nn.BatchNorm2d(hidden)

        # Spatial dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout)

        # Residual head: learns the delta from persistence
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) — normalized input sequence
        Returns: (B, C, H, W) — prediction for next step
        """
        B, T, C, H, W = x.shape

        # Persistence baseline: last frame of input
        persistence = x[:, -1]  # (B, C, H, W)

        # Run through ConvLSTM layers
        state1 = None
        state2 = None
        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            h1, state1 = self.cell1(frame, state1)
            h1 = self.bn1(h1)
            h2, state2 = self.cell2(h1, state2)
            h2 = self.bn2(h2)

        # Apply dropout and compute residual
        h2 = self.dropout(h2)
        residual = self.residual_head(h2)  # (B, C, H, W)

        # Scale residual (favors persistence at initialization)
        residual = residual * self.residual_scale

        # Add to persistence and clip to valid range
        output = torch.clamp(persistence + residual, 0.0, 1.0)
        return output

    def predict_ensemble(self, x: torch.Tensor, n_samples: int = 20) -> dict[str, torch.Tensor]:
        """MC-Dropout ensemble prediction for uncertainty.

        Batches the MC samples through the batch dimension (chunked to bound
        VRAM) so we do ~ceil(n_samples/chunk) forward passes instead of
        n_samples separate ones. Each replicated row gets independent dropout,
        so the samples are still valid MC draws.

        Returns dict with p10, p50, p90, mean, std — all (B, C, H, W).
        """
        self.train()  # Enable dropout
        B = x.shape[0]
        rep_dims = [1] * (x.dim() - 1)
        chunk = max(1, min(n_samples, 8))  # samples per forward pass
        outs = []
        with torch.inference_mode():
            remaining = n_samples
            while remaining > 0:
                k = min(chunk, remaining)
                x_rep = x.repeat(k, *rep_dims)           # (k*B, ...)
                out = self.forward(x_rep)                # (k*B, C, H, W)
                outs.append(out.view(k, B, *out.shape[1:]))
                remaining -= k
        self.eval()

        stacked = torch.cat(outs, dim=0)  # (N, B, C, H, W)
        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "p10": torch.quantile(stacked, 0.10, dim=0),
            "p50": torch.quantile(stacked, 0.50, dim=0),
            "p90": torch.quantile(stacked, 0.90, dim=0),
        }
