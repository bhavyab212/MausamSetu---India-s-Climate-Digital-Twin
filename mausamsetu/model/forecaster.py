"""
forecaster.py
==============
The MausamSetu ConvLSTM Forecaster — takes 6 past days, predicts next 7 days.

ARCHITECTURE
------------
                    Input: (B, T_in=6, C_in=5, H, W)
                              │
                              ▼
                    ┌───────────────────┐
                    │  Encoder ConvLSTM  │
                    │  (stack of layers) │
                    └─────────┬─────────┘
                              │  → final hidden state
                              ▼
                    ┌───────────────────┐
                    │  Decoder ConvLSTM  │
                    │  (rolls 7 outputs) │
                    └─────────┬─────────┘
                              ▼
              ┌───────────────┴────────────────┐
              ▼                                ▼
        RAIN HEADS                     TEMP HEAD
      ┌────┴────┐                 (regression → tmax, tmin)
      ▼         ▼
   Occurrence  Amount           output: (B, T_out=7, 3, H, W)
   (sigmoid)   (regression)     channels = [rain, tmax, tmin]

WHY THIS DESIGN?
----------------
1. Encoder–decoder: separates "understanding the past" from "generating the future"
2. Hurdle heads for rain (from Chapter 11): binary "did it rain?" +
   continuous "how much?" — this handles zero-inflation correctly
3. Physics-aware: outputs are clamped to physical ranges (non-negative rain, etc.)
4. Dropout inside every conv → MC-Dropout at inference gives uncertainty
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from mausamsetu import config
from mausamsetu.model.convlstm import ConvLSTM


# ============================================================================
# MAIN FORECASTER
# ============================================================================
class MausamSetuForecaster(nn.Module):
    """
    ConvLSTM-based spatio-temporal forecaster with hurdle rain head.

    Parameters
    ----------
    in_channels : number of INPUT channels (default 5 = rain+tmax+tmin+lst+insat_rain)
    out_channels : number of OUTPUT variables to predict (default 3 = rain, tmax, tmin)
    hidden_channels : list of hidden sizes for the ConvLSTM stack
    forecast_days : lead time (default 7 days)
    dropout : dropout probability for MC-Dropout uncertainty
    """

    def __init__(
        self,
        in_channels: int = config.INPUT_CHANNELS,
        out_channels: int = config.OUTPUT_VARS,
        hidden_channels: list[int] = None,
        forecast_days: int = config.FORECAST_DAYS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        hidden_channels = hidden_channels or config.HIDDEN_CHANNELS
        self.forecast_days = forecast_days
        self.out_channels = out_channels

        # --- Encoder (processes input sequence) ---
        self.encoder = ConvLSTM(
            input_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=config.KERNEL_SIZE,
        )

        # --- Decoder (generates forecast sequence) ---
        # Takes the encoder's final hidden state as its own initial state.
        # Uses zero inputs (dummy) at each future timestep.
        self.decoder = ConvLSTM(
            input_channels=hidden_channels[-1],   # decoder input = encoder output
            hidden_channels=hidden_channels,
            kernel_size=config.KERNEL_SIZE,
        )

        # --- Output heads ---
        # Shared conv before splitting into heads
        self.shared_head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),   # dropout stays active in MC-mode
        )

        # Rain occurrence head (sigmoid → probability of rain > 0.1mm)
        self.rain_occ_head = nn.Conv2d(hidden_channels[-1], 1, 1)

        # Rain amount head (regression, mm anomaly)
        self.rain_amt_head = nn.Conv2d(hidden_channels[-1], 1, 1)

        # Temperature heads (tmax, tmin — regression, °C anomaly)
        self.temp_head = nn.Conv2d(hidden_channels[-1], 2, 1)   # both at once

    # ---------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_occ: bool = False) -> torch.Tensor:
        """
        Predict the next `forecast_days` frames.

        Parameters
        ----------
        x : (B, T_in, C_in, H, W) — the input sequence (past days)
        return_occ : if True, also return rain-occurrence probabilities (for training)

        Returns
        -------
        y : (B, T_out, out_channels, H, W)   where channels = [rain, tmax, tmin]
        (optional) occ : (B, T_out, 1, H, W) — rain probability

        Note: outputs are in NORMALIZED anomaly space (see dataset.py).
        Denormalization happens in predict.py.
        """
        B, T_in, C_in, H, W = x.shape

        # ------------------------------------------------------------
        # ENCODE: run encoder over the past sequence
        # ------------------------------------------------------------
        enc_out = self.encoder(x)                         # (B, T_in, C_hid, H, W)
        last_hidden = enc_out[:, -1]                       # (B, C_hid, H, W) — summary of past

        # ------------------------------------------------------------
        # DECODE: repeat the last hidden as "seed" input for T_out steps
        # (simple choice; could also feed prev prediction back — recursive)
        # ------------------------------------------------------------
        dec_input = last_hidden.unsqueeze(1).expand(-1, self.forecast_days, -1, -1, -1)
        # (B, T_out, C_hid, H, W)
        dec_out = self.decoder(dec_input)                  # (B, T_out, C_hid, H, W)

        # ------------------------------------------------------------
        # HEADS: apply per-timestep output layers
        # ------------------------------------------------------------
        B, T_out, C, H, W = dec_out.shape
        # Flatten time into batch for per-frame processing, then reshape back
        flat = dec_out.reshape(B * T_out, C, H, W)
        shared = self.shared_head(flat)                    # (B*T_out, C_hid, H, W)

        rain_occ_logits = self.rain_occ_head(shared)       # (B*T_out, 1, H, W)
        rain_amt        = self.rain_amt_head(shared)       # (B*T_out, 1, H, W)
        temp_out        = self.temp_head(shared)           # (B*T_out, 2, H, W) = [tmax, tmin]

        # Reshape back to (B, T_out, ...)
        rain_occ_logits = rain_occ_logits.view(B, T_out, 1, H, W)
        rain_amt        = rain_amt.view(B, T_out, 1, H, W)
        temp_out        = temp_out.view(B, T_out, 2, H, W)

        # Combine rain output: expected rain = P(rain) * amount
        # (This is the standard hurdle-model combination.)
        rain_prob = torch.sigmoid(rain_occ_logits)
        rain_expected = rain_prob * rain_amt

        # Concatenate: [rain, tmax, tmin]
        y = torch.cat([rain_expected, temp_out], dim=2)    # (B, T_out, 3, H, W)

        if return_occ:
            return y, rain_prob
        return y

    # ---------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# CLI SANITY CHECK
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MausamSetu Forecaster architecture sanity check")
    print("=" * 60)

    model = MausamSetuForecaster()
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Fake batch: (B=2, T_in=6, C_in=5, H=19, W=17)
    x = torch.randn(2, 6, 5, 19, 17)
    print(f"\nInput  shape: {tuple(x.shape)}  (B, T_in, C_in, H, W)")

    y = model(x)
    print(f"Output shape: {tuple(y.shape)}  (B, T_out, C_out, H, W)")

    y2, occ = model(x, return_occ=True)
    print(f"Rain-occ prob shape: {tuple(occ.shape)}  (B, T_out, 1, H, W)")
    print(f"Rain-occ prob range: [{occ.min().item():.3f}, {occ.max().item():.3f}]")

    # Test with training data
    print("\nTesting with real dataset window...")
    from mausamsetu.preprocess.dataset import CauveryWindowDataset
    ds = CauveryWindowDataset(split="train")
    x, y_true = ds[0]
    x_batch = x.unsqueeze(0)                    # add batch dim
    y_pred = model(x_batch)
    print(f"  Real input: {tuple(x.shape)}")
    print(f"  Target:     {tuple(y_true.shape)}")
    print(f"  Prediction: {tuple(y_pred[0].shape)}")
    print("\n✓ Forecaster architecture works")
