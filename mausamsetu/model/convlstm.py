"""
convlstm.py
============
ConvLSTM cell + multi-layer ConvLSTM sequence model.

WHAT IS ConvLSTM?
-----------------
An LSTM (Long Short-Term Memory) unit that uses 2-D CONVOLUTIONS instead of
matrix multiplications, so it preserves spatial structure.

  - CNN sees a map (spatial patterns).
  - LSTM has memory (temporal patterns).
  - ConvLSTM = CNN + LSTM = "eyes that remember" — sees maps AND their history.

THE FOUR GATES (each is a 3x3 conv):
  f = sigmoid(Wf * [x, h] + bf)   forget gate    — what to erase from memory
  i = sigmoid(Wi * [x, h] + bi)   input gate     — what new info to write
  g =    tanh(Wg * [x, h] + bg)   candidate      — the new info content
  o = sigmoid(Wo * [x, h] + bo)   output gate    — what to reveal from memory

STATE UPDATE:
  c_new = f * c_old + i * g       cell state (long-term memory)
  h_new = o * tanh(c_new)         hidden state (what we output this step)

Reference: Shi et al. 2015, "Convolutional LSTM Network."
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ============================================================================
# ConvLSTM CELL — one time-step
# ============================================================================
class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell that processes ONE frame at a time."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        # One big conv that outputs 4 * hidden_channels
        # (concat of f, i, g, o gates for efficiency)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,      # (B, C_in, H, W)   — current frame
        h: torch.Tensor,      # (B, C_hid, H, W)  — previous hidden state
        c: torch.Tensor,      # (B, C_hid, H, W)  — previous cell state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step forward. Returns (new_hidden, new_cell)."""
        # Concatenate input and hidden along the channel axis
        combined = torch.cat([x, h], dim=1)              # (B, C_in+C_hid, H, W)
        gates = self.conv(combined)                       # (B, 4*C_hid, H, W)

        # Split into the four gate tensors
        f, i, g, o = torch.split(gates, self.hidden_channels, dim=1)

        # Apply activations
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # State update
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch_size: int, height: int, width: int, device):
        """Zero-init the hidden and cell states."""
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


# ============================================================================
# STACKED ConvLSTM — multi-layer over a whole sequence
# ============================================================================
class ConvLSTM(nn.Module):
    """
    Stack of ConvLSTM cells that processes a SEQUENCE of frames.

    Input:  (B, T, C_in, H, W)
    Output: (B, T, C_last_hidden, H, W)  — one hidden output per timestep
            plus the final (h, c) of each layer.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: list[int],   # e.g. [64, 64, 64]
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels

        cells = []
        ch_in = input_channels
        for hc in hidden_channels:
            cells.append(ConvLSTMCell(ch_in, hc, kernel_size))
            ch_in = hc
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a full sequence.

        Parameters
        ----------
        x : (B, T, C_in, H, W)

        Returns
        -------
        (B, T, C_last, H, W) — outputs at every timestep from the top layer
        """
        B, T, _, H, W = x.shape
        device = x.device

        # Initialise (h, c) for each layer
        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        outputs = []   # will hold top-layer hidden at each timestep
        for t in range(T):
            input_t = x[:, t]     # (B, C_in, H, W)
            for l, cell in enumerate(self.cells):
                h, c = states[l]
                h, c = cell(input_t, h, c)
                states[l] = (h, c)
                input_t = h       # feed to next layer
            outputs.append(h)     # store top-layer hidden
        return torch.stack(outputs, dim=1)   # (B, T, C_last, H, W)
