"""
train.model.film — FiLM (Feature-wise Linear Modulation) 2-D layer.

A FiLM layer takes a feature map ``x`` of shape ``(B, C, H, W)`` and a
per-cell conditioning vector ``z`` of shape ``(B, K, H, W)`` and returns

    y = γ(z) * x  +  β(z)

where γ, β are learned linear maps from K → C. Because the conditioning is
per-cell, boundary cells with mixed soft-membership vectors get a smoothly
interpolated modulation — this is what makes the atmosphere-has-no-hard-
boundary property work in a shared-backbone model.

FiLM has ~an order of magnitude more parameter efficiency than appending
the zone vector as an extra channel and letting a full conv learn the
mapping, and it composes cleanly with any 2-D backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM2d(nn.Module):
    """Feature-wise linear modulation conditioned on a per-cell zone vector."""

    def __init__(self, feature_channels: int, condition_channels: int):
        super().__init__()
        self.feature_channels = feature_channels
        self.condition_channels = condition_channels
        # γ and β are two independent 1×1 convs mapping K → C
        self.gamma = nn.Conv2d(condition_channels, feature_channels, kernel_size=1, bias=True)
        self.beta = nn.Conv2d(condition_channels, feature_channels, kernel_size=1, bias=True)
        # Initialize so FiLM starts as an identity: γ=1, β=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), z: (B, K, H, W)  →  (B, C, H, W)."""
        gamma = self.gamma(z)
        beta = self.beta(z)
        return gamma * x + beta
