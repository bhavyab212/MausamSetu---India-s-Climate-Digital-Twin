"""
train.model.assemble — build the full zone-aware model from a config.

Composition:
    input  (B, T, C_g, H, W) gauge  +  (B, T, C_s, H, W) sat  +  (B, K, H, W) zone_map
      ↓
    GaugeEncoder      → (B, T, hidden, H, W)
    SatelliteEncoder  → (B, T, hidden, H, W)   (with NULL embedding)
      ↓ concat channel-wise → (B, T, 2*hidden, H, W)
    ConvLSTMBackbone (FiLM at every step)     → (B, hidden, H, W)
      ↓
    per-variable zone-conditioned heads:
        rain  → HurdleHead (occurrence-logit + amount)
        tmax  → Regression
        tmin  → Regression
        …
      ↓
    return dict of predictions per variable.

Residual: for tmax/tmin the head predicts an ANOMALY that is added to the
per-cell climatology (a static (H, W) map supplied per variable). For rain
the head is absolute — the hurdle formulation already handles the zero
peak.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from climate_twin.regions import get_zones, ZoneRegistry

from .backbone import ConvLSTMBackbone
from .encoders import GaugeEncoder, SatelliteEncoder
from .heads import HurdleHead, RegressionHead, ZoneConditionedHead


class ZoneAwareModel(nn.Module):
    """The zone-aware model: shared backbone + FiLM + per-variable heads."""

    def __init__(
        self,
        gauge_channels: int,
        sat_channels: int,
        hidden: int,
        zones: ZoneRegistry,
        kernel_size: int = 3,
        dropout: float = 0.2,
        film: bool = True,
        zone_conditioned_heads: bool = True,
        null_dropout: float = 0.5,
        variables: tuple[str, ...] = ("rain", "tmax", "tmin"),
    ):
        super().__init__()
        self.variables = tuple(variables)
        self.hidden = hidden
        self.gauge_channels = gauge_channels
        self.sat_channels = sat_channels
        K = zones.n_zones()
        self.zone_channels = K

        self.gauge_enc = GaugeEncoder(gauge_channels, hidden, kernel_size=kernel_size)
        self.sat_enc = (
            SatelliteEncoder(sat_channels, hidden, kernel_size=kernel_size,
                             null_dropout=null_dropout)
            if sat_channels > 0 else None
        )
        # Backbone consumes concatenation of gauge + sat features (2*hidden or 1*hidden)
        backbone_in = hidden * (2 if self.sat_enc is not None else 1)
        self.backbone = ConvLSTMBackbone(
            input_channels=backbone_in,
            hidden=hidden,
            zone_channels=K,
            kernel_size=kernel_size,
            film=film,
            dropout=dropout,
        )

        # Heads
        heads: dict[str, nn.Module] = {}
        for v in self.variables:
            if v == "rain":
                base = HurdleHead(hidden)
            else:
                base = RegressionHead(hidden)
            if zone_conditioned_heads:
                heads[v] = ZoneConditionedHead(hidden, K, base)
            else:
                heads[v] = base
        self.heads = nn.ModuleDict(heads)

    def forward(
        self,
        gauge: torch.Tensor,                       # (B, T, C_g, H, W)
        zone_map: torch.Tensor,                    # (B, K, H, W)
        sat: torch.Tensor | None = None,           # (B, T, C_s, H, W) or None
        sat_valid: torch.Tensor | None = None,     # (B, T) bool per time step
    ) -> dict[str, Any]:
        g = self.gauge_enc(gauge)                  # (B, T, hidden, H, W)
        if self.sat_enc is not None:
            s = self.sat_enc(sat, sat_valid, gauge_reference=g)
            feats = torch.cat([g, s], dim=2)       # (B, T, 2*hidden, H, W)
        else:
            feats = g
        h = self.backbone(feats, zone_map)         # (B, hidden, H, W)

        out: dict[str, Any] = {}
        for v in self.variables:
            if v == "rain":
                # ZoneConditionedHead(HurdleHead) returns (logit, amount)
                if isinstance(self.heads[v], ZoneConditionedHead):
                    logit, amount = self.heads[v](h, zone_map)
                else:
                    logit, amount = self.heads[v](h)
                out["rain"] = {"logit_occurrence": logit, "amount": amount}
            else:
                if isinstance(self.heads[v], ZoneConditionedHead):
                    y = self.heads[v](h, zone_map)
                else:
                    y = self.heads[v](h)
                out[v] = y
        return out

    def parameter_summary(self) -> dict[str, int]:
        totals = {"total": 0}
        by_block = {}
        for name, mod in self.named_children():
            n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            by_block[name] = n
            totals["total"] += n
        totals.update(by_block)
        return totals


def build_model(cfg, zones: ZoneRegistry | None = None) -> ZoneAwareModel:
    """Instantiate a :class:`ZoneAwareModel` from an :class:`ExperimentConfig`."""
    if zones is None:
        zones = get_zones()
    gauge_channels = len(cfg.data.variables)
    sat_channels = len(cfg.model.satellite_channels) if cfg.data.include_satellite else 0
    return ZoneAwareModel(
        gauge_channels=gauge_channels,
        sat_channels=sat_channels,
        hidden=cfg.model.hidden,
        zones=zones,
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        film=cfg.model.film_conditioning,
        zone_conditioned_heads=cfg.model.zone_conditioned_heads,
        null_dropout=cfg.model.satellite_null_dropout,
        variables=tuple(cfg.data.variables),
    )
