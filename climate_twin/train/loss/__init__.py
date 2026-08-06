"""train.loss — NaN-safe, zone-weighted objectives + per-zone physics penalties."""
from .objectives import (
    hurdle_rain_loss,
    huber_loss,
    total_train_loss,
)
from .physics import (
    tmax_ge_tmin_penalty,
    spatial_smoothness,
    temporal_smoothness,
    zone_physics_bounds_penalty,
)

__all__ = [
    "hurdle_rain_loss",
    "huber_loss",
    "total_train_loss",
    "tmax_ge_tmin_penalty",
    "spatial_smoothness",
    "temporal_smoothness",
    "zone_physics_bounds_penalty",
]
