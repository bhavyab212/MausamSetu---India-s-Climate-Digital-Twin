"""train.model — shared ConvLSTM backbone with FiLM zone conditioning."""
from .film import FiLM2d
from .backbone import ConvLSTMCell, ConvLSTMBackbone
from .heads import ZoneConditionedHead, HurdleHead
from .encoders import GaugeEncoder, SatelliteEncoder
from .assemble import ZoneAwareModel, build_model

__all__ = [
    "FiLM2d",
    "ConvLSTMCell",
    "ConvLSTMBackbone",
    "ZoneConditionedHead",
    "HurdleHead",
    "GaugeEncoder",
    "SatelliteEncoder",
    "ZoneAwareModel",
    "build_model",
]
