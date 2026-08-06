"""
train.ui.dashboards — Streamlit renderers for the two rebuilt tabs.

    training.py    — 🔥 interactive training dashboard (config, live view, actions)
    validation.py  — 🔍 visual + statistical inspection of any checkpoint
    common.py      — shared helpers (device banner, config editor, session-state)
    summary_card.py — deterministic STRONG/MIXED/WEAK verdict
"""
from .training import render_training_dashboard
from .validation import render_validation_dashboard
from .common import DEVICE_BANNER_CSS

__all__ = [
    "render_training_dashboard",
    "render_validation_dashboard",
    "DEVICE_BANNER_CSS",
]
