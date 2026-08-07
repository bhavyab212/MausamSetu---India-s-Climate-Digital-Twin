"""
whatif.report — user-facing rendering (plots + one-page card).

Two public entry points:
    * :func:`payoff_heatmap`     — plotly figure of the payoff matrix
                                    with regret shading + rupee-formatted
                                    tick labels. Divergent colormap
                                    centred on the climatology baseline
                                    (never jet).
    * :func:`export_scenario_card` — one-page HTML card the browser can
                                     Print-to-PDF (reportlab / weasy-
                                     print aren't wired yet; the HTML
                                     is deliberately self-contained).
"""
from __future__ import annotations

from .card import export_scenario_card
from .one_pager import ONE_PAGER_VERSION, render_one_pager
from .payoff_render import payoff_heatmap, format_inr

__all__ = [
    "payoff_heatmap", "format_inr", "export_scenario_card",
    "render_one_pager", "ONE_PAGER_VERSION",
]
