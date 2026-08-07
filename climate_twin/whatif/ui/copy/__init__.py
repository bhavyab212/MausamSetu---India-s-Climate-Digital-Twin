"""
whatif.ui.copy — every user-facing string in one place.

Panels never inline copy. Every displayed line is imported from
here, which makes:
    * Devanagari branding consistent (``MausamSetu मौसम सेतु``).
    * Long-term verbs pinned to "scenario" / "conditional trajectory";
      the module-level test in :mod:`long_term` refuses "predict" and
      "forecast".
    * Recommendation prose slotted through :func:`recommendation.render`.
"""
from . import analogs, banners, context, long_term, recommendation

__all__ = ["analogs", "banners", "context", "long_term", "recommendation"]
