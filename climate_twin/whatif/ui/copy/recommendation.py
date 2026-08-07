"""
whatif.ui.copy.recommendation — templated recommendation prose.

Panels never concatenate their own recommendation sentence.  Slots:
    argmax_label          human decision label
    ev_p50, ev_p10, ev_p90  ₹/ha (already-formatted strings)
    worst_case             ₹/ha (formatted)
    delta_vs_baseline      ₹/ha (formatted, signed)
    delta_pct              % vs baseline (formatted, signed)
    beats_label            second-best decision label
    beats_amount           ₹/ha (formatted)
    downside_reduction     % (formatted)
    confidence_word        "Strong" / "Fair" / "Low"
    analog_year, analog_dist, analog_tier  most-similar analog

Three variants:
    STRONG_ANALOG     — top-1 analog is "strong"
    FAIR_ANALOG       — top-1 is "fair"
    NO_STRONG_ANALOG  — top-1 is "poor"
"""
from __future__ import annotations

STRONG_ANALOG = (
    "**{argmax_label}**\n\n"
    "Expected net revenue **{ev_p50} / ha** "
    "(q10 {ev_p10} · q50 {ev_p50} · q90 {ev_p90}).\n\n"
    "Worst-case (low-rain state): {worst_case} / ha.\n\n"
    "vs. climatology baseline: **{delta_vs_baseline} / ha** "
    "({delta_pct} of baseline).\n\n"
    "This choice beats *{beats_label}* by {beats_amount} / ha in "
    "expectation and reduces downside risk by {downside_reduction}.\n\n"
    "**Confidence: {confidence_word}** — closest analog {analog_year} "
    "(Mahalanobis d² = {analog_dist}, {analog_tier})."
)

FAIR_ANALOG = STRONG_ANALOG        # same template, only badge word differs

NO_STRONG_ANALOG = (
    "**{argmax_label}**\n\n"
    "Expected net revenue **{ev_p50} / ha** "
    "(q10 {ev_p10} · q50 {ev_p50} · q90 {ev_p90}).\n\n"
    "**No strong or fair historical analog exists for this target in "
    "the training record.** Closest match is {analog_year} "
    "(d² = {analog_dist}, poor). The outcome distribution shown above "
    "is unusually uncertain — treat this recommendation as **low "
    "confidence**."
)


def render_recommendation(*, tier: str, **slots) -> str:
    """Return the correct template filled with ``slots``."""
    if tier == "strong":
        tpl = STRONG_ANALOG
    elif tier == "fair":
        tpl = FAIR_ANALOG
    else:
        tpl = NO_STRONG_ANALOG
    return tpl.format(**slots)
