"""
train.ui.dashboards.summary_card — deterministic model-quality verdict.

Rules (from the plan text, §3b):
    🟢 STRONG:  beats persistence in ≥ 8/9 zones AND beats climatology in ≥ 6/9
                AND calibration in [0.75, 0.85] AND physics_violation_pct < 1%
    🟡 MIXED:   beats persistence in ≥ 5/9 zones
    🔴 WEAK:    beats persistence in < 5/9 zones

Card always names best zone, worst zone, and one concrete recommended
action derived from the worst-zone diagnosis. Never returns STRONG if any
zone is below best-of-baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VERDICT_STRONG = "🟢 STRONG"
VERDICT_MIXED = "🟡 MIXED"
VERDICT_WEAK = "🔴 WEAK"


@dataclass
class SummaryCard:
    verdict: str
    beats_persistence: int
    beats_climatology: int
    n_zones: int
    calibration: float | None
    calibration_ok: bool
    physics_violation_pct: float | None
    physics_ok: bool
    best_zone: str
    best_zone_rmse: float | None
    worst_zone: str
    worst_zone_delta: float | None
    recommendation: str
    reasoning: list[str]


def build_summary_card(
    per_zone_model_rmse: dict[str, float],
    per_zone_persistence_rmse: dict[str, float],
    per_zone_climatology_rmse: dict[str, float],
    calibration: float | None = None,
    physics_violation_pct: float | None = None,
) -> SummaryCard:
    """Deterministic verdict card.

    Inputs are per-zone RMSE dicts (all in the same physical units).
    Missing zones are skipped from all counts (never fabricated).
    """
    zones = sorted(per_zone_model_rmse.keys())
    n = len(zones)

    beats_p = 0
    beats_c = 0
    beats_any = 0
    best_zone = ""
    best_rmse = float("inf")
    worst_zone = ""
    worst_delta = -float("inf")

    for z in zones:
        m = per_zone_model_rmse.get(z)
        p = per_zone_persistence_rmse.get(z)
        c = per_zone_climatology_rmse.get(z)
        if not isinstance(m, (int, float)):
            continue
        if isinstance(p, (int, float)) and m < p:
            beats_p += 1
        if isinstance(c, (int, float)) and m < c:
            beats_c += 1
        if isinstance(p, (int, float)) or isinstance(c, (int, float)):
            best_base = min(x for x in (p, c) if isinstance(x, (int, float)))
            if m < best_base:
                beats_any += 1
            delta = m - best_base
            if delta > worst_delta:
                worst_delta = delta
                worst_zone = z
        if m < best_rmse:
            best_rmse = m
            best_zone = z

    calib_ok = calibration is not None and 0.75 <= float(calibration) <= 0.85
    phys_ok = physics_violation_pct is not None and float(physics_violation_pct) < 1.0

    verdict = VERDICT_WEAK
    reasoning: list[str] = []
    strong = (
        n >= 9
        and beats_p >= 8
        and beats_c >= 6
        and calib_ok
        and phys_ok
        and beats_any == n     # every zone strictly beats best-of-baseline
    )
    if strong:
        verdict = VERDICT_STRONG
        reasoning.append(
            f"beats persistence in {beats_p}/{n} zones, "
            f"climatology in {beats_c}/{n}, all zones beat best-of-baseline, "
            f"calibration {calibration:.2f} in [0.75, 0.85], "
            f"physics violations {physics_violation_pct:.2f}% < 1%"
        )
    elif beats_p >= 5:
        verdict = VERDICT_MIXED
        reasoning.append(
            f"beats persistence in {beats_p}/{n} zones. "
            + ("Not STRONG because " if beats_p >= 8 else "")
            + " and ".join([
                *([f"only {beats_c}/{n} zones beat climatology"] if beats_c < 6 else []),
                *(["calibration out of [0.75, 0.85]"] if not calib_ok else []),
                *(["physics violations ≥ 1%"] if not phys_ok else []),
                *([f"{n - beats_any}/{n} zones below best-of-baseline"] if beats_any < n else []),
            ])
        )
    else:
        verdict = VERDICT_WEAK
        reasoning.append(f"beats persistence in only {beats_p}/{n} zones")

    if worst_zone and worst_delta > 0:
        recommendation = (
            f"Investigate the {worst_zone} error map before more training — "
            f"model is {worst_delta:+.2f} mm/day above best baseline there."
        )
    elif worst_zone:
        recommendation = (
            f"Longest training marginal gains likely in {worst_zone} — "
            f"currently the tightest cell vs baseline."
        )
    else:
        recommendation = "No per-zone regression detected; continue training or lock the checkpoint."

    return SummaryCard(
        verdict=verdict,
        beats_persistence=beats_p,
        beats_climatology=beats_c,
        n_zones=n,
        calibration=(None if calibration is None else float(calibration)),
        calibration_ok=calib_ok,
        physics_violation_pct=(None if physics_violation_pct is None else float(physics_violation_pct)),
        physics_ok=phys_ok,
        best_zone=best_zone,
        best_zone_rmse=(None if best_rmse == float("inf") else float(best_rmse)),
        worst_zone=worst_zone,
        worst_zone_delta=(None if worst_delta == -float("inf") else float(worst_delta)),
        recommendation=recommendation,
        reasoning=reasoning,
    )


def render_summary_card_html(card: SummaryCard) -> str:
    color = {"🟢 STRONG": "#28a745", "🟡 MIXED": "#f4a34a", "🔴 WEAK": "#e63946"}[card.verdict]
    calib_str = f"{card.calibration:.2f}" if card.calibration is not None else "—"
    phys_str = f"{card.physics_violation_pct:.2f}%" if card.physics_violation_pct is not None else "—"
    return (
        f'<div style="border:2px solid {color}; padding:14px; border-radius:8px; '
        f'font-family:monospace; background:#0e1522; color:#e5e9f0;">'
        f'<div style="font-size:1.3em; color:{color}; margin-bottom:8px;">'
        f'<b>Overall verdict: {card.verdict}</b></div>'
        f'<div>Beats persistence in <b>{card.beats_persistence} / {card.n_zones}</b> zones</div>'
        f'<div>Beats climatology in <b>{card.beats_climatology} / {card.n_zones}</b> zones</div>'
        f'<div>Ensemble calibration: <b>{calib_str}</b> (target 0.80) '
        f'{"✓" if card.calibration_ok else ""}</div>'
        f'<div>Physics violations: <b>{phys_str}</b> (below 1% threshold) '
        f'{"✓" if card.physics_ok else ""}</div>'
        f'<div style="margin-top:10px;">'
        f'Best zone: <b>{card.best_zone or "—"}</b>'
        f'{f" (RMSE {card.best_zone_rmse:.2f} mm/day)" if card.best_zone_rmse is not None else ""}'
        f'</div>'
        f'<div>Worst zone: <b>{card.worst_zone or "—"}</b>'
        f'{f" ({card.worst_zone_delta:+.2f} mm vs baseline)" if card.worst_zone_delta is not None else ""}'
        f'</div>'
        f'<div style="margin-top:10px; color:#F4A34A;">'
        f'<b>Recommended next action:</b><br>{card.recommendation}</div>'
        f'</div>'
    )
