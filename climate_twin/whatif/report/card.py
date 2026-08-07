"""
whatif.report.card — single-page HTML "scenario card".

Design notes:
    * No PDF dependency. The card is a self-contained HTML the browser
      can Print-to-PDF (Cmd/Ctrl-P). This keeps the whatif package
      free of new heavy deps.
    * Every rupee number goes through :func:`whatif.report.payoff_render.format_inr`
      so the units in the printed card match the units on the screen.
    * Provenance strip is mandatory — the code hash, dataset versions,
      prices SHA, and crops SHA all appear at the foot of the card.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from .payoff_render import format_inr


_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scenario card — {title_esc}</title>
<style>
  @page {{ size: A4; margin: 12mm; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          color: #111; margin: 0; }}
  h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
  h2 {{ font-size: 14px; margin: 12px 0 4px 0; color: #444; }}
  .meta {{ color: #666; font-size: 12px; }}
  .strip {{ background: #f2f2f2; padding: 6px 10px; border-radius: 6px;
            font-family: SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 11px; white-space: pre-wrap; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .rec {{ background: #eef6ff; padding: 10px 12px; border-left: 3px solid #2a6bcc;
          margin: 8px 0; border-radius: 4px; }}
</style>
</head>
<body>

<h1>{title_esc}</h1>
<div class="meta">
  Region: <b>{region}</b> · Created: {created_at} · Scenario id: <code>{run_id}</code>
</div>

<h2>Recommendation</h2>
<div class="rec">
  <b>{rec_label}</b> — EV = {rec_ev} (q10 {rec_q10} · q90 {rec_q90}) ·
  worst-case {rec_wc} · Δ vs climatology <b>{rec_delta}</b>
</div>

<h2>Payoff matrix (₹/ha, q50)</h2>
{payoff_table_html}

<h2>Tornado (top 3 sensitivities)</h2>
<ul>
  {tornado_bullets}
</ul>
<div class="meta">{tornado_sentence}</div>

<h2>Provenance</h2>
<div class="strip">{provenance_block}</div>

<div class="meta" style="margin-top: 16px;">
  Rendered by climate_twin.whatif.report — this card is a diagnostic
  summary. Every ₹ output is a distributional estimate; single numbers
  above are the q50 of that distribution.
</div>

</body>
</html>
"""


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))


def _payoff_table_html(pm) -> str:
    lines = ["<table>", "<thead><tr><th>Decision</th>"]
    for s in pm.states:
        lines.append(f"<th>{_esc(s.label)}<br>p={s.weight:.2f}</th>")
    lines.append("</tr></thead><tbody>")
    for i, dec in enumerate(pm.decisions):
        row = [f"<tr><td>{_esc(dec.label)}</td>"]
        for j in range(pm.n_state):
            row.append(f"<td>{_esc(format_inr(float(pm.payoff_p50[i, j])))}</td>")
        row.append("</tr>")
        lines.append("".join(row))
    lines.append("</tbody></table>")
    return "\n".join(lines)


def export_scenario_card(
    path: str | Path,
    *,
    title: str,
    region: str,
    run_id: str,
    payoff_matrix,
    recommendation: dict,
    tornado_result,
    provenance: dict[str, Any],
    created_at: datetime | None = None,
) -> Path:
    """Render a self-contained HTML card at ``path``.

    Parameters mirror what the What-If page holds in session state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rec = recommendation
    rec_html = {
        "rec_label": _esc(rec.get("argmax_EV_label", "?")),
        "rec_ev": _esc(format_inr(rec.get("EV_q50", float("nan")))),
        "rec_q10": _esc(format_inr(rec.get("EV_q10", float("nan")))),
        "rec_q90": _esc(format_inr(rec.get("EV_q90", float("nan")))),
        "rec_wc": _esc(format_inr(rec.get("worst_case_q50", float("nan")))),
        "rec_delta": _esc(format_inr(rec.get("delta_vs_baseline_EV", float("nan")))),
    }

    tor_rows = tornado_result.rows.head(3)
    bullets = []
    for _, r in tor_rows.iterrows():
        bullets.append(
            f"<li><b>{_esc(str(r['label']))}</b> — "
            f"range {_esc(format_inr(float(r['range'])))} "
            f"({_esc(str(r['units']))})</li>"
        )
    tornado_bullets = "\n".join(bullets) or "<li>—</li>"

    prov_lines = [f"{k}: {v}" for k, v in provenance.items()]
    prov_block = _esc("\n".join(prov_lines))

    html = _TEMPLATE.format(
        title_esc=_esc(title),
        region=_esc(region),
        run_id=_esc(run_id),
        created_at=_esc((created_at or datetime.now()).isoformat(timespec="seconds")),
        payoff_table_html=_payoff_table_html(payoff_matrix),
        tornado_bullets=tornado_bullets,
        tornado_sentence=_esc(tornado_result.sentence()),
        provenance_block=prov_block,
        **rec_html,
    )
    path.write_text(html, encoding="utf-8")
    return path
