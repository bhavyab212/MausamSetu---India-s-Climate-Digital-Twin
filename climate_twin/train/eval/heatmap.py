"""
train.eval.heatmap — small-multiples heatmap grid for the Tier-3 cross-product.

Rows = 9 zones. Columns = 4 seasons. One tiny heatmap panel per metric.
Colour = model skill (0..1 where 1 is perfect). Cells flagged as
``INSUFFICIENT`` render as a hatched grey square, never a blank.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .metrics import INSUFFICIENT, SEASONS


def _skill_score(rmse: float | str, baseline_rmse: float | str | None) -> float | None:
    """1 - rmse/baseline_rmse. None when either is insufficient."""
    if not isinstance(rmse, float) or not isinstance(baseline_rmse, float):
        return None
    if baseline_rmse <= 0:
        return None
    return 1.0 - rmse / baseline_rmse


def render_tier3_heatmap(
    tier3_result: dict[str, Any],
    baseline_result: dict[str, Any] | None,
    zones,
    out_path: Path,
    title: str = "Tier-3 skill",
    metric_key: str = "rmse",
    baseline_source: str = "climatology",   # "climatology" | "persistence"
) -> Path:
    """Render 9 rows × 4 season columns × 1 metric.

    ``baseline_result`` is the same-shape dict from
    ``per_zone_baseline_skill``[baseline_source]; if None, we render raw
    RMSE (colour-coded) instead of skill.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    per_zone_season = tier3_result["per_zone_season"]
    baseline_raw = (baseline_result or {}).get(baseline_source, None)
    # Accept either a flat {zone:{season:{metric}}} dict OR a wrapped
    # {"per_zone_season": {...}} tier-3-shape dict.
    if baseline_raw is not None and isinstance(baseline_raw, dict) and "per_zone_season" in baseline_raw:
        baseline = baseline_raw["per_zone_season"]
    else:
        baseline = baseline_raw

    zone_keys = [z.key for z in zones.zones]
    n_zones = len(zone_keys)
    n_seasons = len(SEASONS)

    values = np.full((n_zones, n_seasons), np.nan, dtype=np.float64)
    labels = np.empty((n_zones, n_seasons), dtype="U16")
    for i, zk in enumerate(zone_keys):
        for j, s in enumerate(SEASONS):
            m = per_zone_season.get(zk, {}).get(s, {}).get(metric_key, INSUFFICIENT)
            if isinstance(m, float) and np.isfinite(m):
                if baseline is not None:
                    b = baseline.get(zk, {}).get(s, {}).get(metric_key, INSUFFICIENT)
                    skill = _skill_score(m, b)
                    if skill is None:
                        labels[i, j] = "n/a"
                    else:
                        values[i, j] = skill
                        labels[i, j] = f"{skill:+.2f}"
                else:
                    values[i, j] = m
                    labels[i, j] = f"{m:.2f}"
            else:
                labels[i, j] = "—"

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    fig.patch.set_facecolor("#050912")
    ax.set_facecolor("#0E1522")

    finite = np.isfinite(values)
    if finite.any():
        if baseline is not None:
            vmin, vmax = -0.5, 0.5
            cmap = "RdYlGn"
        else:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            cmap = "viridis"
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    else:
        im = None

    # Hatched insufficient-data cells
    for i in range(n_zones):
        for j in range(n_seasons):
            if not np.isfinite(values[i, j]):
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#333", edgecolor="#666",
                    hatch="///", linewidth=0.5, zorder=1,
                ))
            ax.text(j, i, labels[i, j],
                    ha="center", va="center", color="white",
                    fontsize=8, zorder=2)

    ax.set_xticks(range(n_seasons))
    ax.set_xticklabels(SEASONS, color="white")
    ax.set_yticks(range(n_zones))
    ax.set_yticklabels(zone_keys, color="white")
    ax.tick_params(colors="white")
    ax.set_title(title + f" ({metric_key}"
                 + (f" vs {baseline_source}" if baseline is not None else "")
                 + ")",
                 color="white", fontsize=11)
    for spine in ax.spines.values():
        spine.set_color("white")

    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.ax.tick_params(colors="white")
        cb.set_label(("skill (1 − model/baseline)" if baseline is not None
                      else metric_key),
                     color="white")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    return out_path
