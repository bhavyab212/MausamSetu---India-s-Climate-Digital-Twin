"""
train.eval.significance — bootstrap CIs + paired Wilcoxon.

Both operate on cell-day-level arrays of a scalar metric (typically per-day
per-zone RMSE contributions). The plan's rule 3g / 4c: every Tier 3/4 number
carries a bootstrap CI; every "we beat baseline" claim needs a paired test.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    samples: np.ndarray,
    stat: callable = np.mean,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` at 1-α confidence.

    Non-parametric bootstrap: resample ``samples`` with replacement ``n_boot``
    times, apply ``stat``, take α/2 and 1-α/2 percentiles.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = samples.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(stat(samples[idx]))
    mean = float(stat(samples))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def paired_wilcoxon(
    ours: np.ndarray, baseline: np.ndarray,
    alternative: str = "less",
) -> dict[str, float | str]:
    """Paired Wilcoxon signed-rank test on error samples.

    ``alternative='less'`` tests the hypothesis ``|ours| < |baseline|``
    (i.e., we're better). Returns ``{'statistic', 'p_value', 'n_pairs',
    'winner'}`` where winner is 'ours' if p < 0.05, 'baseline' if the
    reverse test rejects, else 'tie'.
    """
    from scipy.stats import wilcoxon
    ours = np.asarray(ours, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    finite = np.isfinite(ours) & np.isfinite(baseline)
    ours = ours[finite]
    baseline = baseline[finite]
    if ours.size < 10:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_pairs": int(ours.size), "winner": "insufficient data"}
    # Compare absolute errors; smaller = better.
    diff = np.abs(ours) - np.abs(baseline)
    # If every diff is exactly zero, the two predictors are identical on
    # the paired samples — that is a legitimate tie, not a test failure.
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0,
                "n_pairs": int(ours.size), "winner": "tie"}
    # If diff < 0 → ours smaller error → ours wins.
    try:
        stat, p_less = wilcoxon(diff, alternative="less")
        stat_g, p_greater = wilcoxon(diff, alternative="greater")
    except Exception as e:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_pairs": int(ours.size),
                "winner": f"test failed ({type(e).__name__})"}
    winner = "tie"
    if p_less < 0.05 and p_less <= p_greater:
        winner = "ours"
    elif p_greater < 0.05 and p_greater < p_less:
        winner = "baseline"
    return {
        "statistic": float(stat),
        "p_value": float(min(p_less, p_greater)),
        "n_pairs": int(ours.size),
        "winner": winner,
    }
