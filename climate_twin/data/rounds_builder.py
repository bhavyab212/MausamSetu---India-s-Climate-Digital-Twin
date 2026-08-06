"""
rounds_builder.py — coverage-aware rounds.yaml generator.

Reads the real coverage matrix from the processed cube (via data_source) and
emits a walk-forward schedule where each round declares its OWN variable list:
a variable is included in a round only if every year touched by that round
(train + val) has coverage above ``min_coverage`` (default 0.90).

INSAT LST is partial (2020 8 months, 2021 4 months) — the generator will
naturally drop it from every round at the default threshold. Reservoir /
snow variables can be added later without editing app code: build_cube just
needs to emit them; the generator picks them up automatically.

Outputs
-------
  climate_twin/config/rounds_<region>.yaml        one per region

The schedule / hyperparameters / run sections are preserved verbatim from the
existing ``rounds.yaml`` so the sidebar defaults don't shift. What is new:

  data:
    region: india
    manifest_sig: <12-hex>
    manifest_generated_at_ist: ...
    all_years: [2018, ..., 2025]
    train_years: [2018, 2023]
    variables_available: [rain, tmax, tmin, insat_lst]
    coverage:
      rain: {2018: 1.00, ...}

  rounds:
    - round_num: 1
      train_start: 2018-01-01
      train_end:   2023-12-31
      val_start:   2024-01-01
      val_end:     2024-12-31
      years_touched: [2018, ..., 2024]
      variables: [rain, tmax, tmin]        # ← per-round list (INSAT dropped)
      variables_dropped: [insat_lst]
      drop_reasons: {insat_lst: "coverage 0.00 in years [2022, 2023, 2024]"}

The app can then read the round's ``variables`` and feed exactly that channel
count into the model input adapter (Phase 5d).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Ensure the climate_twin/ package folder is on the path when run as a module
_REPO = Path(__file__).resolve().parent.parent      # climate_twin/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import yaml  # PyYAML

import data_source as DS


CONFIG_DIR = _REPO / "config"

# ---------------------------------------------------------------------------
# Defaults — mirror the existing rounds.yaml so the sidebar keeps its numbers
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = {
    "optimizer": "AdamW",
    "scheduler": "cosine",
    "precision": "bf16",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs_per_round": 30,
    "patience": 8,
    "min_delta": 0.0001,
    "mc_samples": 20,
    "monitor": "val_rmse",
    "seq_length": 10,
}
DEFAULT_RUN = {
    "mode": "manual",
    "train_mode": "warm_start",
    "device": "cuda",
}
DEFAULT_MIN_COVERAGE = 0.90            # per-year threshold for including a var
DEFAULT_INITIAL_WINDOW_YEARS = 6       # 2018-2023 = the manifest's train_years
DEFAULT_WINDOW_MODE = "expanding"
DEFAULT_ROLLING_WINDOW_YEARS = 6
DEFAULT_MAX_ROUNDS = 500


# ---------------------------------------------------------------------------
# Coverage-aware variable selection
# ---------------------------------------------------------------------------
def _variables_for_years(
    coverage: dict[str, dict[int, float]],
    years: list[int],
    min_coverage: float,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (kept, dropped, reasons) given the per-year coverage matrix.

    A variable is kept only if EVERY year in ``years`` has coverage >= threshold.
    """
    kept, dropped, reasons = [], [], {}
    for var, yr_cov in coverage.items():
        bad_years = []
        min_cov_in_range = 1.0
        for y in years:
            c = float(yr_cov.get(int(y), 0.0))
            min_cov_in_range = min(min_cov_in_range, c)
            if c < min_coverage:
                bad_years.append(int(y))
        if bad_years:
            dropped.append(var)
            reasons[var] = (
                f"coverage < {min_coverage:.2f} in years {sorted(bad_years)} "
                f"(min={min_cov_in_range:.2f})"
            )
        else:
            kept.append(var)
    return kept, dropped, reasons


# ---------------------------------------------------------------------------
# Schedule builder (year-granular walk-forward)
# ---------------------------------------------------------------------------
def _build_rounds(
    all_years: list[int],
    initial_window_years: int,
    window_mode: str,
    rolling_window_years: int,
    max_rounds: int,
    coverage: dict[str, dict[int, float]],
    min_coverage: float,
) -> list[dict[str, Any]]:
    all_years = sorted(int(y) for y in all_years)
    if len(all_years) <= initial_window_years:
        return []

    rounds: list[dict[str, Any]] = []
    for k, val_year in enumerate(all_years[initial_window_years:]):
        if k + 1 > max_rounds:
            break
        train_years = (
            [y for y in all_years if y <= val_year - 1][-rolling_window_years:]
            if window_mode == "rolling"
            else [y for y in all_years if y <= val_year - 1]
        )
        years_touched = sorted(set(train_years + [val_year]))
        kept, dropped, reasons = _variables_for_years(coverage, years_touched, min_coverage)

        rounds.append({
            "round_num": k + 1,
            "train_start": date(train_years[0], 1, 1).isoformat(),
            "train_end":   date(train_years[-1], 12, 31).isoformat(),
            "val_start":   date(val_year, 1, 1).isoformat(),
            "val_end":     date(val_year, 12, 31).isoformat(),
            "train_years": train_years,
            "val_year": val_year,
            "years_touched": years_touched,
            "variables": kept,
            "variables_dropped": dropped,
            "drop_reasons": reasons,
        })
    return rounds


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def build_rounds_yaml(
    region: str,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    initial_window_years: int = DEFAULT_INITIAL_WINDOW_YEARS,
    window_mode: str = DEFAULT_WINDOW_MODE,
    rolling_window_years: int = DEFAULT_ROLLING_WINDOW_YEARS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    out_path: Path | None = None,
) -> Path:
    """Generate ``config/rounds_<region>.yaml`` from real cube coverage."""
    info = DS.region_info(region, sig=DS.manifest_sig())
    all_years: list[int] = list(info["all_years"])
    coverage = info["coverage"]

    rounds = _build_rounds(
        all_years=all_years,
        initial_window_years=initial_window_years,
        window_mode=window_mode,
        rolling_window_years=rolling_window_years,
        max_rounds=max_rounds,
        coverage=coverage,
        min_coverage=min_coverage,
    )

    # Count variable-frequency across rounds for the summary
    from collections import Counter
    var_counts = Counter()
    dropped_counts = Counter()
    for r in rounds:
        for v in r["variables"]:
            var_counts[v] += 1
        for v in r["variables_dropped"]:
            dropped_counts[v] += 1

    doc: dict[str, Any] = {
        "_generated_by": "climate_twin/data/rounds_builder.py",
        "_manifest_sig": info["manifest_sig"],
        "_manifest_generated_at_ist": info["manifest_generated_at_ist"],
        "data": {
            "region": region,
            "shape": list(info["shape"]),
            "all_years": all_years,
            "train_years": list(info["train_years"]) if info["train_years"] else None,
            "variables_available": info["variables"],
            "min_coverage_threshold": min_coverage,
            "coverage": {
                v: {int(y): round(float(c), 4) for y, c in yr_cov.items()}
                for v, yr_cov in coverage.items()
            },
            "variable_usage_across_rounds": {
                "kept":    {v: int(var_counts.get(v, 0)) for v in info["variables"]},
                "dropped": {v: int(dropped_counts.get(v, 0)) for v in info["variables"]},
            },
        },
        "schedule": {
            "initial_window_years": initial_window_years,
            "window_mode": window_mode,
            "rolling_window_years": rolling_window_years,
            "step_unit": "years",
            "step_size": 1,
            "max_rounds": max_rounds,
        },
        "hyperparameters": dict(DEFAULT_HYPERPARAMS),
        "run": dict(DEFAULT_RUN),
        "rounds": rounds,
    }

    out_path = out_path or (CONFIG_DIR / f"rounds_{region}.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# rounds_" + region + ".yaml — coverage-aware training schedule.\n")
        f.write("# Generated by climate_twin/data/rounds_builder.py from the current\n")
        f.write("# processed cube. Every round carries its own variable list based on\n")
        f.write("# real per-year data coverage. Regenerate whenever the cube changes.\n\n")
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return out_path


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _summary(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    d = doc["data"]
    print(f"wrote {path}")
    print(f"  region        = {d['region']}")
    print(f"  manifest_sig  = {doc.get('_manifest_sig')}")
    print(f"  all_years     = {d['all_years']}")
    print(f"  min_coverage  = {d['min_coverage_threshold']}")
    print(f"  variables_available = {d['variables_available']}")
    print(f"  variable_usage_across_rounds:")
    for v in d["variables_available"]:
        k = d["variable_usage_across_rounds"]["kept"].get(v, 0)
        r = d["variable_usage_across_rounds"]["dropped"].get(v, 0)
        print(f"    {v:12s} kept in {k:3d} rounds, dropped in {r:3d} rounds")
    print(f"  rounds        = {len(doc['rounds'])}")
    if doc["rounds"]:
        first = doc["rounds"][0]
        last = doc["rounds"][-1]
        print(f"    first: train {first['train_start']}..{first['train_end']}  "
              f"val {first['val_start']}..{first['val_end']}  vars={first['variables']}")
        print(f"    last:  train {last['train_start']}..{last['train_end']}  "
              f"val {last['val_start']}..{last['val_end']}  vars={last['variables']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="all", choices=["india", "cauvery", "all"])
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    ap.add_argument("--initial-window-years", type=int, default=DEFAULT_INITIAL_WINDOW_YEARS)
    ap.add_argument("--window-mode", default=DEFAULT_WINDOW_MODE, choices=["expanding", "rolling"])
    ap.add_argument("--rolling-window-years", type=int, default=DEFAULT_ROLLING_WINDOW_YEARS)
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    args = ap.parse_args()

    regs = ["india", "cauvery"] if args.region == "all" else [args.region]
    for r in regs:
        p = build_rounds_yaml(
            region=r,
            min_coverage=args.min_coverage,
            initial_window_years=args.initial_window_years,
            window_mode=args.window_mode,
            rolling_window_years=args.rolling_window_years,
            max_rounds=args.max_rounds,
        )
        _summary(p)
        print()


if __name__ == "__main__":
    main()
