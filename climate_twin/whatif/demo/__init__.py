"""
whatif.demo — shipped demo scenarios.

Ten pre-cooked, cited, reproducible YAMLs under ``scenarios/``. Each
loads via ``whatif.ui.state.WhatIfState.from_dict`` (extra top-level
fields like ``expected_headline`` are ignored by the state loader).

Rules for demo scenarios (Part 8 §2):
    * They demonstrate the *breadth* of the engine, including its
      honest failures (see 03, 09).
    * They are deterministic — loading the YAML must produce the same
      cache_key on any machine.
    * A scenario with ``status: deferred`` renders a stub in the UI,
      never a fake result.

Programmatic access:

    >>> from whatif.demo import list_scenarios, load_scenario_yaml
    >>> ids = list_scenarios()
    >>> doc = load_scenario_yaml(ids[0])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


def list_scenarios() -> list[str]:
    """Return sorted list of scenario_ids."""
    return sorted(p.stem for p in _SCENARIO_DIR.glob("*.yaml"))


def list_demo_scenarios() -> list[Path]:
    """Return sorted list of scenario YAML paths (Part 8 helper)."""
    return sorted(_SCENARIO_DIR.glob("*.yaml"))


SCENARIOS_DIR = _SCENARIO_DIR


def scenario_path(scenario_id: str) -> Path:
    for p in _SCENARIO_DIR.glob("*.yaml"):
        if p.stem == scenario_id or p.stem.endswith(scenario_id):
            return p
    raise KeyError(f"unknown demo scenario {scenario_id!r}")


def load_scenario_yaml(scenario_id: str) -> dict[str, Any]:
    """Load and return the raw YAML doc — includes ``expected_headline``,
    ``status``, and any other metadata beyond the WhatIfState fields."""
    p = scenario_path(scenario_id)
    return yaml.safe_load(p.read_text(encoding="utf-8"))
