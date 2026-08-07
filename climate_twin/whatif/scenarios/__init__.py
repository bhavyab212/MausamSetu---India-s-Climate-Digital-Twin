"""
whatif.scenarios — orchestration + provenance ledger.

A Scenario is an immutable driver + levers + region; ``run_scenario``
walks L0 → L1 → L2 → L3 → L4 and returns a ``ScenarioResult`` bundle.
Every run also emits a YAML in ``.whatif_runs/<run_id>.yaml`` that
``run_from_yaml`` can replay for reproducibility.

Public surface (Part 1):
    ScenarioResult, run_scenario                       — engine.py
    ProvenanceRecord, record_open, record_add_layer,
        record_close, run_from_yaml                    — provenance.py
    three_pass, assert_single_quantile, MixedQuantiles — quantiles.py
"""
from .engine import ScenarioResult, run_scenario
from .provenance import (
    ProvenanceRecord,
    record_add_layer,
    record_close,
    record_open,
    run_from_yaml,
)
from .quantiles import MixedQuantiles, assert_single_quantile, three_pass

__all__ = [
    "ScenarioResult",
    "run_scenario",
    "ProvenanceRecord",
    "record_add_layer",
    "record_close",
    "record_open",
    "run_from_yaml",
    "MixedQuantiles",
    "assert_single_quantile",
    "three_pass",
]
