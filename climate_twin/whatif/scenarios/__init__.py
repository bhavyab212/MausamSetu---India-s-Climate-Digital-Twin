"""
whatif.scenarios — orchestration + provenance ledger.

A Scenario is an immutable object recording: driver, driver-args,
regions, years, indices requested, seed, timestamp, cube manifest sig,
zone mask sig, model checkpoint (if any). `Scenario.run()` walks
L0 → L1 → L2 → L3 → L4 and returns a `ScenarioResult` bundle plus a
JSON ledger row appended under ``CACHE_DIR/scenarios/``.

# TODO(Part 1): `Scenario` dataclass + `.run()` executor;
#                `.hash()` for reproducible reruns.
"""
