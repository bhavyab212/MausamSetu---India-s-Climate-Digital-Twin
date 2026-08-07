"""
whatif.scenarios.quantiles — three-pass orchestration.

Every DataArray flowing into L4 economics carries a ``quantile`` attr
(one of ``q10``, ``q50``, ``q90``, ``deterministic``). Layers must
propagate that attribute forward unchanged. Any L3/L4 function that
receives inputs with mixed quantiles raises :class:`MixedQuantiles`
rather than silently averaging — the correct pattern is to run three
parallel passes and merge only at the reporting layer.

``three_pass(spec, fn)`` implements exactly that pattern.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from ..drivers.driver import DriverSpec


class MixedQuantiles(RuntimeError):
    """Raised when a function receives inputs with disagreeing quantile tags."""


def three_pass(spec: DriverSpec, fn: Callable[[DriverSpec], Any]) -> dict[str, Any]:
    """Run ``fn(spec_q10)``, ``fn(spec_q50)``, ``fn(spec_q90)`` and
    return the three results keyed ``{"q10", "q50", "q90"}``.

    ``spec`` may already carry a ``quantile``; it is temporarily
    overwritten to each of 0.10 / 0.50 / 0.90. The three resulting
    calls are guaranteed independent — no aggregation happens here.
    """
    out: dict[str, Any] = {}
    for q, key in ((0.10, "q10"), (0.50, "q50"), (0.90, "q90")):
        # DriverSpec is frozen; use dataclasses.replace to build a copy
        sub = replace(spec, quantile=q)
        out[key] = fn(sub)
    return out


def assert_single_quantile(*inputs) -> str:
    """Assert every input carries the same ``attrs["quantile"]``.
    Returns the shared quantile string. Raises :class:`MixedQuantiles`
    on disagreement.

    Inputs may be xr.DataArray, xr.Dataset, or dict-with-attrs-like
    (must have an ``.attrs`` mapping).
    """
    seen = set()
    for x in inputs:
        q = None
        if hasattr(x, "attrs"):
            q = x.attrs.get("quantile")
        if q is not None:
            seen.add(q)
    if len(seen) > 1:
        raise MixedQuantiles(
            f"received inputs with mixed quantile tags {sorted(seen)}. "
            f"Use three_pass() and merge only at the reporting layer."
        )
    return seen.pop() if seen else "deterministic"
