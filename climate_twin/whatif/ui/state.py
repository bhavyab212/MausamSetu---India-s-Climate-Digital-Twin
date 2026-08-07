"""
whatif.ui.state — the single Streamlit state model for the What-If page.

Streamlit reruns the entire page on every widget interaction. Without a
disciplined state model the UI feels laggy and the caching misbehaves.

Rule:
    * All widget changes go through ``update_state(**kwargs)``. No widget
      mutates ``st.session_state["whatif"]`` directly. This is the single
      choke-point that makes "screen ↔ engine ↔ YAML" reproducibility work.
    * ``WhatIfState`` is a frozen-in-behaviour dataclass whose contents
      are hashable via :func:`state_hash` — that's the cache key for the
      engine call.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

_SESSION_KEY = "whatif"


@dataclass
class WhatIfState:
    """Every lever that affects what the engine computes."""
    horizon: Literal["short_term", "long_term"] = "short_term"
    sector: Literal["agriculture", "energy", "water", "health", "disaster"] = "agriculture"
    region_kind: Literal["all_india", "zone", "district", "subbasin", "bbox"] = "bbox"
    region_id: str = "vidarbha"
    # Method
    method: Literal["analog", "perturbation"] = "analog"
    method_caveat_ack: bool = False
    # Decision axis (agriculture)
    crops_selected: tuple[str, ...] = ("paddy_kharif", "bajra_kharif", "arhar_kharif")
    sow_dates: tuple[str, ...] = ("2020-06-15",)
    irrigation: Literal["rainfed", "supplemental"] = "rainfed"
    include_fallow: bool = True
    season: str = "2024-25"
    # Analog axis
    analog_spec_id: str = "vidarbha_JJAS_default"
    analog_target_year: int = 2020
    analog_k: int = 10
    analog_weighting: Literal["uniform", "inv_distance", "softmax"] = "inv_distance"
    analog_bucketing: Literal["tercile", "per_year"] = "tercile"
    # Perturbation axis
    perturbation_rain_scale: float = 1.0
    perturbation_tmax_shift: float = 0.0
    perturbation_tmin_shift: float = 0.0
    # UI-only preferences (do NOT invalidate the engine cache)
    show_baseline_overlay: bool = True
    show_provenance: bool = False
    # Bookkeeping (also UI-only)
    last_run_id: str | None = None

    # Fields that affect the engine cache key. UI-only preferences are
    # excluded so toggling them does not force a rerun.
    _CACHE_KEY_FIELDS: tuple[str, ...] = (
        "horizon", "sector",
        "region_kind", "region_id",
        "method", "method_caveat_ack",
        "crops_selected", "sow_dates", "irrigation", "include_fallow",
        "season",
        "analog_spec_id", "analog_target_year", "analog_k",
        "analog_weighting", "analog_bucketing",
        "perturbation_rain_scale", "perturbation_tmax_shift",
        "perturbation_tmin_shift",
    )

    def cache_key(self) -> str:
        """Deterministic short hash of the lever set only.

        UI-only preferences (``show_baseline_overlay``, ``show_provenance``,
        ``last_run_id``) are deliberately excluded so toggling them
        does not force a rerun. Rule 12: identical levers never re-run.
        """
        parts = {k: getattr(self, k) for k in self._CACHE_KEY_FIELDS}
        # Coerce tuples to lists for stable JSON serialisation
        norm = {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in parts.items()}
        blob = json.dumps(norm, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_CACHE_KEY_FIELDS", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WhatIfState":
        allowed = {k for k in cls.__dataclass_fields__ if k != "_CACHE_KEY_FIELDS"}
        clean = {k: v for k, v in d.items() if k in allowed}
        # Coerce lists back to tuples for the tuple-typed fields
        for k in ("crops_selected", "sow_dates"):
            if k in clean and isinstance(clean[k], list):
                clean[k] = tuple(clean[k])
        return cls(**clean)


def get_state() -> WhatIfState:
    """Return the current WhatIfState, initialising it on first call.

    We import ``streamlit`` lazily so :mod:`state` is testable outside
    a Streamlit script.
    """
    import streamlit as st       # noqa: local import
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = WhatIfState()
    return st.session_state[_SESSION_KEY]


def update_state(**kwargs: Any) -> WhatIfState:
    """Update the WhatIfState in-place. Widgets call this exclusively.

    Returns the mutated state. When a *cache-relevant* field changes,
    ``last_run_id`` is cleared so downstream code knows the current
    ``ScenarioResult`` is stale.
    """
    import streamlit as st
    state = get_state()
    changed_cache_relevant = False
    for k, v in kwargs.items():
        if not hasattr(state, k):
            raise AttributeError(f"WhatIfState has no field {k!r}")
        current = getattr(state, k)
        if current != v:
            setattr(state, k, v)
            if k in WhatIfState._CACHE_KEY_FIELDS:
                changed_cache_relevant = True
    if changed_cache_relevant:
        state.last_run_id = None
    return state


def reset_state() -> WhatIfState:
    """Wipe and rebuild — used by save/load and by tests."""
    import streamlit as st
    st.session_state[_SESSION_KEY] = WhatIfState()
    return st.session_state[_SESSION_KEY]


def load_state_dict(d: dict[str, Any]) -> WhatIfState:
    import streamlit as st
    st.session_state[_SESSION_KEY] = WhatIfState.from_dict(d)
    return st.session_state[_SESSION_KEY]
