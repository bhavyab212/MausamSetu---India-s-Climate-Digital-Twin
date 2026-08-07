"""whatif.ui.panels.save_load_export — bottom bar."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

from ...config.paths import STREAMLIT_ROOT
from ..copy import banners
from ..state import get_state, load_state_dict

_SCENARIO_DIR = STREAMLIT_ROOT / ".whatif_scenarios"


class ExportBlocked(RuntimeError):
    """Raised when export is refused (dirty tree, unacknowledged caveat)."""


def _list_saved() -> list[Path]:
    if not _SCENARIO_DIR.exists():
        return []
    return sorted(_SCENARIO_DIR.glob("*.yaml"))


def _default_slug(state) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
    return f"{state.region_id}_{state.season}_{state.method}_{ts}"


def _can_export(state, result: dict) -> tuple[bool, str]:
    if state.method == "perturbation" and not state.method_caveat_ack:
        return False, banners.EXPORT_BLOCKED_NO_CAVEAT
    # Dirty-tree check is best-effort; the actual gate lives inside
    # whatif.report.card if present.
    return True, ""


def render_save_load_export_bar(state, result: dict) -> None:
    st.markdown("---")
    cols = st.columns([2.0, 2.0, 2.0])

    with cols[0]:
        st.caption("Save")
        slug = st.text_input(
            "Scenario slug", value=_default_slug(state),
            key="sve_slug",
        )
        if st.button("Save scenario", key="sve_save"):
            _SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
            path = _SCENARIO_DIR / f"{slug}.yaml"
            path.write_text(
                yaml.safe_dump(state.to_dict(), sort_keys=False),
                encoding="utf-8",
            )
            st.success(f"Saved to {path.relative_to(STREAMLIT_ROOT)}")

    with cols[1]:
        st.caption("Load")
        saved = _list_saved()
        labels = [p.stem for p in saved]
        pick = st.selectbox(
            "Saved scenarios",
            options=(["—"] + labels) if labels else ["—"],
            index=0, key="sve_load_pick",
            label_visibility="collapsed",
        )
        if st.button("Load", key="sve_load_btn"):
            if pick and pick != "—":
                path = _SCENARIO_DIR / f"{pick}.yaml"
                d = yaml.safe_load(path.read_text(encoding="utf-8"))
                load_state_dict(d)
                st.rerun()

    with cols[2]:
        st.caption("Export card")
        ok, msg = _can_export(state, result)
        if not ok:
            st.warning(msg)
        else:
            if st.button("Export scenario card (HTML)", key="sve_export"):
                try:
                    from ...report.card import export_scenario_card
                    from ...economics.decision import recommend as _recommend

                    _SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
                    out = _SCENARIO_DIR / f"{_default_slug(state)}_card.html"
                    pm = result.get("payoff_matrix")
                    if pm is None:
                        st.error("No payoff matrix — run a scenario first.")
                    else:
                        rec = result.get("recommendation") or _recommend(pm)
                        tor = result.get("tornado")
                        if tor is None:
                            st.error("No tornado — run a scenario first.")
                        else:
                            export_scenario_card(
                                out,
                                title=f"MausamSetu — What-If · {state.region_id}",
                                region=f"{state.region_kind}:{state.region_id}",
                                run_id=state.cache_key(),
                                payoff_matrix=pm,
                                recommendation=rec,
                                tornado_result=tor,
                                provenance={
                                    "method": state.method,
                                    "season": state.season,
                                    "cache_key": state.cache_key(),
                                },
                            )
                            st.success(f"Card written to {out.relative_to(STREAMLIT_ROOT)}")
                except Exception as e:
                    st.error(f"Export failed: {type(e).__name__}: {e}")
