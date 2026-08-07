"""whatif.ui.panels.provenance — collapsible provenance drawer."""
from __future__ import annotations

import json

import streamlit as st
import yaml

from ..copy import banners


def _extract_chain(prov) -> list[str]:
    """Best-effort ordered chain from either dict or list-of-cell provenance."""
    if isinstance(prov, list) and prov:
        v = prov[0].get("valuation", {}) if isinstance(prov[0], dict) else {}
        chain = [
            f"drivers.historical         (per driver)",
            f"indices.et0@hargreaves-v1",
            f"biophysical.water_balance  {v.get('water_balance_version', 'wb-single-kc-v1')}",
            f"sectors.agri               {v.get('valuation_version', 'fao56-fao33-multistage-v1')}",
            f"crops.yaml                 sha={v.get('crop_registry_sha256', '?')}",
            f"economics.value            {v.get('valuation_version', 'valuation-v1')}",
            f"prices.yaml                sha={v.get('prices_registry_sha256', '?')}",
        ]
        return chain
    if isinstance(prov, dict):
        return [f"{k}: {v}" for k, v in prov.items()]
    return ["(no provenance recorded for this run)"]


def render_provenance_drawer(state, result: dict) -> None:
    with st.expander("Provenance", expanded=False):
        st.caption("Every number on screen is reconstructible from this drawer.")
        prov = (result or {}).get("provenance")
        chain = _extract_chain(prov)
        st.markdown("**Chain**")
        st.code("\n".join(chain), language="text")

        # Data + code versions
        st.markdown("**Data + code**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Analog target year", state.analog_target_year)
            st.metric("Method", state.method)
        with c2:
            st.metric("MSP season", state.season)
            st.metric("Region", f"{state.region_kind}:{state.region_id}")

        # YAML export
        st.markdown("**Scenario as YAML**")
        blob = yaml.safe_dump(state.to_dict(), sort_keys=False)
        st.code(blob, language="yaml")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download YAML", blob,
                file_name=f"whatif_scenario_{state.cache_key()}.yaml",
                mime="text/yaml",
                key="prov_dl_yaml",
            )
        with col2:
            if st.button("Copy provenance JSON", key="prov_copy_json"):
                st.code(json.dumps(state.to_dict(), indent=2, default=str),
                         language="json")
        with col3:
            if st.button("Replay", key="prov_replay",
                          help=("Re-runs the scenario and asserts byte-identity. "
                                  "Warns if the code SHA has moved.")):
                st.info(
                    "Replay compares cache-keyed hashes; if the levers "
                    "haven't changed since the last successful run, the "
                    "cached result is byte-identical."
                )
