"""whatif.ui.panels.levers — the left-column lever panel."""
from __future__ import annotations

import streamlit as st

from ...sectors.crops import list_crops
from ..copy import banners
from ..state import update_state


_SECTOR_OPTIONS = [
    ("agriculture",  "Agriculture", True,  "Part 3–5"),
    ("energy",       "Energy",      False, "Part 7"),
    ("water",        "Water",       False, "Part 7"),
    ("health",       "Health",      False, "Part 7"),
    ("disaster",     "Disaster",    False, "Part 7"),
]

_REGION_OPTIONS = [
    ("all_india", "All-India"),
    ("bbox",      "Vidarbha (bbox)"),
    ("subbasin",  "Cauvery basin"),
    ("zone",      "Zone (registry)"),
    ("district",  "District (registry)"),
]


def render_levers(state) -> bool:
    """Draw the lever panel; return True when the user pressed Run."""
    st.subheader("Levers")

    # ── Sector ──
    st.caption("Sector")
    sector_labels = [f"{lbl}" + ("" if enabled else "  ·  " + note)
                      for _, lbl, enabled, note in _SECTOR_OPTIONS]
    current_idx = next(
        (i for i, (k, *_) in enumerate(_SECTOR_OPTIONS) if k == state.sector),
        0,
    )
    picked = st.selectbox("Sector", sector_labels, index=current_idx,
                            label_visibility="collapsed", key="lv_sector")
    picked_key = _SECTOR_OPTIONS[sector_labels.index(picked)][0]
    picked_enabled = _SECTOR_OPTIONS[sector_labels.index(picked)][2]
    if picked_key != state.sector:
        update_state(sector=picked_key)
    if not picked_enabled:
        st.info(f"Arrives in Part 7 — {picked} sector.")
        return False

    st.divider()

    # ── Region ──
    st.caption("Region")
    reg_labels = [lbl for _, lbl in _REGION_OPTIONS]
    reg_keys = [k for k, _ in _REGION_OPTIONS]
    cur_reg_idx = reg_keys.index(state.region_kind) if state.region_kind in reg_keys else 1
    reg_pick = st.selectbox("Region kind", reg_labels, index=cur_reg_idx,
                              label_visibility="collapsed", key="lv_region_kind")
    reg_key = reg_keys[reg_labels.index(reg_pick)]
    if reg_key != state.region_kind:
        update_state(region_kind=reg_key)

    if reg_key == "bbox":
        rid = st.text_input("Region ID", value=state.region_id or "vidarbha",
                              key="lv_region_id")
        if rid != state.region_id:
            update_state(region_id=rid)
        st.caption(banners.RESOLUTION_CEILING_NOTE)
    elif reg_key == "subbasin":
        update_state(region_id="cauvery")
    elif reg_key == "all_india":
        update_state(region_id="all_india")

    st.divider()

    # ── Climate axis ──
    st.caption("Climate axis")
    method = st.radio(
        "Method",
        ["Historical analogs (Method 2)", "Delta perturbation (Method 1)"],
        index=(0 if state.method == "analog" else 1),
        label_visibility="collapsed",
        key="lv_method",
    )
    method_key = "analog" if "analogs" in method else "perturbation"
    if method_key != state.method:
        update_state(method=method_key, method_caveat_ack=False)

    if method_key == "analog":
        st.selectbox(
            "Analog spec", [state.analog_spec_id],
            index=0, key="lv_analog_spec",
            help="Custom specs land in a later part.",
        )
        wt = st.selectbox(
            "Weighting", ["inv_distance", "softmax", "uniform"],
            index=["inv_distance", "softmax", "uniform"].index(state.analog_weighting),
            key="lv_weighting",
        )
        if wt != state.analog_weighting:
            update_state(analog_weighting=wt)
        bk = st.selectbox(
            "Bucketing", ["tercile", "per_year"],
            index=["tercile", "per_year"].index(state.analog_bucketing),
            key="lv_bucketing",
        )
        if bk != state.analog_bucketing:
            update_state(analog_bucketing=bk)
        ty = st.number_input(
            "Target year", min_value=2011, max_value=2022,
            value=int(state.analog_target_year), step=1, key="lv_target_year",
        )
        if int(ty) != state.analog_target_year:
            update_state(analog_target_year=int(ty))
    else:
        st.markdown(banners.PERTURBATION_CAVEAT)
        rs = st.slider("Rain scale", 0.5, 1.5,
                        float(state.perturbation_rain_scale), 0.05,
                        key="lv_rain_scale")
        if rs != state.perturbation_rain_scale:
            update_state(perturbation_rain_scale=float(rs))
        tx = st.slider("Tmax shift (°C)", -3.0, 5.0,
                        float(state.perturbation_tmax_shift), 0.5,
                        key="lv_tmax_shift")
        if tx != state.perturbation_tmax_shift:
            update_state(perturbation_tmax_shift=float(tx))
        tn = st.slider("Tmin shift (°C)", -3.0, 5.0,
                        float(state.perturbation_tmin_shift), 0.5,
                        key="lv_tmin_shift")
        if tn != state.perturbation_tmin_shift:
            update_state(perturbation_tmin_shift=float(tn))
        ack = st.checkbox(
            banners.CAVEAT_ACKNOWLEDGE_CHECKBOX,
            value=bool(state.method_caveat_ack),
            key="lv_caveat_ack",
        )
        if ack != state.method_caveat_ack:
            update_state(method_caveat_ack=bool(ack))

    st.divider()

    # ── Decision axis (agriculture) ──
    st.caption("Decision axis")
    all_crops = list_crops()
    crops = st.multiselect(
        "Crops to compare", all_crops, default=list(state.crops_selected),
        key="lv_crops",
    )
    if tuple(crops) != state.crops_selected:
        update_state(crops_selected=tuple(crops))
    sow = st.text_input(
        "Sow date (ISO)", value=state.sow_dates[0], key="lv_sow_date",
    )
    if (sow,) != state.sow_dates:
        update_state(sow_dates=(sow,))
    irr = st.radio(
        "Irrigation", ["rainfed", "supplemental"],
        index=(0 if state.irrigation == "rainfed" else 1),
        key="lv_irrigation",
    )
    if irr != state.irrigation:
        update_state(irrigation=irr)
    fal = st.checkbox('Include "fallow"', value=bool(state.include_fallow),
                       key="lv_include_fallow")
    if fal != state.include_fallow:
        update_state(include_fallow=bool(fal))

    st.divider()

    run_label = "Run scenario"
    if state.method == "perturbation" and not state.method_caveat_ack:
        st.button(run_label, key="lv_run", disabled=True,
                    help="Acknowledge the perturbation caveat first.")
        return False
    return bool(st.button(run_label, key="lv_run", type="primary"))
