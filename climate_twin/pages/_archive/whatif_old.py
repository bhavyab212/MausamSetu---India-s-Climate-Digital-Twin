"""
whatif_old.py — ARCHIVED (Part 0 of the What-If Engine build).

Preserved verbatim from ``climate_twin/app_v2.py::_render_tab3`` (the
"±1°C What-If Storyline" tab) so future ports can reference the original
UI, its sensitivity-map slider mechanic, and its PAST / PRESENT / FUTURE
tri-column layout.

This file lives under ``pages/_archive/`` on purpose:

    * Streamlit's multipage discovery only registers ``pages/*.py``
      files at the pages-root level. Anything inside a subfolder
      (``_archive/``) is IGNORED by the router. The old page is
      therefore de-registered without being deleted.
    * The archive is INFORMATIONAL. Nothing imports it. The new
      What-If lives in ``pages/30_What_If.py`` and the ``whatif/``
      package.

Original context (extracted from ``app_v2.py``):

    - Renderer name : ``_render_tab3``
    - Label used    : "±1°C What-If"
    - Icon in nav   : "❓"
    - Dependency    : ``rain_data``, ``mask``, ``sens_map`` bound at
                       module level in ``app_v2.py``; ``plotly_vis`` +
                       ``plotly_sensitivity_map`` helpers.
    - Signature     : the sensitivity-map slider assumed a static
                       "rainfall sensitivity to warming" .npy
                       precomputed elsewhere. Was flagged as fragile
                       during Phase 4 (source of the .npy was not
                       versioned; Colab notebook only).

The Part-1+ What-If engine replaces this with proper drivers
(historical, ensemble, perturbation, analog, SSP), indices, biophysical
and sectoral models, economics, and orchestrated provenance-tracked
scenarios.

The verbatim body of the old renderer is intentionally NOT executable as
a Streamlit page (the module-level globals it depends on don't exist in
a page-scoped module). Comments-out the ``@st.fragment`` decorator and
wraps the body in a function you can eyeball but not accidentally
register.
"""
from __future__ import annotations

# ----------------------------------------------------------------------
# ORIGINAL BODY — verbatim from climate_twin/app_v2.py lines 1985-2034
# (kept as text-only reference; do not execute)
# ----------------------------------------------------------------------
ORIGINAL_SOURCE = r'''
@st.fragment
def _render_tab3():
    st.header("Storyline: What If Temperature Changes?")
    st.markdown("PAST / PRESENT / FUTURE comparison")

    if sens_map is not None:
        delta_t = st.slider("Temperature Change (°C)", -2.0, 3.0, 1.0, 0.1, key='wif')

        past_rain = np.nanmean(rain_data[:16], axis=0)
        present_rain = np.nanmean(rain_data[-15:], axis=0)
        future_rain = present_rain + sens_map * delta_t

        st.subheader(f"Rainfall Under {delta_t:+.1f}°C Scenario")
        land_stack = np.where(mask == 1, np.stack([past_rain, present_rain, future_rain]), np.nan)
        p99 = float(np.nanpercentile(land_stack, 99)) if np.any(np.isfinite(land_stack)) else 1500.0
        rain_vmax = max(400.0, min(3000.0, p99 * 1.05))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                plotly_vis.plot_map(
                    past_rain, "PAST (1975-1990)", "rain_annual", custom_range=[0.0, rain_vmax]
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plotly_vis.plot_map(
                    present_rain, "PRESENT (2010-2024)", "rain_annual", custom_range=[0.0, rain_vmax]
                ),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(
                plotly_vis.plot_map(
                    future_rain, f"FUTURE ({delta_t:+.1f}C)", "rain_annual", custom_range=[0.0, rain_vmax]
                ),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Sensitivity Map")
        st.plotly_chart(
            plotly_sensitivity_map(sens_map, "Rainfall Sensitivity (mm per +1C)"),
            use_container_width=True,
        )
        st.info("Red = rainfall decreases with warming. Blue = rainfall increases.")
    else:
        st.warning("Sensitivity map not found. Run the Colab notebook.")
'''
