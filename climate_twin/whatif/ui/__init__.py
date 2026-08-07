"""
whatif.ui — Streamlit component composers for the What-If page.

Nothing here uses ``st.set_page_config`` (only the page file at
``pages/30_What_If.py`` does that). This subpackage exposes render
functions ``render_short_term(...)`` and ``render_long_term(...)`` that
the page calls; both accept a Scenario object and mutate no globals.

# TODO(Part 6): sidebar controls, short-term panel, long-term panel.
"""
