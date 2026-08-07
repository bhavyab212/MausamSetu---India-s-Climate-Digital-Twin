"""
pages/_archive/ — retired page bodies, NOT registered by Streamlit.

Streamlit's multipage router auto-discovers files matching
``pages/*.py`` at the pages-root level. Anything under a subdirectory
(``_archive/`` here) is ignored by the router, which lets us keep
previous page code alongside the current app without accidentally
registering it as a duplicate tab.

Add a note to this file when a page is archived: what page it was,
when it was retired, and where the replacement lives.

Archive log:

    2026-08-07  whatif_old.py
        Retired ``_render_tab3`` ("±1°C What-If Storyline") from
        ``climate_twin/app_v2.py``.
        Replacement: ``pages/30_What_If.py`` + ``climate_twin/whatif/``.
"""
