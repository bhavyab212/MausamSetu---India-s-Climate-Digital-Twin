"""pytest config for the whatif test suite."""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: opt-in tests that walk the full engine or the Streamlit "
        "AppTest harness. Run with `-m slow` to include.",
    )
