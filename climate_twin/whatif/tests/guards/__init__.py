"""whatif.tests.guards — CI-time static scans.

Each script exits non-zero when a rule is broken and prints the
offending file / line to stderr. Pre-commit + GitHub Actions run
these on every push.
"""
