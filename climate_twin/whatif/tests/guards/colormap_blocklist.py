"""Colormap-blocklist static scan.

Any file under ``whatif/ui/`` or ``whatif/report/`` that mentions
``jet``, ``hsv``, ``rainbow``, or ``nipy_spectral`` in a way that
looks like a colorscale assignment fails the build (Rule 4 of Part 6).

We scan for the specific patterns:
    colorscale="jet"       # or single quotes
    "colorscale": "jet"
    cmap="jet"

so incidental occurrences of the word "jet" in a docstring are not
flagged. Exits 0 when clean, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
_SCOPES: tuple[Path, ...] = (
    ROOT / "climate_twin" / "whatif" / "ui",
    ROOT / "climate_twin" / "whatif" / "report",
)
_BLOCKED = ("jet", "hsv", "rainbow", "nipy_spectral", "gnuplot")

_PATTERNS = [
    re.compile(rf"""colorscale\s*=\s*["']({name})["']""")
    for name in _BLOCKED
] + [
    re.compile(rf"""["']colorscale["']\s*:\s*["']({name})["']""")
    for name in _BLOCKED
] + [
    re.compile(rf"""cmap\s*=\s*["']({name})["']""")
    for name in _BLOCKED
]


def _scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for pat in _PATTERNS:
            m = pat.search(line)
            if m:
                hits.append(
                    f"{path}:{i}: blocked colormap {m.group(1)!r} — "
                    f"{line.strip()}"
                )
    return hits


def main() -> int:
    all_hits: list[str] = []
    for scope in _SCOPES:
        if not scope.exists():
            continue
        for f in sorted(scope.rglob("*.py")):
            all_hits.extend(_scan_file(f))
    if all_hits:
        print("colormap-blocklist FAILED (Rule 4 · no jet / hsv / rainbow):",
              file=sys.stderr)
        for line in all_hits:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("colormap-blocklist OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
