"""Docstring citation scan.

Every module under whatif/indices/, whatif/biophysical/, whatif/sectors/,
whatif/economics/, whatif/drivers/downscale.py, whatif/drivers/nex_gddp.py,
whatif/drivers/ssp.py must contain a "Primary source" / "Primary sources"
/ "Citation" phrase in its top-level docstring.

Exits 0 when clean, 1 with a listing when a violation is found.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
_DIR_SCOPES: tuple[Path, ...] = (
    ROOT / "climate_twin" / "whatif" / "indices",
    ROOT / "climate_twin" / "whatif" / "biophysical",
    ROOT / "climate_twin" / "whatif" / "sectors",
    ROOT / "climate_twin" / "whatif" / "economics",
)
_FILE_SCOPES: tuple[Path, ...] = (
    ROOT / "climate_twin" / "whatif" / "drivers" / "downscale.py",
    ROOT / "climate_twin" / "whatif" / "drivers" / "nex_gddp.py",
    ROOT / "climate_twin" / "whatif" / "drivers" / "ssp.py",
    ROOT / "climate_twin" / "whatif" / "drivers" / "analog_features.py",
    ROOT / "climate_twin" / "whatif" / "drivers" / "analogs.py",
    ROOT / "climate_twin" / "whatif" / "drivers" / "perturbation.py",
    ROOT / "climate_twin" / "whatif" / "drivers" / "ensemble_lt.py",
)

# Modules where a citation would be silly (glue __init__, YAML loaders
# whose citations live inline in the yaml, etc.).
_SKIP = {
    "__init__.py",
    "validation_apy.py",         # stub, cites the parquet spec
    "crops.py",                  # loader for cited YAML
    "adaptations.py",            # loader for cited YAML
    "prices.py",                 # loader for cited YAML
    "conftest.py",
}

_TOKENS = ("primary source", "primary sources", "citation", "cited",
           "cite in every")


def _has_citation(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    doc = ast.get_docstring(tree)
    if not doc:
        return False
    low = doc.lower()
    return any(t in low for t in _TOKENS)


def main() -> int:
    misses: list[str] = []
    for scope in _DIR_SCOPES:
        if not scope.exists():
            continue
        for f in sorted(scope.rglob("*.py")):
            if f.name in _SKIP:
                continue
            if not _has_citation(f):
                misses.append(str(f.relative_to(ROOT)))
    for f in _FILE_SCOPES:
        if not f.exists():
            continue
        if not _has_citation(f):
            misses.append(str(f.relative_to(ROOT)))

    if misses:
        print("citation-scan FAILED — modules without a Primary source / Citation:",
              file=sys.stderr)
        for m in misses:
            print(f"  {m}", file=sys.stderr)
        return 1
    print("citation-scan OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
