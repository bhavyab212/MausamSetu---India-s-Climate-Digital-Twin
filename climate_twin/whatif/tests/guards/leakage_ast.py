"""Leakage-guard AST scan.

Refuse any ``.fit(...)`` call under ``whatif/indices/`` or
``whatif/economics/backtest.py`` whose argument list references
``VALID_YEARS``. Fits must be trained on ``TRAIN_YEARS`` only —
Part 2's assert_train_only enforces at runtime; this scan catches the
bug at commit time so a PR with silent leakage never merges.

Exits 0 when clean, 1 with a listing when a violation is found.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
_SCOPES: tuple[Path, ...] = (
    ROOT / "climate_twin" / "whatif" / "indices",
    ROOT / "climate_twin" / "whatif" / "economics" / "backtest.py",
)


def _references_valid_years(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "VALID_YEARS":
            return True
        if (isinstance(child, ast.Attribute)
                and isinstance(child.attr, str)
                and child.attr == "VALID_YEARS"):
            return True
    return False


def _scan_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [f"{path}: syntax error {e}"]
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_fit = False
        if isinstance(func, ast.Attribute) and func.attr in ("fit", "fit_transform"):
            is_fit = True
        elif isinstance(func, ast.Name) and func.id in ("fit_spi", "fit_spei", "fit_r95p"):
            is_fit = True
        if not is_fit:
            continue
        if _references_valid_years(node):
            hits.append(f"{path}:{node.lineno}: fit call references VALID_YEARS")
    return hits


def main() -> int:
    all_hits: list[str] = []
    for scope in _SCOPES:
        if scope.is_dir():
            files = sorted(scope.rglob("*.py"))
        elif scope.is_file():
            files = [scope]
        else:
            continue
        for f in files:
            if "guards" in f.parts:
                continue
            all_hits.extend(_scan_file(f))

    if all_hits:
        print("Leakage-guard FAILED — fits must be TRAIN_YEARS-only:",
              file=sys.stderr)
        for line in all_hits:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("leakage-guard OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
