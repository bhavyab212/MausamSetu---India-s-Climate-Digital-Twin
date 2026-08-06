"""
train.registry.store — filesystem-first zone-aware model registry.

Layout::

    climate_twin/train/registry/models/<region>/<name>/
        weights.pt            — torch.save({model_state_dict, config, zone_mask_sig, ...})
        meta.json             — human-readable metadata
        tier4.json            — final round significance report
        tier3_heatmap.png     — cross-product visual

Load-time assertions (fail loudly, refuse to load):
  * zone_mask_sig must match on-disk ``climate_twin.regions.zone_mask.sha256``
  * variables must match the caller's expected list
  * manifest_sig (optional pin) must match on-disk cube

Lineage:
  * meta.json carries ``parent_name`` — the model this run was resumed from.
  * ``list_models`` returns entries in the same shape the UI expects.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch


IST = timezone(timedelta(hours=5, minutes=30))
_HERE = Path(__file__).resolve().parent
MODELS_ROOT = _HERE / "models"


class RegistryModelIncompatible(RuntimeError):
    """Raised when a stored model can't be loaded under the caller's contract."""


def _safe(name: str) -> str:
    return (
        "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_")
        or "model"
    )[:64]


class ZoneAwareRegistry:
    """Read-only lookups + delete. The trainer writes its own checkpoints
    directly under ``models_root``; the registry is just a browse layer
    over the resulting folder tree."""

    def __init__(self, models_root: Path | None = None):
        self.models_root = Path(models_root) if models_root is not None else MODELS_ROOT
        self.models_root.mkdir(parents=True, exist_ok=True)

    # ── Layout helpers ─────────────────────────────────────────
    def model_dir(self, name: str, region: str) -> Path:
        return self.models_root / _safe(region) / _safe(name)

    def _find_weights(self, folder: Path) -> Path | None:
        w = folder / "weights.pt"
        if w.exists():
            return w
        for pat in ("*.pt", "*.pth"):
            hits = sorted(folder.glob(pat))
            if hits:
                return hits[0]
        return None

    def _read_meta(self, folder: Path) -> dict[str, Any] | None:
        p = folder / "meta.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        # Fall back to checkpoint contents
        w = self._find_weights(folder)
        if w is None:
            return None
        try:
            ck = torch.load(w, map_location="cpu", weights_only=False)
            return {k: v for k, v in ck.items() if k != "model_state_dict"}
        except Exception:
            return None

    # ── Listing / lookup ───────────────────────────────────────
    def list_models(self, region: str | None = None) -> list[dict[str, Any]]:
        regions = ([region] if region else
                   [p.name for p in self.models_root.iterdir() if p.is_dir()])
        out: list[dict[str, Any]] = []
        for reg in regions:
            rdir = self.models_root / _safe(reg)
            if not rdir.exists():
                continue
            for folder in sorted(rdir.iterdir()):
                if not folder.is_dir():
                    continue
                if self._find_weights(folder) is None:
                    continue
                m = self._read_meta(folder)
                if m:
                    m.setdefault("name", folder.name)
                    m.setdefault("region", reg)
                    m.setdefault("path", str(folder))
                    out.append(m)
        out.sort(key=lambda m: m.get("best_zone_weighted_rmse", float("inf")))
        return out

    def get_model(self, name: str, region: str) -> dict[str, Any] | None:
        folder = self.model_dir(name, region)
        if not folder.exists() or self._find_weights(folder) is None:
            return None
        m = self._read_meta(folder)
        if m:
            m.setdefault("name", name)
            m.setdefault("region", region)
            m.setdefault("path", str(folder))
        return m

    # ── Load with contract enforcement ─────────────────────────
    def load_into(
        self,
        name: str,
        region: str,
        model: torch.nn.Module,
        *,
        expected_variables: Sequence[str] | None = None,
        expected_zone_mask_sig: str | None = None,
        expected_manifest_sig: str | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load ``model`` state from the stored checkpoint. Raises
        :class:`RegistryModelIncompatible` on any contract mismatch."""
        folder = self.model_dir(name, region)
        w = self._find_weights(folder)
        if w is None:
            raise FileNotFoundError(f"no weights under {folder}")
        ck = torch.load(w, map_location="cpu", weights_only=False)

        saved_vars = list(ck.get("variables", []) or [])
        saved_zsig = ck.get("zone_mask_sig", "") or ""
        saved_msig = ck.get("manifest_sig", "") or ""

        if strict:
            if expected_variables is not None and list(expected_variables) != saved_vars:
                raise RegistryModelIncompatible(
                    f"{name}: variable list mismatch — expected {list(expected_variables)!r}, "
                    f"saved {saved_vars!r}"
                )
            if expected_zone_mask_sig is not None and expected_zone_mask_sig != saved_zsig:
                raise RegistryModelIncompatible(
                    f"{name}: zone_mask_sig mismatch — expected {expected_zone_mask_sig!r}, "
                    f"saved {saved_zsig!r}"
                )
            if expected_manifest_sig is not None and expected_manifest_sig != saved_msig:
                raise RegistryModelIncompatible(
                    f"{name}: manifest_sig mismatch — expected {expected_manifest_sig!r}, "
                    f"saved {saved_msig!r}"
                )

        state = ck["model_state_dict"] if "model_state_dict" in ck else ck
        model.load_state_dict(state)
        return ck

    # ── Delete ─────────────────────────────────────────────────
    def delete(self, name: str, region: str) -> bool:
        folder = self.model_dir(name, region)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            return True
        return False


@lru_cache(maxsize=1)
def get_registry() -> ZoneAwareRegistry:
    return ZoneAwareRegistry()
