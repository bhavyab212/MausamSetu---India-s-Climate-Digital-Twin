"""
registry.py — filesystem-first, one-folder-per-model registry.

Layout (easy to browse + drop files into):

    climate_twin/models/<region>/<model_name>/
        weights.pt    # torch checkpoint: {model_state_dict, arch, metrics, ...}
        meta.json     # human-readable metadata (arch, metrics, epochs, rounds, ...)
        *             # any extra files you paste are ignored but preserved

Discovery scans the folders directly, so if you paste ANY .pt into a
models/<region>/<name>/ folder (even without meta.json), it will be found:
list_models() synthesizes metadata from the checkpoint. weights are loaded from
weights.pt if present, else the first *.pt in the folder.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Sequence

import torch

from .checkpoints import (
    CheckpointVariableMismatch as ModelVariableMismatch,
    check_variables as _check_variables,
    _norm_vars,
)

IST = timezone(timedelta(hours=5, minutes=30))
_DIR = Path(__file__).resolve().parent            # climate_twin/training
MODELS_DIR = _DIR.parent / "models"               # climate_twin/models   ← browse here
_LEGACY_CKPT_DIR = _DIR / "checkpoints"           # old flat model_<region>_<name>.pt


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)).strip("_")[:48] or "model"


def model_dir(name: str, region: str) -> Path:
    return MODELS_DIR / _safe(region) / _safe(name)


def model_ckpt_path(name: str, region: str) -> Path:
    """Canonical weights path inside the model's folder."""
    return model_dir(name, region) / "weights.pt"


def _find_weights(folder: Path) -> Path | None:
    """weights.pt if present, else the first *.pt / *.pth in the folder."""
    w = folder / "weights.pt"
    if w.exists():
        return w
    for pat in ("*.pt", "*.pth"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _migrate_legacy():
    """Move old flat checkpoints (model_<region>_<name>.pt) into model folders."""
    if not _LEGACY_CKPT_DIR.exists():
        return
    for p in _LEGACY_CKPT_DIR.glob("model_*.pt"):
        stem = p.stem[len("model_"):]            # <region>_<name>
        region = "cauvery" if stem.startswith("cauvery_") else ("india" if stem.startswith("india_") else None)
        if not region:
            continue
        name = stem[len(region) + 1:]
        dest = model_ckpt_path(name, region)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, dest)
            ck = torch.load(dest, map_location="cpu", weights_only=False)
            _write_meta(name, region, ck.get("arch", {}) or {}, ck.get("metrics", {}) or {},
                        epochs=ck.get("epochs", 0) or 0, rounds=ck.get("rounds", 0) or 0,
                        notes="migrated from legacy checkpoint")
        except Exception:
            pass


def init_registry():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy()


def _write_meta(name, region, arch, metrics, epochs, rounds, notes="",
                created_at=None, parent_name="",
                variables: Sequence[str] | None = None,
                manifest_sig: str = "",
                grid_shape: Sequence[int] | None = None):
    d = model_dir(name, region)
    d.mkdir(parents=True, exist_ok=True)
    now = datetime.now(IST).isoformat()
    meta = {
        "name": name, "region": region, "arch": arch, "metrics": metrics,
        "epochs_trained": int(epochs), "rounds_trained": int(rounds),
        "created_at": created_at or now, "updated_at": now, "notes": notes,
        "parent_name": parent_name,
        "variables": _norm_vars(variables),
        "manifest_sig": manifest_sig or "",
        "grid_shape": list(grid_shape) if grid_shape else [],
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def save_model(name, region, model, arch: dict, metrics: dict,
               epochs_add: int, rounds_add: int, notes: str = "", parent_name: str = "",
               variables: Sequence[str] | None = None,
               manifest_sig: str = "",
               grid_shape: Sequence[int] | None = None) -> Path:
    """Persist weights + meta.json into models/<region>/<name>/.

    Accumulates counters and stores parent lineage. Also records the
    ordered ``variables`` list, the ``manifest_sig`` of the cube the model
    was trained on, and the ``grid_shape``. These three fields are what the
    load-time contract checks (see ``load_into``)."""
    d = model_dir(name, region)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "weights.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch": arch, "name": name, "region": region,
        "metrics": metrics, "saved_at": datetime.now(IST).isoformat(),
        "parent_name": parent_name,
        "variables": _norm_vars(variables),
        "manifest_sig": manifest_sig or "",
        "grid_shape": list(grid_shape) if grid_shape else [],
        "checkpoint_format": 2,
    }, path)

    prev = get_model(name, region)
    ep = (prev["epochs_trained"] if prev else 0) + int(epochs_add)
    rd = (prev["rounds_trained"] if prev else 0) + int(rounds_add)
    created = prev["created_at"] if prev else None
    parent = parent_name or (prev.get("parent_name", "") if prev else "")
    # Preserve the previously saved variables/sig unless new values are supplied
    v_final = _norm_vars(variables) or (prev.get("variables", []) if prev else [])
    sig_final = (manifest_sig or "") or (prev.get("manifest_sig", "") if prev else "")
    shape_final = (list(grid_shape) if grid_shape else []) or (prev.get("grid_shape", []) if prev else [])
    _write_meta(name, region, arch, metrics, ep, rd, notes=notes,
                created_at=created, parent_name=parent,
                variables=v_final, manifest_sig=sig_final, grid_shape=shape_final)
    return path


def _load_meta_or_infer(folder: Path, region: str) -> dict | None:
    name = folder.name
    meta_p = folder / "meta.json"
    if meta_p.exists():
        try:
            m = json.loads(meta_p.read_text())
            m.setdefault("name", name)
            m.setdefault("region", region)
            m.setdefault("arch", {})
            m.setdefault("metrics", {})
            m.setdefault("epochs_trained", 0)
            m.setdefault("rounds_trained", 0)
            m.setdefault("parent_name", "")
            m.setdefault("variables", [])
            m.setdefault("manifest_sig", "")
            m.setdefault("grid_shape", [])
            return m
        except Exception:
            pass
    # No meta.json — infer from a checkpoint file (supports pasted models).
    w = _find_weights(folder)
    if w is None:
        return None
    arch, metrics = {}, {}
    parent_name = ""
    variables: list[str] = []
    manifest_sig = ""
    grid_shape: list[int] = []
    try:
        ck = torch.load(w, map_location="cpu", weights_only=False)
        if isinstance(ck, dict):
            arch = ck.get("arch", {}) or {}
            metrics = ck.get("metrics", {}) or {}
            parent_name = ck.get("parent_name", "") or ""
            variables = list(ck.get("variables", []) or [])
            manifest_sig = ck.get("manifest_sig", "") or ""
            grid_shape = list(ck.get("grid_shape", []) or [])
    except Exception:
        pass
    return {"name": name, "region": region, "arch": arch, "metrics": metrics,
            "epochs_trained": 0, "rounds_trained": 0, "parent_name": parent_name,
            "variables": variables, "manifest_sig": manifest_sig, "grid_shape": grid_shape,
            "created_at": "", "updated_at": "", "notes": "discovered from file"}


def get_lineage_tree(region: str) -> list[dict[str, Any]]:
    """Build parent -> child lineage graph for all models in region."""
    models = list_models(region)
    model_map = {m["name"]: m for m in models}
    
    roots = []
    children = {m["name"]: [] for m in models}
    for m in models:
        parent = m.get("parent_name", "")
        if parent and parent in model_map:
            children[parent].append(m["name"])
        else:
            roots.append(m["name"])
            
    def _build_node(name, depth=0):
        m = model_map[name]
        node = {
            "name": name,
            "depth": depth,
            "parent": m.get("parent_name", ""),
            "metrics": m.get("metrics", {}),
            "notes": m.get("notes", ""),
            "children": [_build_node(child, depth + 1) for child in children.get(name, [])]
        }
        return node
        
    return [_build_node(r) for r in roots]


def list_models(region: str | None = None) -> list[dict[str, Any]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    regions = [region] if region else [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    out = []
    for reg in regions:
        rdir = MODELS_DIR / _safe(reg)
        if not rdir.exists():
            continue
        for folder in sorted(rdir.iterdir()):
            if not folder.is_dir():
                continue
            if _find_weights(folder) is None:
                continue
            m = _load_meta_or_infer(folder, reg)
            if m:
                out.append(m)
    out.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return out


def get_model(name: str, region: str) -> dict | None:
    folder = model_dir(name, region)
    if not folder.exists() or _find_weights(folder) is None:
        return None
    return _load_meta_or_infer(folder, region)


def load_into(
    name: str,
    region: str,
    model,
    *,
    expected_variables: Sequence[str] | None = None,
    strict: bool = True,
) -> tuple[bool, dict | None]:
    """Load a model folder's weights into an instance.

    Supports our format, ``{state_dict: ...}``, or a raw state_dict from a
    pasted file. If ``expected_variables`` is supplied and the stored
    ``variables`` list disagrees, raises ``ModelVariableMismatch`` (unless
    ``strict=False``, in which case legacy models without a saved variable
    list are accepted with a warning message on the returned dict).

    Returns ``(ok, checkpoint_dict_or_None)`` where the dict — when present —
    also carries a ``variable_check`` message describing what was verified.
    """
    folder = model_dir(name, region)
    w = _find_weights(folder)
    if w is None:
        return False, None
    ck = torch.load(w, map_location="cpu", weights_only=False)
    saved_vars = ck.get("variables", []) if isinstance(ck, dict) else []
    ok, msg = _check_variables(expected_variables, saved_vars, strict=strict)
    if not ok:
        raise ModelVariableMismatch(f"{name} [{region}]: {msg}")

    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        sd = ck
    else:
        return False, ck
    try:
        model.load_state_dict(sd)
        if isinstance(ck, dict):
            ck["variable_check"] = msg
        return True, (ck if isinstance(ck, dict) else None)
    except Exception:
        return False, (ck if isinstance(ck, dict) else None)


def delete_model(name: str, region: str):
    folder = model_dir(name, region)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)

