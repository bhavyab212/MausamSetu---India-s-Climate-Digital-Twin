"""
whatif.scenarios.provenance — reproducibility ledger.

Every scenario run emits a YAML file under ``.whatif_runs/<run_id>.yaml``
containing everything needed to replay it: driver spec, sector levers,
per-layer versions, dataset versions, code hash (git SHA of whatif/),
mask hash, and a config snapshot.

Contract:
    Two runs with the same YAML must return byte-equal results on any
    machine, forever. ``run_from_yaml(path)`` is the replay entry
    point.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config.paths import STREAMLIT_ROOT

IST = timezone(timedelta(hours=5, minutes=30))
RUNS_DIR = STREAMLIT_ROOT / ".whatif_runs"


def _ulid() -> str:
    """Compact time-ordered id (26-char Base32). We use a lightweight
    fallback if the ``ulid-py`` package isn't installed."""
    try:
        import ulid
        return str(ulid.new())
    except Exception:
        # Fallback: 12-hex sortable
        return datetime.now(IST).strftime("%Y%m%d%H%M%S%f")[:20]


def _git_short_hash(target_dir: Path = STREAMLIT_ROOT / "whatif") -> str:
    """Return the short (12-hex) git SHA of the whatif/ tree, plus a
    ``-dirty`` suffix if the tree has uncommitted changes."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(target_dir), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
    try:
        st = subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(target_dir)],
            cwd=str(target_dir.parent), stderr=subprocess.DEVNULL,
        ).decode().strip()
        if st:
            return f"{h}-dirty"
    except Exception:
        pass
    return h


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _spec_to_dict(spec) -> dict:
    """DriverSpec → plain dict (yaml-safe)."""
    d = {
        "mode": spec.mode,
        "var": spec.var,
        "dates": [spec.start.isoformat(), spec.end.isoformat()],
        "quantile": spec.quantile,
        "extras": spec.extras_dict(),
        "region": {
            "kind": spec.region.kind,
            "id": spec.region.id,
            "bbox": list(spec.region.bbox) if spec.region.bbox else None,
            "point": list(spec.region.point) if spec.region.point else None,
        },
    }
    return d


def _dict_to_spec(d: dict):
    """Round-trip a spec dict back to DriverSpec + RegionSpec."""
    from datetime import date as _d
    from ..config.region import RegionSpec
    from ..drivers.driver import DriverSpec
    r = d["region"]
    region = RegionSpec(
        kind=r["kind"], id=r.get("id"),
        bbox=tuple(r["bbox"]) if r.get("bbox") else None,
        point=tuple(r["point"]) if r.get("point") else None,
    )
    return DriverSpec(
        mode=d["mode"], var=d["var"],
        dates=(_d.fromisoformat(d["dates"][0]), _d.fromisoformat(d["dates"][1])),
        region=region,
        quantile=d.get("quantile"),
        extras=tuple((k, v) for k, v in (d.get("extras") or {}).items()),
    )


# ---------------------------------------------------------------------------
@dataclass
class ProvenanceRecord:
    run_id: str
    created_at: str                       # ISO 8601 IST
    driver: dict                          # DriverSpec as plain dict
    levers: dict = field(default_factory=dict)
    layer_versions: dict[str, str] = field(default_factory=dict)
    dataset_versions: dict[str, str] = field(default_factory=dict)
    code_hash: str = ""
    mask_hash: str = ""
    config_snapshot: dict = field(default_factory=dict)
    layers_ran: list[dict] = field(default_factory=list)
    outputs_summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
def record_open(spec, levers: dict | None = None) -> ProvenanceRecord:
    """Open a new provenance record for a scenario run."""
    from .. import config as _cfg
    from ..config.region import region_mask_array

    mask_hex = ""
    try:
        m = region_mask_array(spec.region)
        if m is not None:
            mask_hex = _sha256_hex(m.tobytes())
    except Exception:
        pass

    config_snapshot = {
        "GRID_DEG": _cfg.GRID_DEG,
        "N_LAT": _cfg.N_LAT,
        "N_LON": _cfg.N_LON,
        "LAT_MIN": _cfg.LAT_MIN,
        "LAT_MAX": _cfg.LAT_MAX,
        "LON_MIN": _cfg.LON_MIN,
        "LON_MAX": _cfg.LON_MAX,
        "TZ": _cfg.TZ,
        "IMD_SENTINELS": list(_cfg.IMD_SENTINELS),
    }
    return ProvenanceRecord(
        run_id=_ulid(),
        created_at=datetime.now(IST).isoformat(),
        driver=_spec_to_dict(spec),
        levers=dict(levers or {}),
        code_hash=_git_short_hash(),
        mask_hash=mask_hex,
        config_snapshot=config_snapshot,
    )


def record_add_layer(
    rec: ProvenanceRecord,
    name: str,
    version: str,
    inputs_summary: dict | None = None,
) -> None:
    """Called at every layer boundary. Records what ran + what it fed on."""
    rec.layer_versions[name] = version
    rec.layers_ran.append({
        "name": name,
        "version": version,
        "inputs": dict(inputs_summary or {}),
    })


def record_close(rec: ProvenanceRecord, outputs_summary: dict | None = None) -> Path:
    """Finalise + persist the record to YAML. Returns the on-disk path."""
    if outputs_summary is not None:
        rec.outputs_summary.update(outputs_summary)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{rec.run_id}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(rec), f, sort_keys=False)
    return path


def run_from_yaml(path: str | Path):
    """Replay entry point. Reads a YAML, reconstructs the DriverSpec +
    levers, and re-invokes ``run_scenario`` with them. Returns the
    ``ScenarioResult`` — bitwise-equal to the original if the code hash
    hasn't drifted."""
    from .engine import run_scenario
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    spec = _dict_to_spec(doc["driver"])
    return run_scenario(spec, levers=doc.get("levers") or None)
