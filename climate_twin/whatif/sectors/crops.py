"""
whatif.sectors.crops — YAML-backed crop parameter registry.

Loads ``crops.yaml`` (sibling of this file) once, validates it against a
Pydantic v2 schema, and exposes a ``Crop`` dataclass to the rest of the
agriculture sector. The SHA-256 of the YAML file lands in every scenario
provenance record so replays can detect drift.

Rule (Part 3, Golden rule #1):
    No crop constant (Kc, Ky, stage length, root depth, depletion p,
    base T, cap T, flowering-stress T, Ymax) may appear inline in
    Python. Everything lives in the YAML with a citation.

Citations for the numbers themselves live inline in ``crops.yaml``
(each crop entry ships a ``citation`` block).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveFloat, model_validator

_YAML_PATH = Path(__file__).resolve().parent / "crops.yaml"


# ── Pydantic schema ─────────────────────────────────────────────────
class _StagesDays(BaseModel):
    ini: int = Field(gt=0)
    dev: int = Field(gt=0)
    mid: int = Field(gt=0)
    late: int = Field(gt=0)


class _Kc(BaseModel):
    ini: float = Field(gt=0)
    mid: float = Field(gt=0)
    end: float = Field(gt=0)


class _KyStage(BaseModel):
    ini: float = Field(ge=0)
    dev: float = Field(ge=0)
    mid: float = Field(ge=0)
    late: float = Field(ge=0)


class _RootDepth(BaseModel):
    ini: PositiveFloat
    max: PositiveFloat


class _CropSchema(BaseModel):
    common_name: str
    citation: dict[str, str]
    stages_days: _StagesDays
    kc: _Kc
    ky_overall: PositiveFloat
    ky_stage: _KyStage
    root_depth_m: _RootDepth
    depletion_p: float = Field(gt=0, lt=1)
    t_base_c: float
    t_cap_c: float
    t_flower_stress_c: float
    flower_stage: str
    ymax_t_per_ha: PositiveFloat
    msp_inr_per_qt: float = Field(ge=0)

    @model_validator(mode="after")
    def _sanity(self) -> "_CropSchema":
        if self.t_cap_c <= self.t_base_c:
            raise ValueError("t_cap_c must exceed t_base_c")
        if self.root_depth_m.max <= self.root_depth_m.ini:
            raise ValueError("root_depth_m.max must exceed root_depth_m.ini")
        if self.flower_stage not in ("ini", "dev", "mid", "late"):
            raise ValueError("flower_stage must be one of ini/dev/mid/late")
        return self


# ── Public dataclass ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Crop:
    """Immutable crop parameter set. Constructed by :func:`load_crop`."""
    key: str
    common_name: str
    citation: dict[str, str]
    stages_days: dict[str, int]
    kc: dict[str, float]
    ky_overall: float
    ky_stage: dict[str, float]
    root_depth_m: dict[str, float]
    depletion_p: float
    t_base_c: float
    t_cap_c: float
    t_flower_stress_c: float
    flower_stage: str
    ymax_t_per_ha: float
    msp_inr_per_qt: float
    registry_version: str
    registry_sha256: str        # yaml SHA — appears in every provenance record

    @property
    def total_days(self) -> int:
        return sum(self.stages_days.values())


# ── YAML loader (cached) ─────────────────────────────────────────────
def _yaml_bytes() -> bytes:
    return _YAML_PATH.read_bytes()


def _yaml_sha256() -> str:
    return hashlib.sha256(_yaml_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    doc = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "version" not in doc:
        raise ValueError("crops.yaml is missing top-level 'version'")
    return doc


def registry_version() -> str:
    return str(_load_yaml()["version"])


def registry_sha256() -> str:
    return _yaml_sha256()


def list_crops() -> list[str]:
    doc = _load_yaml()
    return [k for k in doc.keys() if k != "version"]


def load_crop(key: str) -> Crop:
    doc = _load_yaml()
    if key == "version" or key not in doc:
        raise KeyError(
            f"crop {key!r} not in crops.yaml (known: {list_crops()})"
        )
    validated = _CropSchema.model_validate(doc[key])
    return Crop(
        key=key,
        common_name=validated.common_name,
        citation=dict(validated.citation),
        stages_days=validated.stages_days.model_dump(),
        kc=validated.kc.model_dump(),
        ky_overall=float(validated.ky_overall),
        ky_stage=validated.ky_stage.model_dump(),
        root_depth_m=validated.root_depth_m.model_dump(),
        depletion_p=float(validated.depletion_p),
        t_base_c=float(validated.t_base_c),
        t_cap_c=float(validated.t_cap_c),
        t_flower_stress_c=float(validated.t_flower_stress_c),
        flower_stage=validated.flower_stage,
        ymax_t_per_ha=float(validated.ymax_t_per_ha),
        msp_inr_per_qt=float(validated.msp_inr_per_qt),
        registry_version=registry_version(),
        registry_sha256=registry_sha256(),
    )
