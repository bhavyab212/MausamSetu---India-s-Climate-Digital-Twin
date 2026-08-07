"""
whatif.sectors.adaptations — YAML-backed adaptation option registry.

Every option has three ingredients:
    * what it does biophysically (short description),
    * what it costs (cited capex, opex, effective years),
    * side-effects and CO2 flag for provenance.

Rule 8 (Part 7): options without a cited cost land here with
``cost_complete = False`` and are skipped by :func:`whatif.economics.
npv.adaptation_npv` unless the caller explicitly opts in.

Version: ``adaptations-v1``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveFloat, model_validator

ADAPTATIONS_VERSION = "adaptations-v1"
_YAML_PATH = Path(__file__).resolve().parent / "adaptations.yaml"


class _CitedNumber(BaseModel):
    value: float = Field(ge=0)
    citation: str


class _OptionSchema(BaseModel):
    common_name: str
    sector: str
    applies_to_crops: list[str] | None = None
    applies_to: str | None = None
    biophysical_effect: str
    capex_inr_per_ha: _CitedNumber | None = None
    capex_inr_per_kwh: _CitedNumber | None = None
    opex_inr_per_ha_per_yr: _CitedNumber | None = None
    opex_pct_of_capex_per_yr: float | None = Field(default=None, ge=0, le=1)
    effective_years: int = Field(ge=1)
    co2_kg_per_ha_per_yr: float = 0.0
    side_effects: str | None = None

    @model_validator(mode="after")
    def _check_cost(self) -> "_OptionSchema":
        has_capex = (self.capex_inr_per_ha is not None
                     or self.capex_inr_per_kwh is not None)
        if not has_capex:
            raise ValueError(
                "adaptation option must supply capex_inr_per_ha or "
                "capex_inr_per_kwh (with a citation)"
            )
        return self


@dataclass(frozen=True)
class AdaptationOption:
    key: str
    common_name: str
    sector: str
    applies_to_crops: tuple[str, ...]
    biophysical_effect: str
    capex_inr_per_ha: float                # 0 for non-ha units
    capex_inr_per_kwh: float
    opex_inr_per_ha_per_yr: float
    opex_pct_of_capex_per_yr: float
    effective_years: int
    co2_kg_per_ha_per_yr: float
    side_effects: str
    capex_citation: str
    opex_citation: str
    cost_complete: bool
    registry_version: str
    registry_sha256: str


def _yaml_sha256() -> str:
    return hashlib.sha256(_YAML_PATH.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    doc = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "version" not in doc:
        raise ValueError("adaptations.yaml missing top-level 'version'")
    return doc


def registry_version() -> str:
    return str(_load_yaml()["version"])


def registry_sha256() -> str:
    return _yaml_sha256()


def list_adaptations() -> list[str]:
    return list(_load_yaml()["options"].keys())


class MissingCostCitation(RuntimeError):
    """Raised by NPV when a selected adaptation lacks a cited cost."""


def load_adaptation(key: str) -> AdaptationOption:
    doc = _load_yaml()
    opts = doc.get("options", {})
    if key not in opts:
        raise KeyError(f"adaptation {key!r} not in registry "
                        f"(known: {list(opts)})")
    validated = _OptionSchema.model_validate(opts[key])
    capex_ha = 0.0
    capex_kwh = 0.0
    capex_citation = ""
    if validated.capex_inr_per_ha:
        capex_ha = float(validated.capex_inr_per_ha.value)
        capex_citation = validated.capex_inr_per_ha.citation
    if validated.capex_inr_per_kwh:
        capex_kwh = float(validated.capex_inr_per_kwh.value)
        capex_citation = capex_citation or validated.capex_inr_per_kwh.citation

    opex_ha = 0.0
    opex_citation = ""
    if validated.opex_inr_per_ha_per_yr:
        opex_ha = float(validated.opex_inr_per_ha_per_yr.value)
        opex_citation = validated.opex_inr_per_ha_per_yr.citation

    cost_complete = bool(
        capex_citation and (opex_citation or opex_ha == 0.0
                             or validated.opex_pct_of_capex_per_yr is not None)
    )

    return AdaptationOption(
        key=key,
        common_name=validated.common_name,
        sector=validated.sector,
        applies_to_crops=tuple(validated.applies_to_crops or ()),
        biophysical_effect=validated.biophysical_effect.strip(),
        capex_inr_per_ha=capex_ha,
        capex_inr_per_kwh=capex_kwh,
        opex_inr_per_ha_per_yr=opex_ha,
        opex_pct_of_capex_per_yr=float(validated.opex_pct_of_capex_per_yr or 0.0),
        effective_years=int(validated.effective_years),
        co2_kg_per_ha_per_yr=float(validated.co2_kg_per_ha_per_yr or 0.0),
        side_effects=str(validated.side_effects or ""),
        capex_citation=capex_citation,
        opex_citation=opex_citation,
        cost_complete=cost_complete,
        registry_version=registry_version(),
        registry_sha256=registry_sha256(),
    )
