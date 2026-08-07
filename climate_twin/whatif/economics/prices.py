"""
whatif.economics.prices — YAML-backed price registry.

Loads ``prices.yaml`` once, validates it against a Pydantic v2 schema,
and exposes a ``PriceSet`` dataclass to the rest of the economics
layer. The SHA-256 of the YAML lands in every ``EconomicOutcome``'s
provenance so replays can detect drift.

Primary sources (see prices.yaml citations for per-crop specifics):
    * CACP MSP notifications — Ministry of Agriculture, Government of India.
    * DES Cost of Cultivation of Principal Crops (base 2019-20).
    * FMC-Mkt mandi convention for moisture + transport discounts.
    * Agmarknet — optional live mandi feed accessed via
      :func:`load_agmarknet_series`; not enabled by default.

Rule (Part 4, Golden rule 2):
    *Prices are a lever, not a fact.* Default MSP is what the model
    uses; live mandi prices widen the uncertainty band because they
    add price-forecast noise on top of climate-forecast noise.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field, PositiveFloat, model_validator

_YAML_PATH = Path(__file__).resolve().parent / "prices.yaml"


# ── Pydantic schema ─────────────────────────────────────────────────
class _CostBlock(BaseModel):
    default: PositiveFloat


class _PriceCrop(BaseModel):
    msp_inr_per_qt: dict[str, PositiveFloat]
    cost_of_cultivation_inr_per_ha: _CostBlock
    moisture_discount_pct: float = Field(ge=0, lt=0.5)
    transport_marketing_pct: float = Field(ge=0, lt=0.5)
    citation: dict[str, str]

    @model_validator(mode="after")
    def _sanity(self) -> "_PriceCrop":
        if not self.msp_inr_per_qt:
            raise ValueError("msp_inr_per_qt must have at least one season")
        total_disc = self.moisture_discount_pct + self.transport_marketing_pct
        if total_disc >= 0.5:
            raise ValueError(
                f"moisture + transport discount {total_disc:.2f} ≥ 0.5 is "
                "unphysical for grain mandi trade"
            )
        return self


# ── Public dataclass ─────────────────────────────────────────────────
@dataclass(frozen=True)
class PriceSet:
    """Immutable price parameter set for one crop.

    Emit-time-safe: pickle-hashable, deterministic serialisation.
    Constructed by :func:`load_prices`."""
    crop_key: str
    msp_by_season: dict[str, float]
    cost_of_cultivation_inr_per_ha: float
    moisture_discount_pct: float
    transport_marketing_pct: float
    citation: dict[str, str]
    registry_version: str
    registry_sha256: str

    def msp_for(self, season: str) -> float:
        """MSP in ₹/qt for a named season (e.g., "2024-25"). Raises
        ``KeyError`` if the season is not in the registry — do not fall
        through silently to the latest available."""
        if season not in self.msp_by_season:
            raise KeyError(
                f"season {season!r} not in MSP registry for {self.crop_key} "
                f"(known: {sorted(self.msp_by_season)})"
            )
        return float(self.msp_by_season[season])

    def latest_msp(self) -> tuple[str, float]:
        """Return the most-recent (season, msp) tuple by season key sort.
        Season strings follow ``"YYYY-YY"`` so lexical sort == temporal."""
        latest = sorted(self.msp_by_season)[-1]
        return latest, float(self.msp_by_season[latest])


# ── YAML loader ─────────────────────────────────────────────────────
def _yaml_bytes() -> bytes:
    return _YAML_PATH.read_bytes()


def _yaml_sha256() -> str:
    return hashlib.sha256(_yaml_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    doc = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "version" not in doc:
        raise ValueError("prices.yaml is missing top-level 'version'")
    return doc


def registry_version() -> str:
    return str(_load_yaml()["version"])


def registry_sha256() -> str:
    return _yaml_sha256()


def list_priced_crops() -> list[str]:
    doc = _load_yaml()
    return [k for k in doc.keys() if k != "version"]


def load_prices(crop_key: str) -> PriceSet:
    """Return the validated :class:`PriceSet` for ``crop_key``."""
    doc = _load_yaml()
    if crop_key == "version" or crop_key not in doc:
        raise KeyError(
            f"crop {crop_key!r} not in prices.yaml "
            f"(known: {list_priced_crops()})"
        )
    validated = _PriceCrop.model_validate(doc[crop_key])
    return PriceSet(
        crop_key=crop_key,
        msp_by_season={str(k): float(v) for k, v in validated.msp_inr_per_qt.items()},
        cost_of_cultivation_inr_per_ha=float(validated.cost_of_cultivation_inr_per_ha.default),
        moisture_discount_pct=float(validated.moisture_discount_pct),
        transport_marketing_pct=float(validated.transport_marketing_pct),
        citation=dict(validated.citation),
        registry_version=registry_version(),
        registry_sha256=registry_sha256(),
    )


# ── Optional live mandi feed hook (Part 7) ──────────────────────────
def load_agmarknet_series(
    crop: str, market: str, dates: tuple[date, date],
) -> pd.Series:
    """Fetch modal prices from Agmarknet for ``crop`` @ ``market`` between
    ``dates``. Not wired by default — the caller must set
    ``use_mandi_prices=True`` on the valuation function and provide the
    network credentials.

    This raises until the API client is wired in Part 7. The intent is
    to fail loud so a scenario that thinks it's using live prices but
    silently fell back to MSP is impossible.
    """
    raise NotImplementedError(
        "Agmarknet live-mandi client lands in Part 7. Set "
        "use_mandi_prices=False (default) to run on MSP only."
    )
