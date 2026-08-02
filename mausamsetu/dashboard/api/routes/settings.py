"""Settings routes (threshold configuration)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from mausamsetu.dashboard.api.deps import get_thresholds
from mausamsetu.dashboard.api.schemas import ThresholdConfig, ThresholdsResponse

router = APIRouter(tags=["settings"])


@router.get("/settings/thresholds", response_model=ThresholdsResponse)
def thresholds(config: ThresholdConfig = Depends(get_thresholds)) -> ThresholdsResponse:
    return ThresholdsResponse(thresholds=config)
