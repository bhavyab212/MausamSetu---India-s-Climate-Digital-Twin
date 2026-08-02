"""Dependency helpers for process-scoped MausamSetu resources."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import xarray as xr
from fastapi import HTTPException, Request

from mausamsetu.model.forecaster import MausamSetuForecaster

ApiCache = dict[str, Any]


def get_model(request: Request) -> MausamSetuForecaster:
    return request.app.state.model


def get_datacube(request: Request) -> xr.Dataset:
    return request.app.state.datacube


def get_cache(request: Request) -> ApiCache:
    return request.app.state.cache


def resolve_date(datacube: xr.Dataset, requested: date | None) -> np.datetime64:
    """Return the datacube timestamp matching ``requested`` or the latest day."""
    times = datacube.time.values
    if requested is None:
        return times[-1]
    target = np.datetime64(requested.isoformat())
    matches = np.where(times == target)[0]
    if not matches.size:
        first = str(times[0])[:10]
        last = str(times[-1])[:10]
        raise HTTPException(
            status_code=400,
            detail=f"date {requested.isoformat()} not in datacube ({first}..{last})",
        )
    return times[int(matches[0])]


def resolve_variable(datacube: xr.Dataset, requested: str) -> str:
    if requested not in datacube.data_vars:
        allowed = sorted(datacube.data_vars)
        raise HTTPException(
            status_code=400,
            detail=f"variable '{requested}' not in datacube; known={allowed}",
        )
    return requested


def get_thresholds(request: Request):
    """Return the resolved ThresholdConfig loaded at startup."""
    return request.app.state.thresholds


def get_validation_bundle(request: Request):
    """Return the cached validation bundle, computing it lazily on first use."""
    bundle = request.app.state.validation_bundle
    if bundle is not None:
        return bundle
    from mausamsetu.dashboard.api.validation_service import (
        compute_validation_bundle,
        load_cached_validation_bundle,
        save_cached_validation_bundle,
    )

    bundle = load_cached_validation_bundle()
    if bundle is None:
        bundle = compute_validation_bundle(request.app.state.model, request.app.state.datacube)
        save_cached_validation_bundle(bundle)
    request.app.state.validation_bundle = bundle
    return bundle
