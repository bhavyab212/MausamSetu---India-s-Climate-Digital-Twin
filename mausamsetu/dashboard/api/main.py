"""FastAPI entrypoint for the MausamSetu मौसम सेतु dashboard service."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import xarray as xr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from mausamsetu import config
from mausamsetu.dashboard.api.routes import (
    alerts,
    data,
    forecast,
    impacts,
    reports,
    scenarios,
    settings,
    state,
    validation,
)
from mausamsetu.dashboard.api.settings_service import resolve_thresholds
from mausamsetu.model.predict import load_forecaster

LOGGER = logging.getLogger(__name__)
API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the trained model, datacube, thresholds, and validation bundle once."""
    checkpoint_path = config.CHECKPOINT_DIR / "forecaster_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
    if not config.CAUVERY_NC.is_file():
        raise FileNotFoundError(f"Required datacube not found: {config.CAUVERY_NC}")

    datacube: xr.Dataset | None = None
    try:
        model = load_forecaster(checkpoint_path=checkpoint_path, device="cpu")
        model.eval()
        datacube = xr.open_dataset(config.CAUVERY_NC).load()
        app.state.model = model
        app.state.datacube = datacube
        app.state.cache = {}
        app.state.thresholds = resolve_thresholds(datacube)
        app.state.validation_bundle = None
        LOGGER.info(
            "Loaded forecaster and %d-day datacube; validation bundle deferred to first request.",
            datacube.sizes["time"],
        )
        yield
    finally:
        if datacube is not None:
            datacube.close()


app = FastAPI(
    title="MausamSetu मौसम सेतु API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/livez", include_in_schema=False)
def livez() -> dict[str, str]:
    """Return a lightweight process liveness response without app dependencies."""
    return {"status": "ok"}


app.add_middleware(GZipMiddleware, minimum_size=1000)
_frontend_origins = os.environ.get(
    "MAUSAMSETU_FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _frontend_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for resource_router in (
    state.router,
    forecast.router,
    scenarios.router,
    impacts.router,
    validation.router,
    alerts.router,
    data.router,
    reports.router,
    settings.router,
):
    app.include_router(resource_router, prefix=API_PREFIX)
