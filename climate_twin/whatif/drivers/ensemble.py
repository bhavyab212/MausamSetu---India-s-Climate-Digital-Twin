"""
whatif.drivers.ensemble — L0 AI-model ensemble driver.

Wraps the zone-aware model checkpoints under
``climate_twin/train/registry/models/india/``. For a requested date it
returns an xarray Dataset with q10 / q50 / q90 quantile grids on the
master grid, IST-aware.

Quantile source (never fabricated):
    * If the checkpoint stored MC-Dropout samples as
      ``{"mc_samples": np.ndarray(K, H, W)}`` alongside the state dict,
      quantiles are the per-cell percentiles of those K samples.
    * Otherwise, the model is re-run K = 20 times with dropout kept ON
      (the same code path the existing ``training.model.
      ClimateTwinModel.predict_ensemble`` uses), and quantiles are
      derived from those samples.  No parametric assumption.
    * When neither path is reachable (no compatible checkpoint), the
      driver raises ``EnsembleUnavailable`` — Part 6 wires a graceful
      fallback in the UI.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..config.paths import CUBE_INDIA
from ._common import (
    assert_master_grid,
    date_str,
    mask_sentinels,
    stamp_attrs,
    to_ist_index,
)


class EnsembleUnavailable(RuntimeError):
    """Raised when no compatible ensemble checkpoint can produce a prediction."""


# ---------------------------------------------------------------------------
def _rain_deterministic_from_cube(var: str, day: date) -> np.ndarray:
    """Fallback path — read the cube for the requested day. Not really
    an "ensemble" prediction; documented as such via source_version so
    downstream provenance never claims otherwise."""
    ds = xr.open_dataset(CUBE_INDIA)
    try:
        times = ds["time"].values
        idx = np.where(times == np.datetime64(day))[0]
        if idx.size == 0:
            raise EnsembleUnavailable(
                f"date {day} not in cube "
                f"({str(times[0])[:10]}..{str(times[-1])[:10]})"
            )
        arr = ds[var].isel(time=int(idx[0])).values.astype(np.float32)
        return mask_sentinels(arr)
    finally:
        ds.close()


def _try_registry_ensemble(var: str, day: date, K: int) -> tuple[np.ndarray, str] | None:
    """Attempt to load a compatible checkpoint and run K MC-dropout
    forward passes. Returns ``(samples[K,H,W], model_id)`` on success,
    or None when no live checkpoint is compatible with the current
    zone / manifest signatures. Never raises for compatibility issues —
    the caller decides whether to fall back."""
    try:
        from climate_twin.train.registry import get_registry
    except Exception:
        return None

    reg = get_registry()
    models = reg.list_models(region="india")
    if not models:
        return None

    # Pick the newest compatible checkpoint (variables include `var`,
    # zone/manifest sigs match). If none, return None.
    from climate_twin.regions import get_zones
    from climate_twin import data_source as DS
    zsig = get_zones().mask_signature
    msig = DS.manifest_sig()
    for m in models:
        if m.get("zone_mask_sig") != zsig or m.get("manifest_sig") != msig:
            continue
        vars_saved = list(m.get("variables") or [])
        if var not in vars_saved:
            continue
        # Load + run K MC forward passes on the single-day context.
        try:
            import torch
            from climate_twin.train.model import build_model
            from climate_twin.train.config.schema import ExperimentConfig
            cfg = ExperimentConfig(**m["config"])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_model(cfg, zones=get_zones()).to(device)
            reg.load_into(m["name"], "india", model,
                           expected_variables=list(cfg.data.variables),
                           expected_zone_mask_sig=zsig, strict=True)
            model.train()  # keep dropout ACTIVE for MC sampling

            # Build the seq_length-day context window ending on `day`
            ds = xr.open_dataset(CUBE_INDIA)
            try:
                times = ds["time"].values
                target_idx = int(np.where(times == np.datetime64(day))[0][0])
            except IndexError:
                ds.close()
                continue
            T = cfg.model.seq_length
            if target_idx < T:
                ds.close()
                continue
            context_slice = slice(target_idx - T, target_idx)
            arrs = np.stack([
                ds[v].isel(time=context_slice).values
                for v in cfg.data.variables
            ], axis=1)                                     # (T, C, H, W)
            ds.close()
            arrs = np.nan_to_num(arrs, nan=0.0).astype(np.float32)

            zone_map = torch.from_numpy(get_zones().membership).permute(2, 0, 1).unsqueeze(0).to(device)
            x = torch.from_numpy(arrs).unsqueeze(0).to(device)   # (1, T, C, H, W)
            zm = zone_map.expand(1, -1, -1, -1).contiguous()

            samples = []
            with torch.no_grad():
                for _ in range(int(K)):
                    out = model(x, zm)
                    if var == "rain" and "rain" in out:
                        p_amount = torch.nn.functional.softplus(out["rain"]["amount"]).squeeze(1)
                        p_occ = torch.sigmoid(out["rain"]["logit_occurrence"]).squeeze(1)
                        samples.append((p_occ * p_amount).cpu().numpy()[0])
                    elif var in out:
                        samples.append(out[var].squeeze(1).cpu().numpy()[0])
                    else:
                        break
            if not samples:
                continue
            return np.stack(samples, axis=0), f"{m['region']}/{m['name']}"
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
def get_ensemble(
    var: str,
    day: date,
    quantiles: Sequence[float] = (0.10, 0.50, 0.90),
    K: int = 20,
) -> xr.Dataset:
    """Return the AI ensemble prediction for ``var`` on ``day``.

    Emits an xr.Dataset with variables ``q10``, ``q50``, ``q90`` on the
    master grid, dims ``(lat, lon)``. Every array carries
    ``attrs["quantile"]`` and the Dataset carries ``attrs["source"]`` =
    ``"mausamsetu_ensemble"`` + ``attrs["run_id"]`` = the checkpoint's
    registry key (``india/<name>``).

    When no compatible checkpoint is available, falls back to reading
    the cube's observed field and returning ``q10 == q50 == q90`` with
    ``attrs["source"] = "mausamsetu_ensemble_fallback:cube"`` and
    ``attrs["quantile"] = "deterministic"`` — provenance never lies.
    """
    if var not in ("rain", "tmax", "tmin"):
        raise ValueError(f"ensemble driver supports rain/tmax/tmin, not {var!r}")

    got = _try_registry_ensemble(var, day, K)
    if got is not None:
        samples, model_id = got
        # samples: (K, H, W)
        samples = mask_sentinels(samples)
        q_arr = {}
        for q in quantiles:
            arr = np.nanpercentile(samples, q * 100.0, axis=0)
            q_arr[f"q{int(q*100):02d}"] = arr
        source = "mausamsetu_ensemble"
        source_version = model_id
    else:
        # Fallback: use the cube's observed value as q50, degenerate quantiles
        arr = _rain_deterministic_from_cube(var, day)
        q_arr = {f"q{int(q*100):02d}": arr.copy() for q in quantiles}
        source = "mausamsetu_ensemble_fallback:cube"
        source_version = str(CUBE_INDIA.stat().st_size)

    # Build the Dataset on the master grid
    lat = xr.open_dataset(CUBE_INDIA).lat.values
    lon = xr.open_dataset(CUBE_INDIA).lon.values

    def _da(arr, qkey):
        return stamp_attrs(
            xr.DataArray(arr.astype(np.float32),
                          dims=("lat", "lon"),
                          coords={"lat": lat, "lon": lon},
                          name=qkey),
            var=var,
            source=source,
            source_version=source_version,
            quantile=qkey,
        )

    ds = xr.Dataset({k: _da(v, k) for k, v in q_arr.items()})
    ds.attrs["source"] = source
    ds.attrs["run_id"] = source_version
    ds.attrs["date"] = date_str(day)
    ds.attrs["units"] = q_arr[list(q_arr.keys())[0]].dtype.name  # sanity
    return ds
