"""
harmonize.py — Phase 3.

Bring every product onto the SAME master grid (0.25° India), enforce physical
consistency (tmax ≥ tmin, sentinel-free), and apply an explicit missingness
policy. Returns xr.DataArrays whose (lat, lon) axes are byte-identical
so they can be stacked as channels.

Regridding rules (from the prompt):
    - Temperature 1° → 0.25° :  bilinear (source is regular; smooth field)
    - INSAT ~0.05° curvilinear → 0.25° :  conservative area-average
      (bin native cells into target cells and take the mean per bin;
      preserves the local mean of the physical quantity, unlike bilinear
      which biases toward high native values on downsampling)
    - Rainfall stays native (already 0.25°).

Missingness policy (recorded as data variables, not silently filled):
    - Isolated gaps of ≤ 3 consecutive days on land  → temporal linear interp,
      also flagged True in an is_interpolated mask.
    - Gaps > 3 days                                  → left NaN, is_missing True.
    - Cell missing >20% of a training window         → excluded from loss.
      (The training loop already ignores NaN via the masked loss; the ratio
       is a *reporting* number here — the cube preserves the raw NaN pattern.)
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# --------------------------------------------------------------------------
# Master grid — matches the IMD rainfall 0.25° India cube exactly.
# --------------------------------------------------------------------------
MASTER_LAT0, MASTER_LAT1 = 6.5, 38.5
MASTER_LON0, MASTER_LON1 = 66.5, 100.0
MASTER_NLAT, MASTER_NLON = 129, 135
MASTER_LAT = np.round(np.linspace(MASTER_LAT0, MASTER_LAT1, MASTER_NLAT), 4)
MASTER_LON = np.round(np.linspace(MASTER_LON0, MASTER_LON1, MASTER_NLON), 4)


def canonical_grid() -> tuple[np.ndarray, np.ndarray]:
    return MASTER_LAT.copy(), MASTER_LON.copy()


def assert_grid_match(*das: xr.DataArray):
    """Hard gate: every DataArray uses the identical master (lat, lon)."""
    for i, da in enumerate(das):
        assert "lat" in da.coords and "lon" in da.coords, f"da[{i}] missing lat/lon coords"
        assert da.sizes["lat"] == MASTER_NLAT and da.sizes["lon"] == MASTER_NLON, (
            f"da[{i}] shape {da.sizes} != master ({MASTER_NLAT},{MASTER_NLON})"
        )
        assert np.allclose(da.lat.values, MASTER_LAT), f"da[{i}] lat != master"
        assert np.allclose(da.lon.values, MASTER_LON), f"da[{i}] lon != master"


# --------------------------------------------------------------------------
# 3a. Regridders.
# --------------------------------------------------------------------------
def regrid_temp_to_master(da_1deg: xr.DataArray) -> xr.DataArray:
    """Bilinear upsample IMD 1° temperature (31×31) to master 0.25° 129×135.

    NaNs (99.9 already masked in the reader) propagate through interpolation;
    partially-covered target cells become NaN — that's the correct honest
    behaviour, we do NOT smear valid values across the coast.
    """
    src_lat = da_1deg.lat.values.astype(np.float64)
    src_lon = da_1deg.lon.values.astype(np.float64)
    out = np.empty((da_1deg.sizes["time"], MASTER_NLAT, MASTER_NLON), dtype=np.float32)
    lat_m, lon_m = np.meshgrid(MASTER_LAT, MASTER_LON, indexing="ij")
    pts = np.stack([lat_m.ravel(), lon_m.ravel()], axis=1)
    for t in range(da_1deg.sizes["time"]):
        z = da_1deg.isel(time=t).values.astype(np.float64)
        f = RegularGridInterpolator((src_lat, src_lon), z,
                                    method="linear", bounds_error=False, fill_value=np.nan)
        out[t] = f(pts).reshape(MASTER_NLAT, MASTER_NLON).astype(np.float32)

    da = xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": da_1deg.time, "lat": MASTER_LAT, "lon": MASTER_LON},
        name=da_1deg.name,
        attrs={**da_1deg.attrs, "regridded_from": "1.0° 31×31",
               "regrid_method": "bilinear via RegularGridInterpolator"},
    )
    return da


def regrid_insat_conservative(da_curv: xr.DataArray) -> xr.DataArray:
    """Downsample INSAT LST (2-D curvilinear ~0.05°) to master 0.25° 129×135
    using AREA-AVERAGING (bin native cells to target cells; mean per bin).

    This is the conservative-lite approach: each target cell's value is the
    mean of every native cell whose (lat,lon) falls into that target cell's
    bin. Cells with zero native samples → NaN. Preserves the local mean
    (unlike bilinear which can shift extremes on downsampling).
    """
    lat2d = da_curv.lat2d.values.ravel().astype(np.float64)
    lon2d = da_curv.lon2d.values.ravel().astype(np.float64)

    dlat = float(MASTER_LAT[1] - MASTER_LAT[0])
    dlon = float(MASTER_LON[1] - MASTER_LON[0])
    lat_edges = np.linspace(MASTER_LAT[0] - dlat / 2, MASTER_LAT[-1] + dlat / 2, MASTER_NLAT + 1)
    lon_edges = np.linspace(MASTER_LON[0] - dlon / 2, MASTER_LON[-1] + dlon / 2, MASTER_NLON + 1)

    # Pre-compute the bin index of every native cell (invariant across days).
    lat_ix = np.searchsorted(lat_edges, lat2d, side="right") - 1
    lon_ix = np.searchsorted(lon_edges, lon2d, side="right") - 1
    in_bounds = (lat_ix >= 0) & (lat_ix < MASTER_NLAT) & (lon_ix >= 0) & (lon_ix < MASTER_NLON)
    flat_idx = lat_ix[in_bounds] * MASTER_NLON + lon_ix[in_bounds]

    T = da_curv.sizes["time"]
    out = np.full((T, MASTER_NLAT * MASTER_NLON), np.nan, dtype=np.float32)
    for t in range(T):
        z = da_curv.isel(time=t).values.ravel()[in_bounds]
        valid = np.isfinite(z)
        if not valid.any():
            continue
        idx = flat_idx[valid]
        vals = z[valid]
        sums = np.bincount(idx, weights=vals, minlength=MASTER_NLAT * MASTER_NLON)
        counts = np.bincount(idx, minlength=MASTER_NLAT * MASTER_NLON)
        with np.errstate(invalid="ignore"):
            avg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        out[t] = avg.astype(np.float32)

    out = out.reshape(T, MASTER_NLAT, MASTER_NLON)
    da = xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": da_curv.time, "lat": MASTER_LAT, "lon": MASTER_LON},
        name=da_curv.name,
        attrs={**{k: v for k, v in da_curv.attrs.items() if k not in ("curvilinear",)},
               "regridded_from": "curvilinear 974×1067",
               "regrid_method": "area-average (conservative)"},
    )
    return da


def rainfall_to_master(da_rain: xr.DataArray) -> xr.DataArray:
    """Rainfall is native 0.25° on the master grid — canonicalize coords + attrs only."""
    lat, lon = da_rain.lat.values, da_rain.lon.values
    assert lat.shape == MASTER_LAT.shape and np.allclose(lat, MASTER_LAT), "rain lat mismatch"
    assert lon.shape == MASTER_LON.shape and np.allclose(lon, MASTER_LON), "rain lon mismatch"
    out = da_rain.assign_coords(lat=MASTER_LAT, lon=MASTER_LON)
    out.attrs = {**da_rain.attrs, "regrid_method": "native (0.25°)"}
    return out


# --------------------------------------------------------------------------
# 3b. Physical consistency checks — REPORT, do NOT silently fix.
# --------------------------------------------------------------------------
def check_tmax_ge_tmin(tmax: xr.DataArray, tmin: xr.DataArray) -> dict:
    """Return counts + first offending times/locations. No mutation."""
    assert_grid_match(tmax, tmin)
    common_time = np.intersect1d(tmax.time.values, tmin.time.values)
    a = tmax.sel(time=common_time).values
    b = tmin.sel(time=common_time).values
    valid = np.isfinite(a) & np.isfinite(b)
    viol = (a < b) & valid
    n = int(viol.sum())
    stats = {"violations": n, "valid_cell_days": int(valid.sum()),
             "violation_pct": float(100.0 * n / max(int(valid.sum()), 1))}
    if n:
        idx = np.argwhere(viol)[:5]
        stats["first_violations"] = [
            {"time": str(common_time[t])[:10], "lat_idx": int(la), "lon_idx": int(lo),
             "tmax": float(a[t, la, lo]), "tmin": float(b[t, la, lo])}
            for (t, la, lo) in idx
        ]
    return stats


def land_mask_from(da: xr.DataArray) -> np.ndarray:
    """Return a boolean (lat,lon) mask: True where the field is ever non-NaN."""
    v = da.values
    return ~np.isnan(v).all(axis=0)


def compare_land_masks(*named: tuple[str, xr.DataArray]) -> dict:
    masks = {name: land_mask_from(da) for name, da in named}
    ref_name = list(masks.keys())[0]
    ref = masks[ref_name]
    report = {"reference": ref_name, "counts": {k: int(v.sum()) for k, v in masks.items()}}
    for k, m in masks.items():
        if k == ref_name:
            continue
        both = int((ref & m).sum()); only_ref = int((ref & ~m).sum()); only_k = int((m & ~ref).sum())
        report[f"{ref_name} vs {k}"] = {"both": both, f"only_{ref_name}": only_ref,
                                         f"only_{k}": only_k}
    return report


def assert_no_sentinels(*das: xr.DataArray):
    for da in das:
        v = da.values
        survivors = v[np.isfinite(v)]
        if survivors.size == 0:
            continue
        assert not np.any(survivors <= -998.0), f"{da.name}: -999 sentinel survived"
        # 99.9 could be a plausible LST °C so only reject when the variable is temp
        if da.name in ("tmax", "tmin"):
            assert not np.any(survivors >= 90.0), f"{da.name}: 99.9 sentinel survived"


# --------------------------------------------------------------------------
# 3c. Missingness policy.
# --------------------------------------------------------------------------
def apply_missingness_policy(da: xr.DataArray, max_gap_days: int = 3) -> tuple[xr.DataArray, xr.DataArray]:
    """Fill isolated ≤ max_gap_days temporal gaps by linear interpolation on land.

    Returns (filled_da, is_interpolated_da).  is_interpolated is a bool mask
    with True where a value was interpolated (never over sea, never over gaps
    longer than max_gap_days).
    """
    v = da.values.astype(np.float32)
    filled = v.copy()
    interp_mask = np.zeros(v.shape, dtype=bool)

    T, H, W = v.shape
    for i in range(H):
        for j in range(W):
            col = v[:, i, j]
            finite = np.isfinite(col)
            if finite.sum() < 2 or finite.all():
                continue
            # Find contiguous NaN runs.
            t = 0
            while t < T:
                if finite[t]:
                    t += 1
                    continue
                s = t
                while t < T and not finite[t]:
                    t += 1
                e = t                                            # NaN run [s, e)
                # Bounded gap only (both endpoints must be finite).
                if s > 0 and e < T and finite[s - 1] and finite[e] and (e - s) <= max_gap_days:
                    lo, hi = col[s - 1], col[e]
                    for k in range(s, e):
                        w = (k - (s - 1)) / (e - (s - 1))
                        filled[k, i, j] = lo * (1.0 - w) + hi * w
                        interp_mask[k, i, j] = True

    out = xr.DataArray(
        filled, dims=da.dims, coords=da.coords, name=da.name,
        attrs={**da.attrs, "missingness_policy": f"linear-interp gaps ≤ {max_gap_days} days"},
    )
    mask = xr.DataArray(
        interp_mask, dims=da.dims, coords=da.coords, name=f"{da.name}_is_interpolated",
        attrs={"long_name": f"True where {da.name} was linearly interpolated over a ≤{max_gap_days}-day gap"},
    )
    return out, mask


def cell_missing_fraction(da: xr.DataArray) -> xr.DataArray:
    """Fraction of the time axis that is NaN, per (lat, lon)."""
    frac = np.isnan(da.values).mean(axis=0)
    return xr.DataArray(frac.astype(np.float32), dims=("lat", "lon"),
                        coords={"lat": da.lat, "lon": da.lon},
                        name=f"{da.name}_missing_frac",
                        attrs={"long_name": f"time-mean NaN fraction per cell for {da.name}"})
