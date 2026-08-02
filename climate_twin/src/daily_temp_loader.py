"""
Daily Temperature Loader — Loads a single day from .GRD binary and regrids to 0.25°.
Memory-efficient: reads only one file, extracts only one day.
"""
import os
import numpy as np
import streamlit as st
from scipy.interpolate import RegularGridInterpolator

TEMP_ROWS, TEMP_COLS = 31, 31


@st.cache_data(show_spinner=False)
def get_grd_year_data(year, data_dir='data'):
    """Load the entire GRD binary for a single year and cache it in memory."""
    grd_filename = f'Maxtemp_MaxT_{year}.GRD'
    grd_file = os.path.join(data_dir, grd_filename)
    
    if not os.path.exists(grd_file):
        # Cloud downloads disabled — data now comes from data_source.py (local raw IMD).
        return None

    if not os.path.exists(grd_file):
        return None

    try:
        raw = np.fromfile(grd_file, dtype=np.float32)
        total_days = raw.shape[0] // (TEMP_ROWS * TEMP_COLS)
        daily_3d = raw.reshape(total_days, TEMP_ROWS, TEMP_COLS)
        return daily_3d
    except Exception:
        return None


def load_daily_temp(year, day_index, target_shape, binary_mask,
                    lat_min=6.5, lat_max=37.5, lon_min=67.5, lon_max=97.5,
                    data_dir='data'):
    """
    Load a single day's max temperature from a cached .GRD binary array,
    regrid from 1° (31×31) to the target 0.25° grid, and apply ocean mask.

    Args:
        year: int, e.g. 2020
        day_index: int, 0-based day index (0 = Jan 1)
        target_shape: tuple (H, W) of the target grid (from rainfall grid)
        binary_mask: np.ndarray of shape (H, W), 1=land, 0=ocean
        lat_min, lat_max, lon_min, lon_max: grid bounds
        data_dir: path to the data directory

    Returns:
        np.ndarray of shape (H, W) with daily max temperature, ocean=NaN
        Returns None if file not found or error occurs.
    """
    daily_3d = get_grd_year_data(year, data_dir=data_dir)
    if daily_3d is None:
        return None

    try:
        total_days = daily_3d.shape[0]

        if day_index >= total_days:
            day_index = total_days - 1

        day_grid = daily_3d[day_index].copy()

        # Mask invalid values (IMD uses 99.9 for missing)
        day_grid = np.where(day_grid > 90.0, np.nan, day_grid)

        # Build 1° source grid
        lat_src = np.linspace(lat_min, lat_max, TEMP_ROWS)
        lon_src = np.linspace(lon_min, lon_max, TEMP_COLS)

        # Build 0.25° target grid matching rainfall
        lat_tgt = np.linspace(lat_min, lat_max, target_shape[0])
        lon_tgt = np.linspace(lon_min, lon_max, target_shape[1])

        # Interpolate
        interpolator = RegularGridInterpolator(
            (lat_src, lon_src), day_grid,
            bounds_error=False, fill_value=np.nan
        )
        lat_mesh, lon_mesh = np.meshgrid(lat_tgt, lon_tgt, indexing='ij')
        points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=1)
        regridded = interpolator(points).reshape(target_shape)

        # Interpolate missing source blocks (like grid defects) using pandas
        import pandas as pd
        df = pd.DataFrame(regridded)
        df.interpolate(method='linear', axis=1, limit_direction='both', inplace=True)
        df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
        regridded = df.values

        # Apply ocean mask
        if binary_mask is not None:
            regridded = np.where(binary_mask == 1, regridded, np.nan)

        return regridded

    except Exception:
        return None
