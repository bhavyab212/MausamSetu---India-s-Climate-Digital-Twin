"""
IMD Meteorological Subdivision Zone definitions and extraction utilities.
Based on India Meteorological Department's standard climate zones.
"""
import numpy as np

# Grid bounds (from visualization.py)
LAT_MIN, LAT_MAX = 7.5, 37.5
LON_MIN, LON_MAX = 67.5, 97.5

# IMD standard meteorological subdivisions (approximate lat/lon bounding boxes)
# Each zone: (name, lat_min, lat_max, lon_min, lon_max)
IMD_ZONES = {
    "Northwest India":     (25.0, 37.5, 67.5, 80.0),   # Rajasthan, Punjab, Haryana, J&K, HP
    "Northeast India":     (22.0, 30.0, 88.0, 97.5),   # Assam, Meghalaya, Tripura, Arunachal
    "Central India":       (18.0, 26.0, 74.0, 86.0),   # MP, Chhattisgarh, Vidarbha
    "Peninsular India":    (8.0,  18.0, 74.0, 82.0),   # Tamil Nadu, Karnataka, Kerala, Andhra
    "East Coast & Islands":(8.0,  22.0, 80.0, 90.0),   # Odisha, Andhra coast, W Bengal
    "West Coast":          (8.0,  22.0, 72.5, 77.0),   # Konkan, Goa, Kerala coast
}

def get_zone_mask(lat_dim, lon_dim, lat_min, lat_max, lon_min, lon_max):
    """Return a boolean mask selecting grid cells within a lat/lon bounding box."""
    lat = np.linspace(LAT_MIN, LAT_MAX, lat_dim)
    lon = np.linspace(LON_MIN, LON_MAX, lon_dim)
    Lon, Lat = np.meshgrid(lon, lat)
    return (
        (Lat >= lat_min) & (Lat <= lat_max) &
        (Lon >= lon_min) & (Lon <= lon_max)
    )


def compute_zone_stats(full_data, binary_mask, scalers):
    """
    Compute mean Temperature and Rainfall for each IMD zone over the full time period.

    Returns a dict:
        {zone_name: {'years': [...], 'temp': [...], 'rain': [...]}}
    """
    n_years, lat_dim, lon_dim, _ = full_data.shape
    zone_stats = {}

    for zone_name, (lt_min, lt_max, ln_min, ln_max) in IMD_ZONES.items():
        box_mask = get_zone_mask(lat_dim, lon_dim, lt_min, lt_max, ln_min, ln_max)
        combined  = np.logical_and(binary_mask == 1, box_mask)      # land pixels inside zone

        temps, rains = [], []
        for i in range(n_years):
            t = np.where(combined, full_data[i, ..., 1], np.nan)
            r = np.where(combined, full_data[i, ..., 0], np.nan)
            temps.append(float(np.nanmean(t)) if np.any(combined) else np.nan)
            rains.append(float(np.nanmean(r)) if np.any(combined) else np.nan)

        zone_stats[zone_name] = {'temp': temps, 'rain': rains}

    return zone_stats


def compute_zone_future_stats(future_preds_temp, future_preds_rain, binary_mask, lat_dim, lon_dim):
    """
    Compute per-zone averages from lists of 50 future predicted grids.
    future_preds_temp / future_preds_rain: list of (lat, lon) arrays
    """
    zone_future = {}
    for zone_name, (lt_min, lt_max, ln_min, ln_max) in IMD_ZONES.items():
        box_mask = get_zone_mask(lat_dim, lon_dim, lt_min, lt_max, ln_min, ln_max)
        combined  = np.logical_and(binary_mask == 1, box_mask)

        t_vals, r_vals = [], []
        for tg, rg in zip(future_preds_temp, future_preds_rain):
            t_vals.append(float(np.nanmean(np.where(combined, tg, np.nan))))
            r_vals.append(float(np.nanmean(np.where(combined, rg, np.nan))))

        zone_future[zone_name] = {'temp': t_vals, 'rain': r_vals}

    return zone_future
