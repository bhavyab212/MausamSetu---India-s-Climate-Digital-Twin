import os
import glob
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from src.utils import save_scalers
import re

class DataPreprocessor:
    def __init__(self, data_dir, seq_length=30):
        self.data_dir = data_dir
        self.seq_length = seq_length

    def load_and_align_data(self):
        nc_files = sorted(glob.glob(os.path.join(self.data_dir, 'RF25_ind*.nc')))
        bin_files = sorted(glob.glob(os.path.join(self.data_dir, '*MaxT_*.GRD')))
        
        if not nc_files or not bin_files:
            print(f"Warning: No data found in {self.data_dir}.")
            return None, None
            
        # Load first rainfall file to get target grid
        try:
            sample_ds = xr.open_dataset(nc_files[0])
            rf_var = 'RAINFALL' if 'RAINFALL' in sample_ds.variables else list(sample_ds.data_vars)[0]
            lat_var = 'LATITUDE' if 'LATITUDE' in sample_ds.variables else 'lat'
            lon_var = 'LONGITUDE' if 'LONGITUDE' in sample_ds.variables else 'lon'
            lat_target = sample_ds[lat_var].values
            lon_target = sample_ds[lon_var].values
            # Target shape is [lat, lon], so taking from the last two dimensions
            target_shape = sample_ds[rf_var].values.shape[-2:]
        except Exception as e:
            print(f"Error reading {nc_files[0]}: {e}")
            return None, None

        years_data = []
        for i in range(min(len(nc_files), len(bin_files))):
            ds_rain = xr.open_dataset(nc_files[i])
            daily_rain_grid = ds_rain[rf_var].values
            # Aggregate to annual cumulative rainfall
            # Ignore NaNs (ocean) during sum
            rain_grid = np.nansum(daily_rain_grid, axis=0)
            
            ROWS, COLS = 31, 31 
            try:
                temp_data = np.fromfile(bin_files[i], dtype=np.float32)
                # Reshape to (days, lat, lon)
                days = int(temp_data.shape[0] / (ROWS * COLS))
                temp_grid_1deg_daily = temp_data.reshape(days, ROWS, COLS)
                
                # Mask out the 99.9 invalid values
                temp_grid_1deg_daily = np.where(temp_grid_1deg_daily > 90.0, np.nan, temp_grid_1deg_daily)
                
                # Aggregate to annual maximum temperature
                temp_grid_1deg = np.nanmax(temp_grid_1deg_daily, axis=0)
            except Exception as e:
                print(f"Error reshaping temp file: {e}")
                temp_grid_1deg = np.zeros((ROWS, COLS))
                
            lat_1deg = np.linspace(lat_target.min(), lat_target.max(), ROWS)
            lon_1deg = np.linspace(lon_target.min(), lon_target.max(), COLS)
            
            interpolator = RegularGridInterpolator((lat_1deg, lon_1deg), temp_grid_1deg, bounds_error=False, fill_value=None)
            lat_mesh, lon_mesh = np.meshgrid(lat_target, lon_target, indexing='ij')
            target_points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=1)
            temp_regridded = interpolator(target_points).reshape(target_shape)
            
            year_combined = np.stack([rain_grid, temp_regridded], axis=-1)
            years_data.append(year_combined)
            
        full_data = np.array(years_data)
        
        # Binary mask (1 for land, 0 for ocean)
        # Using the target shape from rainfall. Where rain_grid == 0 after nansum usually implies ocean if it was all NaNs
        # But to be safer, let's use the first day of the first year's raw rainfall data for the mask
        ds_rain_sample = xr.open_dataset(nc_files[0])
        daily_rain_sample = ds_rain_sample[rf_var].values[0]
        binary_mask = (~np.isnan(daily_rain_sample)).astype(np.float32)
        
        # Set oceans to 0 explicitly
        full_data[..., 0] = np.where(binary_mask == 1, full_data[..., 0], 0.0)
        full_data[..., 1] = np.where(binary_mask == 1, full_data[..., 1], 0.0)

        # Handle any residual NaNs inside the land mask (due to interpolation bounds)
        broadcast_mask = np.broadcast_to(binary_mask, full_data[..., 0].shape)
        land_rain = full_data[..., 0][broadcast_mask == 1]
        land_temp = full_data[..., 1][broadcast_mask == 1]
        full_data[..., 0] = np.where(np.isnan(full_data[..., 0]), np.nanmean(land_rain), full_data[..., 0])
        full_data[..., 1] = np.where(np.isnan(full_data[..., 1]), np.nanmean(land_temp), full_data[..., 1])
        
        return full_data, binary_mask

    def normalize_and_split(self, full_data, test_size=5):
        if full_data is None: return None, None, None, None
        
        total_years = full_data.shape[0]
        if total_years <= self.seq_length:
            return None, None, None, None
            
        X, Y = [], []
        for i in range(total_years - self.seq_length):
            X.append(full_data[i : i + self.seq_length])
            Y.append(full_data[i + self.seq_length])
            
        X, Y = np.array(X), np.array(Y)
        
        # Strict temporal split BEFORE calculating scalers
        if len(X) <= test_size:
            X_train, Y_train = X, Y
            X_test, Y_test = X, Y
        else:
            X_train, Y_train = X[:-test_size], Y[:-test_size]
            X_test, Y_test = X[-test_size:], Y[-test_size:]
            
        # We must re-extract the binary mask from the full_data because it was passed separately
        # But wait, full_data has oceans explicitly set to 0.0.
        # It's safer to extract non-zero pixels or use the passed binary mask.
        # Since X_train is fully populated, we can just use np.where(X_train[0, 0, ..., 0] > 0)
        # But let's just use np.nanmin/np.nanmax on non-zero elements to be safe for scalers:
        train_rain_flat = X_train[..., 0].flatten()
        train_temp_flat = X_train[..., 1].flatten()
        
        # Exclude exactly 0.0 because oceans were set to 0.0
        valid_rain = train_rain_flat[train_rain_flat > 0.0001]
        valid_temp = train_temp_flat[train_temp_flat > 0.0001]
        
        rain_min = float(np.nanmin(valid_rain)) if len(valid_rain) > 0 else 0.0
        rain_max = float(np.nanmax(valid_rain)) if len(valid_rain) > 0 else 1.0
        temp_min = float(np.nanmin(valid_temp)) if len(valid_temp) > 0 else 0.0
        temp_max = float(np.nanmax(valid_temp)) if len(valid_temp) > 0 else 1.0
        
        scalers = {
            'rain_min': rain_min, 'rain_max': rain_max,
            'temp_min': temp_min, 'temp_max': temp_max
        }
        save_scalers(scalers, os.path.join(self.data_dir, 'scalers.json'))
        
        rain_range = (rain_max - rain_min) if rain_max != rain_min else 1.0
        temp_range = (temp_max - temp_min) if temp_max != temp_min else 1.0
        
        # Apply normalization
        X_train[..., 0] = (X_train[..., 0] - rain_min) / rain_range
        X_train[..., 1] = (X_train[..., 1] - temp_min) / temp_range
        Y_train[..., 0] = (Y_train[..., 0] - rain_min) / rain_range
        Y_train[..., 1] = (Y_train[..., 1] - temp_min) / temp_range
        
        X_test[..., 0] = (X_test[..., 0] - rain_min) / rain_range
        X_test[..., 1] = (X_test[..., 1] - temp_min) / temp_range
        Y_test[..., 0] = (Y_test[..., 0] - rain_min) / rain_range
        Y_test[..., 1] = (Y_test[..., 1] - temp_min) / temp_range
        
        return X_train, Y_train, X_test, Y_test, scalers

    def load_daily_aligned_data(self, years_back=None):
        """
        Build daily [time, lat, lon, 2] tensor:
        channel-0 rainfall (mm/day), channel-1 max-temp (°C/day), India-only mask.
        """
        nc_files = sorted(glob.glob(os.path.join(self.data_dir, 'RF25_ind*_rfp25.nc')))
        temp_files = sorted(glob.glob(os.path.join(self.data_dir, '*MaxT_*.GRD')))
        if not nc_files or not temp_files:
            print(f"Warning: No daily files found in {self.data_dir}.")
            return None, None, None

        def extract_year(path):
            m = re.search(r'(19|20)\d{2}', os.path.basename(path))
            return int(m.group(0)) if m else None

        rain_by_year = {extract_year(p): p for p in nc_files if extract_year(p) is not None}
        temp_by_year = {extract_year(p): p for p in temp_files if extract_year(p) is not None}
        years = sorted(set(rain_by_year.keys()).intersection(temp_by_year.keys()))
        if not years:
            print("Warning: No overlapping rain/temp years.")
            return None, None, None
        if years_back is not None and years_back > 0 and len(years) > years_back:
            years = years[-years_back:]

        # Target grid from first rainfall file
        ds0 = xr.open_dataset(rain_by_year[years[0]])
        rf_var = 'RAINFALL' if 'RAINFALL' in ds0.variables else list(ds0.data_vars)[0]
        lat_var = 'LATITUDE' if 'LATITUDE' in ds0.variables else 'lat'
        lon_var = 'LONGITUDE' if 'LONGITUDE' in ds0.variables else 'lon'
        lat_target = ds0[lat_var].values
        lon_target = ds0[lon_var].values
        target_shape = ds0[rf_var].values.shape[-2:]
        binary_mask = (~np.isnan(ds0[rf_var].values[0])).astype(np.float32)
        ds0.close()

        ROWS, COLS = 31, 31
        lat_1deg = np.linspace(lat_target.min(), lat_target.max(), ROWS)
        lon_1deg = np.linspace(lon_target.min(), lon_target.max(), COLS)

        all_days = []
        all_meta = []
        for y in years:
            ds_r = xr.open_dataset(rain_by_year[y])
            rf = ds_r[rf_var].values  # [days, lat, lon]
            ds_r.close()

            temp_raw = np.fromfile(temp_by_year[y], dtype=np.float32)
            total_days_t = int(temp_raw.shape[0] / (ROWS * COLS))
            temp_daily = temp_raw.reshape(total_days_t, ROWS, COLS)
            temp_daily = np.where(temp_daily > 90.0, np.nan, temp_daily)

            days = min(rf.shape[0], temp_daily.shape[0], 365)
            for d in range(days):
                rain_d = rf[d]
                interp = RegularGridInterpolator((lat_1deg, lon_1deg), temp_daily[d], bounds_error=False, fill_value=np.nan)
                lat_mesh, lon_mesh = np.meshgrid(lat_target, lon_target, indexing='ij')
                pts = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=1)
                temp_d = interp(pts).reshape(target_shape)

                # Land-only and fill residual NaNs over land.
                rain_d = np.where(binary_mask == 1, rain_d, np.nan)
                temp_d = np.where(binary_mask == 1, temp_d, np.nan)
                rain_fill = np.nanmean(rain_d)
                temp_fill = np.nanmean(temp_d)
                rain_d = np.where(np.isnan(rain_d), rain_fill, rain_d)
                temp_d = np.where(np.isnan(temp_d), temp_fill, temp_d)
                day_pair = np.stack([rain_d, temp_d], axis=-1)
                all_days.append(day_pair)
                all_meta.append((y, d + 1))

        if not all_days:
            return None, None, None
        full_daily = np.array(all_days, dtype=np.float32)
        return full_daily, binary_mask, all_meta

    def normalize_and_split_daily(self, full_daily, test_days=365):
        """
        Sequence split for daily training:
        X: [N, seq, lat, lon, 2], Y: [N, lat, lon, 2]
        """
        total_steps = full_daily.shape[0]
        if total_steps <= self.seq_length + 1:
            return None, None, None, None, None

        sample_count = total_steps - self.seq_length
        est_gib = (
            sample_count
            * self.seq_length
            * full_daily.shape[1]
            * full_daily.shape[2]
            * full_daily.shape[3]
            * 4
        ) / (1024 ** 3)
        if est_gib > 8:
            raise MemoryError(
                f"Daily materialized sequence tensor would require ~{est_gib:.1f} GiB. "
                "Use streaming tf.data pipeline instead of normalize_and_split_daily."
            )

        X, Y = [], []
        for i in range(sample_count):
            X.append(full_daily[i:i + self.seq_length])
            Y.append(full_daily[i + self.seq_length])
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        if len(X) <= test_days:
            X_train, Y_train = X, Y
            X_test, Y_test = X, Y
        else:
            X_train, Y_train = X[:-test_days], Y[:-test_days]
            X_test, Y_test = X[-test_days:], Y[-test_days:]

        train_rain = X_train[..., 0]
        train_temp = X_train[..., 1]
        rain_min, rain_max = float(np.nanmin(train_rain)), float(np.nanmax(train_rain))
        temp_min, temp_max = float(np.nanmin(train_temp)), float(np.nanmax(train_temp))
        rr = (rain_max - rain_min) if rain_max != rain_min else 1.0
        tr = (temp_max - temp_min) if temp_max != temp_min else 1.0
        scalers = {'rain_min': rain_min, 'rain_max': rain_max, 'temp_min': temp_min, 'temp_max': temp_max}
        save_scalers(scalers, os.path.join(self.data_dir, 'scalers_daily.json'))

        X_train[..., 0] = (X_train[..., 0] - rain_min) / rr
        X_train[..., 1] = (X_train[..., 1] - temp_min) / tr
        Y_train[..., 0] = (Y_train[..., 0] - rain_min) / rr
        Y_train[..., 1] = (Y_train[..., 1] - temp_min) / tr
        X_test[..., 0] = (X_test[..., 0] - rain_min) / rr
        X_test[..., 1] = (X_test[..., 1] - temp_min) / tr
        Y_test[..., 0] = (Y_test[..., 0] - rain_min) / rr
        Y_test[..., 1] = (Y_test[..., 1] - temp_min) / tr
        return X_train, Y_train, X_test, Y_test, scalers

    def build_daily_context_features(self, all_meta):
        """
        Build context vectors for each daily sample using (year, doy) metadata.
        Returns np.ndarray [N_days, 7] with:
        [sin_doy, cos_doy, is_monsoon, is_post_monsoon, is_winter, is_pre_monsoon, year_norm]
        """
        if all_meta is None or len(all_meta) == 0:
            return None
        years = np.array([y for y, _ in all_meta], dtype=np.float32)
        doys = np.array([d for _, d in all_meta], dtype=np.float32)
        year_min, year_max = float(np.min(years)), float(np.max(years))
        yr_rng = (year_max - year_min) if year_max != year_min else 1.0
        year_norm = (years - year_min) / yr_rng

        theta = 2.0 * np.pi * (doys / 365.0)
        sin_doy = np.sin(theta)
        cos_doy = np.cos(theta)

        # Seasonal flags for Indian climate cycle.
        # Approx ranges by day-of-year:
        # winter: Dec-Feb (~335-365,1-59), pre-monsoon: Mar-May (~60-151),
        # monsoon: Jun-Sep (~152-273), post-monsoon: Oct-Nov (~274-334).
        is_monsoon = ((doys >= 152) & (doys <= 273)).astype(np.float32)
        is_post_monsoon = ((doys >= 274) & (doys <= 334)).astype(np.float32)
        is_winter = ((doys <= 59) | (doys >= 335)).astype(np.float32)
        is_pre_monsoon = ((doys >= 60) & (doys <= 151)).astype(np.float32)

        return np.stack(
            [sin_doy, cos_doy, is_monsoon, is_post_monsoon, is_winter, is_pre_monsoon, year_norm],
            axis=-1
        ).astype(np.float32)
