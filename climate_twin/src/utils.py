import json
import numpy as np
import os

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

def ensure_data_file(filename):
    """Resolve a data file locally. Cloud downloads are DISABLED — this app now
    reads only the user's local raw IMD data via data_source.py (no external calls)."""
    return os.path.join('data', filename)

def save_scalers(scalers, filepath):
    with open(filepath, 'w') as f:
        json.dump(scalers, f)

def load_scalers(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def denormalize_grid(grid, scalers, is_temp=False):
    if is_temp:
        min_val, max_val = scalers['temp_min'], scalers['temp_max']
    else:
        min_val, max_val = scalers['rain_min'], scalers['rain_max']
    
    val_range = max_val - min_val if max_val != min_val else 1.0
    return grid * val_range + min_val

def masked_rmse(y_true, y_pred, mask):
    """Computes RMSE only over land pixels (mask == 1)."""
    # mask shape: [lat, lon]. We broadcast it if y has time/batch/channel dims
    # y_true and y_pred are already denormalized here
    diff = (y_true - y_pred) ** 2
    
    # Broadcast mask to match the shape of diff
    broadcast_mask = np.broadcast_to(mask, diff.shape)
    
    masked_diff = diff[broadcast_mask == 1]
    if len(masked_diff) == 0: return 0.0
    
    return np.sqrt(np.mean(masked_diff))

def masked_mae(y_true, y_pred, mask):
    """Computes MAE only over land pixels (mask == 1)."""
    diff = np.abs(y_true - y_pred)
    broadcast_mask = np.broadcast_to(mask, diff.shape)
    
    masked_diff = diff[broadcast_mask == 1]
    if len(masked_diff) == 0: return 0.0
    
    return np.mean(masked_diff)

def masked_pearson_r(y_true, y_pred, mask):
    """Computes spatial correlation (Pearson r) over land pixels."""
    broadcast_mask = np.broadcast_to(mask, y_true.shape)

    true_vals = y_true[broadcast_mask == 1]
    pred_vals = y_pred[broadcast_mask == 1]

    if len(true_vals) < 2:
        return 0.0

    if np.nanstd(true_vals) < 1e-12 or np.nanstd(pred_vals) < 1e-12:
        return 0.0

    corr_matrix = np.corrcoef(true_vals, pred_vals)
    r = float(corr_matrix[0, 1])
    if not np.isfinite(r):
        return 0.0
    return r
