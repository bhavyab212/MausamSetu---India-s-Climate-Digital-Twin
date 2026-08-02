"""Headless perf baseline — times the real heavy operations each page runs.
Run from climate_twin/:  ../venv/Scripts/python bench_baseline.py
"""
import os, time, warnings, io
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

def t(fn, *a, **k):
    s = time.perf_counter()
    r = fn(*a, **k)
    return (time.perf_counter() - s) * 1000.0, r

import psutil
proc = psutil.Process(os.getpid())
def rss(): return proc.memory_info().rss / 1e6

results = {}

import data_source as DS

# ---- DATA LOAD (cold = read .nc; warm = cached in-process) ----
dt, _ = t(DS.load_region, "india"); results["data: load_region india (cold)"] = dt
dt, agg = t(DS.load_aggregates, "india"); results["data: load_aggregates india (warm)"] = dt
rain, temp, mask, years = agg
dt, _ = t(DS.daily_rain, "india", 2020); results["data: daily_rain 2020 (cold .grd)"] = dt
dt, _ = t(DS.daily_tmax, "india", 2020); results["data: daily_tmax 2020 (cold .grd+regrid)"] = dt

# ---- MAP FIGURE BUILD (129x135 through the app's PlotlyVisualizer) ----
from src.viz_plotly import PlotlyVisualizer
pv = PlotlyVisualizer(os.path.join("data", "India_States_2024.geojson"), mask,
                      lat_min=6.5, lat_max=38.5, lon_min=66.5, lon_max=100.0)
grid = rain[-1]
dt, _ = t(pv.plot_map, grid, "Annual Rainfall", "rain_annual", [0.0, 2000.0]); results["figures: plot_map (129x135) 1st"] = dt
dt, _ = t(pv.plot_map, grid, "Annual Rainfall", "rain_annual", [0.0, 2000.0]); results["figures: plot_map (129x135) 2nd"] = dt

# ---- ENSEMBLE INFERENCE (TF model load + forward) ----
gpu_touched = False
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
except Exception:
    gpus = []
try:
    from src.ensemble import EnsemblePredictor
    seq = min(30, len(rain) - 1)
    H, W = mask.shape
    # normalize a window like predict_future_annual does
    full = np.stack([rain, temp], axis=-1).astype(np.float64)
    r0, r1 = np.nanmin(full[..., 0]), np.nanmax(full[..., 0])
    t0, t1 = np.nanmin(full[..., 1]), np.nanmax(full[..., 1])
    norm = full.copy()
    norm[..., 0] = (norm[..., 0] - r0) / (r1 - r0 + 1e-8)
    norm[..., 1] = (norm[..., 1] - t0) / (t1 - t0 + 1e-8)
    window = np.clip(norm[-seq:], 0, 1)
    dt, ens = t(EnsemblePredictor, seq, H, W, "data"); results["inference: EnsemblePredictor init+load"] = dt
    inp = np.expand_dims(window, 0)
    dt, _ = t(ens.predict, inp); results["inference: ensemble.predict 1 step (1st)"] = dt
    dt, _ = t(ens.predict, inp); results["inference: ensemble.predict 1 step (2nd)"] = dt
except Exception as e:
    results["inference: ERROR"] = -1
    print("ensemble error:", e)

# ---- MATPLOTLIB ANIMATION (30-day June GIF, the heavy figure path) ----
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from scipy.ndimage import gaussian_filter, zoom
    seq30 = DS.daily_rain("india", 2020)[151:181]  # ~June
    land = (mask == 1)
    vmax = max(6.0, float(np.nanpercentile(seq30[:, land], 96)))
    up = 4; land_up = zoom(land.astype(float), up, order=0) >= 0.5
    def prep(f):
        g = np.nan_to_num(np.array(f, float), nan=0.0)
        g = gaussian_filter(g, 0.9); g = zoom(g, up, order=3)
        g = np.clip(g, 0, vmax); g = np.where(land_up, g, np.nan); return g
    def make_gif():
        frames = [prep(seq30[i]) for i in range(len(seq30))]
        tw = 3; allf = []
        for i in range(len(frames) - 1):
            for k in range(tw):
                allf.append(frames[i] * (1 - k / tw) + frames[i + 1] * (k / tw))
        allf.append(frames[-1])
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        im = ax.imshow(allf[0], origin="lower", cmap="Blues", vmin=0, vmax=vmax, interpolation="bicubic")
        def upd(j): im.set_data(allf[j]); return [im]
        anim = FuncAnimation(fig, upd, frames=len(allf), blit=False)
        p = os.path.join(os.environ.get("TEMP", "/tmp"), "bench_anim.gif")
        anim.save(p, writer=PillowWriter(fps=18), dpi=90); plt.close(fig)
        return os.path.getsize(p)
    dt, sz = t(make_gif); results["figures: matplotlib GIF (30 days June)"] = dt
    results["_gif_bytes"] = sz
except Exception as e:
    print("anim error:", e)

peak = rss()
print("\n=== BASELINE (ms) ===")
for k, v in sorted(results.items(), key=lambda x: -x[1] if x[0] != '_gif_bytes' else 0):
    if k == "_gif_bytes":
        print(f"  (gif size: {v/1e3:.0f} KB)")
    else:
        print(f"  {v:9.1f}  {k}")
print(f"\nPeak RSS: {peak:.0f} MB")
print(f"TF GPUs visible: {len(gpus)}  (gpu_touched flag: {gpu_touched})")
import sys; print("python", sys.version.split()[0])
