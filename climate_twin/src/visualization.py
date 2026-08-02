import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ─── Pre-compute the expensive 300x300 India mask ONCE at module load ──────────
# This means: no matter how many times the slider is dragged, 
# the shapefile and contains_points() are NEVER re-run.
_INDIA_MASK_300   = None   # shape (300, 300) bool
_LON_HI           = None   # shape (300, 300)
_LAT_HI           = None   # shape (300, 300)
_INDIA_PATHS      = None   # list of matplotlib.Path objects

def _build_india_cache():
    global _INDIA_MASK_300, _LON_HI, _LAT_HI, _INDIA_PATHS

    if _INDIA_MASK_300 is not None:
        return  # Already built – skip

    lon_hi = np.linspace(67.5, 97.5, 300)
    lat_hi = np.linspace(7.5,  37.5, 300)
    Lon_hi, Lat_hi = np.meshgrid(lon_hi, lat_hi)
    _LON_HI = Lon_hi
    _LAT_HI = Lat_hi

    if not HAS_CARTOPY:
        _INDIA_MASK_300 = np.ones((300, 300), dtype=bool)
        _INDIA_PATHS = []
        return

    try:
        from cartopy.mpl.path import shapely_to_path
    except ImportError:
        from cartopy.mpl.patch import geos_to_path as shapely_to_path

    reader = shpreader.Reader(
        shpreader.natural_earth('50m', 'cultural', 'admin_0_countries')
    )
    india_geom = [c for c in reader.records()
                  if c.attributes['NAME'] == 'India'][0].geometry

    paths = shapely_to_path(india_geom)
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    _INDIA_PATHS = paths

    points = np.column_stack((Lon_hi.flatten(), Lat_hi.flatten()))
    mask   = np.zeros(len(points), dtype=bool)
    for path in paths:
        mask |= path.contains_points(points)
    _INDIA_MASK_300 = mask.reshape(300, 300)


# Build the cache immediately when the module is imported
if HAS_CARTOPY:
    try:
        _build_india_cache()
    except Exception:
        pass  # Fail silently – will attempt again inside plot_map if needed


# ─── Also cache the NaturalEarth state-lines feature ────────────────────────
_STATES_FEATURE = None

def _get_states_feature():
    global _STATES_FEATURE
    if _STATES_FEATURE is None and HAS_CARTOPY:
        _STATES_FEATURE = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none')
    return _STATES_FEATURE


class Visualizer:
    def __init__(self):
        pass

    def create_figure(self, figsize=(6, 5)):
        fig_kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if HAS_CARTOPY else {}
        return plt.subplots(figsize=figsize, **fig_kw)

    def plot_map(self, ax, grid, title, cmap_name, vmin, vmax, is_discrete=True):
        # Ensure the mask cache is ready (no-op if already built)
        _build_india_cache()

        if HAS_CARTOPY:
            ax.set_extent([66, 99, 6, 39], crs=ccrs.PlateCarree())
            states = _get_states_feature()
            if states is not None:
                ax.add_feature(states, edgecolor='dimgray', linewidth=0.4, linestyle=':')

        # ── Colormap ──────────────────────────────────────────────────────────
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad('white')

        # ── Interpolate raw grid → 300×300 ────────────────────────────────────
        lon_raw = np.linspace(67.5, 97.5, grid.shape[1])
        lat_raw = np.linspace(7.5,  37.5, grid.shape[0])

        interp = RegularGridInterpolator(
            (lat_raw, lon_raw), grid,
            bounds_error=False, fill_value=np.nan
        )
        grid_hi = interp((_LAT_HI, _LON_HI))

        # ── Apply the pre-built India mask ────────────────────────────────────
        if HAS_CARTOPY and _INDIA_MASK_300 is not None:
            grid_final = np.where(_INDIA_MASK_300, grid_hi, np.nan)
        else:
            grid_final = grid_hi

        Lon_final, Lat_final = _LON_HI, _LAT_HI

        # ── Draw India political boundary (crisp black outline) ───────────────
        if HAS_CARTOPY and _INDIA_PATHS:
            for path in _INDIA_PATHS:
                patch = mpatches.PathPatch(
                    path, facecolor='none', edgecolor='black',
                    linewidth=1.5, transform=ccrs.PlateCarree(), zorder=10
                )
                ax.add_patch(patch)

        # ── Normalisation ─────────────────────────────────────────────────────
        if is_discrete:
            levels = plt.MaxNLocator(nbins=12).tick_values(vmin, vmax)
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # ── Plot ──────────────────────────────────────────────────────────────
        if HAS_CARTOPY:
            im = ax.pcolormesh(
                Lon_final, Lat_final, grid_final,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), shading='nearest'
            )
        else:
            im = ax.pcolormesh(
                Lon_final, Lat_final, grid_final,
                cmap=cmap, norm=norm, shading='nearest'
            )

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        return im

    def plot_comparison(self, act_denorm, pred_denorm, var_name="Rainfall", unit=""):
        is_temp = (var_name == "Temperature")
        cmap = "nipy_spectral" if is_temp else "Blues"

        error = np.abs(act_denorm - pred_denorm)
        vmin  = min(np.nanmin(act_denorm), np.nanmin(pred_denorm))
        vmax  = max(np.nanmax(act_denorm), np.nanmax(pred_denorm))

        fig_kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if HAS_CARTOPY else {}
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), **fig_kw)

        im0 = self.plot_map(axes[0], act_denorm,  f"Actual {var_name} {unit}",    cmap,  vmin, vmax)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(f"{var_name} {unit}")

        im1 = self.plot_map(axes[1], pred_denorm, f"Predicted {var_name} {unit}", cmap,  vmin, vmax)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label(f"{var_name} {unit}")

        im2 = self.plot_map(axes[2], error, "Absolute Error Heatmap", "Reds", 0, np.nanmax(error), is_discrete=False)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label("Absolute Error")

        plt.tight_layout()
        return fig
