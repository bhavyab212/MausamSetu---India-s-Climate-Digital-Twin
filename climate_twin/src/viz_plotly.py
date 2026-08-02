"""
Plotly Visualization Layer — High-Fidelity Animated Twin
Static maps: 2× upsample clipped to the IMD land mask only (no ocean bleed, no smear).
"""
import plotly.graph_objects as go
import numpy as np
import json
import os
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage as ndimage

# ── Global Fixed Scales ───────────────────────────────────────────────────────
FIXED_SCALES = {
    'rain_annual':   {'range': [0, 3000],   'colorscale': 'Blues',   'unit': 'mm'},
    'temp_annual':   {'range': [20, 50],    'colorscale': 'YlOrRd', 'unit': '°C'},
    'rain_daily':    {'range': [0, 100],    'colorscale': 'Blues',   'unit': 'mm'},
    'temp_daily':    {'range': [15, 50],    'colorscale': 'YlOrRd', 'unit': '°C'},
    'sensitivity':   {'range': [-50, 50],   'colorscale': 'RdBu_r',   'unit': 'mm/°C'},
    'error':         {'range': [0, 500],    'colorscale': 'Reds',   'unit': ''},
    'uncertainty':   {'range': [0, 5],      'colorscale': 'Purples','unit': 'σ'},
}

MAP_HEIGHT = 550
MAP_WIDTH  = 650


class PlotlyVisualizer:
    def __init__(self, geojson_path, binary_mask,
                 lat_min=6.5, lat_max=37.5, lon_min=67.5, lon_max=97.5):
        self.binary_mask = binary_mask
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max

        if binary_mask is not None:
            h, w = binary_mask.shape
            self.lats = np.linspace(lat_min, lat_max, h)
            self.lons = np.linspace(lon_min, lon_max, w)

        # Load India border as single concatenated trace
        self.border_lons = []
        self.border_lats = []
        if os.path.exists(geojson_path):
            with open(geojson_path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
            self._extract_borders(gj)
        self._init_plotly_upsample_grid()

    def _extract_borders(self, geojson):
        all_lons, all_lats = [], []
        for feature in geojson.get('features', []):
            geom = feature.get('geometry', {})
            gtype = geom.get('type', '')
            coords_list = geom.get('coordinates', [])
            polys = []
            if gtype == 'Polygon':
                polys = [coords_list[0]]
            elif gtype == 'MultiPolygon':
                polys = [ring[0] for ring in coords_list]
            for poly in polys:
                for p in poly:
                    all_lons.append(p[0])
                    all_lats.append(p[1])
                all_lons.append(None)
                all_lats.append(None)
        self.border_lons = all_lons
        self.border_lats = all_lats

    def _init_plotly_upsample_grid(self):
        """2× finer grid + nearest land mask from ``binary_mask`` only (same extent as data)."""
        self._up_lons = None
        self._up_lats = None
        self._up_land = None
        if self.binary_mask is None:
            self._up_fac = 2
            return
        h, w = self.binary_mask.shape
        # Small regions (e.g. Cauvery 19×17) get a much finer display grid so the
        # map isn't blocky; large grids (India) stay light to protect payload.
        cells = h * w
        self._up_fac = 6 if cells <= 1500 else (4 if cells <= 4000 else 2)
        fac = int(self._up_fac)
        lat_c = np.linspace(self.lat_min, self.lat_max, h)
        lon_c = np.linspace(self.lon_min, self.lon_max, w)
        nh, nw = h * fac, w * fac
        self._up_lats = np.linspace(self.lat_min, self.lat_max, nh)
        self._up_lons = np.linspace(self.lon_min, self.lon_max, nw)
        lon_m, lat_m = np.meshgrid(self._up_lons, self._up_lats)
        m_nearest = RegularGridInterpolator(
            (lat_c, lon_c),
            self.binary_mask.astype(np.float64),
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        self._up_land = m_nearest((lat_m, lon_m)) > 0.5

    def _prepare_display_grid(self, grid, upsample=True):
        """
        Optional 2× upsampling for sharper cells at the coast; values exist **only**
        where ``binary_mask`` upsampled is land (no Cartopy mask, no pandas smoothing).
        """
        grid = np.asarray(grid, dtype=np.float64)
        h, w = grid.shape
        if h == 0 or w == 0:
            return grid, self.lons, self.lats

        if self.binary_mask is not None:
            m = self.binary_mask == 1
            z = np.where(m, grid, np.nan)
        else:
            z = np.where(np.isfinite(grid), grid, np.nan)

        if not upsample or self._up_land is None or self._up_lons is None:
            return z, self.lons, self.lats

        lat_c = np.linspace(self.lat_min, self.lat_max, h)
        lon_c = np.linspace(self.lon_min, self.lon_max, w)
        lon_m, lat_m = np.meshgrid(self._up_lons, self._up_lats)
        z_hi = RegularGridInterpolator(
            (lat_c, lon_c), z, method="linear", bounds_error=False, fill_value=np.nan
        )((lat_m, lon_m))
        z_hi = np.where(self._up_land, z_hi, np.nan)
        return z_hi, self._up_lons, self._up_lats

    def _add_india_outline(self, fig, row=None, col=None):
        """Draw state boundaries (clipped to the current extent by fixed axis ranges)
        for geographic context — a light 'map' under the data."""
        if not self.border_lons:
            return
        trace = go.Scatter(
            x=self.border_lons,
            y=self.border_lats,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.35)", width=0.8),
            hoverinfo="skip",
            showlegend=False,
        )
        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    def _heatmap_trace(
        self,
        grid,
        colorscale,
        zmin,
        zmax,
        hovertemplate,
        colorbar=None,
        opacity=1.0,
        showscale=True,
        custom_colorscale=None,
    ):
        # Upsample only small grids (Cauvery) for real resolution; large grids
        # (India) stay native and rely on zsmooth to render smoothly (no payload blowup).
        _g = np.asarray(grid)
        do_up = (_g.shape[0] * _g.shape[1]) <= 1500
        z, x, y = self._prepare_display_grid(grid, upsample=do_up)
        cs = custom_colorscale if custom_colorscale is not None else colorscale
        kw = dict(
            z=z,
            x=x,
            y=y,
            colorscale=cs,
            zmin=zmin,
            zmax=zmax,
            connectgaps=False,
            hoverongaps=False,
            zsmooth="best",   # smooth, high-res look instead of blocky cells
            opacity=opacity,
            showscale=showscale,
            hovertemplate=hovertemplate,
        )
        if colorbar is not None:
            kw["colorbar"] = colorbar
        return go.Heatmap(**kw)

    def plot_map(self, grid, title, scale_key, custom_range=None):
        scale = FIXED_SCALES.get(scale_key, FIXED_SCALES['rain_annual'])
        # Clean the title - remove 'Max' if present
        display_title = title.replace("Max Temp", "Temperature").replace("MaxT", "Temperature")
        if custom_range is not None:
            zmin, zmax = float(custom_range[0]), float(custom_range[1])
        elif 'temp' in scale_key:
            # Default daily-style bounds unless caller passes custom_range
            zmin, zmax = 0.0, 50.0
        else:
            zmin, zmax = scale['range']

        fig = go.Figure()

        fig.add_trace(
            self._heatmap_trace(
                grid,
                colorscale="Turbo" if "temp" in scale_key else scale["colorscale"],
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title=dict(text=scale["unit"]),
                    len=0.75,
                    yanchor="middle",
                    y=0.5,
                ),
                hovertemplate=(
                    "Temperature: %{z:.1f} °C<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>"
                    if "temp" in scale_key
                    else (
                        "Sensitivity: %{z:.2f} mm/°C<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>"
                        if scale_key == "sensitivity"
                        else "Rain: %{z:.1f} mm<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>"
                    )
                ),
            )
        )

        self._add_india_outline(fig)

        fig.update_layout(
            title=dict(
                text=display_title,
                font=dict(size=14, color="white"),
                x=0.05,
                y=0.95
            ),
            xaxis=dict(
                title="Longitude",
                range=[self.lon_min, self.lon_max],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                title="Latitude",
                range=[self.lat_min, self.lat_max],
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False
            ),
            height=MAP_HEIGHT,
            width=MAP_WIDTH,
            margin=dict(l=50, r=20, t=50, b=40),
            template="plotly_dark"
        )
        return fig

    def _axis_ranges(self):
        return [self.lon_min, self.lon_max], [self.lat_min, self.lat_max]

    def plot_sensitivity_map(self, slope_grid):
        return self.plot_map(slope_grid, "Rainfall Sensitivity (mm per +1°C)", 'sensitivity')

    def plot_single_animation(
        self,
        grids,
        day_numbers,
        title,
        scale_key,
        speed_ms=180,
        custom_range=None,
    ):
        """Create a clean day-wise animation for one variable (rain or temp)."""
        scale = FIXED_SCALES.get(scale_key, FIXED_SCALES["rain_daily"])
        zmin, zmax = custom_range if custom_range else scale["range"]

        _, x_plot, y_plot = self._prepare_display_grid(grids[0], upsample=False)

        def frame_hm(g):
            z, _, _ = self._prepare_display_grid(g, upsample=False)
            return go.Heatmap(
                z=z,
                x=x_plot,
                y=y_plot,
                colorscale=scale["colorscale"],
                zmin=zmin,
                zmax=zmax,
                connectgaps=False,
                hoverongaps=False,
                showscale=False,
            )

        fig = go.Figure(
            data=[
                self._heatmap_trace(
                    grids[0],
                    colorscale=scale["colorscale"],
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title=dict(text=scale["unit"]), len=0.8),
                    hovertemplate="Lat: %{y:.2f}<br>Lon: %{x:.2f}<br>Value: %{z:.2f}<extra></extra>",
                )
            ]
        )
        self._add_india_outline(fig)

        frames = []
        for i, d in enumerate(day_numbers):
            frames.append(
                go.Frame(
                    name=str(d),
                    traces=[0],
                    data=[frame_hm(grids[i])],
                )
            )
        fig.frames = frames

        steps = [
            dict(
                method="animate",
                args=[
                    [str(d)],
                    {"mode": "immediate", "frame": {"duration": speed_ms, "redraw": True}, "transition": {"duration": 0}},
                ],
                label=str(d),
            )
            for d in day_numbers
        ]

        lon_r, lat_r = self._axis_ranges()
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis=dict(title="Longitude", range=lon_r),
            yaxis=dict(title="Latitude", range=lat_r, scaleanchor="x", scaleratio=1),
            height=MAP_HEIGHT,
            width=MAP_WIDTH,
            margin=dict(l=60, r=30, t=50, b=50),
            template="plotly_dark",
            sliders=[dict(currentvalue={"prefix": "Day: "}, steps=steps)],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=1.12,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": speed_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
        )
        return fig

    def plot_dual_panel_animation(
        self,
        rain_grids,
        temp_grids,
        day_numbers,
        title,
        rain_range,
        temp_range,
        speed_ms=180,
    ):
        """One timeline, two maps: rainfall + temperature."""
        from plotly.subplots import make_subplots

        _, x_plot, y_plot = self._prepare_display_grid(rain_grids[0], upsample=False)

        def dual_frame_hm(g, colorscale, zmin, zmax):
            z, _, _ = self._prepare_display_grid(g, upsample=False)
            kwargs = {}
            return go.Heatmap(
                z=z,
                x=x_plot,
                y=y_plot,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                connectgaps=False,
                hoverongaps=False,
                showscale=False,
                **kwargs
            )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Rainfall (mm/day)", "Temperature (°C)"),
            horizontal_spacing=0.08,
        )

        fig.add_trace(
            self._heatmap_trace(
                rain_grids[0],
                colorscale=FIXED_SCALES["rain_daily"]["colorscale"],
                zmin=rain_range[0],
                zmax=rain_range[1],
                colorbar=dict(title=dict(text="mm/day"), len=0.75, x=0.46),
                hovertemplate="Rain: %{z:.2f}<br>Lat:%{y:.2f}<br>Lon:%{x:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            self._heatmap_trace(
                temp_grids[0],
                colorscale="Turbo",
                zmin=temp_range[0],
                zmax=temp_range[1],
                colorbar=dict(title=dict(text="°C"), len=0.75, x=1.02),
                hovertemplate="Temp: %{z:.2f}<br>Lat:%{y:.2f}<br>Lon:%{x:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        self._add_india_outline(fig, row=1, col=1)
        self._add_india_outline(fig, row=1, col=2)

        frames = []
        for i, d in enumerate(day_numbers):
            frames.append(
                go.Frame(
                    name=str(d),
                    traces=[0, 1],
                    data=[
                        dual_frame_hm(
                            rain_grids[i],
                            FIXED_SCALES["rain_daily"]["colorscale"],
                            rain_range[0],
                            rain_range[1]
                        ),
                        dual_frame_hm(
                            temp_grids[i],
                            "Turbo",
                            temp_range[0],
                            temp_range[1]
                        ),
                    ],
                )
            )
        fig.frames = frames

        steps = [
            dict(
                method="animate",
                args=[[str(d)], {"mode": "immediate", "frame": {"duration": speed_ms, "redraw": True}, "transition": {"duration": 0}}],
                label=str(d),
            )
            for d in day_numbers
        ]

        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            template="plotly_dark",
            height=600,
            margin=dict(l=40, r=40, t=60, b=40),
            sliders=[dict(currentvalue={"prefix": "Day: "}, steps=steps)],
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.01, y=1.12, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": speed_ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]),
                ]
            )],
        )

        lon_r, lat_r = self._axis_ranges()
        fig.update_xaxes(range=lon_r, title="Longitude", row=1, col=1)
        fig.update_yaxes(range=lat_r, title="Latitude", scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_xaxes(range=lon_r, title="Longitude", row=1, col=2)
        fig.update_yaxes(range=lat_r, title="Latitude", scaleanchor="x2", scaleratio=1, row=1, col=2)
        return fig

    def plot_destine_climate_spiral(
        self,
        df,
        *,
        variable="Temp",
        baseline_label="1975–1990 India land mean",
        title_main="ISRO India Climate Digital Twin",
        title_sub="Monthly 2m temperature anomaly spiral (1975–2075)",
    ):
        """
        DestinE-inspired 2D climate spiral (Ed Hawkins / Copernicus style).
        df columns: Year, Month, Value (°C anomaly vs baseline).
        """
        import plotly.graph_objects as go

        df = df.sort_values(["Year", "Month"]).copy()
        min_year = int(df["Year"].min())
        max_year = int(df["Year"].max())
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        # Polar angle: January at top (12 o'clock)
        df["theta"] = (df["Month"] - 1) / 12.0 * 2.0 * np.pi - (np.pi / 2.0)
        df["r"] = df["Value"].astype(float)

        def _xy(rr, tt):
            return rr * np.cos(tt), rr * np.sin(tt)

        df["x"], df["y"] = _xy(df["r"].values, df["theta"].values)

        # Dynamic plot bounds from data + reference rings
        r_data_max = float(np.nanmax(df["r"])) if len(df) else 2.0
        r_plot_max = max(2.4, r_data_max * 1.15, 2.0)
        label_r = r_plot_max * 1.12

        bg = "#071322"
        fig = go.Figure()

        # Reference anomaly rings (Paris 1.5°C, 2°C targets)
        ring_specs = [
            (0.0, "0 °C", "rgba(72, 220, 160, 0.95)", 2.2),
            (1.0, "1 °C", "rgba(255, 255, 255, 0.85)", 1.8),
            (1.5, "1.5 °C (Paris)", "rgba(120, 200, 255, 0.9)", 1.6),
            (2.0, "2 °C", "rgba(255, 150, 90, 0.9)", 1.6),
        ]
        t_circ = np.linspace(0, 2 * np.pi, 360)
        for r_ref, _lbl, col, lw in ring_specs:
            if r_ref > r_plot_max * 1.05:
                continue
            cx, cy = _xy(np.full_like(t_circ, r_ref), t_circ)
            fig.add_trace(
                go.Scatter(
                    x=cx,
                    y=cy,
                    mode="lines",
                    line=dict(color=col, width=lw),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Month labels around the rim
        for mi, name in enumerate(months):
            ang = mi / 12.0 * 2.0 * np.pi - (np.pi / 2.0)
            lx, ly = _xy(label_r, ang)
            fig.add_trace(
                go.Scatter(
                    x=[lx],
                    y=[ly],
                    mode="text",
                    text=[name],
                    textfont=dict(size=13, color="#e879f9", family="Arial Black"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        years = sorted(df["Year"].unique())

        def _spiral_trace(sub):
            cd = np.column_stack([sub["Year"], sub["Month"], sub["r"]])
            return go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="lines+markers",
                line=dict(width=2.4, color="rgba(170, 200, 255, 0.55)"),
                marker=dict(
                    size=4,
                    color=sub["Year"].to_numpy(),
                    colorscale=[
                        [0.0, "#1e3a8a"],
                        [0.35, "#0ea5e9"],
                        [0.65, "#fbbf24"],
                        [1.0, "#dc2626"],
                    ],
                    cmin=min_year,
                    cmax=max_year,
                    line=dict(width=0.5, color="rgba(255,255,255,0.35)"),
                ),
                hovertemplate="Year %{customdata[0]}<br>Month %{customdata[1]}<br>Anomaly %{customdata[2]:.2f} °C<extra></extra>",
                customdata=cd,
                showlegend=False,
            )

        def _year_annos(year):
            return [
                dict(
                    text=f"<b>{int(year)}</b>",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=56, color="#ffffff", family="Arial Black"),
                ),
                dict(
                    text="Global monthly temperature anomaly — India land mean (°C)",
                    x=0.5,
                    y=-0.06,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#a8b8d8"),
                ),
                dict(
                    text=f"<span style='color:#4ea8ff'>{min_year}</span>"
                    " — — — "
                    f"<span style='color:#ff6b4a'>{max_year}</span>",
                    x=0.5,
                    y=-0.11,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=11, color="#8899bb"),
                ),
            ]

        sub0 = df[df["Year"] <= years[0]]
        fig.add_trace(_spiral_trace(sub0))
        spiral_idx = len(fig.data) - 1

        frames = []
        for year in years:
            sub = df[df["Year"] <= year]
            frames.append(
                go.Frame(
                    name=str(int(year)),
                    data=[_spiral_trace(sub)],
                    traces=[spiral_idx],
                    layout=go.Layout(annotations=_year_annos(year)),
                )
            )

        fig.frames = frames

        # Timeline slider + play/pause
        steps = []
        for y in years:
            steps.append(
                dict(
                    method="animate",
                    args=[
                        [str(int(y))],
                        dict(
                            mode="immediate",
                            frame=dict(duration=120, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=str(int(y)),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            title=dict(
                text=(
                    f"<b>{title_main}</b><br>"
                    f"<span style='font-size:14px;color:#9fb3d9;'>{title_sub}</span><br>"
                    f"<span style='font-size:12px;color:#7d91b3;'>Baseline: {baseline_label}</span>"
                ),
                x=0.5,
                xanchor="center",
                y=0.98,
                font=dict(size=20, color="#ffffff"),
            ),
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
            margin=dict(l=30, r=30, t=110, b=90),
            height=720,
            showlegend=False,
            annotations=_year_annos(years[0]),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.12,
                    y=1.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(12, 24, 48, 0.85)",
                    bordercolor="#3d5a80",
                    font=dict(color="#e8eef8", size=12),
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=120, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="❚❚ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.12,
                    y=0.02,
                    len=0.76,
                    xanchor="left",
                    bgcolor="rgba(20, 35, 60, 0.6)",
                    bordercolor="#3d5a80",
                    tickcolor="#9fb3d9",
                    font=dict(color="#c5d4ef", size=10),
                    currentvalue=dict(
                        prefix="Year: ",
                        visible=True,
                        xanchor="center",
                        font=dict(size=13, color="#ffffff"),
                    ),
                    steps=steps,
                )
            ],
        )
        return fig

    def get_climate_spiral_html(self, df, variable='Temperature', title="Global monthly temperature anomaly"):
        """
        Generates a perfectly smooth, 60fps HTML5 Canvas animation matching the DestinE spiral exactly.
        df should have columns: ['Year', 'Month', 'Value']
        """
        import json
        import numpy as np
        
        # 1. Prepare data
        min_val = df['Value'].min()
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        
        if 'Temp' in variable:
            offset = 1.5
            suffix = '°C'
            color_mode = 'temp'
            ticks = [-1, 0, 1, 2]
            max_rad = 4.0
        else:
            offset = 10.0
            suffix = ' mm'
            color_mode = 'rain'
            ticks = [0, 50, 100, 150]
            max_rad = max(float(df['Value'].max()) + offset, 200.0)
            
        df['Radius'] = df['Value'] + offset
        
        # Interpolate purely in JS for performance, or pass full data.
        # It's better to pass monthly data and let JS interpolate at 60fps!
        data_json = df[['Year', 'Month', 'Value', 'Radius']].to_dict(orient='records')
        
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {{ 
                background: radial-gradient(circle at center, #1e293b 0%, #020617 100%); 
                margin: 0; padding: 0; 
                display: flex; justify-content: center; align-items: center; 
                height: 100vh; overflow: hidden;
                font-family: 'Outfit', -apple-system, sans-serif;
            }}
            canvas {{
                max-width: 100%;
                max-height: 100%;
            }}
        </style>
        </head>
        <body>
        <canvas id="spiral" width="800" height="850"></canvas>
        <script>
            const data = {json.dumps(data_json)};
            const canvas = document.getElementById('spiral');
            const ctx = canvas.getContext('2d');
            
            const cx = 400;
            const cy = 400;
            const maxR = 300;
            
            const minYear = {min_year};
            const maxYear = {max_year};
            const colorMode = '{color_mode}';
            const suffix = '{suffix}';
            const offset = {offset};
            const title = '{title}';
            const maxDataRad = {max_rad};
            
            // Generate smoothly interpolated path
            const path = [];
            for(let i=0; i<data.length-1; i++) {{
                let p1 = data[i];
                let p2 = data[i+1];
                
                // 30 interpolated points per month
                let steps = 30;
                for(let s=0; s<steps; s++) {{
                    let t = s / steps;
                    
                    let y = p1.Year + t * (p2.Year - p1.Year);
                    let val = p1.Value + t * (p2.Value - p1.Value);
                    let rad = p1.Radius + t * (p2.Radius - p1.Radius);
                    
                    // Angle: Jan=0 (top), going clockwise
                    // Month is 1..12
                    let m1 = p1.Month - 1;
                    let m2 = p2.Month - 1;
                    // Fix wraparound
                    if(m2 < m1) m2 += 12;
                    let m_interp = m1 + t * (m2 - m1);
                    
                    let angle = (m_interp / 12) * Math.PI * 2 - Math.PI/2 + (y - minYear) * 0.05;
                    
                    // Map rad to pixels using dynamic maxDataRad
                    let pixelR = (Math.max(0, rad) / maxDataRad) * maxR;
                    
                    path.push({{
                        x: cx + pixelR * Math.cos(angle),
                        y: cy + pixelR * Math.sin(angle),
                        year: y,
                        val: val
                    }});
                }}
            }}
            
            function getColor(year) {{
                let norm = (year - minYear) / Math.max(1, (maxYear - minYear));
                if (colorMode === 'temp') {{
                    // Blue to Red
                    if (norm < 0.5) {{
                        let r = Math.floor(norm * 2 * 255);
                        let g = Math.floor(norm * 2 * 255);
                        return `rgb(${{r}},${{g}},255)`;
                    }} else {{
                        let b = Math.floor((1 - norm) * 2 * 255);
                        let g = Math.floor((1 - norm) * 2 * 255);
                        return `rgb(255,${{g}},${{b}})`;
                    }}
                }} else {{
                    // Cyan to Dark Blue
                    let r = Math.floor((1 - norm) * 0);
                    let g = Math.floor((1 - norm) * 255);
                    let b = 255;
                    return `rgb(${{r}},${{g}},${{b}})`;
                }}
            }}
            
            let currentFrame = 0;
            const drawSpeed = 25; // points per frame
            
            function draw() {{
                ctx.clearRect(0, 0, 800, 850);
                
                // Draw path segments FIRST so grid scales are drawn ON TOP of them and remain clearly visible
                ctx.lineWidth = 3;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                
                // We add glow to the line
                ctx.shadowBlur = 10;
                
                for(let i=1; i<currentFrame; i++) {{
                    ctx.beginPath();
                    ctx.moveTo(path[i-1].x, path[i-1].y);
                    ctx.lineTo(path[i].x, path[i].y);
                    
                    let color = getColor(path[i].year);
                    ctx.strokeStyle = color;
                    // Shadow color matches stroke for neon effect
                    ctx.shadowColor = color;
                    ctx.stroke();
                }}
                
                // Reset shadow for grid & text
                ctx.shadowBlur = 0;
                ctx.shadowColor = 'transparent';
                
                // Draw Grid (Scales) ON TOP of the blue paths
                ctx.lineWidth = 1;
                const tickValues = {ticks};
                for(let v of tickValues) {{
                    let rad = v + offset;
                    let pixelR = (Math.max(0, rad) / maxDataRad) * maxR;
                    if(pixelR > 0 && pixelR < 380) {{
                        ctx.beginPath();
                        ctx.arc(cx, cy, pixelR, 0, 2*Math.PI);
                        ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
                        ctx.setLineDash([5, 5]);
                        ctx.stroke();
                        
                        // Label
                        ctx.fillStyle = '#ffffff';
                        ctx.font = '500 12px Outfit, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'bottom';
                        ctx.fillText(v + suffix, cx, cy - pixelR - 2);
                    }}
                }}
                ctx.setLineDash([]);
                
                // Months
                const months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
                for(let i=0; i<12; i++) {{
                    let angle = (i / 12) * Math.PI * 2 - Math.PI/2;
                    let px = cx + (maxR + 30) * Math.cos(angle);
                    let py = cy + (maxR + 30) * Math.sin(angle);
                    ctx.fillStyle = '#cbd5e1';
                    ctx.font = '600 14px Outfit, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(months[i], px, py);
                }}
                
                // Draw tip
                if(currentFrame > 0 && currentFrame < path.length) {{
                    let p = path[currentFrame-1];
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 4, 0, 2*Math.PI);
                    ctx.fillStyle = '#fff';
                    ctx.shadowColor = '#fff';
                    ctx.fill();
                    
                    // Draw central text
                    ctx.fillStyle = '#ffffff';
                    ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
                    ctx.shadowBlur = 15;
                    ctx.font = '700 45px Outfit, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(Math.floor(p.year), cx, cy);
                }} else if (currentFrame >= path.length) {{
                    let p = path[path.length-1];
                    ctx.fillStyle = '#ffffff';
                    ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
                    ctx.shadowBlur = 15;
                    ctx.font = '700 45px Outfit, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(Math.floor(p.year), cx, cy);
                }}
                
                // Title
                ctx.fillStyle = '#e2e8f0';
                ctx.font = '600 24px Outfit, sans-serif';
                ctx.textAlign = 'center';
                ctx.shadowBlur = 0;
                ctx.fillText(title, 400, 800);
                
                if (currentFrame < path.length) {{
                    currentFrame = Math.min(currentFrame + drawSpeed, path.length);
                    requestAnimationFrame(draw);
                }}
            }}
            
            draw();
        </script>
        </body>
        </html>
        """
        return html_code

    def plot_orthographic_globe_dual_animation(
        self,
        rain_grids,
        temp_grids,
        day_numbers,
        title,
        rain_range,
        temp_range,
        speed_ms=180,
        downsample=2,
        show_clouds=True,
        show_humidity=True,
        show_aqi=True,
        show_cyclone=True,
        show_wind=True,
    ):
        """
        DestinE-style immersive globe twin (Plotly-only, robust in Streamlit).
        Includes:
        - Rainfall heatmap (left) + Temp heatmap (right) with separate scales
        - Cloud coverage (derived from rain) overlay
        - Humidity + AQI overlays (derived from rain/temp; no extra colorbars to avoid overlap)
        - Cyclone path + moving marker (procedural)
        - Wind overlay (procedural, rotates with timeline)
        """
        from plotly.subplots import make_subplots

        if self.binary_mask is None:
            raise ValueError("binary_mask is required for globe plotting.")

        ds = max(1, int(downsample))
        mask_ds = self.binary_mask[::ds, ::ds] == 1

        lat_patch = self.lats[::ds]
        lon_patch = self.lons[::ds]
        Lon2d, Lat2d = np.meshgrid(lon_patch, lat_patch)
        lat_rad = np.deg2rad(Lat2d)
        lon_rad = np.deg2rad(Lon2d)

        # Slightly raise India weather surface above base globe for cinematic separation.
        R_patch = 1.005
        X_patch = R_patch * np.cos(lat_rad) * np.cos(lon_rad)
        Y_patch = R_patch * np.cos(lat_rad) * np.sin(lon_rad)
        Z_patch = R_patch * np.sin(lat_rad)

        # Atmosphere shell.
        R_atm = 1.03
        X_atm = R_atm * np.cos(lat_rad) * np.cos(lon_rad)
        Y_atm = R_atm * np.cos(lat_rad) * np.sin(lon_rad)
        Z_atm = R_atm * np.sin(lat_rad)

        rain_colorscale = FIXED_SCALES["rain_daily"]["colorscale"]
        temp_colorscale = FIXED_SCALES["temp_daily"]["colorscale"]

        # Better looking base globe.
        lat_base = np.linspace(-90, 90, 48)
        lon_base = np.linspace(-180, 180, 96)
        LonB, LatB = np.meshgrid(lon_base, lat_base)
        latB_rad = np.deg2rad(LatB)
        lonB_rad = np.deg2rad(LonB)
        X_base = np.cos(latB_rad) * np.cos(lonB_rad)
        Y_base = np.cos(latB_rad) * np.sin(lonB_rad)
        Z_base = np.sin(latB_rad)
        # Faux earth-tone shading by latitude.
        base_color = (np.cos(2 * latB_rad) + 1.0) / 2.0

        def prep_patch_day(grid):
            g = np.array(grid[::ds, ::ds], dtype=np.float64)
            g = np.where(mask_ds, g, np.nan)
            return np.where(np.isfinite(g), g, np.nan)

        def derive_cloud_from_rain(r2d):
            r = np.where(np.isfinite(r2d), r2d, 0.0)
            cloud = (r - 1.0) / max(1.0, rain_range[1] * 0.25)
            cloud = np.clip(cloud, 0.0, 1.0)
            return np.where(np.isfinite(r2d), cloud, np.nan)

        def derive_humidity_from_rt(r2d, t2d):
            r = np.where(np.isfinite(r2d), r2d, 0.0)
            t = np.where(np.isfinite(t2d), t2d, 25.0)
            hum = 58.0 + 0.60 * r - 0.30 * (t - 25.0)
            hum = np.clip(hum, 0.0, 100.0)
            return np.where(np.isfinite(r2d) | np.isfinite(t2d), hum, np.nan)

        def derive_aqi_from_rt(r2d, t2d):
            r = np.where(np.isfinite(r2d), r2d, 0.0)
            t = np.where(np.isfinite(t2d), t2d, 25.0)
            wet = np.clip(r / max(1e-6, rain_range[1]), 0.0, 1.0)
            hot = np.clip((t - 25.0) / 10.0, 0.0, 1.0)
            aqi = 65.0 + 300.0 * (1.0 - wet) + 110.0 * hot
            aqi = np.clip(aqi, 0.0, 500.0)
            return np.where(np.isfinite(r2d) | np.isfinite(t2d), aqi, np.nan)

        # Cyclone path in Bay of Bengal toward land.
        path_lats = np.array([8, 10, 12, 14, 16, 18, 20, 22, 23, 24], dtype=np.float64)
        path_lons = np.array([92, 90, 88, 86, 84, 82, 79, 76, 73, 70], dtype=np.float64)
        plr = np.deg2rad(path_lats)
        pgr = np.deg2rad(path_lons)
        Xp = 1.01 * np.cos(plr) * np.cos(pgr)
        Yp = 1.01 * np.cos(plr) * np.sin(pgr)
        Zp = 1.01 * np.sin(plr)

        # Wind glyph anchors.
        wind_lats = np.array([10, 14, 18, 22, 26, 30], dtype=np.float64)
        wind_lons = np.array([70, 76, 82, 88, 94], dtype=np.float64)
        wind_lat2, wind_lon2 = np.meshgrid(wind_lats, wind_lons, indexing="ij")
        wlats = wind_lat2.flatten()
        wlons = wind_lon2.flatten()
        wlr = np.deg2rad(wlats)
        wgr = np.deg2rad(wlons)
        Wsx = 1.01 * np.cos(wlr) * np.cos(wgr)
        Wsy = 1.01 * np.cos(wlr) * np.sin(wgr)
        Wsz = 1.01 * np.sin(wlr)

        def build_wind_lines(frame_frac, arrow_len=0.07):
            theta = 2.0 * np.pi * frame_frac

            east_x = -np.sin(wgr)
            east_y = np.cos(wgr)
            east_z = np.zeros_like(east_x)
            north_x = -np.sin(wlr) * np.cos(wgr)
            north_y = -np.sin(wlr) * np.sin(wgr)
            north_z = np.cos(wlr)

            u_e = 0.8 * np.cos(theta) + 0.2
            u_n = 0.8 * np.sin(theta) + 0.1
            dir_x = u_e * east_x + u_n * north_x
            dir_y = u_e * east_y + u_n * north_y
            dir_z = u_e * east_z + u_n * north_z

            norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2) + 1e-12
            dir_x /= norm
            dir_y /= norm
            dir_z /= norm

            ex = Wsx + arrow_len * dir_x
            ey = Wsy + arrow_len * dir_y
            ez = Wsz + arrow_len * dir_z

            e_norm = np.sqrt(ex**2 + ey**2 + ez**2) + 1e-12
            ex /= e_norm
            ey /= e_norm
            ez /= e_norm

            x_arr, y_arr, z_arr = [], [], []
            for k in range(len(Wsx)):
                x_arr.extend([float(Wsx[k]), float(ex[k]), None])
                y_arr.extend([float(Wsy[k]), float(ey[k]), None])
                z_arr.extend([float(Wsz[k]), float(ez[k]), None])
            return x_arr, y_arr, z_arr

        n_days = len(day_numbers)
        subframes = 2  # smooth flow between days

        def interp_grid(g0, g1, a):
            return (1.0 - a) * g0 + a * g1

        rain0 = prep_patch_day(rain_grids[0])
        temp0 = prep_patch_day(temp_grids[0])
        cloud0 = derive_cloud_from_rain(rain0) if show_clouds else np.full_like(rain0, np.nan)
        hum0 = derive_humidity_from_rt(rain0, temp0) if show_humidity else np.full_like(temp0, np.nan)
        aqi0 = derive_aqi_from_rt(rain0, temp0) if show_aqi else np.full_like(temp0, np.nan)

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            horizontal_spacing=0.04,
        )

        # Base spheres.
        fig.add_trace(
            go.Surface(
                x=X_base, y=Y_base, z=Z_base,
                surfacecolor=base_color,
                colorscale=[[0.0, "#14315e"], [0.45, "#2f5f99"], [1.0, "#95b6d7"]],
                cmin=0, cmax=1,
                showscale=False,
                opacity=1.0, hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Surface(
                x=X_base, y=Y_base, z=Z_base,
                surfacecolor=base_color,
                colorscale=[[0.0, "#14315e"], [0.45, "#2f5f99"], [1.0, "#95b6d7"]],
                cmin=0, cmax=1,
                showscale=False,
                opacity=1.0, hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # Rain patch (trace index 2)
        fig.add_trace(
            go.Surface(
                x=X_patch, y=Y_patch, z=Z_patch,
                surfacecolor=rain0,
                colorscale=rain_colorscale,
                cmin=rain_range[0], cmax=rain_range[1],
                showscale=True,
                colorbar=dict(title="Rain (mm/day)", len=0.56, thickness=12, x=0.47, y=0.48, xanchor="left"),
                opacity=0.98,
                hovertemplate="Rain: %{surfacecolor:.1f} mm/day<extra></extra>",
            ),
            row=1, col=1,
        )
        # Temp patch (trace index 3)
        fig.add_trace(
            go.Surface(
                x=X_patch, y=Y_patch, z=Z_patch,
                surfacecolor=temp0,
                colorscale=temp_colorscale,
                cmin=temp_range[0], cmax=temp_range[1],
                showscale=True,
                colorbar=dict(title="Temp (°C)", len=0.56, thickness=12, x=0.995, y=0.48, xanchor="right"),
                opacity=0.98,
                hovertemplate="Temp: %{surfacecolor:.1f} °C<extra></extra>",
            ),
            row=1, col=2,
        )

        # Cloud overlay (trace index 4)
        fig.add_trace(
            go.Surface(
                x=X_patch, y=Y_patch, z=Z_patch,
                surfacecolor=cloud0,
                colorscale=[
                    [0.0, "rgba(255,255,255,0.00)"],
                    [0.4, "rgba(255,255,255,0.28)"],
                    [1.0, "rgba(255,255,255,0.95)"],
                ],
                cmin=0, cmax=1, showscale=False,
                opacity=0.95 if show_clouds else 0.0,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

        # Atmospheric glow shell (visual polish).
        fig.add_trace(
            go.Surface(
                x=X_atm, y=Y_atm, z=Z_atm,
                surfacecolor=np.ones_like(Z_atm),
                colorscale=[[0.0, "rgba(120,170,255,0.0)"], [1.0, "rgba(120,170,255,0.35)"]],
                cmin=0, cmax=1,
                showscale=False,
                opacity=0.28,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Surface(
                x=X_atm, y=Y_atm, z=Z_atm,
                surfacecolor=np.ones_like(Z_atm),
                colorscale=[[0.0, "rgba(120,170,255,0.0)"], [1.0, "rgba(120,170,255,0.35)"]],
                cmin=0, cmax=1,
                showscale=False,
                opacity=0.28,
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # Humidity overlay (trace index 5)
        fig.add_trace(
            go.Surface(
                x=X_patch, y=Y_patch, z=Z_patch,
                surfacecolor=hum0,
                colorscale="YlGnBu",
                cmin=0, cmax=100, showscale=False,
                opacity=0.22 if show_humidity else 0.0,
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # AQI overlay (trace index 6)
        fig.add_trace(
            go.Surface(
                x=X_patch, y=Y_patch, z=Z_patch,
                surfacecolor=aqi0,
                colorscale="RdYlGn_r",
                cmin=0, cmax=500, showscale=False,
                opacity=0.16 if show_aqi else 0.0,
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # Cyclone path line and marker.
        fig.add_trace(
            go.Scatter3d(
                x=Xp, y=Yp, z=Zp,
                mode="lines",
                line=dict(color="rgba(255,170,0,0.85)", width=5),
                hoverinfo="skip",
                showlegend=False,
                visible=True if show_cyclone else False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[float(Xp[0])], y=[float(Yp[0])], z=[float(Zp[0])],
                mode="markers",
                marker=dict(size=7, color="orange"),
                hoverinfo="skip",
                showlegend=False,
                visible=True if show_cyclone else False,
            ),
            row=1, col=1,
        )

        # Wind arrows (trace index 9)
        wx0, wy0, wz0 = build_wind_lines(0.0) if show_wind else ([], [], [])
        fig.add_trace(
            go.Scatter3d(
                x=wx0, y=wy0, z=wz0,
                mode="lines",
                line=dict(color="rgba(120,220,255,0.75)", width=4),
                hoverinfo="skip",
                showlegend=False,
                visible=True if show_wind else False,
            ),
            row=1, col=1,
        )

        # Trace indices to update in frames:
        # 2 rain patch, 3 temp patch, 4 cloud, 5 hum, 6 aqi, 8 cyclone marker, 9 wind
        frames = []
        frame_names = []
        day_to_frame_name = []

        for i in range(n_days):
            g_r0 = prep_patch_day(rain_grids[i])
            g_t0 = prep_patch_day(temp_grids[i])
            if i < n_days - 1:
                g_r1 = prep_patch_day(rain_grids[i + 1])
                g_t1 = prep_patch_day(temp_grids[i + 1])
            else:
                g_r1 = g_r0
                g_t1 = g_t0

            for s in range(subframes + 1):
                a = s / float(subframes + 1)
                gr = interp_grid(g_r0, g_r1, a)
                gt = interp_grid(g_t0, g_t1, a)
                cloud_i = derive_cloud_from_rain(gr) if show_clouds else cloud0
                hum_i = derive_humidity_from_rt(gr, gt) if show_humidity else hum0
                aqi_i = derive_aqi_from_rt(gr, gt) if show_aqi else aqi0

                # Global progress (0..1) for animated overlays/camera.
                prog = (i + a) / max(1e-9, (n_days - 1))

                if show_cyclone:
                    kf = prog * (len(path_lats) - 1)
                    k0 = int(np.floor(kf))
                    k1 = min(len(path_lats) - 1, k0 + 1)
                    ak = kf - k0
                    cx = (1.0 - ak) * Xp[k0] + ak * Xp[k1]
                    cy = (1.0 - ak) * Yp[k0] + ak * Yp[k1]
                    cz = (1.0 - ak) * Zp[k0] + ak * Zp[k1]
                else:
                    cx, cy, cz = float(Xp[0]), float(Yp[0]), float(Zp[0])

                wx, wy, wz = build_wind_lines(prog) if show_wind else (wx0, wy0, wz0)

                # Slow cinematic camera sweep.
                rot = -0.45 + 0.9 * prog
                eye = dict(x=1.15 * np.cos(rot), y=1.15 * np.sin(rot), z=0.45)

                fname = f"{day_numbers[i]}_{s}"
                if s == 0:
                    day_to_frame_name.append(fname)
                frame_names.append(fname)
                frames.append(
                    go.Frame(
                        name=fname,
                        traces=[2, 3, 4, 5, 6, 8, 9],
                        data=[
                            go.Surface(surfacecolor=gr),
                            go.Surface(surfacecolor=gt),
                            go.Surface(surfacecolor=cloud_i),
                            go.Surface(surfacecolor=hum_i),
                            go.Surface(surfacecolor=aqi_i),
                            go.Scatter3d(x=[float(cx)], y=[float(cy)], z=[float(cz)]),
                            go.Scatter3d(x=wx, y=wy, z=wz),
                        ],
                        layout=go.Layout(
                            scene=dict(camera=dict(projection=dict(type="orthographic"), eye=eye)),
                            scene2=dict(camera=dict(projection=dict(type="orthographic"), eye=eye)),
                        ),
                    )
                )
        fig.frames = frames

        steps = [
            dict(
                method="animate",
                args=[
                    [day_to_frame_name[idx]],
                    {"mode": "immediate", "frame": {"duration": speed_ms, "redraw": True}, "transition": {"duration": 0}},
                ],
                label=str(d),
            )
            for idx, d in enumerate(day_numbers)
        ]

        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            template="plotly_dark",
            height=780,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="#0b1f46",
            plot_bgcolor="#0b1f46",
            sliders=[dict(currentvalue={"prefix": "Day: "}, steps=steps)],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.01,
                    y=1.14,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[frame_names, {"frame": {"duration": max(35, int(speed_ms / (subframes + 1))), "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
        )

        # Orthographic-like projection settings and camera.
        scene_cam = dict(
            projection=dict(type="orthographic"),
            eye=dict(x=1.15, y=0.0, z=0.45),
        )
        fig.update_layout(
            scene=dict(
                domain=dict(x=[0.00, 0.48], y=[0.08, 0.96]),
                bgcolor="#0b1f46",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=scene_cam,
            ),
            scene2=dict(
                domain=dict(x=[0.52, 1.00], y=[0.08, 0.96]),
                bgcolor="#0b1f46",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=scene_cam,
            ),
        )

        return fig

    def plot_timeseries(self, years, values, title, ylabel, color='steelblue',
                        fill_upper=None, fill_lower=None, fill_color='rgba(70,130,180,0.2)'):
        fig = go.Figure()
        if fill_upper is not None and fill_lower is not None:
            fig.add_trace(go.Scatter(
                x=list(years) + list(years)[::-1],
                y=list(fill_upper) + list(fill_lower)[::-1],
                fill='toself', fillcolor=fill_color,
                line=dict(color='rgba(0,0,0,0)'),
                name='±1σ Uncertainty', hoverinfo='skip'
            ))
        fig.add_trace(go.Scatter(
            x=years, y=values, mode='lines+markers',
            line=dict(color=color, width=2.5),
            marker=dict(size=3), name=title
        ))
        fig.update_layout(
            title=title, xaxis_title='Year', yaxis_title=ylabel,
            height=400, template='plotly_dark',
            margin=dict(l=50, r=20, t=50, b=40)
        )
        return fig

    def plot_dual_animation(
        self,
        temp_grids,
        rain_grids,
        day_numbers,
        year_label,
        month_label,
        speed_ms=200,
    ):
        """
        High-fidelity animated twin for the Indian map.
        Initially shows a plain map. When 'Play' is clicked, it uses sub-frame 
        linear interpolation and Gaussian blurring to create a true, fluid 
        'smoky floating' effect.
        """
        import scipy.ndimage as ndimage

        temp_scale = FIXED_SCALES["temp_daily"]
        rain_scale = FIXED_SCALES["rain_daily"]

        def mask_temp(g):
            g2 = np.array(g, dtype=np.float64)
            if self.binary_mask is not None:
                g2 = np.where(self.binary_mask == 1, g2, np.nan)
            g2 = np.where(np.isfinite(g2), g2, np.nan)
            return g2

        def mask_rain(g):
            g2 = np.array(g, dtype=np.float64)
            if self.binary_mask is not None:
                g2 = np.where(self.binary_mask == 1, g2, np.nan)
            g2 = np.where(g2 > 0.25, g2, np.nan)
            g2 = np.where(np.isfinite(g2), g2, np.nan)
            return g2

        def smoky_temp(raw):
            """Apply atmospheric smoothing without distorting coastal temperatures."""
            import pandas as pd

            df = pd.DataFrame(raw)
            df.interpolate(method="linear", axis=1, limit_direction="both", inplace=True)
            df.interpolate(method="linear", axis=0, limit_direction="both", inplace=True)
            clean = np.nan_to_num(df.values, nan=30.0)
            return mask_temp(ndimage.gaussian_filter(clean, sigma=0.8))

        def smoky_rain(raw):
            clean = np.nan_to_num(raw, nan=0.0)
            return mask_rain(ndimage.gaussian_filter(clean, sigma=0.5))

        # Robust monthly ranges (prevents persistent blue/flat coloring).
        t_vals = np.where(np.isfinite(temp_grids), temp_grids, np.nan)
        r_vals = np.where(np.isfinite(rain_grids), rain_grids, np.nan)
        t_p5 = float(np.nanpercentile(t_vals, 5)) if np.isfinite(np.nanmean(t_vals)) else 15.0
        t_p95 = float(np.nanpercentile(t_vals, 95)) if np.isfinite(np.nanmean(t_vals)) else 45.0
        r_p99 = float(np.nanpercentile(r_vals, 99)) if np.isfinite(np.nanmean(r_vals)) else 50.0
        temp_zmin = max(10.0, min(35.0, t_p5 - 1.0))
        temp_zmax = min(52.0, max(temp_zmin + 6.0, t_p95 + 1.0))
        rain_zmax = min(100.0, max(10.0, r_p99))

        t0 = smoky_temp(temp_grids[0])
        r0 = smoky_rain(rain_grids[0])

        cloud_colors = [
            [0.0, "rgba(255, 255, 255, 0.0)"],
            [0.05, "rgba(255, 255, 255, 0.35)"],
            [0.2, "rgba(255, 255, 255, 0.60)"],
            [1.0, "rgba(255, 255, 255, 1.0)"],
        ]

        def anim_hm(z, colorscale, zmin, zmax, opacity, colorbar, hovertemplate):
            kw = dict(
                z=z,
                x=self.lons,
                y=self.lats,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                connectgaps=False,
                hoverongaps=False,
                opacity=opacity,
                showscale=colorbar is not None,
                hovertemplate=hovertemplate,
            )
            if colorbar is not None:
                kw["colorbar"] = colorbar
            return go.Heatmap(**kw)

        data = [
            anim_hm(
                t0,
                "Turbo",
                temp_zmin,
                temp_zmax,
                0.9,
                dict(title=dict(text="Temp (°C)"), len=0.75, x=1.0),
                "Temp: %{z:.1f} °C<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
            ),
            anim_hm(
                r0,
                cloud_colors,
                0,
                rain_zmax,
                0.7,
                dict(title=dict(text="Rain (mm)"), len=0.75, x=1.15),
                "Rain: %{z:.1f} mm/day<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
            ),
        ]

        # Build frames with 'Color Flow' Sub-frame Interpolation
        # We generate 2 intermediate frames between each day for a liquid effect
        frames = []
        num_subframes = 2 
        
        # Temporal smoothing to remove day-to-day 'jumps'
        def get_smoothed_grid(grids, idx):
            # 3-day weighted average for fluid temporal flow
            g = grids[idx] * 0.6
            if idx > 0: g += grids[idx-1] * 0.2
            if idx < len(grids)-1: g += grids[idx+1] * 0.2
            return g

        for i in range(len(day_numbers) - 1):
            d_start = day_numbers[i]
            d_end = day_numbers[i+1]
            
            g_t_start = get_smoothed_grid(temp_grids, i)
            g_r_start = get_smoothed_grid(rain_grids, i)
            g_t_end = get_smoothed_grid(temp_grids, i+1)
            g_r_end = get_smoothed_grid(rain_grids, i+1)

            # Main Day Frame
            ti = smoky_temp(g_t_start)
            ri = smoky_rain(g_r_start)
            frames.append(
                go.Frame(
                    name=str(d_start),
                    traces=[0, 1],
                    data=[
                        anim_hm(
                            ti,
                            "Turbo",
                            temp_zmin,
                            temp_zmax,
                            0.9,
                            None,
                            "Temp: %{z:.1f} °C<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
                        ),
                        anim_hm(
                            ri,
                            cloud_colors,
                            0,
                            rain_zmax,
                            0.7,
                            None,
                            "Rain: %{z:.1f} mm/day<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
                        ),
                    ],
                )
            )

            # Liquid 'Flow' Sub-frames
            for s in range(1, num_subframes + 1):
                frac = s / (num_subframes + 1)
                # Linear interpolation for the 'flow'
                interp_t = g_t_start * (1 - frac) + g_t_end * frac
                interp_r = g_r_start * (1 - frac) + g_r_end * frac
                
                ti_s = smoky_temp(interp_t)
                ri_s = smoky_rain(interp_r)
                frames.append(
                    go.Frame(
                        name=f"{d_start}_{s}",
                        traces=[0, 1],
                        data=[
                            anim_hm(
                                ti_s,
                                "Turbo",
                                temp_zmin,
                                temp_zmax,
                                0.9,
                                None,
                                "Temp: %{z:.1f} °C<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
                            ),
                            anim_hm(
                                ri_s,
                                cloud_colors,
                                0,
                                rain_zmax,
                                0.7,
                                None,
                                "Rain: %{z:.1f} mm/day<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
                            ),
                        ],
                    )
                )

        # Add the final day
        ti_final = smoky_temp(get_smoothed_grid(temp_grids, -1))
        ri_final = smoky_rain(get_smoothed_grid(rain_grids, -1))
        frames.append(
            go.Frame(
                name=str(day_numbers[-1]),
                traces=[0, 1],
                data=[
                    anim_hm(
                        ti_final,
                        "Turbo",
                        temp_zmin,
                        temp_zmax,
                        0.9,
                        None,
                        "Temp: %{z:.1f} °C<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
                    ),
                    anim_hm(
                        ri_final,
                        cloud_colors,
                        0,
                        rain_zmax,
                        0.7,
                        None,
                        "Rain: %{z:.1f} mm/day<br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>",
                    ),
                ],
            )
        )

        fig = go.Figure(data=data, frames=frames)
        self._add_india_outline(fig)

        # Update steps and buttons to use sub-frame speed
        subframe_speed = speed_ms // (num_subframes + 1)

        steps = []
        for i, d in enumerate(day_numbers):
            steps.append(
                dict(
                    method="animate",
                    args=[
                        [str(d)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": speed_ms, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                    label=str(d),
                )
            )

        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.12,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": subframe_speed, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ]

        fig.update_layout(
            title=dict(
                text=f"Animated Indian Twin — Temperature + Rainfall ({month_label} {year_label})",
                font=dict(size=14, color="white"),
            ),
            xaxis=dict(
                title="Longitude",
                range=[self.lon_min, self.lon_max],
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="Latitude",
                range=[self.lat_min, self.lat_max],
                scaleanchor="x",
                scaleratio=1,
                showgrid=False,
                zeroline=False,
            ),
            height=MAP_HEIGHT,
            width=MAP_WIDTH + 80,
            margin=dict(l=60, r=100, t=70, b=50),
            template="plotly_dark",  # Dark mode makes vibrant Turbo/Magenta colors pop
            sliders=[dict(currentvalue={"prefix": "Day: "}, steps=steps)],
            updatemenus=updatemenus,
        )

        return fig
