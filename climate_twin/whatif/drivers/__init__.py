"""
whatif.drivers — L0 scenario input generators.

Each module here returns an xarray Dataset shaped (time, lat, lon) on the
master grid, carrying rain / tmax / tmin (+ satellite where available).
Callers should not care whether the source is history, an AI ensemble,
an analog year, or an SSP projection — the return shape is identical.
"""
