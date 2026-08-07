"""
whatif.drivers.historical — L0, historical driver.

Reads raw IMD gridded fields via the existing readers in
``climate_twin.data.readers`` (imd_rainfall_nc, imd_temperature) and
returns an xarray Dataset on the master grid.

# TODO(Part 1): implement `load(years, variables, region)` returning a
#               provenance-tagged xarray Dataset. Sentinels must be
#               masked BEFORE any arithmetic; see
#               ``whatif.config.constants.IMD_SENTINELS``.
"""
