"""
whatif.drivers.ssp — L0, NEX-GDDP-CMIP6 SSP driver.

Loads NASA's downscaled CMIP6 projections (SSP1-2.6, 2-4.5, 3-7.0,
5-8.5), bias-corrected against IMD gauge climatology, on the master
grid. Cache to netCDF under ``CACHE_DIR`` — the source is remote.

# TODO(Part 7): implement `load(ssp, model, year, variables)` with
#               local caching and a bias-correction QC report.
"""
