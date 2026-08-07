"""
whatif.indices — L1, climate indices computed from L0 fields.

Every index accepts an xarray Dataset (rain / tmax / tmin) and returns
an xarray DataArray on the same master grid, with a unit label from
``whatif.config.constants.UNITS``.

# TODO(Part 2): SPI, SPEI, GDD, CDD, HDD, WBGT, HI, VPD, ET0 (FAO-56).
"""
