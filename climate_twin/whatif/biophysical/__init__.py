"""
whatif.biophysical — L2, process models operating on L1 indices.

Each model runs on (time × zone × cell), returns an xarray DataArray
with the target quantity in physical units. Deterministic; no
model-in-the-loop training here — the trained pieces live in
``climate_twin.train``.

# TODO(Part 3): crop-yield reducer (AquaCrop-style water × heat),
#               hydrological runoff, energy-demand curve.
"""
