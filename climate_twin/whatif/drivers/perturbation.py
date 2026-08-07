"""
whatif.drivers.perturbation — L0, Method 1: parametric perturbation.

Shifts + scales a historical base window by (ΔT, ΔP%) drawn from a
user-chosen distribution (uniform / IPCC-band / user-defined). Preserves
spatial structure; documents everything applied so provenance is
reproducible.

# TODO(Part 5): implement `perturb(base_ds, delta_t, delta_p_pct, seed)`
#               returning an xarray Dataset + a scenario-record dict.
"""
