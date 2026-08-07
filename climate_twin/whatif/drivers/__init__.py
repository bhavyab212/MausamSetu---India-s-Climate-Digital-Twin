"""
whatif.drivers — L0 scenario input generators.

Public entry point: ``load_driver(DriverSpec) → xr.DataArray``.
Concrete drivers (historical, ensemble, perturbation, analog, ssp)
live as submodules but should NOT be called directly by anything
outside this package — go through ``driver.load_driver`` so the
uniform (time, lat, lon) / IST / attrs contract stays enforced.
"""
from .driver import DriverSpec, load_driver
from .ensemble import EnsembleUnavailable, get_ensemble
from .historical import get_historical, get_tmean

__all__ = [
    "DriverSpec",
    "load_driver",
    "get_historical",
    "get_tmean",
    "get_ensemble",
    "EnsembleUnavailable",
]
