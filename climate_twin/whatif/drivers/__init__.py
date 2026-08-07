"""
whatif.drivers — L0 scenario input generators.

Public entry point: ``load_driver(DriverSpec) → xr.DataArray``.

Concrete drivers (historical, ensemble, perturbation, analog, ssp)
live as submodules but should NOT be called directly by anything
outside this package — go through ``driver.load_driver`` so the
uniform (time, lat, lon) / IST / attrs contract stays enforced.

The Part-5 modules (analog_features, analogs, analog_outcomes,
climate_states, analog_backtest, perturbation) are reachable via their
full submodule paths, e.g.
    ``from climate_twin.whatif.drivers.analogs import find_analogs``.
They are NOT imported here because ``analog_features`` depends on
``indices.reference`` which itself depends on this package's
``historical`` submodule — eager loading them here creates a
circular import at package init.
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
