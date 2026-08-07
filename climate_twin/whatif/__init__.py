"""
climate_twin.whatif — Scenario Engine (Part-1+ build).

Layered structure:

    L0  drivers      — scenario input generators (historical, ensemble,
                        perturbation, analog, SSP)
    L1  indices      — climate indices (SPI, GDD, CDD, WBGT, ET0, …)
    L2  biophysical  — process models (crop, hydrology, energy demand)
    L3  sectors      — sector aggregators (agri, water, power, health)
    L4  economics    — INR / MWh translations, cost curves
    UI              — Streamlit page composers
    scenarios       — orchestration + provenance ledger

Part 0 ships EMPTY skeleton modules only. Every module carries a
``# TODO(Part N)`` marker naming the phase that fills it in.
"""
