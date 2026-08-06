"""
climate_twin.train — zone-aware training system (Phase 2+).

Config-driven, fully reproducible, zone-aware at every layer. The registry
of zones (mask + membership + per-zone stats) lives one folder up in
``climate_twin.regions`` and is imported here — nothing in ``train/`` may
redefine a zone.
"""
