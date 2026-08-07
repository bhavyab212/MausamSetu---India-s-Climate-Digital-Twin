"""
whatif.drivers.ensemble — L0, AI-model ensemble driver.

Loads the ensemble prediction artifact produced by
``climate_twin.train.loop.trainer`` (the zone-aware model checkpoints
committed under ``climate_twin/train/registry/models/india/``). Emits
p10 / p50 / p90 fields on the master grid.

# TODO(Part 1): implement `load(model_name, dates, region)`. Every
#               array carries `zone_mask_sig` and `manifest_sig`; refuse
#               to serve a prediction whose sigs disagree with the
#               live registry (RegistryModelIncompatible).
"""
