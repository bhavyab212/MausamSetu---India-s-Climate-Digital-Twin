"""
whatif.drivers.analogs — L0, Method 2: analog-year resampling.

Given a target (temperature, precipitation) signature, find the K
closest historical years by Mahalanobis distance in the (annual anomaly)
subspace. Concatenate their fields as a synthetic future block that is
guaranteed physically consistent (it happened).

# TODO(Part 5): implement `analogs(target_signature, k, years_pool)`.
"""
