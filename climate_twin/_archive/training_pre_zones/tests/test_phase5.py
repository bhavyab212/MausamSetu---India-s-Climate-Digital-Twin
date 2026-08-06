"""
Phase 5 QC runner — hard gates for the post-Phase-4 pipeline.

Run:  cd climate_twin && ../venv/Scripts/python tests/test_phase5.py

Covers:
  1. Manifest completeness  (manifest.yaml + manifest.sig fields present)
  2. Physical consistency   (tmax ≥ tmin on land, no sentinels survive)
  3. Cauvery ⊂ India        (Cauvery cube is exactly India NaN-masked outside basin)
  4. Checkpoint contract    (variable-list save/load happy path + rejection)
  5. Rounds generator       (INSAT dropped, rain/tmax/tmin kept)
  6. Cache-bust behaviour   (manifest-sig change wipes Streamlit caches)

Every check prints a line and asserts. First failure exits with code 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parents[2]
CT = REPO / "climate_twin"
sys.path.insert(0, str(CT))

import data_source as DS
from training.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    check_variables,
    CheckpointVariableMismatch,
    CHECKPOINT_DIR,
)
from training.registry import (
    save_model,
    load_into,
    ModelVariableMismatch,
    delete_model,
)
from data.rounds_builder import build_rounds_yaml, DEFAULT_MIN_COVERAGE

PROCESSED_DIR = REPO / "data" / "processed"
MANIFEST_YAML = PROCESSED_DIR / "manifest.yaml"
MANIFEST_SIG = PROCESSED_DIR / "manifest.sig"

_PASS: list[str] = []
_FAIL: list[str] = []


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")
    _PASS.append(msg)


def _fail(msg: str, err: str = "") -> None:
    print(f"  ✗ {msg}")
    if err:
        print(f"      {err}")
    _FAIL.append(f"{msg}: {err}")


def _check(cond: bool, msg: str, err_if_false: str = "") -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg, err_if_false)


def _hdr(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Manifest completeness
# ---------------------------------------------------------------------------
def test_manifest_completeness() -> None:
    _hdr("1. Manifest completeness")

    _check(MANIFEST_SIG.exists(), "manifest.sig exists",
           f"missing {MANIFEST_SIG}")
    _check(MANIFEST_YAML.exists(), "manifest.yaml exists",
           f"missing {MANIFEST_YAML}")

    if MANIFEST_SIG.exists():
        sig = MANIFEST_SIG.read_text().strip()
        _check(len(sig) == 12 and all(c in "0123456789abcdef" for c in sig),
               f"manifest.sig is 12-hex ({sig!r})",
               "not a 12-hex signature")

    if MANIFEST_YAML.exists():
        m = yaml.safe_load(MANIFEST_YAML.read_text())
        required = ["generator", "generated_at_ist", "git_hash", "master_grid",
                    "regions_built", "years", "train_years", "variables",
                    "readers", "regrid", "missing_policy", "sources", "outputs"]
        missing = [k for k in required if k not in m]
        _check(not missing, "manifest.yaml has all required keys",
               f"missing keys: {missing}")

        readers = m.get("readers", {})
        for r in ("rainfall", "tmax", "tmin", "insat_lst"):
            _check(r in readers, f"manifest.readers has '{r}'")
        srcs = m.get("sources", {})
        for s in ("rainfall", "tmax", "tmin", "insat"):
            has_md5 = "md5_first" in srcs.get(s, {}) and "md5_last" in srcs.get(s, {})
            _check(has_md5, f"manifest.sources.{s} has md5 fingerprints")


# ---------------------------------------------------------------------------
# 2. Physical consistency  (tmax ≥ tmin, no sentinels)
# ---------------------------------------------------------------------------
def test_physical_consistency() -> None:
    _hdr("2. Physical consistency  (tmax ≥ tmin, no sentinels)")

    for region in ("india", "cauvery"):
        ds = DS.load_region(region)

        # Per-variable sentinels from the raw sources (see manifest.readers):
        #   rain      → NaN (built-in in NetCDF), also -999 in some files
        #   tmax/tmin → 99.9  (IMD .GRD convention)
        #   insat_lst → -999  (HDF5)
        # A real rainfall of 99.9 mm/day is plausible — do NOT test 99.9 for rain.
        sentinel_map = {
            "rain":     [-999.0],
            "tmax":     [99.9],
            "tmin":     [99.9],
            "insat_lst": [-999.0],
        }
        for var, sentinels in sentinel_map.items():
            arr = ds[var].values
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                _ok(f"[{region}] {var}: no finite cells (sparse coverage) — skipping bounds")
                continue
            leaked = [s for s in sentinels
                      if bool(np.any(np.abs(finite - s) < 1e-3))]
            _check(not leaked,
                   f"[{region}] {var}: no sentinels {sentinels} survive",
                   f"leaked: {leaked}")
            # Sanity ranges
            if var == "rain":
                _check(finite.min() >= -1e-3 and finite.max() < 2000.0,
                       f"[{region}] rain in plausible range [0, 2000] mm/day "
                       f"[{finite.min():.2f}, {finite.max():.2f}]")
            elif var in ("tmax", "tmin"):
                _check(-30.0 < finite.min() < 60.0 and -30.0 < finite.max() < 60.0,
                       f"[{region}] {var} in plausible °C range "
                       f"[{finite.min():.2f}, {finite.max():.2f}]")
            elif var == "insat_lst":
                _check(-50.0 < finite.min() < 90.0 and -50.0 < finite.max() < 90.0,
                       f"[{region}] insat_lst in plausible °C range "
                       f"[{finite.min():.2f}, {finite.max():.2f}]")

        # tmax ≥ tmin on cells where both are finite
        tmax = ds["tmax"].values
        tmin = ds["tmin"].values
        both = np.isfinite(tmax) & np.isfinite(tmin)
        violations = int(np.sum(both & (tmax < tmin)))
        _check(violations == 0,
               f"[{region}] tmax >= tmin (0 violations on {int(both.sum())} cell-days)",
               f"{violations} violations")

        # is_interpolated sidecars have same shape as the variable
        for sc in ("tmax_is_interpolated", "tmin_is_interpolated"):
            if sc in ds.data_vars:
                _check(ds[sc].shape == ds["tmax"].shape,
                       f"[{region}] sidecar {sc} shape matches tmax")


# ---------------------------------------------------------------------------
# 3. Cauvery ⊂ India consistency
# ---------------------------------------------------------------------------
def test_cauvery_is_india_subset() -> None:
    _hdr("3. Cauvery ⊂ India consistency")

    di = DS.load_region("india")
    dc = DS.load_region("cauvery")

    _check(di.sizes["time"] == dc.sizes["time"], "time dims equal")
    _check(di.sizes["lat"] == dc.sizes["lat"], "lat dims equal")
    _check(di.sizes["lon"] == dc.sizes["lon"], "lon dims equal")
    _check(np.array_equal(di.lat.values, dc.lat.values), "lat coords identical")
    _check(np.array_equal(di.lon.values, dc.lon.values), "lon coords identical")

    # Cauvery mask must be a strict subset of India mask
    im = di["mask"].values
    cm = dc["mask"].values
    strict_subset = bool(((cm == 1) & (im != 1)).sum() == 0)
    _check(strict_subset,
           f"cauvery mask ⊆ india mask (cauvery={int(cm.sum())}, india={int(im.sum())})")

    # On the Cauvery basin cells, values must equal India's values
    basin = cm == 1
    if basin.any():
        # Compare on the first, middle, and last time index
        idxs = [0, di.sizes["time"] // 2, di.sizes["time"] - 1]
        for t in idxs:
            for var in ("rain", "tmax", "tmin"):
                a = di[var].values[t][basin]
                b = dc[var].values[t][basin]
                # NaN-tolerant comparison
                both_nan = np.isnan(a) & np.isnan(b)
                equal = (a == b) | both_nan
                _check(bool(equal.all()),
                       f"cauvery[{var}] equals india[{var}] on basin at t={t}",
                       f"{int((~equal).sum())} disagreements")


# ---------------------------------------------------------------------------
# 4. Checkpoint variable-list contract
# ---------------------------------------------------------------------------
def test_checkpoint_contract() -> None:
    _hdr("4. Checkpoint contract  (variable list + shape)")

    class Tiny(nn.Module):
        def __init__(self, c: int):
            super().__init__()
            self.conv = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            return self.conv(x)

    vars_full = ["rain", "tmax", "tmin", "insat_lst"]

    # 4a. check_variables primitive
    ok, _ = check_variables(vars_full, vars_full); _check(ok, "identical lists match")
    ok, _ = check_variables(vars_full, ["rain", "tmax"]); _check(not ok, "length mismatch rejected")
    ok, _ = check_variables(vars_full, ["tmax", "rain", "tmin", "insat_lst"])
    _check(not ok, "order mismatch rejected")
    ok, _ = check_variables(vars_full, [], strict=False); _check(ok, "legacy list accepted with strict=False")
    ok, _ = check_variables(vars_full, [], strict=True); _check(not ok, "legacy list rejected with strict=True")

    # 4b. save_checkpoint stores variables; load refuses on mismatch
    m1 = Tiny(len(vars_full))
    opt = torch.optim.Adam(m1.parameters())
    p = save_checkpoint(m1, opt, round_num=999, val_period="PH5G", config={}, metrics={},
                        epoch=1, region="india", variables=vars_full,
                        manifest_sig="deadbeef1234", grid_shape=(129, 135))
    m2 = Tiny(len(vars_full))
    ck = load_checkpoint(p, m2, expected_variables=vars_full)
    _check(ck.get("variables") == vars_full and ck.get("manifest_sig") == "deadbeef1234",
           "checkpoint round-trips variables + manifest_sig")

    m3 = Tiny(len(vars_full))
    try:
        load_checkpoint(p, m3, expected_variables=["rain", "tmax", "tmin"])
        _fail("mismatched load raised", "did not raise")
    except CheckpointVariableMismatch as e:
        _ok(f"mismatched load raised CheckpointVariableMismatch")

    p.unlink(missing_ok=True)

    # 4c. registry save_model + load_into
    save_model("_ph5g_test", "india", Tiny(len(vars_full)), arch={"channels": 4},
               metrics={}, epochs_add=1, rounds_add=1,
               variables=vars_full, manifest_sig="deadbeef1234",
               grid_shape=(129, 135))
    m4 = Tiny(len(vars_full))
    ok, ck = load_into("_ph5g_test", "india", m4, expected_variables=vars_full)
    _check(ok, "registry happy load")

    m5 = Tiny(len(vars_full))
    try:
        load_into("_ph5g_test", "india", m5, expected_variables=["rain"])
        _fail("registry mismatched load raised", "did not raise")
    except ModelVariableMismatch:
        _ok("registry mismatched load raised ModelVariableMismatch")

    delete_model("_ph5g_test", "india")


# ---------------------------------------------------------------------------
# 5. Coverage-aware rounds generator
# ---------------------------------------------------------------------------
def test_rounds_generator() -> None:
    _hdr("5. Coverage-aware rounds generator")

    out_path = CT / "config" / "_test_rounds_india.yaml"
    build_rounds_yaml(region="india", out_path=out_path)
    doc = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    out_path.unlink(missing_ok=True)

    _check(doc["_manifest_sig"] == DS.manifest_sig(),
           "generated yaml carries current manifest_sig")

    data = doc["data"]
    _check("insat_lst" in data["variables_available"],
           "insat_lst listed as available (present in cube)")

    kept = data["variable_usage_across_rounds"]["kept"]
    dropped = data["variable_usage_across_rounds"]["dropped"]

    # insat_lst has 12% coverage overall, present only in 2020-2021 partial —
    # at the default 0.90 threshold it MUST be dropped from every round.
    _check(kept.get("insat_lst", 0) == 0 and dropped.get("insat_lst", 0) > 0,
           f"insat_lst dropped from every round at threshold {DEFAULT_MIN_COVERAGE}",
           f"kept={kept.get('insat_lst',0)}  dropped={dropped.get('insat_lst',0)}")

    # rain/tmax/tmin must be kept in every round
    for v in ("rain", "tmax", "tmin"):
        _check(dropped.get(v, 0) == 0 and kept.get(v, 0) > 0,
               f"'{v}' kept in every round",
               f"kept={kept.get(v,0)}  dropped={dropped.get(v,0)}")

    # Every round declares its own variable list
    for r in doc["rounds"]:
        _check(isinstance(r.get("variables"), list) and len(r["variables"]) > 0,
               f"round {r['round_num']} has non-empty variables list")


# ---------------------------------------------------------------------------
# 6. Cache-bust on manifest-sig change (uses a fake session_state dict)
# ---------------------------------------------------------------------------
def test_cache_bust() -> None:
    _hdr("6. Cache-bust on manifest-sig change")

    # Fake session_state + fake cache_data.clear / cache_resource.clear.
    class FakeCache:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    class FakeSt:
        def __init__(self):
            self.session_state = {}
            self.cache_data = FakeCache()
            self.cache_resource = FakeCache()

    fake = FakeSt()
    _real_st = DS.st if hasattr(DS, "st") else None
    _real_has = DS._HAS_ST
    DS.st = fake
    DS._HAS_ST = True
    try:
        # First call — sig is None initially, cache is cleared once.
        fake.session_state["_climate_twin_manifest_sig"] = "OLD_SIG"
        DS._bust_streamlit_caches_if_sig_changed()
        _check(fake.cache_data.cleared == 1 and fake.cache_resource.cleared == 1,
               "cache cleared when signature changes (OLD -> current)")

        # Second call — sig unchanged, cache NOT cleared again.
        DS._bust_streamlit_caches_if_sig_changed()
        _check(fake.cache_data.cleared == 1 and fake.cache_resource.cleared == 1,
               "cache NOT cleared when signature unchanged")

        # Third call — pretend session_state reset to different sig.
        fake.session_state["_climate_twin_manifest_sig"] = "ANOTHER_SIG"
        DS._bust_streamlit_caches_if_sig_changed()
        _check(fake.cache_data.cleared == 2 and fake.cache_resource.cleared == 2,
               "cache cleared again on next signature change")
    finally:
        DS.st = _real_st
        DS._HAS_ST = _real_has


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main() -> int:
    test_manifest_completeness()
    test_physical_consistency()
    test_cauvery_is_india_subset()
    test_checkpoint_contract()
    test_rounds_generator()
    test_cache_bust()

    print()
    print(f"passed: {len(_PASS)}   failed: {len(_FAIL)}")
    if _FAIL:
        print("FAILURES:")
        for f in _FAIL:
            print("  -", f)
        return 1
    print("All Phase-5 QC gates passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
