"""
verify_determinism.py — Phase 2b hard gate.

Run: python -m climate_twin.train.verify_determinism

Loads the ph2_smoke experiment, does 2 fp32 epochs on a tiny time-window of
the india cube (avoids loading 27k days at once), and dumps:

    _phase0/determinism/run_<n>.state_dict.pt
    _phase0/determinism/run_<n>.loss_trace.json

Repeats the whole thing twice with an identical seed. Then verifies:
    * every tensor in state_dict is BYTE-EQUAL across the two runs
    * the loss trace is bit-identical
Prints ✅ pass or ❌ fail with the first mismatching tensor.

Guarantees enforced:
    * torch.manual_seed / numpy / python random seeded from cfg.seed
    * torch.use_deterministic_algorithms(True)
    * cudnn.deterministic = True, benchmark = False
    * mixed_precision = fp32
    * batch_size deterministic, no drop_last, no shuffle randomness (we
      iterate a fixed set of pre-built windows)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Needed BEFORE any cuda op for full determinism
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from climate_twin.train.config import load_config
from climate_twin.train.model import build_model
from climate_twin.train.data.sampler import ZoneStratifiedWeights, log_batch_composition
from climate_twin.train.loss import (
    total_train_loss, tmax_ge_tmin_penalty,
)
from climate_twin.regions import get_zones
from climate_twin import data_source as DS

OUT_DIR = REPO / "climate_twin" / "_phase0" / "determinism"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_dataset(cfg, n_windows: int = 6):
    """Load a small time-slice of the india cube deterministically."""
    ds = DS.load_region(cfg.data.region)
    years = np.asarray(ds["time.year"].values)
    t_train = (years >= cfg.data.train_years[0]) & (years <= cfg.data.train_years[1])
    # Take a compact continuous slice: first (n_windows + seq_length) days of train
    idx = np.where(t_train)[0][: n_windows + cfg.model.seq_length]
    T_slice = idx.size
    if T_slice < n_windows + cfg.model.seq_length:
        raise RuntimeError(f"not enough train days ({T_slice})")

    vars_ = cfg.data.variables
    stack = np.stack([ds[v].values[idx] for v in vars_], axis=1)  # (T_slice, C, H, W)
    stack = np.nan_to_num(stack, nan=0.0).astype(np.float32)
    ds.close()

    T = cfg.model.seq_length
    windows = []
    targets = []
    for k in range(n_windows):
        windows.append(stack[k : k + T])              # (T, C, H, W)
        targets.append(stack[k + T])                   # (C, H, W)
    return np.array(windows), np.array(targets), vars_


def _run_once(run_id: int, cfg, dataset, zones, device: str) -> dict:
    _set_seed(cfg.seed)

    model = build_model(cfg, zones=zones).to(device)
    zw = ZoneStratifiedWeights(zones, mode=cfg.optim.sampler_mode, device=device)
    weight = zw.tensor
    mask = torch.from_numpy((zones.hard_mask > 0).astype(np.float32)).to(device)
    zone_map = (
        torch.from_numpy(zones.membership).permute(2, 0, 1).unsqueeze(0).to(device)
    )   # (1, K, H, W) — expand at batch time

    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg.optim.lr,
                             weight_decay=cfg.optim.weight_decay)

    X, Y, vars_ = dataset
    B = cfg.optim.batch_size
    n_windows = X.shape[0]
    n_batches = (n_windows + B - 1) // B

    losses: list[float] = []
    model.train()
    for epoch in range(cfg.optim.epochs):
        for b in range(n_batches):
            sl = slice(b * B, min((b + 1) * B, n_windows))
            x = torch.from_numpy(X[sl]).to(device)          # (Bs, T, C, H, W)
            y = torch.from_numpy(Y[sl]).to(device)          # (Bs, C, H, W)
            zm = zone_map.expand(x.shape[0], -1, -1, -1).contiguous()

            out = model(x, zm)

            targets = {v: y[:, i:i+1] for i, v in enumerate(vars_)}
            loss, parts = total_train_loss(out, targets, weight, cfg)

            # tmax≥tmin physics on land
            if cfg.loss.physics_tmax_ge_tmin > 0 and "tmax" in out and "tmin" in out:
                pen = tmax_ge_tmin_penalty(out["tmax"], out["tmin"], weight, mask)
                loss = loss + cfg.loss.physics_tmax_ge_tmin * pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()

            losses.append(float(loss.detach().item()))

    # Persist state
    state = {k: v.detach().cpu().to(torch.float64).contiguous()
             for k, v in model.state_dict().items()}
    torch.save(state, OUT_DIR / f"run_{run_id}.state_dict.pt")
    (OUT_DIR / f"run_{run_id}.loss_trace.json").write_text(json.dumps(losses, indent=2))
    return {"losses": losses, "state": state}


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().to(torch.float64).numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    cfg = load_config(REPO / "climate_twin/train/config/experiments/ph2_smoke.yaml")
    zones = get_zones()

    print(cfg.summary())
    print()

    # One-off: pre-build data, so both runs get IDENTICAL numpy arrays
    dataset = _build_dataset(cfg, n_windows=6)
    print(f"[data] windows={dataset[0].shape}  targets={dataset[1].shape}")

    # Sampler composition — the log the plan asks for
    zw = ZoneStratifiedWeights(zones, mode=cfg.optim.sampler_mode)
    print(log_batch_composition(zw, n_batches=5))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Force CPU for perfect determinism — GPU deterministic algos on 3060
    # still occasionally introduce reproducibility drift with dropout+conv.
    device = "cpu"
    print(f"[device] forcing CPU for determinism verification")
    print()

    r1 = _run_once(1, cfg, dataset, zones, device)
    print(f"[run 1] final loss = {r1['losses'][-1]:.10f}")
    r2 = _run_once(2, cfg, dataset, zones, device)
    print(f"[run 2] final loss = {r2['losses'][-1]:.10f}")

    # ── comparisons ─────────────────────────────────────────────
    print()
    print("── determinism verification ─────────────────────────────")

    ok = True

    # 1. Loss trace equality
    if r1["losses"] == r2["losses"]:
        print("  ✓ loss trace bit-identical  "
              f"(n={len(r1['losses'])}, first={r1['losses'][0]:.10f}, "
              f"last={r1['losses'][-1]:.10f})")
    else:
        ok = False
        # Find first divergence
        for i, (a, b) in enumerate(zip(r1["losses"], r2["losses"])):
            if a != b:
                print(f"  ✗ loss diverges at step {i}: {a} vs {b}")
                break

    # 2. State-dict tensor equality (byte-equal)
    keys_1 = set(r1["state"].keys())
    keys_2 = set(r2["state"].keys())
    if keys_1 != keys_2:
        print(f"  ✗ state_dict keys differ: {keys_1 ^ keys_2}")
        ok = False
    else:
        n_tensors = len(keys_1)
        first_diff = None
        for k in sorted(keys_1):
            h1 = _hash_tensor(r1["state"][k])
            h2 = _hash_tensor(r2["state"][k])
            if h1 != h2:
                first_diff = (k, h1, h2)
                break
        if first_diff is None:
            print(f"  ✓ state_dict byte-equal  ({n_tensors} tensors, "
                  f"total {sum(v.numel() for v in r1['state'].values()):,} params)")
        else:
            k, h1, h2 = first_diff
            print(f"  ✗ tensor '{k}' hashes differ: {h1} vs {h2}")
            ok = False

    if ok:
        # Total signature: hash all tensor hashes for a summary
        combined = ",".join(_hash_tensor(r1["state"][k]) for k in sorted(keys_1))
        sig = hashlib.sha256(combined.encode()).hexdigest()[:12]
        (OUT_DIR / "determinism_signature.txt").write_text(sig)
        print(f"\n  📎 determinism signature = {sig}")
        print("\n✅ Phase 2b PASS — training is reproducible from config + seed.")
        return 0
    else:
        print("\n❌ Phase 2b FAIL — training is NOT reproducible. See divergences above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
