# Archived: pre-official-data artifacts

These checkpoints, registry models, and processed cubes were trained on the
**OLD** annual-aggregate cube (51 years of yearly rain/temp aggregates, 2–3
channels). They are **INCOMPATIBLE** with the new official IMD + INSAT cube
built in Phases 2–4:

- **Old cube:** annual, `(51 years, H, W)`, channels `rain + tmax [+ tmin]`.
- **New cube:** daily, `(2922 days, H, W)`, channels `rain + tmax + tmin + insat_lst`.

Every checkpoint here stores weights whose input adapter expects the OLD
channel count and OLD normalization. Loading any of them into the current
pipeline would silently produce wrong output.

**Do not restore.** Kept only for reference / lineage. Train fresh models on
the new cube via the 🔥 Training tab.

Archived: $(date --iso-8601=seconds 2>/dev/null || python -c "from datetime import datetime; print(datetime.now().isoformat())")
Git hash at archive: $(cd .. && git rev-parse HEAD 2>/dev/null || echo unknown)
