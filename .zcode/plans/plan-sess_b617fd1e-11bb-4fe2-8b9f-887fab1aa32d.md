## Phase 0 diagnosis — approval to proceed to Phase 1

Phase 0 is READ-ONLY diagnosis; no code was written. Findings above show three broken layers:

1. **`_render_tab9`** in `app_v2.py` — early-return description page, tells user to launch a terminal (`streamlit run …`, `python -m …`). ~860 lines of dead code below the return. Zero buttons in the reachable body.
2. **`train/ui/app.py`** — closer to a dashboard (does spawn a real background thread), but missing auto-refresh, Pause/Stop, GPU meter, live tier-3/4 heatmap.
3. **`train/loop/trainer.py`** — silently falls back to CPU (line 134), `num_workers=0` on every DataLoader, no per-batch event surface, no pause primitive.

**Phase 1 preview** (per your rebuild plan text — NOT part of this Phase 0 output, provided so you can approve the direction):

- Create `climate_twin/runtime/` with `executor.py`, `event_queue.py`, `run_state.py`, `device.py`
- Add `Trainer.pause() / resume() / stop()` primitives (batch-boundary check, not just epoch)
- Convert trainer to emit through a thread-safe `queue.Queue` (batch/tier1/tier2/tier3/tier4/checkpoint/done/error events), not just `LiveState` mutation
- GPU refusal policy — hard fail at startup when `torch.cuda.is_available() == False`; disable the Start button and show a red banner
- DataLoader: `num_workers=4, pin_memory=True` (kept configurable in the schema)
- Then prove it works with a minimal Python test (no Streamlit yet) — the plan's "prove no subprocess is spawned" STOP gate

**What I will NOT do in Phase 1:**
- Not touch `web/`, FastAPI, or `mausamsetu/` core (per your rule)
- Not touch existing checkpoints (all 5 stay compatible with `242f813af71b`)
- Not build the Streamlit UI yet — that's Phase 2
- Not touch the Validation dashboard — that's Phase 3

Approving this ExitPlanMode means: acknowledge the diagnosis and clear me to start Phase 1 (execution spine). I will STOP again at the end of Phase 1 with the required proof of the minimal `Trainer.start()` invocation from a Python test.

If you want me to adjust the diagnosis or the Phase 1 approach before I begin, say so now.