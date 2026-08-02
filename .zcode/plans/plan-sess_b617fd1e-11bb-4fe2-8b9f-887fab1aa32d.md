Build a Train/Validate workflow with a named, resumable model registry in the 🔥 Training tab.

1. New `training/registry.py`: persistent named-model registry (runs.db table + weights at training/checkpoints/model_<region>_<name>.pt). save/load/list/get/delete; accumulates epochs & rounds across re-training.

2. Rewrite `_render_tab9` into two modes (toggle at top):
   - 🏋️ Train: choose "New model" (name it) or "Continue existing" (pick from registry → resume weights); region/shape-guarded. Keep schedule + hyperparameters. Add a scientific live console (monospace scrolling log narrating data build → sequence windowing → model build → per-epoch loss/val/grad/LR/GPU-mem/throughput → validation → checkpoint), plus the existing live progress panel + loss curve. On finish, save under the model name and show a summary vs persistence/climatology baselines.
   - 🔬 Validate: select a saved model (region-filtered), pick a year, run no-grad MC-dropout inference, and show a full report — metric table (ours vs persistence vs climatology), reliability diagram, metric bars, skill-by-lead-time, sample predictions, error map, and model provenance.

3. Wire resume (registry.load_into before training) and save (registry.save_model after); validation reuses the same region-aware data prep. GPU-only guard stays.

Files: new training/registry.py; edit app_v2.py (_render_tab9); minor loops.py log hook if needed. No other tabs or the map/animation affected.