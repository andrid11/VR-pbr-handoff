# Results

This directory contains lightweight metadata for each shipped run:
`args.json`, `history.json`, and `previews/` (sample qualitative outputs at training milestones).

The full training and EMA checkpoints (`best.pt`, `best_ema.pt`) are hosted on a
**private** Hugging Face Hub model repo:

**`https://huggingface.co/Andrid1/vrtest-pbr-handoff`**

To download (after `pip install huggingface_hub` and authenticating with `huggingface-cli login`):

```python
from huggingface_hub import snapshot_download

# Single run
snapshot_download(
    repo_id="Andrid1/vrtest-pbr-handoff",
    local_dir="checkpoints",
    allow_patterns=["S4_baseline/**"],
)

# All runs
snapshot_download(
    repo_id="Andrid1/vrtest-pbr-handoff",
    local_dir="checkpoints",
)
```

See `../examples/demo.py` for a full end-to-end inference example.

## What's in each run directory

- `args.json` — the exact CLI flags used for training. For Stage 1-3 runs, these were
  reconstructed from `run_stage*.bat` launch scripts (the original train.py predates
  args.json persistence). The `_backfilled_from` field marks reconstructed entries.
- `history.json` — per-epoch metrics: train_loss, val_loss, val_maps (per-map L1),
  learning rate, wall time, plus discriminator metrics for GAN runs.
- `previews/` — sample qualitative outputs against the fixed comparison set at
  training milestones (`--preview-every` epochs).

See `SUMMARY.md` for an overview table.
