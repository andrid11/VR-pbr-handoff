# Reproduction guide

## 1. Environment

- Python 3.11+
- A CUDA-capable GPU is recommended (Stage 4 runs at batch 16 / 256 px need ~10 GB VRAM)
- Tested on Windows 11 + RTX 3060 Ti (8 GB) and on CPU for inference only

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Hugging Face Hub authentication

The model checkpoints (and the MatSynth dataset, if you pre-cache) live on
Hugging Face Hub. Authenticate once:

```bash
pip install -U huggingface_hub
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
```

The shipped checkpoints are in a **private** repo. Whoever is running this
needs read access (granted by the repo owner) for the
`Andrid1/vrtest-pbr-handoff` model repo.

## 3. Quick smoke test (no training, no dataset)

```bash
python examples/demo.py --image examples/inputs/wood.png --run S4_gan_mid
```

This downloads the checkpoint from HF Hub on first run (cached to
`.cache/checkpoints/`) and produces a 4-panel grid at
`examples/output/wood__S4_gan_mid__predicted.png`. About 30 seconds end-to-end
on CPU.

Use `--run S4_gan_light` for the lowest-val_loss checkpoint, or any other shipped
run (`S1_bce`, `S1B_bce_long`, `S1B_bce_gan`, `S2_dual_w10`, `S3_rw1`,
`S4_baseline`, `S4_gan_heavy`).

## 4. Dataset

The code targets [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth)
on the Hugging Face Hub. Streaming-mode dataloading is supported but slow;
pre-caching to local disk as `.pt` files is the recommended path.

### Pre-cache to local disk

```bash
python scripts/predownload.py --out-dir data/processed2/train_256 --size 256
```

Downloads ~5,700 samples and writes one `.pt` file per sample. Total disk
usage ≈ 10 GB. The `train_256` directory is the cache path referenced by all
shipped run configurations.

### Re-create the splits used by the shipped runs

```bash
python scripts/select_comparison_set.py --out outputs/comparison_set.json
python scripts/make_stage4_split.py --source <existing-split> --out outputs/stage4_split.json --seed 4242
```

The shipped Stage 4 runs used `seed=4242` and the comparison set in
`results/`. The splits are 5,130 train / 285 val / 285 held-out test.

## 5. Reproducing a shipped run

The exact CLI flags for every shipped run are in
`results/<run>/args.json`. The Stage 4 baseline (the headline-by-val_loss
recipe before the GAN was added) command is:

```bash
python scripts/train.py \
  --cache-dir data/processed2/train_256 \
  --batch-size 16 --use-category --epochs 150 \
  --separate-normal-decoder \
  --normal-weight 0.1 --roughness-weight 0.1 --metallic-weight 0.2 \
  --metallic-bce 1.0 --normal-cosine 0.0 --roughness-ssim 0.0 --normal-loss l1 \
  --render-loss 1.0 \
  --r1-gamma 10.0 --adv-warmup-epochs 5 \
  --comparison-set outputs/comparison_set.json \
  --split-file outputs/stage4_split.json \
  --out-dir outputs/S4_baseline_repro
```

For the Stage 4 GAN variants, add `--adversarial <weight>`:

```bash
python scripts/train.py [... same flags ...] --adversarial 0.005 --out-dir outputs/S4_gan_light_repro
python scripts/train.py [... same flags ...] --adversarial 0.01  --out-dir outputs/S4_gan_mid_repro
```

For Stage 2 — the project's largest single recovery on the flat-normal
metric (REPORT §4.7) — drop the Stage 4 flags and use the Stage 2 weight scheme:

```bash
python scripts/train.py \
  --cache-dir data/processed2/train_256 \
  --batch-size 16 --normal-loss mse --use-category --epochs 100 \
  --metallic-bce 1.0 --normal-weight 10.0 \
  --separate-normal-decoder \
  --comparison-set outputs/comparison_set.json \
  --out-dir outputs/S2_dual_w10_repro
```

Other run configurations: copy the CLI from each `results/<run>/args.json`.

### Earlier stages

For Stages 1, 1B, 2, 3, the per-run `args.json` files were reconstructed from
the launch scripts (the original `train.py` did not yet persist args). The
`_backfilled_from` field marks reconstructed entries. The reconstructed args
match what was actually run, but if you want byte-for-byte match with the
original command lines, the source `.bat` scripts are not in this
package — request them from the repo owner.

## 6. Evaluation

### Quick per-run validation metrics

These are already in each `results/<run>/history.json` — the entry with the
lowest `val_loss` is the model state in `best_ema.pt` (Stage 4) or `best.pt`
(Stage 1-3). No re-run needed.

### Held-out test-set evaluation (recommended next action — see REPORT §7.2)

The 285-sample test set in `outputs/stage4_split.json["test"]` was reserved
during training. None of the shipped checkpoints have been evaluated against
it. Run:

```bash
python scripts/eval_full.py \
  --cache-dir data/processed2/train_256 \
  --split-file outputs/stage4_split.json \
  --split test \
  --checkpoint <path-to-downloaded-best_ema.pt-or-best.pt> \
  --out outputs/eval_test_<run-name>.json
```

Repeat for each shipped checkpoint. Per-category metrics (REPORT §7.4) come
out of this same script.

### Reproducing the flat-normals analysis (REPORT §4.7)

The flat-normals measurement in REPORT §4.7 is produced by the bundled
`scripts/analyze_flat_normals.py`. After authenticating with the HF Hub
(see §2 above) and pre-caching the dataset (§4), run:

```bash
python scripts/analyze_flat_normals.py
```

It downloads each shipped checkpoint from the HF Hub model repo on first
call (cached in `.cache/checkpoints/`), runs inference on the 285-sample
test set, computes per-sample predicted-normal std and L1 vs GT, and writes
the resulting table to `results/FLAT_NORMALS_ANALYSIS.md`. The bundled
`results/FLAT_NORMALS_ANALYSIS.md` is the output of this exact command;
re-running should reproduce it numerically (the methodology is
deterministic given the frozen split).

A future improvement would be to fold this measurement into
`scripts/eval_full.py` directly (see REPORT §7.1 for the recommended
permanent fix: add `val_normal_std` to `validate()` in `scripts/train.py`,
so the metric is logged at training time rather than only post-hoc).

## 7. Hardware notes

The Stage 4 heavy adversarial config (`adv=0.05` + R1 + render-loss + EMA at
batch 16 / 256 px) draws transient GPU power spikes ≥ 1.5× TDP for
milliseconds at a time. On a RTX 3060 Ti with a marginal PSU, this caused
repeated full-system power-offs (REPORT §4.5 documents the diagnosis).

Mitigation:

```cmd
nvidia-smi -pl 170    # admin shell; caps GPU at 170 W, ~5-8% throughput cost
```

This is sufficient to complete S4_gan_heavy on the development hardware.
For the lighter Stage 4 runs (`adv ≤ 0.01`) the cap is unnecessary.
