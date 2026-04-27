# VR-pbr-handoff — PBR material prediction from basecolor

> Predict normal, roughness, and metallic maps from a single basecolor image.
> A 4-stage research project on the MatSynth PBR dataset.

**Author:** Andrii Ridzel
**Period:** April 2026

## Where to look

| Path | What it is |
|---|---|
| **`docs/REPORT.md`** | **Start here.** Full research report — problem, method, stage-by-stage findings, results, limitations. Reads end-to-end in ~15 minutes. |
| `docs/reproduction.md` | Environment setup, dataset download, training and evaluation commands. |
| `src/` | Model package (PBRUNet, dataset loaders, losses, GGX rendering, PatchGAN, EMA). |
| `scripts/` | Entry points: `train.py`, `eval.py`, `predict.py`, `qualitative_grid.py`, `eval_full.py`, etc. |
| `tests/` | 44 unit tests covering the core modules. Run with `pytest tests/`. |
| `results/` | Per-run lightweight metadata: `args.json`, `history.json`, qualitative `previews/`. **Heavy checkpoints are on Hugging Face Hub** — see `results/README.md`. |
| `examples/` | One-shot demo (`examples/demo.py`) and sample input images. |

## 60-second smoke test

```bash
pip install -r requirements.txt
huggingface-cli login    # private checkpoint repo, requires read access
python examples/demo.py --image examples/inputs/wood.png --run S4_gan_mid
```

You should get `examples/output/wood__S4_gan_mid__predicted.png` — a 4-panel
grid showing the input basecolor and the three predicted PBR maps.

## Headline result

Nine training runs are shipped, covering the full research arc. Re-evaluation
on the 285-sample held-out test set produced a non-monotonic flat-normal
trajectory and one clear standout:

- **`S4_gan_light/best_ema.pt`** — best on **render loss (0.1364)** and
  normal L1 (0.0406). The strongest single checkpoint for the headline
  practical use case ("feed in a basecolor, get a renderable PBR set").
- **`S2_dual_w10/best.pt`** — best on **rendered-LPIPS (perceptual rendered
  quality)**, best on **flat-normal recovery** (42% of GT spatial variation
  vs 14-16% in Stages 1/1B), and best on normal/roughness PSNR. The
  perceptually strongest run, often under-credited because its training-time
  val_loss is artificially inflated by the 10× normal weight (REPORT §4.6).
- **`S4_gan_mid/best_ema.pt`** — middle ground; competitive on most metrics
  but does not lead on any.
- **`S1B_bce_long/best.pt`** — best on roughness L1 and metallic L1, but its
  normals are flat. A specialised baseline rather than a general
  recommendation.

**No single recommended checkpoint dominates on every metric.** Choose based
on the downstream task. Full per-map / per-category benchmark on the
held-out 285-sample test set is in
[`results/EVAL_REPORT.md`](results/EVAL_REPORT.md); the flat-normal
analysis is in [`results/FLAT_NORMALS_ANALYSIS.md`](results/FLAT_NORMALS_ANALYSIS.md).
**No shipped run reaches GT-level normal spatial variation;** REPORT §7
documents the open problems for future work.

## What the project achieved (one paragraph)

A four-stage research arc on a single underlying U-Net architecture, starting
from a shared-trunk U-Net that produced flat normals and ending with a
separate-decoder U-Net trained against a GGX rendering loss with optional
PatchGAN sharpening. **Progress on the flat-normal problem was non-monotonic
and held-out test-set evaluation reframed it.** Stage 2's separate-decoder +
10× normal-weight change is the project's largest single recovery on the
flat-normal metric (42% of GT spatial variation, up from 14-16% in
Stages 1/1B). Stage 3's loss-scheme pivot — reducing per-map weights to let
the render loss dominate — silently regressed normal sharpness back to 20%
while val_loss continued to drop, a methodological cautionary tale described
in REPORT §4.4. Stage 4's extended training partially recovered to 22%, and
the PatchGAN at light/mid adversarial weight added another 6-9 percentage
points (28-31%), but **no Stage 4 run matches Stage 2 on flat-normal
recovery, and no shipped run reaches GT-level spatial variation.** This
finding only became visible after the 285-sample test-set re-run; an earlier
8-sample preliminary analysis had overstated the Stage 4 GAN contribution
because of sample selection bias and a category-label bug (REPORT §4.7).
See REPORT §7.1 for the two-line fix (adding `val_normal_std` as a
training-time metric) that would have caught the Stage 3 regression at the
time.

## Where the heavy stuff lives

- **Model checkpoints** (~200 MB each, 9 runs): private Hugging Face Hub repo
  `Andrid1/vrtest-pbr-handoff`. See [`results/README.md`](results/README.md)
  for download instructions.
- **Dataset:** public MatSynth on Hugging Face Hub. See `docs/reproduction.md` §4.

## Tests

```bash
pytest tests/ -v
```

44 tests, ~12 seconds end-to-end on CPU. Coverage: model architectures,
losses (L1, FFT, gradient, render), discriminator, EMA, GAN-stability
machinery, height-to-normal conversion, run summarization.

## License

[MIT](LICENSE).
