# PBR Material Prediction from Basecolor — Research Report

**Author:** Andrii Ridzel
**Period:** April 2026
**Status:** Stages 1-3 complete; Stage 4 partially complete (S4_gan_heavy halted at epoch 21 of 150 due to a hardware power-budget constraint — see §4.5).

---

## 1. Problem statement

Given a single basecolor (albedo) image of a material, predict the three accompanying
PBR maps required to render it under a physically-based shader: **normal**,
**roughness**, and **metallic**. The downstream goal is to author renderable 3D
materials from a single diffuse photograph, instead of painting each map by hand.

The problem is fundamentally **underdetermined**: many distinct surface geometries
and material parameter sets can produce the same albedo. It is also **multi-task** with
sharply different output statistics — normals are roughly Gaussian around the
view-up direction (0, 0, 1), roughness is broad-spectrum, and metallic is bimodal
(materials are mostly either dielectric or metal, with little in between).

**The flat-normal failure mode** is the recurring problem this project ran into
and only **partially recovered from** — not solved. Because the ground-truth
normal distribution is roughly Gaussian around (0, 0, 1), per-pixel L1 between
prediction and target is minimized by predicting the per-image mean — i.e.
predicting a flat normal map gives a *low* L1 score even though it has solved
nothing. Standard val_loss metrics therefore can hide the failure entirely.
§4.7 measures predicted-normal spatial std on the held-out 285-sample test
set, comparing each prediction sample-by-sample to ground truth. The
trajectory was non-monotonic: Stage 1/1B sit at 14-16% of GT (essentially
flat); **Stage 2's separate decoder + 10× normal weight produces the
project's largest single improvement, jumping to 42% of GT** with no L1 cost;
Stage 3's loss-scheme pivot regressed back to 20%; Stage 4 baseline partially
recovers to 22%; and the Stage 4 PatchGAN at light/mid weight adds another
6-9 percentage points (28-31%). No shipped run reaches GT-level spatial
variation — even the best (S2_dual_w10 at 42%) leaves predicted normals
about 2.4× smoother than ground truth. The qualitative previews confirm
this: the GAN runs show somewhat more structure than the no-GAN baselines,
but every shipped run remains visibly smoother than ground truth.

## 2. Dataset

We use [**MatSynth**](https://huggingface.co/datasets/gvecchio/MatSynth), a curated
public PBR material dataset on the Hugging Face Hub. After filtering for
completeness (all four maps present), we work with ~5,700 samples spanning 14
material categories. The eight evaluation-relevant categories are Wood, Metal,
Fabric, Stone, Ceramic, Concrete, Ground, and Plaster.

**Splits.** From Stage 4 onwards, all runs use a frozen split file
(`outputs/stage4_split.json`, seed `4242`):

- 5,130 train  /  285 val  /  285 held-out test

The held-out test set is reserved for final evaluation and was not seen by any
training run reported here.

**Comparison set.** Eight representative samples — one per evaluation category —
are pinned in `outputs/comparison_set.json` and used by every run for visual
preview generation. This makes it possible to compare any two runs side-by-side
on the *same* inputs at the *same* training milestones, which became essential
once the experiment count grew past a handful.

## 3. Method

### 3.1 Architecture

`PBRUNet` (`src/model.py`) is a U-Net built on
[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
with a ResNet-34 encoder pretrained on ImageNet. Input is the basecolor image
(B, 3, H, W); output is a dict with `normal` (3 channels), `roughness` (1 channel),
and `metallic` (1 channel), each in `[0, 1]`.

Three architectural variants are configurable at construction time and were
each tested during the research:

- **Single shared trunk** (default): one U-Net producing all three maps from a common
  decoder. Used in Stages 1 and 1B.
- **Separate normal decoder** (`--separate-normal-decoder`): two parallel U-Nets,
  one dedicated to the normal map, one to roughness + metallic. Each has its own
  ResNet-34 encoder, so the model has **2.00× the parameters** of the shared
  trunk (48.9 M vs 24.4 M) and roughly **1.25× the wall-time per epoch** at
  batch 16 / 256 px on a single RTX 3060 Ti (Stage 2: 4.0 h vs 3.2 h). VRAM
  during training is ~2× because both U-Nets' activations are held simultaneously
  for backprop. Adopted from Stage 2 onwards because the qualitative gain on
  normals justified the cost; quantitatively the val-loss improvement at this
  stage was modest (see §4.3).
- **Height-map branch** (`--predict-height`): predict a single-channel height map,
  then derive the normal analytically by finite differences (`src/height_to_normal.py`).
  Tested in Stage 2; not adopted.

Optionally, the model accepts a category index per sample, which is converted
to a learned embedding (`category_embed_dim=8`) and concatenated as extra input
channels (`--use-category`, used in Stages 2-4).

### 3.2 Loss components

The training loss is a weighted sum of per-map and global terms, all gated by
CLI flags so individual contributions can be ablated:

| Term | Source | Purpose |
|---|---|---|
| L1 per map | `scripts/train.py` | Default per-pixel supervision. Available variants: `l1`, `mse`. |
| BCE on metallic | `--metallic-bce` | Handles the bimodal metallic distribution (Stage 1 finding — see §4.1). |
| SSIM on roughness | `--roughness-ssim` | Preserves local roughness texture detail. |
| Cosine distance on normal | `--normal-cosine` | Penalizes orientation error independently of magnitude. |
| Sobel gradient on normal | `--normal-gradient` | Penalizes flat / low-detail normal predictions. |
| FFT spectral loss | `--fft-weight` | Penalizes missing high-frequency detail (Stage 2; not adopted). |
| LPIPS perceptual | `--lpips-weight` | Perceptual similarity (Stage 1). |
| **GGX rendering loss** | `--render-loss` | Differentiable Cook-Torrance render (`src/rendering_loss.py`) under stochastic point lighting (3 diffuse + 6 near-specular configurations resampled each call), comparing renders of predicted vs ground-truth maps in log space. **Primary signal from Stage 3 onwards.** |
| Adversarial (PatchGAN + R1) | `--adversarial`, `--r1-gamma` | Perceptual sharpness via 70x70 PatchGAN discriminator (`src/discriminator.py`) with R1 gradient penalty for stability. Stage 4. |

Per-map weights (`--normal-weight`, `--roughness-weight`, `--metallic-weight`) scale
the contribution of each map's loss after all per-map components are accumulated.
These weights were re-tuned across stages (see §4.6) and are the reason raw
val-loss values are not directly comparable between Stages 1-2 and Stages 3-4.

### 3.3 Training infrastructure

- **EMA weights** (`src/ema.py`, decay 0.999). The shadow checkpoint
  `best_ema.pt` is the recommended inference checkpoint from Stage 4 onwards;
  in practice it consistently produces sharper qualitative output than `best.pt`.
- **Adversarial warmup** (`--adv-warmup-epochs`): the discriminator is trained but
  produces no gradient signal to the generator for the first N epochs, letting
  the discriminator catch up before it shapes the generator.
- **R1 gradient penalty** (`--r1-gamma`) on real samples to stabilize discriminator
  training. Applied every step in this work; lazy R1 was not implemented (see §7).
- **Resume support**, fixed comparison-set previews every `--preview-every` epochs,
  args.json + history.json + split_indices.json saved per run.

The full 4-stage research arc, stage-by-stage findings, and quantitative results
follow in §4-5.

---

## 4. Stage-by-stage findings

The work proceeded in four staged sweeps. Each stage tested a single hypothesis
and inherited the configuration from the previous stage. The arc is:

1. **Stage 1 / 1B** — *which per-pixel loss is the right baseline?* Result: L1 + BCE-on-metallic, trained for 100 epochs, is the strongest non-architectural baseline. Normals are flat (per-sample median ratio of predicted std to GT std at 14-16% across all six runs on the test set).
2. **Stage 2** — *what architectural change reduces the flat-normal collapse?* Result: separate normal decoder + 10× normal-loss weight. **Per-sample median ratio jumps to 42% of GT — the project's largest single improvement on the flat-normal problem, by a wide margin.** This finding has been chronically under-credited because (a) the val_loss number for this configuration looks high (1.08), an artefact of the 10× weight scaling (§4.6); and (b) earlier preliminary measurements on the 8-sample comparison set understated the gap to Stage 4.
3. **Stage 3** — *can the GGX rendering loss become the primary training signal?* Result: yes by val_loss; **but the lower per-map weights silently regressed normal sharpness from 42% back to 20% of GT**. The val_loss number became the lowest in the project to date but the underlying problem got worse.
4. **Stage 4** — *do extended training and adversarial loss still help?* Result: small but real. S4_baseline (no GAN) partially recovers to 22% on the flat-normal metric (extended training + EMA help). The PatchGAN at light/mid weight adds another 6-9 percentage points (28-31%). Stage 4 GAN's contribution to flat-normal recovery is real but **substantially smaller than Stage 2's earlier 26-point jump**. No Stage 4 run matches Stage 2 on this metric.

The val_loss numbers tell a different story than the flatness numbers, and
both are necessary to interpret what the project achieved. §4.6 explains why
val_loss is not directly comparable across stages, and §4.7 presents the
flatness analysis in full.

Each stage is documented in full below — hypothesis, sweep table, finding,
what carried forward.

### 4.1 Stage 1 — loss screening (50 epochs, 6 runs)

**Hypothesis:** With a fixed shared-trunk PBRUNet, identify the single most
effective per-pixel loss addition over a plain L1 baseline.

**Configuration (held constant):** `--batch-size 16 --normal-loss mse --use-category --epochs 50`. Each variant adds one CLI flag.

| run | extra flag | best val_loss |
|---|---|---|
| S1_bce | `--metallic-bce 1.0` | **0.7846** |
| S1_baseline | (none) | 0.7886 |
| S1_cosine | `--normal-cosine 1.0` | 0.8223 |
| S1_render | `--render-loss 0.25` | 0.8236 |
| S1_lpips | `--lpips-weight 0.1` | 0.8577 |
| S1_gradient | `--normal-gradient 0.5` | 0.9203 |

**Finding.** BCE on metallic is the best single addition — it correctly handles
the bimodal metallic distribution (most pixels are either pure-metal or
pure-dielectric; per-pixel L1 averages them). Render-loss as the *only* extra signal
**underperforms** at this stage (the model has not yet been given the capacity to
satisfy it). Sobel gradient loss actively hurts. Cosine and LPIPS are competitive
with baseline but not improvements.

**Carry-forward:** `--metallic-bce 1.0` becomes part of every subsequent run.

### 4.2 Stage 1B — loss combinations and first GAN (100 epochs, 4 runs)

**Hypothesis:** Stack the Stage 1 winner (BCE) with each of: longer training,
gradient regularization, render loss, and adversarial loss.

**Configuration:** Stage 1 config + `--metallic-bce 1.0 --epochs 100`.

| run | extras | best val_loss |
|---|---|---|
| S1B_bce_long | (none) | **0.7417** |
| S1B_bce_gan | `--adversarial 0.01` | 0.7557 |
| S1B_bce_render | `--render-loss 0.1` | 0.7590 |
| S1B_bce_grad | `--normal-gradient 0.05` | 0.7754 |

**Finding.** Just running BCE for 100 epochs instead of 50 outperformed every
addition. The first PatchGAN attempt at adv=0.01 was nearly neutral — adversarial
loss without the architectural fixes that come in Stage 2 doesn't help.
Render-loss as a small auxiliary (weight 0.1) gave a marginal contribution.

**Critical context:** *the qualitative output at this point still has flat
normals.* The val-loss numbers improved because BCE drove metallic sharpness,
but per-category preview images showed normal predictions collapsing toward
(0, 0, 1). This is the failure mode that motivated Stage 2.

### 4.3 Stage 2 — normal-prediction architecture (100 epochs, 6 runs)

**Hypothesis.** The flat-normal collapse has two roots: (a) **multi-task loss
imbalance** — empirically, the normal-map loss contributed only ~5.6% of the
total while roughness contributed ~76%; (b) **decoder capacity competition** —
in a shared trunk, the decoder weights that would produce sharp normals are
also responsible for roughness and metallic, and the gradients pull in
different directions. Test five mitigations in one sweep:
(1) reweight normals 10×, (2) separate decoder head for normals,
(3) XY-only normal prediction with Z derived analytically,
(4) FFT spectral penalty for missing high-frequency detail,
(5) height-map parameterization with normals derived by finite differences.

**Configuration:** Stage 1B config + `--normal-weight 10 --metallic-bce 1.0`.

| run | additional flags | best val_loss |
|---|---|---|
| S2_dual_w10 | `--separate-normal-decoder` | **1.0808** |
| S2_weight10 | (none — weight only) | 1.0882 |
| S2_weight10_xy | `--normal-xy` | 1.1170 |
| S2_height | `--separate-normal-decoder --predict-height` | 1.1510 |
| S2_dual_w10_fft | `--separate-normal-decoder --fft-weight 0.1` | 2.4641 |
| S2_height_fft | `--separate-normal-decoder --predict-height --fft-weight 0.1` | 2.5194 |

**Findings.**
- The 10× normal weight alone (S2_weight10) does most of the work; the additional
  separate decoder gives a small further improvement (1.0882 → 1.0808).
- **XY-only prediction does not help.** The `--normal-xy` flag enforces an
  analytic Z but adds constraint without unlocking detail.
- **Height-map prediction does not help.** Predicting a 1-channel height and
  taking finite differences should bias toward physically plausible normals,
  but the gradient flow through the differentiation step appears to be
  the issue.
- **FFT spectral loss made things dramatically worse** (1.08 → 2.46). The likely
  explanation is that with per-pixel L1 still dominating, the FFT term penalizes
  any high-frequency content the model produces, including correct content,
  faster than it rewards missing content. We did not pursue FFT further.

**Carry-forward:** `--separate-normal-decoder --normal-weight 10` becomes the
architecture from this point on. The val-loss values are *higher* than Stage 1B
not because the model got worse, but because the 10× normal-weight scaled the
loss number up — qualitative previews at Stage 2 had visibly sharper normals
than Stage 1B for the first time.

### 4.4 Stage 3 — render-loss as primary signal (100 epochs, 4 runs)

**Hypothesis.** With the architecture fixed, pivot the loss scheme: **reduce**
per-map weights so they become stabilizers, and let the GGX rendering loss
become the dominant supervision signal. Sweep its weight to find the operating
point.

**Configuration:** Stage 2 architecture + `--normal-weight 0.1 --roughness-weight 0.1 --metallic-weight 0.2 --metallic-bce 1.0 --normal-cosine 0.0 --roughness-ssim 0.0 --normal-loss l1`. The four runs vary only `--render-loss`.

| run | `--render-loss` | best val_loss |
|---|---|---|
| **S3_rw1** | **1.0** | **0.1924** |
| S3_rw2 | 2.0 | 0.3430 |
| S3_rw5 | 5.0 | 0.7567 |
| S3_rw10 | 10.0 | 1.4535 |

**Finding.** Strong, monotonic, and decisive. Render-loss weight 1.0 is
optimal; every increase past 1.0 makes results worse. Beyond the point where
render loss balances the per-map L1 supervision, it begins to dominate the
gradient with high-variance physics-aware updates, and convergence breaks down.

**Why this looked like the headline result at the time.** Stage 3's best run (S3_rw1)
reached val_loss 0.19 — roughly **75% lower than Stage 2's best (1.08)**. Most of
that drop is artefactual: the per-map weights are now smaller, so the loss
numerator shrinks (see §4.6).

**A finding that does not show up in val_loss.** The lower per-map weights also
weakened the gradient that had been holding normals away from flat. Stage 3
in fact **regressed on the flat-normal problem** relative to Stage 2 — on
the held-out test set, S3_rw1 has a per-sample median ratio of 20% of GT
spatial variation, down from Stage 2's 42% (§4.7). This is masked by
val_loss because flat predictions *are* the per-pixel L1 minimum on a
roughly-Gaussian target. Stage 4 partially recovers (22-31%) but never
matches Stage 2 on this metric.

**Carry-forward:** `--render-loss 1.0` and the `0.1/0.1/0.2` per-map weight
scheme become the Stage 4 baseline. The flat-normal regression was not yet
identified at this point — it surfaced only when Stage 4's GAN runs produced
visibly different qualitative output.

### 4.5 Stage 4 — extended training + GAN (150 epochs, 4 runs)

**Hypothesis.** With architecture (§4.3) and loss scheme (§4.4) settled, the
remaining sources of improvement are (a) longer training and (b) a PatchGAN
discriminator for perceptual sharpening. Sweep adversarial weight on the
otherwise-frozen Stage 3 winner.

**Configuration:** Stage 3 winner config + `--epochs 150 --separate-normal-decoder --r1-gamma 10.0 --adv-warmup-epochs 5` + EMA + frozen 5130/285/285 split. Variants differ only in `--adversarial`.

| run | `--adversarial` | best val_loss | epochs completed |
|---|---|---|---|
| **S4_gan_light** | **0.005** | **0.1828** | 150 / 150 |
| S4_baseline | 0 (control) | 0.1923 | 150 / 150 |
| S4_gan_mid | 0.01 | 0.2030 | 150 / 150 |
| S4_gan_heavy | 0.05 | 0.2171 (e20) | **21 / 150 — incomplete** |

**Findings (val_loss).**
- **S4_baseline at 150 epochs is approximately tied with S3_rw1 at 100 epochs**
  (0.1923 vs 0.1924). Extended training alone, on this architecture and loss
  scheme, gave essentially nothing.
- **Light adversarial (adv=0.005)** has the lowest val_loss of the project at
  0.1828 — but the margin over the no-GAN baseline is **~5%**, which on its own
  would be uninteresting.
- **Mid adversarial (adv=0.01)** has higher val_loss (0.2030) but is qualitatively
  competitive and arguably the better checkpoint (see §4.7).
- **Heavy adversarial (adv=0.05)** underperforms in val_loss and was halted at
  21 epochs by hardware constraints.
- The EMA checkpoint (`best_ema.pt`) consistently produces visibly smoother
  qualitative output than the non-EMA `best.pt`. We recommend `best_ema.pt`
  for inference.

**Findings (flat-normal recovery — see §4.7 for details).** Re-evaluation on
the held-out 285-sample test set shows Stage 4's contribution to the
flat-normal problem is real but smaller than initial 8-sample estimates
suggested. S4_baseline alone (no GAN, just longer training + EMA) reaches a
per-sample median ratio of 22% of GT — already a partial recovery from
Stage 3's 20%. The PatchGAN adds another 6-9 percentage points: S4_gan_light
at 28%, S4_gan_mid at 31%. **None of the Stage 4 runs match Stage 2's 42%**
(§4.7). The PatchGAN's role is best characterized as a small marginal
sharpening on top of the architecture and loss-scheme decisions made in
earlier stages, not as the primary fix for flat normals.

**S4_gan_heavy incompleteness.** This run was halted at epoch 21 of 150 after
two hardware-level crashes (full PC power-off, not driver/software). Diagnosis
traced this to GPU power-transient spikes exceeding the development PSU's
stable headroom on the heaviest configuration: R1 gradient penalty with
`create_graph=True` (double-backward) plus GGX rendering pass plus PatchGAN at
adv=0.05, all in one training step. The transient power profile of this combination
is the worst case in the experiment matrix. Mitigation tested:
`nvidia-smi -pl 170` (cap GPU at 170 W, ~5-8% throughput cost) reduced
spikes in standalone testing. The decision was to ship the partial run as-is
rather than block this release on a multi-day re-train. The run's
`best_ema.pt` is from epoch 20; `latest.pt` (resume point) is also shipped on
the HF Hub for anyone continuing this work.

### 4.6 On comparing val_loss across stages

The val_loss numbers above are **not directly comparable across all stages**.
The per-map weights in `compute_loss` are CLI flags that were re-tuned between
Stages 1B → 2 → 3:

- Stages 1, 1B: per-map weights default `1.0`. Total loss is the sum of three roughly equal map losses.
- Stage 2: `--normal-weight 10` adds ~10× the normal contribution. Numerator scales up — *higher* val_loss values do not mean *worse* models.
- Stage 3, 4: `--normal-weight 0.1 --roughness-weight 0.1 --metallic-weight 0.2`. Numerator scales down — Stage 3 numbers (~0.2) are not directly an order of magnitude better than Stage 2 (~1.1).

For genuinely comparable numbers across stages, compare the per-map raw L1
values stored in each `history.json` under `val_maps_raw.<map>` — those are
unweighted and represent the same quantity in every run. The
`history.json` files are bundled in `results/<run>/`. A future evaluation run
of `scripts/eval_full.py` against the held-out test set would produce these
values cleanly for every checkpoint in one place.

### 4.7 The flat-normal problem — empirical measurement on the held-out test set

The flat-normal collapse described in §1 was the central failure mode the early
stages were designed to fix, but neither val_loss nor raw normal L1 actually
measure flatness. **Per-pixel L1 between predicted and ground-truth normals is
minimized by predicting the per-image mean** (since the ground-truth normal
distribution is roughly Gaussian around the view-up direction). A model that
correctly predicts a flat normal map gets very low L1 — and therefore low
val_loss — even though it has solved nothing.

To see what the project actually achieved on this problem, we run inference
for each shipped checkpoint on the **285-sample held-out test set**
(`outputs/stage4_split.json`, seed 4242), compute per-channel spatial standard
deviation of the predicted normal map, and compare it sample-by-sample to the
ground-truth normal's spatial std on the same image. The metric is reported
as the **per-sample median ratio** of `predicted_std / gt_std`. 100% would
mean predictions match ground-truth spatial variation on the typical sample;
near 0% means the prediction is essentially constant (flat).

Samples with GT std < 0.005 (essentially flat ground truth, e.g. smooth
ceramic — 42 of 285 samples) are excluded from the median; the ratio is
ill-defined when both sides approach zero. The remaining **243 samples**
form the basis of the table below.

| Run | per-sample median ratio | aggregated ratio | L1 vs GT (mean ± SE) |
|---|---|---|---|
| S1B_bce_long (100 ep) | 14% | 15.1% | 0.0435 ± 0.0024 |
| S1B_bce_gan (100 ep) | 16% | 15.0% | 0.0430 ± 0.0024 |
| S1_bce (50 ep) | 16% | 16.0% | 0.0436 ± 0.0024 |
| S4_gan_heavy (21 ep) | 16% | 18.5% | 0.0548 ± 0.0020 |
| S3_rw1 (100 ep) | 20% | 22.2% | 0.0415 ± 0.0025 |
| S4_baseline (150 ep) | 22% | 30.1% | 0.0420 ± 0.0025 |
| S4_gan_light (150 ep) | 28% | 32.2% | **0.0405 ± 0.0024** |
| S4_gan_mid (150 ep) | 31% | 32.1% | 0.0411 ± 0.0024 |
| **S2_dual_w10 (100 ep)** | **42%** | **42.0%** | **0.0407 ± 0.0021** |

The aggregated column is the older `mean(predicted_std) / mean(gt_std)`
metric. It is included for transparency but is biased toward samples with
high GT std (textured wood etc.); the per-sample median is the more honest
number when GT spatial variation is itself heavily skewed across samples
(test-set GT std median 0.0417, mean 0.0512, range [0, 0.1948]).

**What this table reveals.**

- **Stage 2 (S2_dual_w10) is the project's clearest single advance on the
  flat-normal problem.** Per-sample median ratio of 42% is roughly **2.6× the
  Stage 1/1B baseline** (14-16%), and it achieves this with L1 (0.0407) tied
  with the very best run in the table. The Stage 2 finding has been
  systematically under-credited because (a) the val_loss number for this
  configuration looks high (1.08), an artefact of the 10× normal weight (§4.6),
  and (b) earlier preliminary measurements on the 8-sample comparison set
  understated the gap to Stage 4.
- **Stage 1 / 1B all produce essentially flat normals** (per-sample median
  ratio 14-16% of GT). The first PatchGAN attempt (`S1B_bce_gan`, adv=0.01
  with shared trunk) did not change this — adversarial supervision at that
  stage had nothing meaningful to discriminate.
- **Stage 3 regressed on flatness** (S3_rw1: 20%, down from S2's 42%).
  Reducing the per-map weights from 10 to 0.1 to give the render loss room
  to dominate also stripped out the gradient that had been pushing normals
  toward sharpness. The val_loss kept dropping because the render loss alone
  is satisfied by smooth, plausibly-shaped surfaces — a silent regression
  invisible in standard training metrics.
- **Stage 4 baseline (no GAN) partially recovers** to 22%, presumably from
  150 vs 100 epochs and EMA shadow weights. This was hidden by the
  preliminary 8-sample measurement which had reported 14%.
- **The Stage 4 PatchGAN's contribution is real but small.** S4_gan_light
  (28%) and S4_gan_mid (31%) lift the ratio another 6-9 percentage points
  over S4_baseline. This is a meaningful improvement, but **substantially
  smaller than initial measurements suggested**, and substantially smaller
  than Stage 2's earlier 26-point jump.
- **No shipped run reaches GT-level spatial variation.** Even the best
  (S2_dual_w10 at 42%) leaves predicted normals about 2.4× smoother than
  ground truth. The project reduced flat-normal collapse by a factor of
  roughly 2-3× over baseline; it did not eliminate it.
- **S4_gan_heavy (21 ep, incomplete) is in the bottom group.** Its
  flatness is similar to Stages 1/1B and its L1 is the worst (0.0548) — a
  combination of "early stopping" and "the heavy adversarial weight had not
  yet stabilized."

**Practical implication for inference choice.**

| If you care most about… | Use |
|---|---|
| **Non-flat normals** (max spatial variation at no L1 cost) | `S2_dual_w10/best.pt` — 42% of GT, L1 0.0407 |
| **Lowest L1 vs GT** | `S4_gan_light/best_ema.pt` — L1 0.0405, std 28% |
| **Stage 4 sharpening on top of S4_baseline architecture** | `S4_gan_mid/best_ema.pt` — std 31%, L1 0.0411 |

S2_dual_w10 is the conservative recommendation — the metric pair (std + L1)
that matters most for downstream rendering favours it, and it is robust to
methodology choices (per-sample median 42% and aggregated 42% agree exactly).
The Stage 4 GAN runs add small marginal sharpness over S4_baseline at
competitive L1, but they do not match Stage 2 on the flatness metric.

**Methodology and caveats.**

- The metric is a sharpness lower bound, not an upper bound on correctness.
  A model that outputs random noise at the right magnitude would also score
  high on per-channel std. The L1-vs-GT column is the partner metric: low
  std → flat (regardless of L1); high std + high L1 → noisy/incorrect; high
  std + low L1 → genuine improvement. S4_gan_heavy sits in the noisy/early
  regime; everything else is in either the flat or the genuine-improvement
  regime.
- A preliminary version of this analysis was run on the 8-sample
  comparison set (`outputs/comparison_set.json`) and produced numbers that
  overstated Stage 4 GAN's contribution. Two methodology issues caused this:
  the comparison-set samples skew toward smoother materials (lower GT std,
  inflating "% of GT" for low-variance predictions), and a script bug fed
  all category-aware Stage 4 runs the `'unknown'` category for every
  inference. The test-set re-run reported here uses the correct category
  labels and the full held-out test set; its numbers supersede the
  preliminary ones.
- The methodology is reproducible from the bundled checkpoints with
  `python scripts/analyze_flat_normals.py` (full table in
  `results/FLAT_NORMALS_ANALYSIS.md`). See `docs/reproduction.md`.

---

## 5. Final results

Nine runs are shipped: one or two per stage covering the
research arc plus the four Stage 4 sweep runs. Lightweight metadata
(`args.json`, `history.json`, qualitative `previews/`) is in
`results/<run>/`; full checkpoints are on the private Hugging Face Hub repo
`Andrid1/vrtest-pbr-handoff` (see `results/README.md`).

| Run | Stage | Epochs | Best val_loss | Flat-normal median (% GT) | L1 vs GT | Notes |
|---|---|---|---|---|---|---|
| S1_bce | 1 | 50 / 50 | 0.7846 | 16% | 0.0436 | Stage 1 anchor (flat normals) |
| S1B_bce_long | 1B | 100 / 100 | 0.7417 | 14% | 0.0435 | BCE + long training (flat) |
| S1B_bce_gan | 1B | 100 / 100 | 0.7557 | 16% | 0.0430 | First GAN attempt (flat — pre-architecture-fix) |
| **S2_dual_w10** | 2 | 100 / 100 | 1.0808 | **42%** | **0.0407** | **Project's largest flat-normal recovery** |
| S3_rw1 | 3 | 100 / 100 | 0.1924 | 20% | 0.0415 | Render-loss-primary; silent flatness regression |
| S4_baseline | 4 | 150 / 150 | 0.1923 | 22% | 0.0420 | Stage 4 control (no GAN) |
| **S4_gan_light** | 4 | 150 / 150 | **0.1828** | 28% | **0.0405** | **Lowest L1 of all runs** |
| **S4_gan_mid** | 4 | 150 / 150 | 0.2030 | 31% | 0.0411 | Stage 4 GAN sharpening |
| S4_gan_heavy | 4 | **21 / 150** | 0.2171 | 16% | 0.0548 | Incomplete (PSU constraint) |

Flat-normal median ratio = per-sample median of `predicted_std / gt_std` on
the 285-sample held-out test set, samples where GT std > 0.005 only (n=243).
100% would mean prediction matches GT spatial variation.

### Headline result

**The project's strongest run on flat-normal recovery is `S2_dual_w10`.** It
reaches 42% of GT spatial variation — the largest recovery in the project
by a wide margin — while paying no L1 cost (0.0407, tied with the best
runs in the table). The Stage 4 sweep produced two competitive checkpoints
on different metrics: `S4_gan_light` has the lowest L1 (0.0405) and modest
flatness recovery (28%), and `S4_gan_mid` has middle-of-the-pack L1 (0.0411)
with somewhat better flatness (31%). All three are defensible inference
choices depending on the downstream task; **none reach GT-level normal
spatial variation, and `S2_dual_w10` is the only one that exceeds 31%.**

| If you care most about… | Use | Why |
|---|---|---|
| **Non-flat normals** | `S2_dual_w10/best.pt` | Best flat-normal metric (42% of GT) at tied-best L1; supersedes the Stage 4 GAN runs on this dimension |
| **Lowest L1 vs ground truth** | `S4_gan_light/best_ema.pt` | Lowest L1 (0.0405); modest flatness recovery (28%) |
| **Stage 4 sharpening on top of S4_baseline** | `S4_gan_mid/best_ema.pt` | Moderate L1 (0.0411); 31% flatness — best Stage 4 run |

**No single recommended checkpoint dominates on every metric.** This is the
honest reading of the data; see §4.7 for the full flat-normal analysis and §7
for the remaining open problems.

### Test-set evaluation (eval_full.py)

The full test-set evaluation has been run for every shipped checkpoint
(`scripts/eval_full.py` against `outputs/stage4_split.json[test]`, n=285).
Per-map L1 / SSIM / PSNR, GGX render loss, rendered-LPIPS, and
per-category breakdowns are in
[`results/EVAL_REPORT.md`](../results/EVAL_REPORT.md). Key findings:

| Metric | Winner | Value |
|---|---|---|
| **Render loss** (renderer-fidelity) | `S4_gan_light/best_ema` | 0.1364 |
| **Rendered-LPIPS** (perceptual rendered quality) | `S2_dual_w10/best` | 0.1498 |
| **Normal L1** | `S4_gan_light/best_ema` | 0.0406 |
| **Roughness L1** | `S1B_bce_long/best` | 0.1518 |
| **Metallic L1** | `S1B_bce_long/best` | 0.0257 |
| **Normal PSNR** | `S2_dual_w10/best` | 23.61 dB |
| **Roughness PSNR** | `S2_dual_w10/best` | 14.39 dB |
| **Metallic PSNR** | `S1B_bce_long/best` | 28.20 dB |

Practical reading: **`S4_gan_light/best_ema` wins on render loss and normal L1**,
making it the strongest single checkpoint for "feed in a basecolor and get a
renderable PBR set" — the headline practical use case. **`S2_dual_w10/best` wins
on perceptual metrics** (rendered-LPIPS, normal/roughness PSNR) and on the
flat-normal recovery (§4.7). `S1B_bce_long` wins on roughness/metallic L1 — but
its flat-normal recovery is essentially zero (14% of GT, §4.7), so it is a
specialised baseline rather than a general recommendation. `S4_gan_heavy` is
worst on most metrics, consistent with its incomplete training (§4.5).

### What's available in this package

- **Per-run training trajectory** (`results/<run>/history.json`),
  hyperparameters (`args.json`), and qualitative previews (`previews/*.png`)
  for every shipped run.
- **One-shot demo** (`examples/demo.py --image examples/inputs/wood.png --run S4_gan_light`)
  for end-to-end inference in roughly 30 seconds — downloads a checkpoint,
  runs it, saves a 4-panel grid.
- **Reproduction commands** for every shipped run in `docs/reproduction.md`.

---

## 6. Qualitative results

Each shipped run includes a `previews/` directory with side-by-side images
generated against the fixed 8-sample comparison set at training milestones
(every `--preview-every` epochs). These are the visual ground-truth for the
quantitative findings in §4.

The most informative comparisons:

- **S2_dual_w10/previews/** vs **S1B_bce_long/previews/** — Stage 2's
  architectural fix vs the Stage 1B no-architecture baseline. Side-by-side
  shows the first stage where predicted normals show local detail rather than
  a uniform pale-blue. Quantitatively the largest jump in the project (§4.7).
- **S3_rw1/previews/** vs **S2_dual_w10/previews/** — Stage 3's render-loss
  pivot. Roughness and metallic become visibly more accurate; normals are
  smoother (this is the regression §4.4 / §4.7 quantify).
- **S4_gan_mid/previews/** vs **S4_baseline/previews/** — the PatchGAN's
  effect on normal sharpness, holding everything else constant. The
  improvement is visible but modest, consistent with the +9-percentage-point
  gain in §4.7.
- **S4_gan_mid/previews/** vs **S2_dual_w10/previews/** — the comparison
  that calibrates expectations: even the project's strongest run (S2) leaves
  predicted normals visibly smoother than the ground truth shown alongside
  in each preview.

A unified qualitative grid across all 9 shipped runs can be generated by
`scripts/qualitative_grid.py` once checkpoints are downloaded from HF Hub
(see `docs/reproduction.md`).

---

## 7. Limitations & open problems

For a future engineer continuing this work.

### 7.1 Methodology — track a sharpness metric during training

The flat-normal regression at Stage 3 (§4.7) was caught only by post-hoc
measurement after Stage 4 completed. **The val_loss number kept dropping
through the regression**, so log-watching gave no signal. Per-pixel L1 has a
degenerate minimum at "predict the mean" for the roughly-Gaussian normal
distribution, and any metric that does not measure spatial diversity will
miss this.

**Concrete recommendation:** add `val_normal_std` to `validate()` in
`scripts/train.py` and log it in `history.json` alongside `val_maps_raw`.
The implementation is two lines:

```python
# inside validate(), per-batch:
normal_std_batch = pred["normal"].std(dim=(2, 3)).mean().item()
# aggregate across val batches, store in val_maps_raw["normal_std"]
```

This is the single highest-leverage methodological change to the codebase. It
would have prevented committing the Stage 3 sweep + Stage 4 baseline to a
silent regression. Cost during training: negligible (a single reduction).

### 7.2 Test-set evaluation has been run; per-category breakdown follows

`scripts/eval_full.py` has been run against the held-out 285-sample test
set (`outputs/stage4_split.json[test]`) for every shipped checkpoint. Per-map
L1 / SSIM / PSNR, GGX render loss, rendered-LPIPS, and per-category
breakdowns are in [`results/EVAL_REPORT.md`](../results/EVAL_REPORT.md);
per-run JSON output is also under each run's `eval_report.json` in the
source repo's `outputs/` (not shipped here — they are reproducible from the
checkpoints + test set).

Per-category normal L1 (best 3 runs, full table in EVAL_REPORT.md) shows that
the model handles **Marble, Plastic, and Metal** best (~0.01-0.02 L1) and
**Stone, Ground, Wood** worst (~0.06+ L1) — the textured high-frequency
categories. Rare categories (Marble, Misc — 8 samples each) are noisier but
do not appear systematically degraded relative to their nearest neighbours.

### 7.3 S4_gan_heavy is incomplete

Halted at epoch 21 of 150 due to GPU power transients exceeding PSU headroom
(see §4.5). To complete it, `nvidia-smi -pl 170` (admin) caps the spikes at
~5-8% throughput cost. The `latest.pt` resume point is included on HF Hub.
Whether finishing this run is informative depends on whether the heavy
adversarial weight (0.05) is hypothesized to recover from its early
under-performance — current Stage 4 evidence (§4.5) suggests it would not.

### 7.4 Per-category metrics — open work on hard categories

Per-category breakdowns from `eval_full.py` (see §7.2 / EVAL_REPORT.md) reveal
the categories the model handles worst by normal L1: **Stone (n=44), Ground
(n=11), Wood (n=51), Plaster (n=14)** — the high-frequency textured
materials. **Marble (n=8), Plastic (n=14), Metal (n=28)** are handled best.

The pattern is consistent across S2_dual_w10, S4_gan_light, S4_gan_mid:
their per-category L1 numbers track each other within ~0.005 L1 on every
category. **No shipped run has a substantially-different per-category profile**;
the variation between checkpoints is dominated by the per-map decisions
(separate decoder, render loss weight, GAN), not category-specific
specialisation. Future work targeting the hard-category gap is plausibly
where further per-pixel improvements live.

### 7.5 Discriminator architecture is fixed

The PatchGAN topology (4 layers, 70x70 receptive field) was chosen for
simplicity; multi-scale, U-Net-disc, and ProjectionGAN variants were not
explored. Given that the discriminator is the loss term that resolved the
flat-normal problem (§4.7), its architecture is one of the higher-leverage
hyperparameters to revisit.

### 7.6 Single-image inference only

The model conditions on one basecolor image. Multi-view or sequence-based
inputs would constrain the underdetermined inverse problem (§1) significantly
— two views of the same surface from different angles place strong geometric
constraints on the normal map.

### 7.7 GGX rendering loss uses simple stochastic point lights

`src/rendering_loss.py` resamples 3 diffuse + 6 near-specular point-light
configurations per call. The randomness reduces overfitting to any one
lighting setup, but the lighting is still a small handful of point sources
in flat space. A learned environment map or HDR-IBL lighting setup might
give richer gradients, especially for materials whose appearance is
dominated by reflections off complex environments rather than direct
point sources.

### 7.8 Domain gap to real photographs

Tested only on MatSynth's curated synthetic basecolors. Real-world photos
have lighting baked in, colour-cast, sensor noise, and JPEG artefacts that
the model never saw during training. A domain-adaptation step (or training
on a real-photo dataset like AmbientCG photographs) is the prerequisite for
deployment beyond demo use.

### 7.9 No shipped run reaches GT-level normal spatial variation

The best run on the flat-normal metric is `S2_dual_w10` at 42% of GT
spatial variation; everything else is at 31% or below (§4.7). The
qualitative previews confirm this: predicted normals at every shipped run
remain visibly smoother than the ground truth, even on textured materials
where the GT has rich spatial detail. Closing the remaining ~60% gap is
the most important open problem on this work.

Plausible directions, none yet tested:

- **Higher discriminator capacity / different topology.** A multi-scale
  discriminator (StyleGAN-style) or a U-Net discriminator may push the
  generator to produce variation at scales the 70×70 PatchGAN cannot see.
- **Discriminator on rendered images, not raw maps.** The current discriminator
  scores the predicted maps directly. A discriminator that sees the *render*
  (a rendered image of the predicted maps under various lighting) might
  reward physically-plausible normal variation more directly.
- **Higher adversarial weights with stable training.** S4_gan_heavy at adv=0.05
  was halted at epoch 21 by hardware constraints; if completed it might lift
  std further (or not — current evidence is ambiguous since adv=0.05 also
  hurt val_loss). With lazy R1 (§7.10) the heavier-adv configurations become
  computationally tractable.
- **Augment the loss with an explicit variation reward** — e.g. penalize
  KL-divergence between predicted-normal and GT-normal per-pixel-direction
  histograms.

### 7.10 R1 is applied every step

The current code applies the R1 gradient penalty (with `create_graph=True`,
hence double-backward) at every discriminator step. The original
StyleGAN2 paper's "lazy R1" applies it every 16 steps with no measurable
quality loss. Adding `--r1-every N` would cut Stage 4's per-step compute
significantly and likely also relieve the power-transient issue from §4.5
without needing the `nvidia-smi` workaround.

---

## 8. Reproduction

See [`docs/reproduction.md`](reproduction.md) for environment setup, dataset
download / pre-cache instructions, training commands for each shipped run,
inference / demo usage, and the test-set evaluation that §7.2 recommends.

---

## Appendix — files and where to find them

- **Code:** `src/` (model, losses, rendering, discriminator, EMA, transforms),
  `scripts/` (train, eval, predict, qualitative_grid, predownload, etc.)
- **Tests:** `tests/` (44 test cases covering core modules)
- **Per-run metadata:** `results/<run>/{args.json, history.json, previews/}`
- **Cross-run summary:** `results/SUMMARY.md`
- **Flat-normal test-set analysis:** `results/FLAT_NORMALS_ANALYSIS.md`
  (reproduce with `scripts/analyze_flat_normals.py`)
- **Full test-set evaluation:** `results/EVAL_REPORT.md`
  (reproduce with `scripts/eval_full.py`)
- **Checkpoints:** `https://huggingface.co/Andrid1/vrtest-pbr-handoff` (private)
- **Demo:** `examples/demo.py`, sample inputs in `examples/inputs/`
- **This report:** `docs/REPORT.md`
