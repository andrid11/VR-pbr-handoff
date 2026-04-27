# Full test-set evaluation (eval_full.py)

Evaluation of all 9 shipped checkpoints on the held-out 285-sample test set (`outputs/stage4_split.json` under `"test"`, frozen seed 4242).

Metrics:

- **n/r/m _L1**: per-pixel L1 distance to ground-truth normal / roughness / metallic.
- **render**: GGX rendering loss (mean over batches, deterministic seed). Lower = predicted maps render closer to GT-rendered images.
- **rLPIPS**: rendered-LPIPS, perceptual distance between renders of predicted vs GT maps under one fixed light/view. Lower = visually closer rendered output.
- **n/r/m _PSNR**: per-map peak signal-to-noise ratio in dB. Higher = closer.

## Overall metrics, sorted by render loss (lowest first)

| run / ckpt | n_L1 | r_L1 | m_L1 | render | rLPIPS | n_PSNR | r_PSNR | m_PSNR |
|---|---|---|---|---|---|---|---|---|
| S4_gan_light/best_ema | 0.0406 | 0.1721 | 0.0300 | **0.1364** | 0.1511 | 23.24 | 13.54 | 27.54 |
| S4_gan_light/best | 0.0407 | 0.1740 | 0.0369 | **0.1407** | 0.1520 | 23.24 | 13.44 | 26.48 |
| S1B_bce_long/best | 0.0436 | 0.1518 | 0.0257 | **0.1424** | 0.1603 | 22.70 | 14.24 | 28.20 |
| S2_dual_w10/best | 0.0408 | 0.1520 | 0.0316 | **0.1426** | 0.1498 | 23.61 | 14.39 | 26.33 |
| S3_rw1/best | 0.0416 | 0.1716 | 0.0358 | **0.1428** | 0.1553 | 22.90 | 13.51 | 25.48 |
| S4_baseline/best | 0.0419 | 0.1721 | 0.0323 | **0.1433** | 0.1555 | 22.88 | 13.53 | 25.70 |
| S4_gan_mid/best | 0.0413 | 0.1754 | 0.0386 | **0.1437** | 0.1535 | 23.04 | 13.36 | 25.89 |
| S4_baseline/best_ema | 0.0421 | 0.1733 | 0.0343 | **0.1439** | 0.1562 | 22.84 | 13.46 | 25.10 |
| S4_gan_mid/best_ema | 0.0412 | 0.1776 | 0.0397 | **0.1447** | 0.1536 | 23.06 | 13.28 | 25.36 |
| S1B_bce_gan/best | 0.0430 | 0.1520 | 0.0284 | **0.1453** | 0.1590 | 22.61 | 14.27 | 26.87 |
| S1_bce/best | 0.0436 | 0.1609 | 0.0393 | **0.1538** | 0.1618 | 22.72 | 14.02 | 25.22 |
| S4_gan_heavy/best_ema | 0.0548 | 0.1864 | 0.0500 | **0.1543** | 0.1601 | 21.81 | 13.07 | 23.01 |
| S4_gan_heavy/best | 0.0432 | 0.1907 | 0.0470 | **0.1548** | 0.1610 | 22.66 | 12.91 | 23.03 |

## Best run by metric

- **render loss**: `S4_gan_light/best_ema` (0.1364)
- **rendered-LPIPS (perceptual)**: `S2_dual_w10/best` (0.1498)
- **normal L1**: `S4_gan_light/best_ema` (0.0406)
- **roughness L1**: `S1B_bce_long/best` (0.1518)
- **metallic L1**: `S1B_bce_long/best` (0.0257)
- **normal PSNR**: `S2_dual_w10/best` (23.6107)
- **roughness PSNR**: `S2_dual_w10/best` (14.3908)
- **metallic PSNR**: `S1B_bce_long/best` (28.2027)

## Test-set category distribution

| category | n |
|---|---|
| Ceramic | 37 |
| Concrete | 13 |
| Fabric | 24 |
| Ground | 11 |
| Leather | 17 |
| Marble | 8 |
| Metal | 28 |
| Misc | 8 |
| Plaster | 14 |
| Plastic | 14 |
| Stone | 44 |
| Terracotta | 16 |
| Wood | 51 |
| **TOTAL** | **285** |

## Per-category normal L1 — three best runs

| category | n | S2_dual_w10 / best | S4_gan_light / best_ema | S4_gan_mid / best_ema |
|---|---|---|---|---|
| Ceramic | 37 | 0.0465 | 0.0454 | 0.0457 |
| Concrete | 13 | 0.0246 | 0.0250 | 0.0218 |
| Fabric | 24 | 0.0293 | 0.0313 | 0.0310 |
| Ground | 11 | 0.0604 | 0.0635 | 0.0643 |
| Leather | 17 | 0.0294 | 0.0263 | 0.0255 |
| Marble | 8 | 0.0121 | 0.0107 | 0.0110 |
| Metal | 28 | 0.0235 | 0.0208 | 0.0242 |
| Misc | 8 | 0.0345 | 0.0342 | 0.0343 |
| Plaster | 14 | 0.0463 | 0.0472 | 0.0474 |
| Plastic | 14 | 0.0186 | 0.0161 | 0.0163 |
| Stone | 44 | 0.0616 | 0.0631 | 0.0637 |
| Terracotta | 16 | 0.0468 | 0.0437 | 0.0439 |
| Wood | 51 | 0.0452 | 0.0462 | 0.0474 |

## Notes

- The render loss is computed with the same `GGXRenderingLoss` used in training but with a fixed seed so per-checkpoint comparisons are deterministic.
- The rendered-LPIPS metric renders predicted and GT maps under one fixed light/view (lit from front-above, view straight-down) using `_render_fixed` in `scripts/eval_full.py`, then compares the two RGB images via SqueezeNet-LPIPS.
- The test set contains only 8 samples in each of the categories `Marble` and `Misc`; per-category numbers for those are noisier than the others.
- Reproduce with `python scripts/eval_full.py --cache-dir data/processed2/train_256 --runs <run-dirs> --ckpts best.pt best_ema.pt`. Per-run JSON output is written to each `run/eval_report.json`.
