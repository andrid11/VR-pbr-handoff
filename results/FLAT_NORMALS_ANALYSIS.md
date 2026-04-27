# Flat-normal analysis on the held-out test set

Test set size: **285 samples** (frozen split, seed 4242).

Ground-truth predicted-normal std reference: mean=0.0512, median=0.0417, range=[0.0000, 0.1948].

Samples with GT std < 0.005 (essentially flat ground truth — e.g. smooth ceramic) are excluded from the per-sample median ratio: **243 of 285 samples used**.

**Per-sample median ratio** = median over samples of `predicted_std / gt_std` on the same image. 100% would mean predictions match ground-truth spatial variation. **Aggregated ratio** = `mean(predicted_std) / mean(gt_std)`; biased toward high-GT-std samples.

| run | per-sample median ratio | aggregated ratio | L1 vs GT (mean ± SE) |
|---|---|---|---|
| S1B_bce_long | 14% | 15.1% | 0.0435 ± 0.00238 |
| S1B_bce_gan | 16% | 15.0% | 0.0430 ± 0.00244 |
| S1_bce | 16% | 16.0% | 0.0436 ± 0.00238 |
| S4_gan_heavy | 16% | 18.5% | 0.0548 ± 0.00197 |
| S3_rw1 | 20% | 22.2% | 0.0415 ± 0.00250 |
| S4_baseline | 22% | 30.1% | 0.0420 ± 0.00246 |
| S4_gan_light | 28% | 32.2% | 0.0405 ± 0.00240 |
| S4_gan_mid | 31% | 32.1% | 0.0411 ± 0.00243 |
| **S2_dual_w10** | **42%** | **42.0%** | **0.0407 ± 0.00214** |

Lower per-sample median ratio = flatter / more collapsed normal output. Reproduce with `python scripts/analyze_flat_normals.py`.
