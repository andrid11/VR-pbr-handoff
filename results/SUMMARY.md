# Run Summary


## Stage 1 - loss screening

| run | epochs | best val_loss | best render_loss | adv | r1 | render_w | sep_norm | hours |
|---|---|---|---|---|---|---|---|---|
| S1_bce | 50 | 0.7846 (e49) | - | 0 | 0 | 0 | N | 1.5 |

## Stage 1B - loss combinations + first GAN

| run | epochs | best val_loss | best render_loss | adv | r1 | render_w | sep_norm | hours |
|---|---|---|---|---|---|---|---|---|
| S1B_bce_long | 100 | 0.7417 (e92) | - | 0 | 0 | 0 | N | 3.2 |
| S1B_bce_gan | 100 | 0.7557 (e100) | - | 0.01 | 0 | 0 | N | 4.0 |

## Stage 2 - normal-prediction architecture

| run | epochs | best val_loss | best render_loss | adv | r1 | render_w | sep_norm | hours |
|---|---|---|---|---|---|---|---|---|
| S2_dual_w10 | 100 | 1.0808 (e77) | - | 0 | 0 | 0 | Y | 4.0 |

## Stage 3 - render-loss as primary signal

| run | epochs | best val_loss | best render_loss | adv | r1 | render_w | sep_norm | hours |
|---|---|---|---|---|---|---|---|---|
| S3_rw1 | 100 | 0.1924 (e74) | - | 0 | 0 | 1 | Y | 5.3 |

## Stage 4 - extended training + GAN sweep

| run | epochs | best val_loss | best render_loss | adv | r1 | render_w | sep_norm | hours |
|---|---|---|---|---|---|---|---|---|
| S4_gan_light | 150 | 0.1828 (e106) | - | 0.005 | 10 | 1 | Y | 30.2 |
| S4_baseline | 150 | 0.1923 (e119) | - | 0 | 10 | 1 | Y | 7.6 |
| S4_gan_mid | 150 | 0.2030 (e117) | - | 0.01 | 10 | 1 | Y | 30.1 |
| S4_gan_heavy | 21 | 0.2171 (e20) | - | 0.05 | 10 | 1 | Y | 3.6 |
