"""
Train PBR prediction model: basecolor -> normal + roughness + metallic.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 8 --lr 1e-3
    python scripts/train.py --cache-dir data/processed/train_256 --resume outputs/checkpoints/latest.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_msssim import ssim as compute_ssim
from src.model import PBRUNet, category_to_index
from src.dataset import CachedMatSynthDataset
from src.transforms import MAP_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Train PBR prediction model")
    # Data
    p.add_argument("--cache-dir", default="data/processed/train_256")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit dataset to first N samples (useful for overfit tests)")
    p.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation")
    p.add_argument("--split-file", type=str, default=None,
                   help="Global split JSON with train/val/test indices. "
                        "If set, overrides per-run split creation and uses "
                        "these exact indices. Test indices are ignored by training.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--augment", action="store_true",
                   help="Enable PBR-aware augmentation (flips + 90° rotations)")
    # Model
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--use-category", action="store_true",
                   help="Condition model on material category")
    p.add_argument("--normal-xy", action="store_true",
                   help="Predict only XY normal channels, derive Z analytically")
    p.add_argument("--separate-normal-decoder", action="store_true",
                   help="Use separate U-Net decoder for normals (dedicated capacity)")
    p.add_argument("--predict-height", action="store_true",
                   help="Predict height map and derive normals (physically structured)")
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--metallic-bce", type=float, default=1.0,
                   help="BCE weight for metallic loss (0 = L1 only, 1.0 = BCE + L1)")
    p.add_argument("--normal-cosine", type=float, default=1.0,
                   help="Cosine similarity loss weight for normals (0 = L1 only)")
    p.add_argument("--normal-gradient", type=float, default=0.0,
                   help="Sobel gradient loss weight for normals (0 = disabled)")
    p.add_argument("--fft-weight", type=float, default=0.0,
                   help="FFT spectral loss weight for normals (0 = disabled). Penalizes missing detail.")
    p.add_argument("--lpips-weight", type=float, default=0.0,
                   help="LPIPS perceptual loss weight for normals+roughness (0 = disabled)")
    p.add_argument("--render-loss", type=float, default=0.0,
                   help="GGX rendering loss weight (0 = disabled). Adds physics-based supervision.")
    p.add_argument("--adversarial", type=float, default=0.0,
                   help="Adversarial loss weight (0 = disabled, enables PatchGAN discriminator)")
    p.add_argument("--disc-lr", type=float, default=2e-4,
                   help="Discriminator learning rate (only used with --adversarial)")
    p.add_argument("--adv-warmup-epochs", type=int, default=0,
                   help="Skip discriminator updates for the first N epochs "
                        "(generator trains on direct+render losses only).")
    p.add_argument("--r1-gamma", type=float, default=0.0,
                   help="R1 gradient penalty weight on the discriminator "
                        "(0 = disabled, 10.0 is a standard value).")
    p.add_argument("--roughness-ssim", type=float, default=1.0,
                   help="SSIM loss weight for roughness (0 = L1 only)")
    p.add_argument("--freeze-bn", action="store_true",
                   help="Freeze BatchNorm layers (use running stats from pretrained)")
    p.add_argument("--only-map", type=str, default=None, choices=["normal", "roughness", "metallic"],
                   help="Train on only one target map (for debugging)")
    p.add_argument("--normal-loss", type=str, default="l1", choices=["l1", "mse"],
                   help="Base loss for normals: l1 or mse")
    p.add_argument("--normal-weight", type=float, default=1.0,
                   help="Multiplier for normal loss to fix multi-task imbalance (try 10-15)")
    p.add_argument("--roughness-weight", type=float, default=1.0,
                   help="Multiplier for roughness loss (use <1 to deweight as stabilizer)")
    p.add_argument("--metallic-weight", type=float, default=1.0,
                   help="Multiplier for metallic loss (use <1 to deweight as stabilizer)")
    # Output
    p.add_argument("--out-dir", default="outputs/checkpoints")
    p.add_argument("--log-every", type=int, default=25, help="Print every N batches")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--comparison-set", type=str, default=None,
                   help="Path to comparison_set.json for fixed cross-run previews")
    p.add_argument("--preview-every", type=int, default=50, help="Save preview images every N epochs")
    p.add_argument("--preview-dir", default=None,
                   help="Directory for preview images (default: <out-dir>/previews)")
    p.add_argument("--n-previews", type=int, default=3, help="Number of validation samples to preview")
    return p.parse_args()


class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps a dataset with a transform applied at __getitem__ time."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample)


TARGET_MAPS = ("normal", "roughness", "metallic")

SPLIT_SEED = 42


def load_global_split(dataset, split_file: str):
    """Load a global split file (train/val/test) and return (train_ds, val_ds).

    The test indices are intentionally ignored here — they're consumed by
    eval_full.py after training.
    """
    with open(split_file) as f:
        data = json.load(f)
    if data["dataset_size"] != len(dataset):
        raise ValueError(
            f"Split file expects {data['dataset_size']} samples "
            f"but dataset has {len(dataset)}. Regenerate the split."
        )
    train_ds = torch.utils.data.Subset(dataset, data["train_indices"])
    val_ds = torch.utils.data.Subset(dataset, data["val_indices"])
    print(f"Loaded global split from {split_file} "
          f"({len(data['train_indices'])} train, {len(data['val_indices'])} val, "
          f"{len(data['test_indices'])} test [held out])")
    return train_ds, val_ds


def load_or_create_split(dataset, val_fraction: float, split_path: str):
    """Load saved split indices, or create and save a new split.

    If split_path exists, reuses the saved indices (verifying dataset size matches).
    Otherwise creates a new random split and persists the indices.
    """
    n_total = len(dataset)
    n_val = int(n_total * val_fraction) if val_fraction > 0 else 0
    n_val = max(n_val, 1) if val_fraction > 0 else 0
    n_train = n_total - n_val

    if os.path.isfile(split_path):
        with open(split_path, "r") as f:
            saved = json.load(f)
        if saved["dataset_size"] != n_total:
            raise ValueError(
                f"Split file expects {saved['dataset_size']} samples "
                f"but dataset has {n_total}. Delete {split_path} to recreate."
            )
        train_indices = saved["train_indices"]
        val_indices = saved["val_indices"]
        print(f"Loaded split from {split_path} "
              f"({len(train_indices)} train, {len(val_indices)} val)")
    else:
        all_indices = list(range(n_total))
        gen = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(n_total, generator=gen).tolist()
        val_indices = sorted(perm[:n_val])
        train_indices = sorted(perm[n_val:])

        split_info = {
            "seed": SPLIT_SEED,
            "dataset_size": n_total,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
        }
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"Created split: {len(train_indices)} train, {len(val_indices)} val "
              f"-> {split_path}")

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    return train_ds, val_ds


def load_history(history_path: str) -> list[dict]:
    """Load existing history from disk, returning [] if missing or corrupt."""
    if not os.path.isfile(history_path):
        return []
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


@torch.no_grad()
def save_previews(
    model, dataset, indices: list[int], device: torch.device,
    epoch: int, preview_dir: str,
):
    """Save side-by-side comparison panels for a fixed set of validation samples."""
    model.eval()
    os.makedirs(preview_dir, exist_ok=True)

    # columns: basecolor | pred_normal | gt_normal | pred_rough | gt_rough | pred_metal | gt_metal
    col_labels = [
        "basecolor", "pred normal", "GT normal",
        "pred roughness", "GT roughness", "pred metallic", "GT metallic",
    ]

    for si, idx in enumerate(indices):
        sample = dataset[idx]
        basecolor = sample["basecolor"].unsqueeze(0).to(device)
        cat_idx = torch.tensor([category_to_index(sample.get("category", "unknown"))],
                               dtype=torch.long, device=device)
        preds = model(basecolor, category=cat_idx)

        # Collect images as (H, W, 3) or (H, W) numpy in [0, 1]
        panels = [sample["basecolor"].permute(1, 2, 0).cpu().numpy()]
        for map_name in TARGET_MAPS:
            pred_t = preds[map_name][0].cpu().clamp(0, 1)
            gt_t = sample[map_name]
            # GT from cache is 3-ch for roughness/metallic — take ch 0 to match pred
            if map_name in ("roughness", "metallic") and gt_t.shape[0] == 3:
                gt_t = gt_t[:1]
            # 1-channel maps: squeeze to (H, W) for grayscale display
            if pred_t.shape[0] == 1:
                pred_img = pred_t[0].numpy()
            else:
                pred_img = pred_t.permute(1, 2, 0).numpy()
            if gt_t.shape[0] == 1:
                gt_img = gt_t[0].numpy()
            else:
                gt_img = gt_t.permute(1, 2, 0).numpy()
            panels.append(pred_img)
            panels.append(gt_img)

        fig, axes = plt.subplots(1, 7, figsize=(21, 3))
        for ax, img, label in zip(axes, panels, col_labels):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch}  |  {sample.get('name', '')}", fontsize=10)
        fig.tight_layout()

        path = os.path.join(preview_dir, f"epoch_{epoch:03d}_sample_{si:02d}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss for normal maps.

    Converts [0,1] -> [-1,1], normalizes to unit vectors, computes
    1 - cos_sim. This penalizes angular deviation, forcing the model
    to predict correct surface directions instead of averaging to flat.
    """
    # (B, 3, H, W) [0,1] -> [-1,1]
    p = pred * 2.0 - 1.0
    t = target * 2.0 - 1.0
    # Normalize to unit vectors along channel dim
    p = nn.functional.normalize(p, dim=1, eps=1e-6)
    t = nn.functional.normalize(t, dim=1, eps=1e-6)
    # cos_sim per pixel, mean over all
    cos_sim = (p * t).sum(dim=1)  # (B, H, W)
    return (1.0 - cos_sim).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM loss. Penalizes structural/contrast degradation (blur)."""
    return 1.0 - compute_ssim(pred, target, data_range=1.0, size_average=True)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel gradient loss: penalizes differences in spatial gradients.

    Computes Sobel edges of pred and target, returns L1 between them.
    Flat predictions (zero gradients) get penalized when GT has detail.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    c = pred.shape[1]
    sobel_x = sobel_x.repeat(c, 1, 1, 1)
    sobel_y = sobel_y.repeat(c, 1, 1, 1)

    pred_gx = torch.nn.functional.conv2d(pred, sobel_x, padding=1, groups=c)
    pred_gy = torch.nn.functional.conv2d(pred, sobel_y, padding=1, groups=c)
    target_gx = torch.nn.functional.conv2d(target, sobel_x, padding=1, groups=c)
    target_gy = torch.nn.functional.conv2d(target, sobel_y, padding=1, groups=c)

    return (torch.nn.functional.l1_loss(pred_gx, target_gx)
            + torch.nn.functional.l1_loss(pred_gy, target_gy))


def compute_loss(
    preds: dict, batch: dict,
    metallic_bce_weight: float = 1.0,
    normal_cosine_weight: float = 0.0,
    roughness_ssim_weight: float = 0.0,
    normal_gradient_weight: float = 0.0,
    fft_weight: float = 0.0,
    lpips_fn=None,
    lpips_weight: float = 0.0,
    render_loss_fn=None,
    render_loss_weight: float = 0.0,
    basecolor: torch.Tensor | None = None,
    only_map: str | None = None,
    normal_loss_type: str = "l1",
    normal_weight: float = 1.0,
    roughness_weight: float = 1.0,
    metallic_weight: float = 1.0,
) -> tuple[torch.Tensor, dict, dict]:
    """Combined loss: L1/MSE + optional cosine/gradient/LPIPS/render (normal) + SSIM (roughness) + BCE (metallic).

    Returns:
        (total_loss, weighted_dict, raw_dict) where weighted_dict has weight-scaled
        per-map losses and raw_dict has unweighted per-map losses.
    """
    losses = {}
    raw_losses = {}
    maps_to_train = [only_map] if only_map else list(TARGET_MAPS)

    # Normal: L1 or MSE + optional cosine similarity
    if "normal" in maps_to_train:
        if normal_loss_type == "mse":
            base_normal = nn.functional.mse_loss(preds["normal"], batch["normal"])
        else:
            base_normal = nn.functional.l1_loss(preds["normal"], batch["normal"])
        losses["normal"] = base_normal
        if normal_cosine_weight > 0:
            cos = cosine_loss(preds["normal"], batch["normal"])
            losses["normal"] = losses["normal"] + normal_cosine_weight * cos
        if normal_gradient_weight > 0:
            grad = gradient_loss(preds["normal"], batch["normal"])
            losses["normal"] = losses["normal"] + normal_gradient_weight * grad
        if fft_weight > 0:
            from src.losses import fft_loss
            fft = fft_loss(preds["normal"], batch["normal"])
            losses["normal"] = losses["normal"] + fft_weight * fft
        if lpips_weight > 0 and lpips_fn is not None:
            lp = lpips_fn(preds["normal"] * 2 - 1, batch["normal"] * 2 - 1).mean()
            losses["normal"] = losses["normal"] + lpips_weight * lp
        raw_losses["normal"] = losses["normal"].item()
        losses["normal"] = losses["normal"] * normal_weight

    # Roughness: L1 + SSIM
    if "roughness" in maps_to_train:
        l1_rough = nn.functional.l1_loss(preds["roughness"], batch["roughness"])
        losses["roughness"] = l1_rough
        if roughness_ssim_weight > 0:
            ssim_r = ssim_loss(preds["roughness"], batch["roughness"])
            losses["roughness"] = losses["roughness"] + roughness_ssim_weight * ssim_r
        if lpips_weight > 0 and lpips_fn is not None:
            rough_pred_3ch = preds["roughness"].expand(-1, 3, -1, -1)
            rough_gt_3ch = batch["roughness"].expand(-1, 3, -1, -1)
            lp_r = lpips_fn(rough_pred_3ch * 2 - 1, rough_gt_3ch * 2 - 1).mean()
            losses["roughness"] = losses["roughness"] + lpips_weight * lp_r
        raw_losses["roughness"] = losses["roughness"].item()
        losses["roughness"] = losses["roughness"] * roughness_weight

    # Metallic: BCE + L1
    if "metallic" in maps_to_train:
        l1_metal = nn.functional.l1_loss(preds["metallic"], batch["metallic"])
        if metallic_bce_weight > 0:
            bce = nn.functional.binary_cross_entropy(
                preds["metallic"], batch["metallic"], reduction="mean"
            )
            losses["metallic"] = metallic_bce_weight * bce + l1_metal
        else:
            losses["metallic"] = l1_metal
        raw_losses["metallic"] = losses["metallic"].item()
        losses["metallic"] = losses["metallic"] * metallic_weight

    total = sum(losses.values())

    # Rendering loss: physics-based supervision across all maps jointly
    render_loss_val = 0.0
    if render_loss_weight > 0 and render_loss_fn is not None and basecolor is not None:
        r_loss = render_loss_fn(
            preds["normal"], preds["roughness"], preds["metallic"],
            batch["normal"], batch["roughness"], batch["metallic"],
            basecolor,
        )
        total = total + render_loss_weight * r_loss
        render_loss_val = r_loss.item()

    # Build both weighted and raw result dicts
    weighted_result = {}
    raw_result = {}
    for k in TARGET_MAPS:
        weighted_result[k] = losses[k].item() if k in losses else 0.0
        raw_result[k] = raw_losses.get(k, 0.0)

    weighted_result["render"] = render_loss_val * render_loss_weight
    raw_result["render"] = render_loss_val

    return total, weighted_result, raw_result


def r1_gradient_penalty(discriminator, real_input, gamma: float = 10.0) -> torch.Tensor:
    """R1 regularizer: gamma/2 * ||grad_D(real)||^2 averaged over the batch.

    Reference: Mescheder et al. 2018, "Which Training Methods for GANs do
    actually Converge?". Stabilizes GAN training at a low cost.
    """
    real_input = real_input.detach().clone().requires_grad_(True)
    real_score = discriminator(real_input)
    grad = torch.autograd.grad(
        outputs=real_score.sum(), inputs=real_input,
        create_graph=True, retain_graph=True,
    )[0]
    # Squared L2 norm per sample, mean over batch
    per_sample = grad.pow(2).view(grad.shape[0], -1).sum(dim=1)
    return 0.5 * gamma * per_sample.mean()


def adversarial_step(
    discriminator, disc_optimizer, basecolor, targets, preds,
    adv_weight: float,
    r1_gamma: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """Run one discriminator + generator adversarial step.

    Returns: (generator_adv_loss, disc_loss_value, gen_score_value)
    """
    def concat_maps(base, maps_dict):
        return torch.cat([base, maps_dict["normal"],
                          maps_dict["roughness"], maps_dict["metallic"]], dim=1)

    real_input = concat_maps(basecolor, targets)
    fake_input = concat_maps(basecolor, {k: v.detach() for k, v in preds.items()})

    # Train discriminator
    disc_optimizer.zero_grad()
    real_score = discriminator(real_input)
    fake_score = discriminator(fake_input)

    # LSGAN loss (MSE-based, more stable than BCE)
    disc_loss = 0.5 * (
        nn.functional.mse_loss(real_score, torch.ones_like(real_score))
        + nn.functional.mse_loss(fake_score, torch.zeros_like(fake_score))
    )
    if r1_gamma > 0:
        disc_loss = disc_loss + r1_gradient_penalty(
            discriminator, real_input, gamma=r1_gamma,
        )
    disc_loss.backward()
    disc_optimizer.step()

    # Generator adversarial loss (fool discriminator)
    fake_for_gen = concat_maps(basecolor, preds)
    gen_score = discriminator(fake_for_gen)
    gen_adv_loss = adv_weight * nn.functional.mse_loss(
        gen_score, torch.ones_like(gen_score)
    )

    return gen_adv_loss, disc_loss.item(), gen_score.mean().item()


def freeze_batchnorm(model):
    """Set all BatchNorm layers to eval mode (use running stats, don't update)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()


def train_one_epoch(model, loader, optimizer, device, log_every,
                    metallic_bce=1.0, normal_cosine=0.0, roughness_ssim=0.0,
                    normal_gradient=0.0, fft_weight=0.0,
                    lpips_fn=None, lpips_weight=0.0,
                    render_loss_fn=None, render_loss_weight=0.0,
                    discriminator=None, disc_optimizer=None, adversarial=0.0,
                    ema=None, freeze_bn=False, only_map=None, normal_loss_type="l1",
                    normal_weight=1.0, roughness_weight=1.0, metallic_weight=1.0,
                    adv_warmup_epochs=0, current_epoch=0,
                    r1_gamma=0.0):
    model.train()
    if freeze_bn:
        freeze_batchnorm(model)
    total_loss = 0.0
    map_losses = {k: 0.0 for k in TARGET_MAPS}
    map_losses_raw = {k: 0.0 for k in TARGET_MAPS}
    n_batches = 0
    disc_sums = {"disc_loss": 0.0, "gen_score": 0.0, "gen_adv_loss": 0.0}
    disc_steps = 0

    for i, batch in enumerate(loader):
        basecolor = batch["basecolor"].to(device)
        category = batch["category_idx"].to(device)
        targets = {k: batch[k].to(device) for k in TARGET_MAPS}

        preds = model(basecolor, category=category)
        loss, per_map, per_map_raw = compute_loss(preds, targets,
                                     metallic_bce_weight=metallic_bce,
                                     normal_cosine_weight=normal_cosine,
                                     roughness_ssim_weight=roughness_ssim,
                                     normal_gradient_weight=normal_gradient,
                                     fft_weight=fft_weight,
                                     lpips_fn=lpips_fn,
                                     lpips_weight=lpips_weight,
                                     render_loss_fn=render_loss_fn,
                                     render_loss_weight=render_loss_weight,
                                     basecolor=basecolor,
                                     only_map=only_map,
                                     normal_loss_type=normal_loss_type,
                                     normal_weight=normal_weight,
                                     roughness_weight=roughness_weight,
                                     metallic_weight=metallic_weight)

        optimizer.zero_grad()

        if adversarial > 0 and discriminator is not None and current_epoch >= adv_warmup_epochs:
            adv_loss, d_loss, g_score = adversarial_step(
                discriminator, disc_optimizer, basecolor, targets, preds, adversarial,
                r1_gamma=r1_gamma,
            )
            total = loss + adv_loss
            total.backward()
            disc_sums["disc_loss"] += d_loss
            disc_sums["gen_score"] += g_score
            disc_sums["gen_adv_loss"] += adv_loss.item()
            disc_steps += 1
        else:
            loss.backward()

        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        for k in TARGET_MAPS:
            map_losses[k] += per_map[k]
            map_losses_raw[k] += per_map_raw[k]
        n_batches += 1

        if (i + 1) % log_every == 0:
            avg = total_loss / n_batches
            print(f"    batch {i + 1}: loss={avg:.4f}  "
                  + "  ".join(f"{k}={map_losses[k] / n_batches:.4f}" for k in TARGET_MAPS))

    avg_loss = total_loss / max(n_batches, 1)
    avg_maps = {k: map_losses[k] / max(n_batches, 1) for k in TARGET_MAPS}
    avg_maps_raw = {k: map_losses_raw[k] / max(n_batches, 1) for k in TARGET_MAPS}
    disc_avg = {k: (v / disc_steps) if disc_steps > 0 else 0.0
                for k, v in disc_sums.items()}
    return avg_loss, avg_maps, avg_maps_raw, disc_avg


@torch.no_grad()
def validate(model, loader, device, metallic_bce=1.0, normal_cosine=0.0, roughness_ssim=0.0,
             normal_gradient=0.0, fft_weight=0.0, lpips_fn=None, lpips_weight=0.0,
             render_loss_fn=None, render_loss_weight=0.0,
             only_map=None, normal_loss_type="l1",
             normal_weight=1.0, roughness_weight=1.0, metallic_weight=1.0):
    model.eval()
    total_loss = 0.0
    map_losses = {k: 0.0 for k in TARGET_MAPS}
    map_losses_raw = {k: 0.0 for k in TARGET_MAPS}
    n_batches = 0

    for batch in loader:
        basecolor = batch["basecolor"].to(device)
        category = batch["category_idx"].to(device)
        targets = {k: batch[k].to(device) for k in TARGET_MAPS}

        preds = model(basecolor, category=category)
        loss, per_map, per_map_raw = compute_loss(preds, targets,
                                     metallic_bce_weight=metallic_bce,
                                     normal_cosine_weight=normal_cosine,
                                     roughness_ssim_weight=roughness_ssim,
                                     normal_gradient_weight=normal_gradient,
                                     fft_weight=fft_weight,
                                     lpips_fn=lpips_fn,
                                     lpips_weight=lpips_weight,
                                     render_loss_fn=render_loss_fn,
                                     render_loss_weight=render_loss_weight,
                                     basecolor=basecolor,
                                     only_map=only_map,
                                     normal_loss_type=normal_loss_type,
                                     normal_weight=normal_weight,
                                     roughness_weight=roughness_weight,
                                     metallic_weight=metallic_weight)

        total_loss += loss.item()
        for k in TARGET_MAPS:
            map_losses[k] += per_map[k]
            map_losses_raw[k] += per_map_raw[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_maps = {k: map_losses[k] / max(n_batches, 1) for k in TARGET_MAPS}
    avg_maps_raw = {k: map_losses_raw[k] / max(n_batches, 1) for k in TARGET_MAPS}
    return avg_loss, avg_maps, avg_maps_raw


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Default preview dir lives inside the run directory
    if args.preview_dir is None:
        args.preview_dir = os.path.join(args.out_dir, "previews")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────
    full_dataset = CachedMatSynthDataset(args.cache_dir)

    # Optionally limit dataset size (e.g. for overfit tests)
    if args.max_samples is not None:
        n = min(args.max_samples, len(full_dataset))
        full_dataset = torch.utils.data.Subset(full_dataset, list(range(n)))

    if args.split_file:
        train_ds, val_ds = load_global_split(full_dataset, args.split_file)
    else:
        split_path = os.path.join(args.out_dir, "split_indices.json")
        train_ds, val_ds = load_or_create_split(full_dataset, args.val_split, split_path)

    # Apply augmentation to train set only
    if args.augment:
        from src.transforms import PBRAugmentation
        augment = PBRAugmentation()
        train_ds = AugmentedDataset(train_ds, augment)
        print("Augmentation: ON (hflip + vflip + 90° rotations)")

    def collate_fn(batch):
        result = {}
        for key in MAP_NAMES:
            stacked = torch.stack([b[key] for b in batch])
            # Roughness and metallic: take channel 0 -> (B, 1, H, W)
            if key in ("roughness", "metallic"):
                stacked = stacked[:, :1, :, :]
            result[key] = stacked
        result["name"] = [b["name"] for b in batch]
        result["category"] = [b["category"] for b in batch]
        result["category_idx"] = torch.tensor(
            [category_to_index(b["category"]) for b in batch], dtype=torch.long
        )
        return result

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = None
    if len(val_ds) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )

    print(f"Data: {len(train_ds)} train, {len(val_ds)} val (from {args.cache_dir})")

    # Fixed preview indices (from comparison set, or default to first N val samples)
    preview_ds = val_ds if len(val_ds) > 0 else train_ds
    if args.comparison_set and os.path.isfile(args.comparison_set):
        with open(args.comparison_set) as f:
            comp = json.load(f)
        # Map comparison set dataset-level indices to val_ds subset indices
        val_orig_set = set(val_ds.indices) if hasattr(val_ds, 'indices') else set()
        preview_indices = []
        for ci in comp["indices"]:
            if ci in val_orig_set:
                for vi, orig_idx in enumerate(val_ds.indices):
                    if orig_idx == ci:
                        preview_indices.append(vi)
                        break
        print(f"Using fixed comparison set: {len(preview_indices)} samples")
    else:
        n_previews = min(args.n_previews, len(preview_ds))
        preview_indices = list(range(n_previews))

    # ── Model ────────────────────────────────────────────────
    model = PBRUNet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        use_category=args.use_category,
        normal_xy_only=args.normal_xy,
        separate_normal_decoder=args.separate_normal_decoder,
        predict_height=args.predict_height,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: PBRUNet ({args.encoder}), {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    from src.ema import EMA
    ema = EMA(model, decay=0.999)

    # ── Rendering loss (no learned params, pure math) ─────
    render_loss_fn = None
    if args.render_loss > 0:
        from src.rendering_loss import GGXRenderingLoss
        render_loss_fn = GGXRenderingLoss(n_diffuse=3, n_specular=6).to(device)
        print(f"GGX rendering loss enabled (weight={args.render_loss})")

    # ── LPIPS (lazy init, only if weight > 0) ───────────────
    lpips_fn = None
    if args.lpips_weight > 0:
        import lpips
        lpips_fn = lpips.LPIPS(net="squeeze", verbose=False).to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print(f"LPIPS loss enabled (weight={args.lpips_weight})")

    # ── Adversarial setup ───────────────────────────────────
    discriminator = None
    disc_optimizer = None
    if args.adversarial > 0:
        from src.discriminator import PatchGANDiscriminator
        discriminator = PatchGANDiscriminator(in_channels=8).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.999),
        )
        disc_n_params = sum(p.numel() for p in discriminator.parameters())
        print(f"PatchGAN discriminator: {disc_n_params / 1e6:.1f}M params (lr={args.disc_lr})")

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_render = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_render = ckpt.get("best_val_render", float("inf"))
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    # ── Training loop ────────────────────────────────────────
    history_path = os.path.join(args.out_dir, "history.json")
    history = load_history(history_path)
    # Remove any entries from epochs we're about to retrain (handles partial resumes)
    history = [h for h in history if h["epoch"] < start_epoch + 1]
    print(f"\nTraining for {args.epochs} epochs (lr={args.lr})")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        train_loss, train_maps, train_maps_raw, disc_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.log_every,
            metallic_bce=args.metallic_bce,
            normal_cosine=args.normal_cosine,
            roughness_ssim=args.roughness_ssim,
            normal_gradient=args.normal_gradient,
            fft_weight=args.fft_weight,
            lpips_fn=lpips_fn,
            lpips_weight=args.lpips_weight,
            render_loss_fn=render_loss_fn,
            render_loss_weight=args.render_loss,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            adversarial=args.adversarial,
            ema=ema,
            freeze_bn=args.freeze_bn,
            only_map=args.only_map,
            normal_loss_type=args.normal_loss,
            normal_weight=args.normal_weight,
            roughness_weight=args.roughness_weight,
            metallic_weight=args.metallic_weight,
            adv_warmup_epochs=args.adv_warmup_epochs,
            current_epoch=epoch,
            r1_gamma=args.r1_gamma,
        )
        if val_loader is not None:
            val_loss, val_maps, val_maps_raw = validate(model, val_loader, device,
                                          metallic_bce=args.metallic_bce,
                                          normal_cosine=args.normal_cosine,
                                          roughness_ssim=args.roughness_ssim,
                                          normal_gradient=args.normal_gradient,
                                          fft_weight=args.fft_weight,
                                          lpips_fn=lpips_fn,
                                          lpips_weight=args.lpips_weight,
                                          render_loss_fn=render_loss_fn,
                                          render_loss_weight=args.render_loss,
                                          only_map=args.only_map,
                                          normal_loss_type=args.normal_loss,
                                          normal_weight=args.normal_weight,
                                          roughness_weight=args.roughness_weight,
                                          metallic_weight=args.metallic_weight)
        else:
            val_loss, val_maps, val_maps_raw = train_loss, train_maps, train_maps_raw
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  time={elapsed:.1f}s")

        # Save history
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_maps": train_maps,
            "val_maps": val_maps,
            "train_maps_raw": train_maps_raw,
            "val_maps_raw": val_maps_raw,
            "disc": disc_metrics,
            "lr": lr_now,
            "time": elapsed,
        }
        history.append(entry)

        # Save previews
        if args.preview_every > 0 and (epoch + 1) % args.preview_every == 0:
            save_previews(model, preview_ds, preview_indices, device,
                          epoch + 1, args.preview_dir)

        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            torch.save(ema.state_dict(), os.path.join(args.out_dir, "best_ema.pt"))
            print(f"  -> New best val_loss: {val_loss:.4f}")

        if args.render_loss > 0:
            val_render_raw = val_maps_raw.get("render", 0.0)
            if val_render_raw > 0 and val_render_raw < best_val_render:
                best_val_render = val_render_raw
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best_render.pt"))
                print(f"  -> New best val_render: {val_render_raw:.4f}")

        # Save latest checkpoint (for resume)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_render": best_val_render,
            "ema": ema.state_dict(),
        }, os.path.join(args.out_dir, "latest.pt"))

        # Save history after each epoch (survives interrupts)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Best model:  {os.path.join(args.out_dir, 'best.pt')}")
    print(f"Latest ckpt: {os.path.join(args.out_dir, 'latest.pt')}")
    print(f"History:     {history_path}")


if __name__ == "__main__":
    main()
