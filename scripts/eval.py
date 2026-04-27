"""
Evaluate a PBR model checkpoint on the cached dataset.

Metrics:
  - L1 (per map)
  - SSIM (roughness, normal)
  - Angular error in degrees (normal)
  - LPIPS (roughness, normal — optional, use --lpips)

Usage:
    python scripts/eval.py --checkpoint outputs/run_category/best.pt --use-category
    python scripts/eval.py --checkpoint outputs/run_category/best.pt --use-category --lpips
    python scripts/eval.py --checkpoint outputs/run_category/best.pt --use-category --save-previews
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ssim as compute_ssim
from src.model import PBRUNet, category_to_index
from src.dataset import CachedMatSynthDataset
from src.transforms import MAP_NAMES

TARGET_MAPS = ("normal", "roughness", "metallic")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PBR model checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt (state_dict)")
    p.add_argument("--cache-dir", default="data/processed/train_256")
    p.add_argument("--split-file", default=None,
                   help="Path to split_indices.json (default: same dir as checkpoint)")
    p.add_argument("--split", choices=["train", "val", "all"], default="val",
                   help="Which split to evaluate on")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--use-category", action="store_true",
                   help="Model was trained with category conditioning")
    p.add_argument("--normal-xy", action="store_true",
                   help="Model was trained with XY-only normal prediction")
    p.add_argument("--lpips", action="store_true",
                   help="Compute LPIPS (slower, requires lpips package)")
    # Output
    p.add_argument("--save-previews", action="store_true",
                   help="Save per-sample preview images")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <checkpoint_dir>/eval_<split>)")
    p.add_argument("--n-previews", type=int, default=8,
                   help="Max number of preview images to save")
    return p.parse_args()


def collate_fn(batch):
    result = {}
    for key in MAP_NAMES:
        stacked = torch.stack([b[key] for b in batch])
        if key in ("roughness", "metallic"):
            stacked = stacked[:, :1, :, :]
        result[key] = stacked
    result["name"] = [b["name"] for b in batch]
    result["category"] = [b["category"] for b in batch]
    result["category_idx"] = torch.tensor(
        [category_to_index(b["category"]) for b in batch], dtype=torch.long
    )
    return result


def angular_error_degrees(pred, gt):
    """Per-pixel angular error between predicted and GT normal maps.

    Args:
        pred, gt: (B, 3, H, W) tensors in [0, 1]
    Returns:
        (B,) tensor of mean angular error in degrees per sample
    """
    # Convert [0,1] -> [-1,1]
    pred_n = pred * 2.0 - 1.0
    gt_n = gt * 2.0 - 1.0
    # Normalize to unit vectors
    pred_n = nn.functional.normalize(pred_n, dim=1)
    gt_n = nn.functional.normalize(gt_n, dim=1)
    # Cosine similarity per pixel
    dot = (pred_n * gt_n).sum(dim=1).clamp(-1.0, 1.0)  # (B, H, W)
    angles = torch.acos(dot) * (180.0 / torch.pi)  # degrees
    return angles.mean(dim=(1, 2))  # (B,)


def compute_ssim_batch(pred, gt):
    """SSIM per sample in a batch.

    Args:
        pred, gt: (B, 3, H, W) tensors in [0, 1]
    Returns:
        (B,) tensor of SSIM values
    """
    B = pred.shape[0]
    ssim_vals = []
    for i in range(B):
        val = compute_ssim(
            pred[i:i+1], gt[i:i+1],
            data_range=1.0, size_average=True,
        )
        ssim_vals.append(val.item())
    return torch.tensor(ssim_vals)


@torch.no_grad()
def evaluate(model, loader, device, lpips_fn=None):
    """Run evaluation, return per-sample metrics."""
    model.eval()
    results = []

    for batch in loader:
        basecolor = batch["basecolor"].to(device)
        category = batch["category_idx"].to(device)
        targets = {k: batch[k].to(device) for k in TARGET_MAPS}
        preds = model(basecolor, category=category)

        # Normal angular error
        ang_err = angular_error_degrees(preds["normal"], targets["normal"])

        # SSIM for normal and roughness
        ssim_normal = compute_ssim_batch(preds["normal"], targets["normal"])
        ssim_roughness = compute_ssim_batch(preds["roughness"], targets["roughness"])

        # LPIPS (optional) — requires 3-channel input
        lpips_normal = None
        lpips_roughness = None
        if lpips_fn is not None:
            lpips_normal = lpips_fn(preds["normal"], targets["normal"]).squeeze()
            # Expand 1-channel roughness to 3-channel for LPIPS
            pred_r3 = preds["roughness"].expand(-1, 3, -1, -1)
            gt_r3 = targets["roughness"].expand(-1, 3, -1, -1)
            lpips_roughness = lpips_fn(pred_r3, gt_r3).squeeze()
            if lpips_normal.dim() == 0:
                lpips_normal = lpips_normal.unsqueeze(0)
            if lpips_roughness.dim() == 0:
                lpips_roughness = lpips_roughness.unsqueeze(0)

        B = basecolor.shape[0]
        for i in range(B):
            sample_metrics = {"name": batch["name"][i], "category": batch["category"][i]}
            total_l1 = 0.0
            for name in TARGET_MAPS:
                l1 = nn.functional.l1_loss(preds[name][i], targets[name][i]).item()
                sample_metrics[f"{name}_l1"] = l1
                total_l1 += l1
            sample_metrics["total_l1"] = total_l1
            sample_metrics["normal_angular_error"] = ang_err[i].item()
            sample_metrics["normal_ssim"] = ssim_normal[i].item()
            sample_metrics["roughness_ssim"] = ssim_roughness[i].item()
            if lpips_fn is not None:
                sample_metrics["normal_lpips"] = lpips_normal[i].item()
                sample_metrics["roughness_lpips"] = lpips_roughness[i].item()
            results.append(sample_metrics)

    return results


@torch.no_grad()
def save_preview(model, sample, device, path, sample_name=""):
    """Save a single side-by-side comparison panel."""
    model.eval()
    basecolor = sample["basecolor"].unsqueeze(0).to(device)
    cat_idx = torch.tensor([category_to_index(sample.get("category", "unknown"))],
                           dtype=torch.long, device=device)
    preds = model(basecolor, category=cat_idx)

    col_labels = [
        "basecolor", "pred normal", "GT normal",
        "pred roughness", "GT roughness", "pred metallic", "GT metallic",
    ]

    panels = [sample["basecolor"].permute(1, 2, 0).cpu().numpy()]
    for map_name in TARGET_MAPS:
        pred_t = preds[map_name][0].cpu().clamp(0, 1)
        gt_t = sample[map_name]
        if map_name in ("roughness", "metallic") and gt_t.shape[0] == 3:
            gt_t = gt_t[:1]
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
    fig.suptitle(sample_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def print_results(results, split, lpips_enabled):
    """Print aggregate and per-category results."""
    n = len(results)

    # Aggregate
    agg = {}
    for name in TARGET_MAPS:
        agg[f"{name}_l1"] = float(np.mean([r[f"{name}_l1"] for r in results]))
    agg["total_l1"] = float(np.mean([r["total_l1"] for r in results]))
    agg["normal_angular_error"] = float(np.mean([r["normal_angular_error"] for r in results]))
    agg["normal_ssim"] = float(np.mean([r["normal_ssim"] for r in results]))
    agg["roughness_ssim"] = float(np.mean([r["roughness_ssim"] for r in results]))
    if lpips_enabled:
        agg["normal_lpips"] = float(np.mean([r["normal_lpips"] for r in results]))
        agg["roughness_lpips"] = float(np.mean([r["roughness_lpips"] for r in results]))

    print(f"\n{'=' * 60}")
    print(f"Results on {split} ({n} samples):")
    print(f"{'=' * 60}")
    print(f"  Total L1:              {agg['total_l1']:.4f}")
    for name in TARGET_MAPS:
        print(f"  {name:12s} L1:      {agg[f'{name}_l1']:.4f}")
    print(f"  ---")
    print(f"  Normal angular error:  {agg['normal_angular_error']:.2f} deg")
    print(f"  Normal SSIM:           {agg['normal_ssim']:.4f}")
    print(f"  Roughness SSIM:        {agg['roughness_ssim']:.4f}")
    if lpips_enabled:
        print(f"  Normal LPIPS:          {agg['normal_lpips']:.4f}")
        print(f"  Roughness LPIPS:       {agg['roughness_lpips']:.4f}")
    print(f"{'=' * 60}")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    per_category = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        entry = {"n_samples": len(cat_results)}
        entry["total_l1"] = float(np.mean([r["total_l1"] for r in cat_results]))
        for m in TARGET_MAPS:
            entry[f"{m}_l1"] = float(np.mean([r[f"{m}_l1"] for r in cat_results]))
        entry["normal_angular_error"] = float(np.mean([r["normal_angular_error"] for r in cat_results]))
        entry["normal_ssim"] = float(np.mean([r["normal_ssim"] for r in cat_results]))
        entry["roughness_ssim"] = float(np.mean([r["roughness_ssim"] for r in cat_results]))
        if lpips_enabled:
            entry["normal_lpips"] = float(np.mean([r["normal_lpips"] for r in cat_results]))
            entry["roughness_lpips"] = float(np.mean([r["roughness_lpips"] for r in cat_results]))
        per_category[cat] = entry

    if len(categories) > 1:
        print(f"\nPer-category breakdown:")
        header = f"  {'category':<20s} {'n':>4s}  {'L1':>7s}  {'ang_err':>7s}  {'n_ssim':>6s}  {'r_ssim':>6s}"
        if lpips_enabled:
            header += f"  {'n_lpips':>7s}  {'r_lpips':>7s}"
        print(header)
        print(f"  {'-'*20} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}" +
              (f"  {'-'*7}  {'-'*7}" if lpips_enabled else ""))
        for cat in categories:
            c = per_category[cat]
            line = (f"  {cat:<20s} {c['n_samples']:4d}  {c['total_l1']:7.4f}  "
                    f"{c['normal_angular_error']:7.2f}  {c['normal_ssim']:6.4f}  "
                    f"{c['roughness_ssim']:6.4f}")
            if lpips_enabled:
                line += f"  {c['normal_lpips']:7.4f}  {c['roughness_lpips']:7.4f}"
            print(line)

    return agg, per_category


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────
    model = PBRUNet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        use_category=args.use_category,
        normal_xy_only=args.normal_xy,
    ).to(device)
    state_dict = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── LPIPS (optional) ────────────────────────────────────
    lpips_fn = None
    if args.lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        print("LPIPS enabled (AlexNet)")

    # ── Load data + split ───────────────────────────────────
    full_dataset = CachedMatSynthDataset(args.cache_dir)

    # Find split file
    split_file = args.split_file
    if split_file is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        split_file = os.path.join(ckpt_dir, "split_indices.json")

    if args.split == "all":
        eval_ds = full_dataset
        print(f"Evaluating on all {len(eval_ds)} samples")
    elif os.path.isfile(split_file):
        with open(split_file, "r") as f:
            split_info = json.load(f)
        indices = split_info[f"{args.split}_indices"]
        eval_ds = torch.utils.data.Subset(full_dataset, indices)
        print(f"Evaluating on {args.split} split: {len(eval_ds)} samples "
              f"(from {split_file})")
    else:
        print(f"Warning: split file not found at {split_file}, evaluating on all data")
        eval_ds = full_dataset

    loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # ── Evaluate ────────────────────────────────────────────
    if len(eval_ds) == 0:
        print(f"No samples in {args.split} split. Nothing to evaluate.")
        return

    results = evaluate(model, loader, device, lpips_fn=lpips_fn)

    # Print and collect aggregates
    agg, per_category = print_results(results, args.split, args.lpips)

    # ── Save results ────────────────────────────────────────
    out_dir = args.out_dir
    if out_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    output = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": len(results),
        "aggregate": agg,
        "per_category": per_category,
        "per_sample": results,
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Save previews
    if args.save_previews:
        preview_dir = os.path.join(out_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        ranked = sorted(range(len(results)), key=lambda i: results[i]["total_l1"], reverse=True)
        n_save = min(args.n_previews, len(results))
        for rank, idx in enumerate(ranked[:n_save]):
            sample = eval_ds[idx]
            name = results[idx]["name"]
            l1 = results[idx]["total_l1"]
            path = os.path.join(preview_dir, f"rank{rank:02d}_{name}_l1_{l1:.4f}.png")
            save_preview(model, sample, device, path, sample_name=f"{name} (L1={l1:.4f})")
        print(f"Saved {n_save} previews (worst-first): {preview_dir}")


if __name__ == "__main__":
    main()
