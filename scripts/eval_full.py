"""Full test-set evaluation for Stage 4 runs.

Metrics per map (normal / roughness / metallic):
    - L1
    - SSIM (pytorch_msssim, data_range=1.0)
    - PSNR

Joint metrics:
    - Render loss (GGXRenderingLoss, seed-fixed)
    - Rendered-LPIPS (LPIPS on a deterministic rendered pair)

Also reports per-category breakdowns for each metric.

Usage:
    python scripts/eval_full.py \
        --cache-dir data/processed2/train_256 \
        --split outputs/stage4_split.json \
        --runs outputs/S4_baseline outputs/S4_gan_light outputs/S4_gan_mid outputs/S4_gan_heavy \
        --ckpts best.pt best_ema.pt
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from pytorch_msssim import ssim as compute_ssim

from src.model import PBRUNet, category_to_index
from src.dataset import CachedMatSynthDataset
from src.transforms import MAP_NAMES
from src.rendering_loss import GGXRenderingLoss

TARGET_MAPS = ("normal", "roughness", "metallic")


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


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _build_model(run_dir: str, device) -> PBRUNet:
    """Reconstruct the model using args.json if present, otherwise S3 defaults."""
    args_path = os.path.join(run_dir, "args.json")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            a = json.load(f)
        kwargs = dict(
            encoder_name=a.get("encoder", "resnet34"),
            encoder_weights=None,  # weights get overwritten anyway
            use_category=a.get("use_category", False),
            normal_xy_only=a.get("normal_xy", False),
            separate_normal_decoder=a.get("separate_normal_decoder", False),
            predict_height=a.get("predict_height", False),
        )
    else:
        kwargs = dict(
            encoder_name="resnet34", encoder_weights=None,
            use_category=True, separate_normal_decoder=True,
        )
    return PBRUNet(**kwargs).to(device)


@torch.no_grad()
def eval_one(run_dir, ckpt_name, dataset, split_data, device, render_fn, lpips_fn, batch_size):
    model = _build_model(run_dir, device)
    state = torch.load(os.path.join(run_dir, ckpt_name), weights_only=False, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    test_ds = torch.utils.data.Subset(dataset, split_data["test_indices"])
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # Running sums: overall + per-category
    overall = {f"{m}_{name}": 0.0 for m in TARGET_MAPS for name in ("l1", "ssim", "psnr")}
    overall["render"] = 0.0
    overall["rendered_lpips"] = 0.0
    n_batches = 0

    by_cat: dict[str, dict[str, float]] = {}
    cat_counts: dict[str, int] = {}

    # Seed the render for deterministic per-checkpoint comparison
    torch.manual_seed(0)

    for batch in loader:
        basecolor = batch["basecolor"].to(device)
        category = batch["category_idx"].to(device)
        tgt = {k: batch[k].to(device) for k in TARGET_MAPS}
        preds = model(basecolor, category=category)

        for m in TARGET_MAPS:
            l1 = nn.functional.l1_loss(preds[m], tgt[m]).item()
            ss = compute_ssim(preds[m], tgt[m], data_range=1.0, size_average=True).item()
            ps = psnr(preds[m], tgt[m])
            overall[f"{m}_l1"] += l1
            overall[f"{m}_ssim"] += ss
            overall[f"{m}_psnr"] += ps

        overall["render"] += render_fn(
            preds["normal"], preds["roughness"], preds["metallic"],
            tgt["normal"], tgt["roughness"], tgt["metallic"], basecolor,
        ).item()

        # Rendered-LPIPS: render one fixed view/light for each, compare in LPIPS
        r_pred = _render_fixed(render_fn, preds, basecolor, device)
        r_tgt = _render_fixed(render_fn, tgt, basecolor, device)
        overall["rendered_lpips"] += lpips_fn(
            r_pred.clamp(0, 1) * 2 - 1, r_tgt.clamp(0, 1) * 2 - 1,
        ).mean().item()

        # Per-category accounting
        for i, cat in enumerate(batch["category"]):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            by_cat.setdefault(cat, {f"{m}_l1": 0.0 for m in TARGET_MAPS})
            for m in TARGET_MAPS:
                by_cat[cat][f"{m}_l1"] += nn.functional.l1_loss(
                    preds[m][i:i+1], tgt[m][i:i+1],
                ).item()

        n_batches += 1

    overall = {k: v / max(n_batches, 1) for k, v in overall.items()}
    per_cat = {
        cat: {k: v / cat_counts[cat] for k, v in metrics.items()}
        for cat, metrics in by_cat.items()
    }
    return {
        "overall": overall,
        "per_category": per_cat,
        "cat_counts": cat_counts,
        "n_test": len(test_ds),
    }


def _render_fixed(render_fn, maps, basecolor, device):
    """Render under a fixed light+view so LPIPS comparisons are apples-to-apples.

    Uses the same `_render_single` math as GGXRenderingLoss but with deterministic
    directions. Returns (B, 3, H, W).
    """
    bc = basecolor.permute(0, 2, 3, 1)
    n = torch.nn.functional.normalize(maps["normal"].permute(0, 2, 3, 1) * 2 - 1, dim=-1)
    r = maps["roughness"].permute(0, 2, 3, 1)
    m = maps["metallic"].permute(0, 2, 3, 1)
    diff, spec = render_fn._metallic_to_specular(bc, m)

    B = basecolor.shape[0]
    wi = torch.tensor([[0.3, 0.3, 0.9]], device=device).expand(B, 3)
    wo = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(B, 3)
    wi = wi.unsqueeze(1).unsqueeze(1)
    wo = wo.unsqueeze(1).unsqueeze(1)
    img = render_fn._render_single(diff, spec, r, n, wi, wo)  # (B,H,W,3)
    return img.permute(0, 3, 1, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--split", default="outputs/stage4_split.json")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ckpts", nargs="+", default=["best.pt", "best_ema.pt", "best_render.pt"])
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = CachedMatSynthDataset(args.cache_dir)
    with open(args.split) as f:
        split_data = json.load(f)
    print(f"Dataset: {len(full_dataset)} total, test={len(split_data['test_indices'])}")

    render_fn = GGXRenderingLoss(n_diffuse=3, n_specular=6).to(device)
    import lpips
    lpips_fn = lpips.LPIPS(net="squeeze", verbose=False).to(device).eval()
    for q in lpips_fn.parameters():
        q.requires_grad = False

    all_results = {}
    for run in args.runs:
        run_name = os.path.basename(os.path.normpath(run))
        run_results = {}
        for ckpt in args.ckpts:
            ckpt_path = os.path.join(run, ckpt)
            if not os.path.isfile(ckpt_path):
                print(f"[skip] {run_name}/{ckpt} not found")
                continue
            print(f"\n=== {run_name}/{ckpt} ===")
            res = eval_one(run, ckpt, full_dataset, split_data, device,
                           render_fn, lpips_fn, args.batch_size)
            for k, v in res["overall"].items():
                print(f"  {k:20s}: {v:.4f}")
            run_results[ckpt] = res
        out_path = os.path.join(run, "eval_report.json")
        with open(out_path, "w") as f:
            json.dump(run_results, f, indent=2)
        print(f"  -> wrote {out_path}")
        all_results[run_name] = run_results

    # Summary table
    print("\n" + "=" * 92)
    print(f"{'run/ckpt':<32} {'n_l1':>8} {'r_l1':>8} {'m_l1':>8} "
          f"{'render':>9} {'rLPIPS':>8}")
    print("-" * 92)
    for run_name, run_res in all_results.items():
        for ckpt, res in run_res.items():
            o = res["overall"]
            label = f"{run_name}/{ckpt}"
            print(f"{label:<32} {o['normal_l1']:8.4f} {o['roughness_l1']:8.4f} "
                  f"{o['metallic_l1']:8.4f} {o['render']:9.4f} {o['rendered_lpips']:8.4f}")


if __name__ == "__main__":
    main()
