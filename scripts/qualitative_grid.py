"""Qualitative comparison grid for a Stage 4 run.

For each of N fixed test samples, emits a row:
    basecolor | GT normal | pred normal | GT rough | pred rough
    | GT metal | pred metal | GT render(L1) | pred render(L1)
    | GT render(L2) | pred render(L2) | GT render(L3) | pred render(L3)

Three lighting configs (L1/L2/L3) are fixed for reproducibility.

Usage:
    python scripts/qualitative_grid.py \
        --cache-dir data/processed2/train_256 \
        --split outputs/stage4_split.json \
        --run-dir outputs/S4_gan_mid \
        --ckpt best_ema.pt \
        --n-samples 8 \
        --out outputs/S4_gan_mid/qualitative.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.dataset import CachedMatSynthDataset
from src.model import PBRUNet, category_to_index
from src.rendering_loss import GGXRenderingLoss


LIGHTS = [
    ("front",  [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
    ("grazing",[0.8, 0.0, 0.3], [0.0, 0.0, 1.0]),
    ("side",   [0.3, 0.6, 0.7], [0.0, 0.0, 1.0]),
]


def _render(render_fn, maps, basecolor, wi_vec, wo_vec, device):
    bc = basecolor.permute(0, 2, 3, 1)
    n = torch.nn.functional.normalize(maps["normal"].permute(0, 2, 3, 1) * 2 - 1, dim=-1)
    r = maps["roughness"].permute(0, 2, 3, 1)
    m = maps["metallic"].permute(0, 2, 3, 1)
    diff, spec = render_fn._metallic_to_specular(bc, m)
    B = basecolor.shape[0]
    wi = torch.tensor([wi_vec], device=device).expand(B, 3).unsqueeze(1).unsqueeze(1)
    wo = torch.tensor([wo_vec], device=device).expand(B, 3).unsqueeze(1).unsqueeze(1)
    img = render_fn._render_single(diff, spec, r, n, wi, wo)
    return img.permute(0, 3, 1, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--split", default="outputs/stage4_split.json")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best_ema.pt")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.split) as f:
        split_data = json.load(f)
    dataset = CachedMatSynthDataset(args.cache_dir)
    test_indices = split_data["test_indices"][: args.n_samples]

    # Rebuild model from args.json
    args_path = os.path.join(args.run_dir, "args.json")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            a = json.load(f)
        model = PBRUNet(
            encoder_name=a.get("encoder", "resnet34"),
            encoder_weights=None,
            use_category=a.get("use_category", False),
            normal_xy_only=a.get("normal_xy", False),
            separate_normal_decoder=a.get("separate_normal_decoder", False),
            predict_height=a.get("predict_height", False),
        ).to(device)
    else:
        model = PBRUNet(encoder_name="resnet34", encoder_weights=None,
                        use_category=True, separate_normal_decoder=True).to(device)

    state = torch.load(os.path.join(args.run_dir, args.ckpt), weights_only=False, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    render_fn = GGXRenderingLoss(n_diffuse=3, n_specular=6).to(device)

    cols = ["basecolor", "GT n", "pr n", "GT r", "pr r", "GT m", "pr m"]
    for lbl, *_ in LIGHTS:
        cols += [f"GT {lbl}", f"pr {lbl}"]

    fig, axes = plt.subplots(len(test_indices), len(cols),
                             figsize=(1.6 * len(cols), 1.6 * len(test_indices)))
    if len(test_indices) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row, idx in enumerate(test_indices):
            sample = dataset[idx]
            basecolor = sample["basecolor"].unsqueeze(0).to(device)
            cat = torch.tensor([category_to_index(sample.get("category", "unknown"))],
                               dtype=torch.long, device=device)
            preds = model(basecolor, category=cat)

            gt = {
                "normal":    sample["normal"].unsqueeze(0).to(device),
                "roughness": sample["roughness"][:1].unsqueeze(0).to(device),
                "metallic":  sample["metallic"][:1].unsqueeze(0).to(device),
            }

            panels = [sample["basecolor"].permute(1, 2, 0).cpu().numpy()]
            for m in ("normal", "roughness", "metallic"):
                g = gt[m][0]
                p_ = preds[m][0].clamp(0, 1)
                if g.shape[0] == 1:
                    panels.append(g[0].cpu().numpy())
                    panels.append(p_[0].cpu().numpy())
                else:
                    panels.append(g.permute(1, 2, 0).cpu().numpy())
                    panels.append(p_.permute(1, 2, 0).cpu().numpy())

            for lbl, wi_vec, wo_vec in LIGHTS:
                r_gt = _render(render_fn, gt, basecolor, wi_vec, wo_vec, device)[0]
                r_pr = _render(render_fn, preds, basecolor, wi_vec, wo_vec, device)[0]
                panels.append(r_gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                panels.append(r_pr.clamp(0, 1).permute(1, 2, 0).cpu().numpy())

            for c, (ax, img) in enumerate(zip(axes[row], panels)):
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                ax.axis("off")
                if row == 0:
                    ax.set_title(cols[c], fontsize=8)

    fig.suptitle(f"{os.path.basename(args.run_dir)} / {args.ckpt}", fontsize=10)
    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
