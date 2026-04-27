"""Single-image PBR inference.

Given a basecolor image path and a checkpoint, writes three PNG outputs:
  <out>/<stem>_normal.png
  <out>/<stem>_roughness.png
  <out>/<stem>_metallic.png

Usage:
    python scripts/predict.py \
        --input path/to/basecolor.png \
        --ckpt outputs/S4_gan_mid/best_ema.pt \
        --run-dir outputs/S4_gan_mid \
        --out predictions/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
import numpy as np

from src.model import PBRUNet, category_to_index


def load_basecolor(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def save_map(tensor: torch.Tensor, path: str) -> None:
    # tensor: (C, H, W) in [0, 1]
    arr = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    if arr.shape[0] == 1:
        Image.fromarray(arr[0], mode="L").save(path)
    else:
        Image.fromarray(arr.transpose(1, 2, 0), mode="RGB").save(path)


def build_model_from_args_json(run_dir: str, device) -> PBRUNet:
    args_path = os.path.join(run_dir, "args.json")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            a = json.load(f)
        kwargs = dict(
            encoder_name=a.get("encoder", "resnet34"),
            encoder_weights=None,
            use_category=a.get("use_category", False),
            normal_xy_only=a.get("normal_xy", False),
            separate_normal_decoder=a.get("separate_normal_decoder", False),
            predict_height=a.get("predict_height", False),
        )
    else:
        print(f"[warn] no args.json in {run_dir}, falling back to S3 defaults")
        kwargs = dict(
            encoder_name="resnet34", encoder_weights=None,
            use_category=True, separate_normal_decoder=True,
        )
    return PBRUNet(**kwargs).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run-dir", required=True,
                   help="Dir containing args.json (for model-arch reconstruction)")
    p.add_argument("--out", default="predictions")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--category", default="unknown")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_args_json(args.run_dir, device)
    state = torch.load(args.ckpt, weights_only=False, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    basecolor = load_basecolor(args.input, args.size).to(device)
    cat = torch.tensor([category_to_index(args.category)], dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model(basecolor, category=cat)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    for name in ("normal", "roughness", "metallic"):
        save_map(preds[name][0], os.path.join(args.out, f"{stem}_{name}.png"))

    print(f"Wrote: {args.out}/{stem}_{{normal,roughness,metallic}}.png")


if __name__ == "__main__":
    main()
