"""End-to-end demo: download a checkpoint from HF Hub, predict PBR maps, save a preview grid.

Usage:
    python examples/demo.py --image examples/inputs/wood.png --run S4_baseline

Produces:
    examples/output/<image-stem>__<run>__predicted.png  (4-panel grid:
    basecolor | predicted normal | predicted roughness | predicted metallic)

The demo downloads the run's args.json + checkpoint from a private HF Hub model
repo, reconstructs the PBRUNet with the matching architecture flags, and runs a
single forward pass on the supplied basecolor image.

For Stage 4 runs (best_ema.pt available), the EMA weights are preferred. For
Stage 1-3 runs only best.pt is published, and the demo falls back to that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model import PBRUNet, category_to_index  # noqa: E402

DEFAULT_HF_REPO = "Andrid1/vrtest-pbr-handoff"
DEFAULT_RUN = "S4_baseline"


def _hf_download(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Wrapper around hf_hub_download that returns a Path."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir),
    )
    return Path(path)


def _download_checkpoint(run: str, hf_repo: str, cache_dir: Path) -> Path:
    """Fetch best_ema.pt for the requested run, falling back to best.pt for
    older Stage 1-3 runs that did not export an EMA checkpoint."""
    last_err: Exception | None = None
    for filename in (f"{run}/best_ema.pt", f"{run}/best.pt"):
        try:
            return _hf_download(hf_repo, filename, cache_dir)
        except Exception as exc:  # noqa: BLE001 (we genuinely want any failure)
            last_err = exc
    raise FileNotFoundError(
        f"Neither best_ema.pt nor best.pt found for run '{run}' in {hf_repo} "
        f"(last error: {last_err})"
    )


def _download_args(run: str, hf_repo: str, cache_dir: Path) -> dict:
    """Fetch args.json for the run; return empty dict on failure (we'll fall
    back to S3-style defaults, which works for all keeper runs)."""
    try:
        path = _hf_download(hf_repo, f"{run}/args.json", cache_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not fetch {run}/args.json: {exc}; using defaults")
        return {}
    with open(path) as f:
        return json.load(f)


def _build_model(run_args: dict, device: str) -> PBRUNet:
    """Construct PBRUNet using flags from args.json (matches scripts/predict.py)."""
    if run_args:
        kwargs = dict(
            encoder_name=run_args.get("encoder", "resnet34"),
            encoder_weights="none",  # we'll load weights from the checkpoint
            use_category=run_args.get("use_category", False),
            normal_xy_only=run_args.get("normal_xy", False),
            separate_normal_decoder=run_args.get("separate_normal_decoder", False),
            predict_height=run_args.get("predict_height", False),
        )
    else:
        # Conservative S3-era defaults (matches scripts/predict.py fallback)
        kwargs = dict(
            encoder_name="resnet34",
            encoder_weights="none",
            use_category=True,
            separate_normal_decoder=True,
        )
    return PBRUNet(**kwargs).to(device)


def _load_state(model: PBRUNet, ckpt_path: Path, device: str) -> None:
    """Load checkpoint state dict, handling several common wrapping conventions."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            sd = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            sd = state["state_dict"]
        else:
            sd = state
    else:
        sd = state
    model.load_state_dict(sd)


def _load_image(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def _predict(
    model: PBRUNet,
    basecolor: torch.Tensor,
    device: str,
    use_category: bool,
    category: str = "unknown",
) -> dict[str, torch.Tensor]:
    x = basecolor.unsqueeze(0).to(device)
    cat = None
    if use_category:
        cat = torch.tensor(
            [category_to_index(category)], dtype=torch.long, device=device
        )
    with torch.no_grad():
        out = model(x, category=cat)
    if not isinstance(out, dict):
        raise RuntimeError(f"Unexpected model output type: {type(out)}")
    return {k: v[0].detach().cpu() for k, v in out.items()}


def _to_image(t: torch.Tensor) -> Image.Image:
    """Convert a (C,H,W) or (1,H,W) or (H,W) tensor in [0,1] to a 3-channel PIL image."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _save_grid(
    basecolor: torch.Tensor,
    pred: dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    panels = [_to_image(basecolor)]
    for key in ("normal", "roughness", "metallic"):
        if key in pred:
            panels.append(_to_image(pred[key]))
    w, h = panels[0].size
    grid = Image.new("RGB", (w * len(panels), h))
    for i, p in enumerate(panels):
        grid.paste(p, (i * w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Input basecolor image (RGB)")
    p.add_argument("--run", default=DEFAULT_RUN, help="Run name on HF Hub repo")
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HF Hub model repo id")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "checkpoints",
        help="Local cache for downloaded checkpoints",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "output",
        help="Where to write the preview grid PNG",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    p.add_argument(
        "--category",
        default="unknown",
        help="Material category hint for use_category models (e.g. Wood, Metal). "
             "Ignored if the model was trained without category conditioning.",
    )
    p.add_argument("--size", type=int, default=256, help="Input/output image size")
    args = p.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Input image not found: {args.image}")

    print(f"[demo] Fetching args.json + checkpoint for run='{args.run}' from {args.hf_repo}")
    run_args = _download_args(args.run, args.hf_repo, args.cache_dir)
    ckpt_path = _download_checkpoint(args.run, args.hf_repo, args.cache_dir)
    print(f"[demo] Checkpoint: {ckpt_path.name}")

    model = _build_model(run_args, args.device)
    _load_state(model, ckpt_path, args.device)
    model.eval()

    print(f"[demo] Predicting maps for {args.image} on {args.device}")
    basecolor = _load_image(args.image, size=args.size)
    pred = _predict(
        model,
        basecolor,
        args.device,
        use_category=run_args.get("use_category", False),
        category=args.category,
    )

    grid_path = args.out_dir / f"{args.image.stem}__{args.run}__predicted.png"
    _save_grid(basecolor, pred, grid_path)
    print(f"[demo] Wrote {grid_path}")


if __name__ == "__main__":
    main()
