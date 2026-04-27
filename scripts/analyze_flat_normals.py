"""Empirical flat-normal analysis on the held-out test set.

For each shipped run (downloaded from the HF Hub model repo on first call),
runs inference on the 285-sample test set, computes per-sample predicted-
normal std and L1 vs GT, and emits a markdown table with two metrics:

  * **Per-sample median ratio** = median over samples of
    ``predicted_std / gt_std`` (paired comparison on the same image, then
    aggregated). 100% would mean predictions match GT spatial variation
    on the typical sample.
  * **Aggregated ratio** = ratio of means
    (``mean(predicted_std) / mean(gt_std)``). Older metric; biased toward
    high-GT-std samples but simpler to explain.

Samples whose GT normal map has near-zero spatial variation
(``gt_std < --gt-floor``, e.g. perfectly smooth ceramics) are excluded
from the per-sample median, since the ratio is ill-defined when both
sides approach zero.

Usage:
    python scripts/analyze_flat_normals.py \\
        --cache-dir data/processed2/train_256 \\
        --split-file outputs/stage4_split.json

(Requires the dataset cache and a logged-in `huggingface-cli` for the
private checkpoint repo. See docs/reproduction.md.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.model import PBRUNet, category_to_index  # noqa: E402
from src.dataset import CachedMatSynthDataset  # noqa: E402

HF_REPO = "Andrid1/vrtest-pbr-handoff"
KEEPERS = [
    "S1_bce", "S1B_bce_gan", "S1B_bce_long",
    "S2_dual_w10", "S3_rw1",
    "S4_baseline", "S4_gan_heavy",
    "S4_gan_light", "S4_gan_mid",
]


def _download(run: str, cache_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download
    args_path = Path(hf_hub_download(HF_REPO, f"{run}/args.json",
                                     local_dir=str(cache_dir)))
    for fname in (f"{run}/best_ema.pt", f"{run}/best.pt"):
        try:
            ckpt = Path(hf_hub_download(HF_REPO, fname, local_dir=str(cache_dir)))
            return args_path, ckpt
        except Exception:
            continue
    raise FileNotFoundError(f"No checkpoint for {run}")


def _build(args: dict, device: str) -> PBRUNet:
    return PBRUNet(
        encoder_name=args.get("encoder", args.get("encoder_name", "resnet34")),
        encoder_weights="none",
        use_category=bool(args.get("use_category", False)),
        normal_xy_only=bool(args.get("normal_xy", False)),
        separate_normal_decoder=bool(args.get("separate_normal_decoder", False)),
        predict_height=bool(args.get("predict_height", False)),
    ).to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/processed2/train_256", type=Path)
    p.add_argument("--split-file", default="outputs/stage4_split.json", type=Path)
    p.add_argument("--ckpt-cache", default=".cache/checkpoints", type=Path)
    p.add_argument("--out", default="results/FLAT_NORMALS_ANALYSIS.md", type=Path)
    p.add_argument("--gt-floor", default=0.005, type=float)
    p.add_argument("--batch", default=32, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()

    sp = json.loads(cli.split_file.read_text())
    test_idx = sp["test_indices"]
    print(f"[analyze] {len(test_idx)} test samples on {cli.device}")

    ds = CachedMatSynthDataset(str(cli.cache_dir))
    samples = [{
        "category": (s := ds[i]).get("category", "unknown"),
        "basecolor": s["basecolor"],
        "normal_gt": s["normal"],
    } for i in test_idx]

    gt_per = np.array([float(np.mean(s["normal_gt"].numpy().std(axis=(1, 2))))
                       for s in samples])
    n_kept = int((gt_per > cli.gt_floor).sum())
    print(f"[analyze] GT std mean={gt_per.mean():.4f} median={np.median(gt_per):.4f}; "
          f"keeping {n_kept}/{len(gt_per)} samples for median ratio")

    cli.ckpt_cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in KEEPERS:
        args_path, ckpt_path = _download(run, cli.ckpt_cache)
        run_args = json.loads(args_path.read_text())
        m = _build(run_args, cli.device)
        state = torch.load(ckpt_path, map_location=cli.device, weights_only=False)
        sd = state.get("model", state) if isinstance(state, dict) else state
        m.load_state_dict(sd, strict=False)

        pred_std, pred_l1 = [], []
        for start in range(0, len(samples), cli.batch):
            chunk = samples[start:start + cli.batch]
            x = torch.stack([s["basecolor"] for s in chunk]).to(cli.device)
            gt = torch.stack([s["normal_gt"] for s in chunk]).to(cli.device)
            cat = None
            if run_args.get("use_category", False):
                cat = torch.tensor(
                    [category_to_index(s["category"]) for s in chunk],
                    dtype=torch.long, device=cli.device,
                )
            with torch.no_grad():
                out = m(x, category=cat)
            pn = out["normal"]
            pred_std.extend(pn.std(dim=(2, 3)).mean(dim=1).cpu().numpy().tolist())
            pred_l1.extend((pn - gt).abs().mean(dim=(1, 2, 3)).cpu().numpy().tolist())

        del m
        if cli.device == "cuda":
            torch.cuda.empty_cache()

        p_std = np.array(pred_std)
        p_l1 = np.array(pred_l1)
        mask = gt_per > cli.gt_floor
        median_ratio = float(np.median(p_std[mask] / gt_per[mask]))
        aggregated = float(p_std.mean() / gt_per.mean())
        rows.append({
            "run": run,
            "median_ratio": median_ratio,
            "aggregated_ratio": aggregated,
            "l1_mean": float(p_l1.mean()),
            "l1_se": float(p_l1.std() / np.sqrt(len(p_l1))),
        })
        print(f"  {run}: median={median_ratio*100:.0f}% agg={aggregated*100:.1f}% "
              f"L1={p_l1.mean():.4f}")

    lines = [
        "# Flat-normal analysis on the held-out test set",
        "",
        f"Test set size: **{len(samples)} samples** (frozen split, seed 4242).",
        "",
        f"Ground-truth predicted-normal std reference: mean={gt_per.mean():.4f}, "
        f"median={np.median(gt_per):.4f}, "
        f"range=[{gt_per.min():.4f}, {gt_per.max():.4f}].",
        "",
        f"Samples with GT std < {cli.gt_floor} (essentially flat ground truth — "
        f"e.g. smooth ceramic) are excluded from the per-sample median ratio: "
        f"**{n_kept} of {len(samples)} samples used**.",
        "",
        "**Per-sample median ratio** = median over samples of "
        "`predicted_std / gt_std` on the same image. 100% would mean predictions "
        "match ground-truth spatial variation. **Aggregated ratio** = "
        "`mean(predicted_std) / mean(gt_std)`; biased toward high-GT-std samples.",
        "",
        "| run | per-sample median ratio | aggregated ratio | L1 vs GT (mean ± SE) |",
        "|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["median_ratio"]):
        lines.append(
            f"| {r['run']} | {r['median_ratio']*100:.0f}% | "
            f"{r['aggregated_ratio']*100:.1f}% | "
            f"{r['l1_mean']:.4f} ± {r['l1_se']:.4f} |"
        )
    lines.append("")
    lines.append("Lower per-sample median ratio = flatter / more collapsed normal "
                 "output. Reproduce with `python scripts/analyze_flat_normals.py`.")

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[analyze] wrote {cli.out}")


if __name__ == "__main__":
    main()
