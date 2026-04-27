"""Select a fixed comparison set of validation samples for cross-run visual comparison.

Picks one sample per category from the validation split, saves indices to a JSON file.
All training runs should use these same indices for preview generation.

Usage:
    python scripts/select_comparison_set.py --cache-dir data/processed2/train_256 --out-dir outputs
    python scripts/select_comparison_set.py --cache-dir data/processed2/train_256 --out-dir outputs --split-file outputs/run_full_baseline/split_indices.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import CachedMatSynthDataset

TARGET_CATEGORIES = ["Wood", "Metal", "Fabric", "Stone", "Ceramic", "Concrete", "Ground", "Plaster"]


def main():
    p = argparse.ArgumentParser(description="Select fixed comparison set")
    p.add_argument("--cache-dir", default="data/processed2/train_256")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--split-file", default=None,
                   help="Path to split_indices.json. If not given, uses full dataset indices.")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset = CachedMatSynthDataset(args.cache_dir)

    # Determine validation indices
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file) as f:
            splits = json.load(f)
        val_indices = splits["val"]
        print(f"Loaded {len(val_indices)} val indices from {args.split_file}")
    else:
        # Generate same split as training would
        import torch
        n = len(dataset)
        n_val = int(n * args.val_split)
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=generator).tolist()
        val_indices = perm[:n_val]
        print(f"Generated {len(val_indices)} val indices (seed={args.seed})")

    # Pick one sample per target category
    selected = {}
    for idx in val_indices:
        sample = dataset[idx]
        cat = sample.get("category", "unknown")
        if cat in TARGET_CATEGORIES and cat not in selected:
            selected[cat] = {"index": idx, "name": sample.get("name", ""), "category": cat}
        if len(selected) == len(TARGET_CATEGORIES):
            break

    # Report what we found
    found = list(selected.keys())
    missing = [c for c in TARGET_CATEGORIES if c not in selected]
    print(f"Found: {found}")
    if missing:
        print(f"Missing (not in val set): {missing}")

    result = {
        "description": "Fixed comparison set for cross-run visual evaluation",
        "samples": list(selected.values()),
        "indices": [s["index"] for s in selected.values()],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "comparison_set.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
