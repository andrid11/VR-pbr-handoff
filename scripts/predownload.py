"""
Pre-download and cache MatSynth samples to disk as resized tensors.

Saves each sample as a .pt file containing:
    {"basecolor": (3,H,W), "normal": (3,H,W), "roughness": (3,H,W),
     "metallic": (3,H,W), "name": str, "category": str}

Usage:
    python scripts/predownload.py                        # 1000 samples, 256px
    python scripts/predownload.py --n 500 --size 512
    python scripts/predownload.py --resume               # skip already downloaded
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from PIL import Image

from src.transforms import get_resize_transform, MAP_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Pre-download MatSynth samples")
    p.add_argument("--n", type=int, default=1000, help="Number of samples to download")
    p.add_argument("--size", type=int, default=256, help="Resize resolution")
    p.add_argument("--split", type=str, default="train", help="Dataset split")
    p.add_argument("--out", type=str, default="data/processed", help="Output directory")
    p.add_argument("--resume", action="store_true", help="Skip existing files")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument("--no-shuffle", action="store_true", help="Disable shuffle (faster first sample)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join(args.out, f"{args.split}_{args.size}")
    os.makedirs(out_dir, exist_ok=True)

    # Track what's already downloaded
    existing = {f.replace(".pt", "") for f in os.listdir(out_dir) if f.endswith(".pt")}
    saved = len(existing) if args.resume else 0
    if args.resume:
        print(f"Resuming: {len(existing)} samples already cached in {out_dir}")

    # ── Load streaming dataset ──────────────────────────────
    # On resume, check for a pre-scanned shard list to load only what's needed.
    data_files = None
    missing_shards_path = os.path.join(out_dir, "_missing_shards.json")
    if args.resume and os.path.exists(missing_shards_path):
        with open(missing_shards_path, "r") as f:
            shard_info = json.load(f)
        shard_indices = shard_info["shards"]
        total_shards = 431
        data_files = [
            f"data/{args.split}-{i:05d}-of-{total_shards:05d}.parquet"
            for i in shard_indices
        ]
        print(f"Loading {len(data_files)} shards with missing samples "
              f"({shard_info['missing_count']} to download)")

    print(f"Loading stream (split={args.split})...")
    t0 = time.perf_counter()
    ds = load_dataset(
        "gvecchio/MatSynth",
        split=args.split,
        streaming=True,
        data_files=data_files,
    )

    # Drop heavy columns we don't need
    keep = {"name", "metadata", *MAP_NAMES}
    try:
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
    except Exception:
        pass

    if not args.no_shuffle:
        ds = ds.shuffle(seed=args.seed, buffer_size=20)

    print(f"Stream ready in {time.perf_counter() - t0:.1f}s")

    transform = get_resize_transform(args.size)
    t_start = time.perf_counter()
    resume_saved = saved
    errors = 0
    skipped = 0

    manifest = []

    for i, sample in enumerate(ds):
        if saved >= args.n:
            break

        name = sample.get("name", f"sample_{i}")
        safe_name = name.replace("/", "_").replace("\\", "_")

        # Skip if already cached
        if safe_name in existing:
            skipped += 1
            continue

        # Process 4 maps
        tensors = {}
        ok = True
        for map_name in MAP_NAMES:
            img = sample.get(map_name)
            if img is None or not isinstance(img, Image.Image):
                ok = False
                break
            img = img.convert("RGB")
            tensors[map_name] = transform(img)

        if not ok:
            errors += 1
            continue

        # Extract category from metadata (string)
        meta = sample.get("metadata", {})
        category = meta.get("category", "unknown")
        if isinstance(category, (list, dict)):
            category = str(category)

        tensors["name"] = name
        tensors["category"] = category

        # Save
        out_path = os.path.join(out_dir, f"{safe_name}.pt")
        torch.save(tensors, out_path)
        saved += 1
        existing.add(safe_name)

        manifest.append({"name": name, "category": category, "file": f"{safe_name}.pt"})

        # Progress
        elapsed = time.perf_counter() - t_start
        new_downloaded = saved - resume_saved
        rate = elapsed / new_downloaded if new_downloaded > 0 else 0
        remaining = args.n - saved
        eta = rate * remaining
        print(f"  [{saved}/{args.n}] {name}  "
              f"(skipped={skipped}) elapsed={elapsed:.0f}s  rate={rate:.1f}s/sample  ETA={eta:.0f}s")

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            old = json.load(f)
        old_names = {e["name"] for e in old}
        for entry in manifest:
            if entry["name"] not in old_names:
                old.append(entry)
        manifest = old

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = time.perf_counter() - t_start
    print(f"\nDone: {saved} saved, {errors} errors")
    print(f"Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"Output: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
