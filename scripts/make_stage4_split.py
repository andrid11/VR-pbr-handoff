"""Carve a frozen test set from an existing Stage 3 split.

Reads the val indices produced by Stage 3 training and splits them in half
(seeded) into new val and test lists. Writes a global split file referenced
by all Stage 4 runs.

Usage:
    python scripts/make_stage4_split.py \
        --source outputs/S3_rw1/split_indices.json \
        --output outputs/stage4_split.json
"""

import argparse
import json
import os

import torch

TEST_SEED = 4242


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="Existing split_indices.json (e.g. from S3_rw1)")
    p.add_argument("--output", default="outputs/stage4_split.json")
    p.add_argument("--test-fraction", type=float, default=0.5,
                   help="Fraction of val to move into test (default 0.5)")
    args = p.parse_args()

    with open(args.source) as f:
        src = json.load(f)

    train = list(src["train_indices"])
    val_all = list(src["val_indices"])

    n_test = int(round(len(val_all) * args.test_fraction))
    assert n_test > 0 and n_test < len(val_all), (
        f"test fraction {args.test_fraction} yields {n_test} samples "
        f"from val={len(val_all)} — pick something in (0, 1)"
    )

    gen = torch.Generator().manual_seed(TEST_SEED)
    perm = torch.randperm(len(val_all), generator=gen).tolist()
    test_indices = sorted(val_all[i] for i in perm[:n_test])
    val_indices = sorted(val_all[i] for i in perm[n_test:])

    # Sanity: sets disjoint and cover the originals
    assert set(train).isdisjoint(val_indices)
    assert set(train).isdisjoint(test_indices)
    assert set(val_indices).isdisjoint(test_indices)
    assert set(val_indices) | set(test_indices) == set(val_all)

    payload = {
        "seed": TEST_SEED,
        "source": os.path.abspath(args.source),
        "dataset_size": src["dataset_size"],
        "train_size": len(train),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "train_indices": train,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.output}")
    print(f"  train={len(train)}  val={len(val_indices)}  test={len(test_indices)}")


if __name__ == "__main__":
    main()
