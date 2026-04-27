"""
Plot training curves from history.json.

Usage:
    python scripts/plot_history.py
    python scripts/plot_history.py --history outputs/checkpoints/history.json
    python scripts/plot_history.py --history outputs/run1/history.json outputs/run2/history.json --labels run1 run2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_MAPS = ("normal", "roughness", "metallic")


def parse_args():
    p = argparse.ArgumentParser(description="Plot training history curves")
    p.add_argument("--history", nargs="+", default=["outputs/checkpoints/history.json"],
                   help="Path(s) to history.json files")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels for each history file")
    p.add_argument("--out", default=None,
                   help="Output image path (default: <history_dir>/curves.png)")
    p.add_argument("--per-map", action="store_true",
                   help="In comparison mode, also plot per-map val losses")
    return p.parse_args()


def load_history(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Empty or invalid history: {path}")
    return data


def plot_single(history: list[dict], out_path: str, label: str = ""):
    """Plot curves for a single training run (4-panel figure)."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    lr = [h["lr"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) Train vs Val loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train", linewidth=1.5)
    ax.plot(epochs, val_loss, label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Per-map train losses
    ax = axes[0, 1]
    for name in TARGET_MAPS:
        values = [h["train_maps"][name] for h in history]
        ax.plot(epochs, values, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Per-Map Train Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Per-map val losses
    ax = axes[1, 0]
    for name in TARGET_MAPS:
        values = [h["val_maps"][name] for h in history]
        ax.plot(epochs, values, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Per-Map Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, lr, color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    title = "Training Curves"
    if label:
        title += f" — {label}"
    best_epoch = min(history, key=lambda h: h["val_loss"])
    title += f"\nBest val_loss: {best_epoch['val_loss']:.4f} @ epoch {best_epoch['epoch']}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_comparison(histories: list[list[dict]], labels: list[str], out_path: str,
                    per_map: bool = False):
    """Overlay multiple runs on a single figure for comparison."""
    if per_map:
        # 2 rows x 3 cols: top = train/val total, bottom = val normal/roughness/metallic
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))

        # (0,0) Train loss
        ax = axes[0, 0]
        for hist, label in zip(histories, labels):
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["train_loss"] for h in hist], label=label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,1) Val loss
        ax = axes[0, 1]
        for hist, label in zip(histories, labels):
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["val_loss"] for h in hist], label=label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Val Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,2) empty — hide it
        axes[0, 2].axis("off")

        # Bottom row: per-map val losses
        for col, map_name in enumerate(TARGET_MAPS):
            ax = axes[1, col]
            for hist, label in zip(histories, labels):
                epochs = [h["epoch"] for h in hist]
                values = [h["val_maps"][map_name] for h in hist]
                ax.plot(epochs, values, label=label, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Val {map_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Train loss comparison
        ax = axes[0]
        for hist, label in zip(histories, labels):
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["train_loss"] for h in hist], label=label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Val loss comparison
        ax = axes[1]
        for hist, label in zip(histories, labels):
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["val_loss"] for h in hist], label=label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Val Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Run Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()

    histories = [load_history(p) for p in args.history]
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.history]

    # Pad labels if needed
    while len(labels) < len(histories):
        labels.append(f"run{len(labels)}")

    if len(histories) == 1:
        # Single run: detailed 4-panel plot
        out_path = args.out or os.path.join(
            os.path.dirname(args.history[0]), "curves.png"
        )
        plot_single(histories[0], out_path, label=labels[0])
    else:
        # Multiple runs: comparison overlay + individual plots
        out_path = args.out or "outputs/comparison.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plot_comparison(histories, labels, out_path, per_map=args.per_map)

        # Also generate individual detailed plots
        for hist, label, path in zip(histories, labels, args.history):
            individual_path = os.path.join(os.path.dirname(path), "curves.png")
            plot_single(hist, individual_path, label=label)


if __name__ == "__main__":
    main()
