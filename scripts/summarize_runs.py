"""Walk outputs/, summarize each run, emit markdown table + keepers JSON."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunInfo:
    name: str
    path: str
    epochs_completed: int
    best_val_loss: float
    best_epoch: int
    best_render_loss: Optional[float]
    best_render_epoch: Optional[int]
    total_time_hours: float
    batch_size: int
    adversarial: float
    render_loss: float
    r1_gamma: float
    separate_normal_decoder: bool
    normal_weight: float
    roughness_weight: float
    metallic_weight: float
    stage: str
    has_best_pt: bool
    has_best_ema_pt: bool


def classify_run(name: str) -> str:
    """Stage classification by name prefix."""
    if name.startswith("S1B_"):
        return "stage1b"
    for n in (1, 2, 3, 4):
        if name.startswith(f"S{n}_"):
            return f"stage{n}"
    if name.startswith("overfit_") or name.startswith("run_"):
        return "exploration"
    return "exploration"


def summarize_run(run_dir: Path) -> Optional[RunInfo]:
    """Read args.json + history.json from run_dir, return RunInfo or None."""
    args_path = run_dir / "args.json"
    hist_path = run_dir / "history.json"
    if not args_path.exists() or not hist_path.exists():
        return None
    args = json.loads(args_path.read_text())
    hist = json.loads(hist_path.read_text())
    if not isinstance(hist, list) or not hist:
        return None

    val_losses = [(e.get("epoch", i), e.get("val_loss")) for i, e in enumerate(hist)
                  if e.get("val_loss") is not None]
    if not val_losses:
        return None
    best_epoch, best_val = min(val_losses, key=lambda x: x[1])

    render_losses = [(e.get("epoch", i), e.get("val_render_loss"))
                     for i, e in enumerate(hist)
                     if e.get("val_render_loss") is not None]
    best_render_epoch, best_render = (None, None)
    if render_losses:
        best_render_epoch, best_render = min(render_losses, key=lambda x: x[1])

    total_time = sum(float(e.get("time", 0.0) or 0.0) for e in hist) / 3600.0

    return RunInfo(
        name=run_dir.name,
        path=str(run_dir),
        epochs_completed=len(hist),
        best_val_loss=float(best_val),
        best_epoch=int(best_epoch),
        best_render_loss=float(best_render) if best_render is not None else None,
        best_render_epoch=int(best_render_epoch) if best_render_epoch is not None else None,
        total_time_hours=total_time,
        batch_size=int(args.get("batch_size", 0)),
        adversarial=float(args.get("adversarial", 0.0) or 0.0),
        render_loss=float(args.get("render_loss", 0.0) or 0.0),
        r1_gamma=float(args.get("r1_gamma", 0.0) or 0.0),
        separate_normal_decoder=bool(args.get("separate_normal_decoder", False)),
        normal_weight=float(args.get("normal_weight", 1.0) or 1.0),
        roughness_weight=float(args.get("roughness_weight", 1.0) or 1.0),
        metallic_weight=float(args.get("metallic_weight", 1.0) or 1.0),
        stage=classify_run(run_dir.name),
        has_best_pt=(run_dir / "best.pt").exists(),
        has_best_ema_pt=(run_dir / "best_ema.pt").exists(),
    )


def render_markdown(runs: list[RunInfo]) -> str:
    """Group runs by stage, emit a markdown table per stage."""
    out: list[str] = ["# Run Summary\n"]
    stages = ["stage1", "stage1b", "stage2", "stage3", "stage4", "exploration"]
    headings = {
        "stage1": "Stage 1 - loss screening",
        "stage1b": "Stage 1B - loss combinations + first GAN",
        "stage2": "Stage 2 - normal-prediction architecture",
        "stage3": "Stage 3 - render-loss as primary signal",
        "stage4": "Stage 4 - extended training + GAN sweep",
        "exploration": "Exploration / sanity (discard candidates)",
    }
    for stage in stages:
        in_stage = [r for r in runs if r.stage == stage]
        if not in_stage:
            continue
        out.append(f"\n## {headings[stage]}\n")
        out.append("| run | epochs | best val_loss | best render_loss | adv | r1 | render_w | sep_norm | hours |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(in_stage, key=lambda x: x.best_val_loss):
            render_str = f"{r.best_render_loss:.4f}" if r.best_render_loss is not None else "-"
            out.append(
                f"| {r.name} | {r.epochs_completed} | {r.best_val_loss:.4f} (e{r.best_epoch}) | "
                f"{render_str} | {r.adversarial:g} | {r.r1_gamma:g} | {r.render_loss:g} | "
                f"{'Y' if r.separate_normal_decoder else 'N'} | {r.total_time_hours:.1f} |"
            )
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", default="outputs", type=Path)
    p.add_argument("--summary-out", default="outputs/run_summary.md", type=Path)
    p.add_argument("--keepers-out", default="outputs/keepers.json", type=Path)
    args = p.parse_args()

    runs: list[RunInfo] = []
    for d in sorted(args.outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        info = summarize_run(d)
        if info is not None:
            runs.append(info)

    args.summary_out.write_text(render_markdown(runs), encoding="utf-8")

    candidates = {r.name: False for r in runs}
    args.keepers_out.write_text(
        json.dumps({"_instructions": "Set runs to true to ship them in handoff.",
                    "runs": candidates}, indent=2),
        encoding="utf-8",
    )
    print(f"Summarized {len(runs)} runs -> {args.summary_out}")
    print(f"Keepers template -> {args.keepers_out}")


if __name__ == "__main__":
    main()
