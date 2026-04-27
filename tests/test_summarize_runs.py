"""Tests for scripts.summarize_runs."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_run_dir(tmp_path):
    """Create a minimal fake run directory mimicking outputs/<run>/ layout."""
    run = tmp_path / "S_fake"
    run.mkdir()
    (run / "args.json").write_text(json.dumps({
        "out_dir": str(run),
        "epochs": 50,
        "batch_size": 16,
        "adversarial": 0.01,
        "render_loss": 1.0,
        "separate_normal_decoder": True,
        "r1_gamma": 10.0,
    }))
    (run / "history.json").write_text(json.dumps([
        {"epoch": 0, "train_loss": 0.5, "val_loss": 0.4, "time": 700.0},
        {"epoch": 1, "train_loss": 0.3, "val_loss": 0.25, "time": 700.0},
        {"epoch": 2, "train_loss": 0.2, "val_loss": 0.22, "time": 700.0},
    ]))
    return run


def test_summarize_single_run_extracts_metrics(fake_run_dir):
    from scripts.summarize_runs import summarize_run
    info = summarize_run(fake_run_dir)
    assert info.name == "S_fake"
    assert info.epochs_completed == 3
    assert info.best_val_loss == pytest.approx(0.22)
    assert info.best_epoch == 2
    assert info.batch_size == 16
    assert info.adversarial == 0.01
    assert info.separate_normal_decoder is True


def test_summarize_handles_missing_history(tmp_path):
    from scripts.summarize_runs import summarize_run
    run = tmp_path / "broken"
    run.mkdir()
    (run / "args.json").write_text("{}")
    info = summarize_run(run)
    assert info is None


def test_classify_run_by_name():
    from scripts.summarize_runs import classify_run
    assert classify_run("S1_baseline") == "stage1"
    assert classify_run("S1B_bce_gan") == "stage1b"
    assert classify_run("S2_dual_w10") == "stage2"
    assert classify_run("S3_rw1") == "stage3"
    assert classify_run("S4_baseline") == "stage4"
    assert classify_run("overfit_test") == "exploration"
    assert classify_run("run_baseline") == "exploration"
