"""Tests for custom loss functions in train.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


def test_sobel_gradient_loss_flat_vs_detailed():
    """Flat prediction should have higher gradient loss than matching GT."""
    from scripts.train import gradient_loss

    torch.manual_seed(42)
    gt = torch.rand(2, 3, 64, 64)

    loss_match = gradient_loss(gt, gt)

    flat = torch.full_like(gt, 0.5)
    loss_flat = gradient_loss(flat, gt)

    assert loss_match < loss_flat, (
        f"Matching pred should have lower gradient loss than flat. "
        f"match={loss_match:.4f}, flat={loss_flat:.4f}"
    )
    assert loss_match < 1e-6, f"Identical tensors should have ~0 gradient loss, got {loss_match:.4f}"


def test_sobel_gradient_loss_shape():
    """Gradient loss should work with various batch sizes and be a scalar."""
    from scripts.train import gradient_loss

    for batch in [1, 4]:
        pred = torch.rand(batch, 3, 32, 32)
        target = torch.rand(batch, 3, 32, 32)
        loss = gradient_loss(pred, target)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"


def test_sobel_gradient_loss_gpu():
    """Gradient loss should work on GPU if available."""
    from scripts.train import gradient_loss

    if not torch.cuda.is_available():
        pytest.skip("No GPU")

    pred = torch.rand(2, 3, 32, 32, device="cuda")
    target = torch.rand(2, 3, 32, 32, device="cuda")
    loss = gradient_loss(pred, target)
    assert loss.device.type == "cuda"


def test_lpips_loss_integration():
    """LPIPS should return a scalar loss, and blurry pred should score worse."""
    import lpips
    loss_fn = lpips.LPIPS(net="squeeze", verbose=False)

    torch.manual_seed(42)
    gt = torch.rand(2, 3, 64, 64)

    loss_match = loss_fn(gt * 2 - 1, gt * 2 - 1).mean()

    blurry = torch.nn.functional.avg_pool2d(gt, 5, stride=1, padding=2)
    loss_blurry = loss_fn(blurry * 2 - 1, gt * 2 - 1).mean()

    assert loss_match < loss_blurry, (
        f"Matching should have lower LPIPS than blurry. "
        f"match={loss_match:.4f}, blurry={loss_blurry:.4f}"
    )


def test_lpips_on_single_channel():
    """LPIPS expects 3-ch input. Single-channel maps should be expanded to 3-ch."""
    import lpips
    loss_fn = lpips.LPIPS(net="squeeze", verbose=False)

    gt = torch.rand(2, 1, 64, 64)
    pred = torch.rand(2, 1, 64, 64)

    gt_3ch = gt.expand(-1, 3, -1, -1)
    pred_3ch = pred.expand(-1, 3, -1, -1)

    loss = loss_fn(pred_3ch * 2 - 1, gt_3ch * 2 - 1).mean()
    assert loss.dim() == 0
    assert loss >= 0


def test_normal_weight_scales_loss():
    """Normal weight multiplier should scale the normal component of total loss."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.train import compute_loss

    torch.manual_seed(42)
    preds = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }
    targets = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }

    loss_1x, maps_1x, _ = compute_loss(preds, targets, normal_weight=1.0)
    loss_10x, maps_10x, _ = compute_loss(preds, targets, normal_weight=10.0)

    assert loss_10x.item() > loss_1x.item()


def test_roughness_weight_scales_loss():
    """Roughness weight multiplier should scale the roughness component of total loss."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.train import compute_loss

    torch.manual_seed(42)
    preds = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }
    targets = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }

    loss_1x, _, _ = compute_loss(preds, targets, roughness_weight=1.0)
    loss_01x, _, _ = compute_loss(preds, targets, roughness_weight=0.1)

    assert loss_01x.item() < loss_1x.item()


def test_metallic_weight_scales_loss():
    """Metallic weight multiplier should scale the metallic component of total loss."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.train import compute_loss

    torch.manual_seed(42)
    preds = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }
    targets = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }

    loss_1x, _, _ = compute_loss(preds, targets, metallic_weight=1.0)
    loss_01x, _, _ = compute_loss(preds, targets, metallic_weight=0.1)

    assert loss_01x.item() < loss_1x.item()


def test_compute_loss_returns_raw_and_weighted():
    """compute_loss should return both weighted and raw per-map losses."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.train import compute_loss

    torch.manual_seed(42)
    preds = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }
    targets = {
        "normal": torch.rand(2, 3, 32, 32),
        "roughness": torch.rand(2, 1, 32, 32),
        "metallic": torch.rand(2, 1, 32, 32),
    }

    total, weighted, raw = compute_loss(preds, targets,
                                         normal_weight=10.0,
                                         roughness_weight=0.1,
                                         metallic_weight=0.2)

    # Weighted normal should be ~10x raw normal
    assert abs(weighted["normal"] - raw["normal"] * 10.0) < 1e-4
    # Weighted roughness should be ~0.1x raw roughness
    assert abs(weighted["roughness"] - raw["roughness"] * 0.1) < 1e-4
    # Weighted metallic should be ~0.2x raw metallic
    assert abs(weighted["metallic"] - raw["metallic"] * 0.2) < 1e-4
    # Raw values should be positive
    for k in ("normal", "roughness", "metallic"):
        assert raw[k] > 0


def test_fft_loss_flat_vs_detailed():
    """Flat prediction should have higher FFT loss than matching GT."""
    from src.losses import fft_loss

    torch.manual_seed(42)
    gt = torch.rand(2, 3, 64, 64)

    loss_match = fft_loss(gt, gt)
    assert loss_match < 1e-5, f"Identical inputs should have ~0 FFT loss, got {loss_match:.6f}"

    flat = torch.full_like(gt, 0.5)
    loss_flat = fft_loss(flat, gt)
    assert loss_flat > loss_match


def test_fft_loss_preserves_gradients():
    """FFT loss should be differentiable."""
    from src.losses import fft_loss

    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    loss = fft_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_fft_loss_gpu():
    """FFT loss should work on GPU."""
    from src.losses import fft_loss

    if not torch.cuda.is_available():
        pytest.skip("No GPU")

    pred = torch.rand(2, 3, 32, 32, device="cuda")
    target = torch.rand(2, 3, 32, 32, device="cuda")
    loss = fft_loss(pred, target)
    assert loss.device.type == "cuda"
