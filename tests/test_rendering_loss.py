"""Tests for the differentiable GGX rendering loss."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


def test_rendering_loss_identical_maps_low_loss():
    """Identical pred and GT should produce near-zero loss."""
    from src.rendering_loss import GGXRenderingLoss

    torch.manual_seed(42)
    loss_fn = GGXRenderingLoss(n_diffuse=2, n_specular=3)

    B, H, W = 2, 64, 64
    basecolor = torch.rand(B, 3, H, W)
    normal = torch.rand(B, 3, H, W) * 0.4 + 0.3  # around 0.5 (flat-ish but varied)
    roughness = torch.rand(B, 1, H, W) * 0.8 + 0.1
    metallic = torch.rand(B, 1, H, W)

    loss = loss_fn(normal, roughness, metallic, normal, roughness, metallic, basecolor)

    assert loss.dim() == 0, f"Loss should be scalar, got {loss.shape}"
    assert loss.item() < 1e-5, f"Identical maps should give ~0 loss, got {loss.item():.6f}"


def test_rendering_loss_flat_normals_higher_loss():
    """Flat (average) normal prediction should produce higher loss than correct normals."""
    from src.rendering_loss import GGXRenderingLoss

    torch.manual_seed(42)
    loss_fn = GGXRenderingLoss(n_diffuse=2, n_specular=4)

    B, H, W = 2, 64, 64
    basecolor = torch.rand(B, 3, H, W)
    # GT: varied normals with real surface detail
    gt_normal = torch.rand(B, 3, H, W) * 0.6 + 0.2
    roughness = torch.rand(B, 1, H, W) * 0.5 + 0.2
    metallic = (torch.rand(B, 1, H, W) > 0.8).float()

    # Pred 1: correct normals (same as GT)
    loss_correct = loss_fn(gt_normal, roughness, metallic,
                           gt_normal, roughness, metallic, basecolor)

    # Pred 2: flat normals (constant 0.5, 0.5, 1.0 = pointing straight up)
    flat_normal = torch.zeros(B, 3, H, W)
    flat_normal[:, 0, :, :] = 0.5  # X = 0
    flat_normal[:, 1, :, :] = 0.5  # Y = 0
    flat_normal[:, 2, :, :] = 1.0  # Z = 1 (straight up)
    loss_flat = loss_fn(flat_normal, roughness, metallic,
                        gt_normal, roughness, metallic, basecolor)

    assert loss_flat > loss_correct, (
        f"Flat normals should have higher rendering loss. "
        f"flat={loss_flat:.4f}, correct={loss_correct:.6f}"
    )


def test_rendering_loss_wrong_roughness_detected():
    """Wrong roughness should produce higher loss (specular highlight differs)."""
    from src.rendering_loss import GGXRenderingLoss

    torch.manual_seed(42)
    loss_fn = GGXRenderingLoss(n_diffuse=1, n_specular=4)

    B, H, W = 2, 64, 64
    basecolor = torch.rand(B, 3, H, W)
    normal = torch.full((B, 3, H, W), 0.5)
    normal[:, 2, :, :] = 1.0  # flat normal, straight up
    gt_roughness = torch.full((B, 1, H, W), 0.3)  # smooth
    metallic = torch.zeros(B, 1, H, W)

    loss_correct = loss_fn(normal, gt_roughness, metallic,
                           normal, gt_roughness, metallic, basecolor)

    wrong_roughness = torch.full((B, 1, H, W), 0.9)  # very rough
    loss_wrong = loss_fn(normal, wrong_roughness, metallic,
                         normal, gt_roughness, metallic, basecolor)

    assert loss_wrong > loss_correct, (
        f"Wrong roughness should be detected. "
        f"wrong={loss_wrong:.4f}, correct={loss_correct:.6f}"
    )


def test_rendering_loss_gradient_flows():
    """Gradients should flow through the rendering loss to predicted maps."""
    from src.rendering_loss import GGXRenderingLoss

    loss_fn = GGXRenderingLoss(n_diffuse=1, n_specular=2)

    B, H, W = 1, 32, 32
    basecolor = torch.rand(B, 3, H, W)
    pred_normal = torch.rand(B, 3, H, W, requires_grad=True)
    pred_roughness = torch.rand(B, 1, H, W, requires_grad=True)
    pred_metallic = torch.rand(B, 1, H, W, requires_grad=True)
    gt_normal = torch.rand(B, 3, H, W)
    gt_roughness = torch.rand(B, 1, H, W)
    gt_metallic = torch.rand(B, 1, H, W)

    loss = loss_fn(pred_normal, pred_roughness, pred_metallic,
                   gt_normal, gt_roughness, gt_metallic, basecolor)
    loss.backward()

    assert pred_normal.grad is not None, "No gradient on normal"
    assert pred_roughness.grad is not None, "No gradient on roughness"
    assert pred_metallic.grad is not None, "No gradient on metallic"
    assert pred_normal.grad.abs().sum() > 0, "Zero gradient on normal"
    assert pred_roughness.grad.abs().sum() > 0, "Zero gradient on roughness"


def test_rendering_loss_gpu():
    """Should work on GPU if available."""
    from src.rendering_loss import GGXRenderingLoss

    if not torch.cuda.is_available():
        pytest.skip("No GPU")

    loss_fn = GGXRenderingLoss(n_diffuse=1, n_specular=2).cuda()

    B, H, W = 2, 32, 32
    basecolor = torch.rand(B, 3, H, W, device="cuda")
    normal = torch.rand(B, 3, H, W, device="cuda")
    roughness = torch.rand(B, 1, H, W, device="cuda")
    metallic = torch.rand(B, 1, H, W, device="cuda")

    loss = loss_fn(normal, roughness, metallic, normal, roughness, metallic, basecolor)
    assert loss.device.type == "cuda"
    assert loss.item() < 1e-5
