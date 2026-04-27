"""Tests for PBRUNet model, focusing on separate_normal_decoder feature."""

import torch
import pytest
from src.model import PBRUNet


# Use resnet18 + no pretrained weights for speed
ENCODER = "resnet18"
WEIGHTS = None
B, H, W = 2, 64, 64


def _make_input(use_category=False):
    basecolor = torch.rand(B, 3, H, W)
    category = torch.randint(0, 5, (B,)) if use_category else None
    return basecolor, category


def test_dual_decoder_output_shapes():
    model = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                    separate_normal_decoder=True)
    model.eval()
    basecolor, _ = _make_input()
    with torch.no_grad():
        out = model(basecolor)
    assert out["normal"].shape == (B, 3, H, W)
    assert out["roughness"].shape == (B, 1, H, W)
    assert out["metallic"].shape == (B, 1, H, W)


def test_dual_decoder_values_in_range():
    model = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                    separate_normal_decoder=True)
    model.eval()
    basecolor, _ = _make_input()
    with torch.no_grad():
        out = model(basecolor)
    for k in ("normal", "roughness", "metallic"):
        assert out[k].min() >= 0.0, f"{k} has values < 0"
        assert out[k].max() <= 1.0, f"{k} has values > 1"


def test_dual_decoder_with_category():
    model = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                    use_category=True, separate_normal_decoder=True)
    model.eval()
    basecolor, category = _make_input(use_category=True)
    with torch.no_grad():
        out = model(basecolor, category=category)
    assert out["normal"].shape == (B, 3, H, W)
    assert out["roughness"].shape == (B, 1, H, W)
    assert out["metallic"].shape == (B, 1, H, W)


def test_dual_decoder_with_xy_only():
    model = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                    normal_xy_only=True, separate_normal_decoder=True)
    model.eval()
    basecolor, _ = _make_input()
    with torch.no_grad():
        out = model(basecolor)
    # XY-only still outputs 3-ch normal (XY + derived Z)
    assert out["normal"].shape == (B, 3, H, W)
    assert out["normal"].min() >= 0.0
    assert out["normal"].max() <= 1.0


def test_single_decoder_still_works():
    model = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                    separate_normal_decoder=False)
    model.eval()
    basecolor, _ = _make_input()
    with torch.no_grad():
        out = model(basecolor)
    assert out["normal"].shape == (B, 3, H, W)
    assert out["roughness"].shape == (B, 1, H, W)
    assert out["metallic"].shape == (B, 1, H, W)


def test_height_mode_output_shapes():
    """Height mode outputs normal (3ch) + height (1ch)."""
    model = PBRUNet(encoder_name=ENCODER, encoder_weights=None,
                    separate_normal_decoder=True, predict_height=True)
    model.eval()
    x = torch.rand(B, 3, H, W)
    with torch.no_grad():
        out = model(x)
    assert out["normal"].shape == (B, 3, H, W)
    assert out["height"].shape == (B, 1, H, W)
    assert out["roughness"].shape == (B, 1, H, W)
    assert out["metallic"].shape == (B, 1, H, W)


def test_height_mode_single_decoder():
    """Height mode works without separate decoder."""
    model = PBRUNet(encoder_name=ENCODER, encoder_weights=None, predict_height=True)
    model.eval()
    x = torch.rand(B, 3, H, W)
    with torch.no_grad():
        out = model(x)
    assert out["normal"].shape == (B, 3, H, W)
    assert out["height"].shape == (B, 1, H, W)


def test_height_mode_normals_valid():
    """Height-derived normals in [0,1] and roughly unit-length."""
    model = PBRUNet(encoder_name=ENCODER, encoder_weights=None, predict_height=True)
    model.eval()
    x = torch.rand(B, 3, H, W)
    with torch.no_grad():
        out = model(x)
    n = out["normal"]
    assert n.min() >= 0.0
    assert n.max() <= 1.0
    n_signed = n * 2 - 1
    lengths = n_signed.norm(dim=1)
    assert (lengths - 1.0).abs().max() < 0.05


def test_dual_decoder_more_params():
    single = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                     separate_normal_decoder=False)
    dual = PBRUNet(encoder_name=ENCODER, encoder_weights="none",
                   separate_normal_decoder=True)
    single_params = sum(p.numel() for p in single.parameters())
    dual_params = sum(p.numel() for p in dual.parameters())
    assert dual_params > single_params, (
        f"Dual ({dual_params}) should have more params than single ({single_params})"
    )
