"""Tests for height_to_normal conversion."""

import torch
import pytest
from src.height_to_normal import height_to_normal


def test_flat_height_gives_up_normal():
    """Flat height -> normals pointing up [0.5, 0.5, ~1.0]."""
    height = torch.full((1, 1, 16, 16), 0.5)
    normal = height_to_normal(height)
    # Check interior pixels (avoid border effects)
    interior = normal[:, :, 2:-2, 2:-2]
    assert interior[:, 0].allclose(torch.tensor(0.5), atol=1e-5), "X should be ~0.5"
    assert interior[:, 1].allclose(torch.tensor(0.5), atol=1e-5), "Y should be ~0.5"
    assert interior[:, 2].allclose(torch.tensor(1.0), atol=1e-5), "Z should be ~1.0"


def test_sloped_height_gives_tilted_normal():
    """Linear ramp in X -> X normal deviates from 0.5."""
    height = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    normal = height_to_normal(height, intensity=1.0)
    interior = normal[:, :, 2:-2, 2:-2]
    # X gradient is positive, so nx = -dh_dx < 0, stored as < 0.5
    mean_x = interior[:, 0].mean().item()
    assert mean_x < 0.48, f"Expected tilted X normal < 0.48, got {mean_x}"


def test_height_to_normal_differentiable():
    """Must be differentiable for backprop."""
    height = torch.rand(1, 1, 16, 16, requires_grad=True)
    normal = height_to_normal(height)
    loss = normal.sum()
    loss.backward()
    assert height.grad is not None
    assert height.grad.shape == height.shape


def test_height_to_normal_in_range():
    """Output must be in [0, 1]."""
    height = torch.rand(2, 1, 32, 32)
    normal = height_to_normal(height, intensity=5.0)
    assert normal.min() >= 0.0
    assert normal.max() <= 1.0
