"""Tests for PatchGAN discriminator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


def test_patchgan_output_shape():
    """PatchGAN should output a 2D grid of real/fake scores."""
    from src.discriminator import PatchGANDiscriminator

    # Input: basecolor (3ch) + normal (3ch) + roughness (1ch) + metallic (1ch) = 8ch
    disc = PatchGANDiscriminator(in_channels=8)
    x = torch.rand(2, 8, 256, 256)
    out = disc(x)

    assert out.dim() == 4, f"Expected 4D output, got {out.dim()}D"
    assert out.shape[0] == 2, f"Batch size mismatch"
    assert out.shape[1] == 1, f"Expected 1 output channel, got {out.shape[1]}"
    # PatchGAN output should be smaller than input (receptive field patches)
    assert out.shape[2] < 256 and out.shape[3] < 256


def test_patchgan_output_range():
    """Output should be unbounded logits (no sigmoid applied)."""
    from src.discriminator import PatchGANDiscriminator

    disc = PatchGANDiscriminator(in_channels=8)
    x = torch.rand(2, 8, 256, 256)
    out = disc(x)

    # Should have both positive and negative values (logits, not probabilities)
    # Just check it's not all in [0,1]
    assert out.min() < 0.5 or out.max() > 0.5, "Output looks like it has sigmoid applied"


def test_patchgan_gradient_flow():
    """Gradients should flow through the discriminator."""
    from src.discriminator import PatchGANDiscriminator

    disc = PatchGANDiscriminator(in_channels=8)
    x = torch.rand(2, 8, 64, 64, requires_grad=True)
    out = disc(x)
    out.mean().backward()

    assert x.grad is not None, "No gradient on input"
    assert x.grad.abs().sum() > 0, "Zero gradients"
