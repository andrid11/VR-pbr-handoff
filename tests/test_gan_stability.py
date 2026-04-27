"""Tests for GAN stability flags: warmup and R1 penalty."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


def test_warmup_skips_adversarial_step():
    """When epoch < warmup, train_one_epoch should not touch the discriminator."""
    from scripts.train import train_one_epoch
    from src.discriminator import PatchGANDiscriminator

    torch.manual_seed(0)

    # Tiny fake dataset yielding the shape the loop expects
    class FakeLoader:
        def __iter__(self):
            for _ in range(2):
                yield {
                    "basecolor":     torch.rand(2, 3, 32, 32),
                    "normal":        torch.rand(2, 3, 32, 32),
                    "roughness":     torch.rand(2, 1, 32, 32),
                    "metallic":      torch.rand(2, 1, 32, 32),
                    "category_idx":  torch.zeros(2, dtype=torch.long),
                    "name":          ["a", "b"],
                    "category":      ["unknown", "unknown"],
                }

    from src.model import PBRUNet
    model = PBRUNet(encoder_name="resnet18", encoder_weights=None,
                    use_category=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    disc = PatchGANDiscriminator(in_channels=8)
    disc_opt = torch.optim.SGD(disc.parameters(), lr=1e-3)

    disc_before = {k: v.clone() for k, v in disc.state_dict().items()}

    # Simulate "during warmup": pass current_epoch < adv_warmup_epochs
    train_one_epoch(
        model, FakeLoader(), opt, torch.device("cpu"), log_every=999,
        discriminator=disc, disc_optimizer=disc_opt, adversarial=0.01,
        adv_warmup_epochs=5, current_epoch=0,
    )

    for k in disc_before:
        assert torch.equal(disc_before[k], disc.state_dict()[k]), (
            f"Discriminator param {k} changed during warmup — it should not"
        )


def test_warmup_allows_adversarial_after_threshold():
    from scripts.train import train_one_epoch
    from src.discriminator import PatchGANDiscriminator
    from src.model import PBRUNet

    torch.manual_seed(0)

    class FakeLoader:
        def __iter__(self):
            for _ in range(2):
                yield {
                    "basecolor":     torch.rand(2, 3, 32, 32),
                    "normal":        torch.rand(2, 3, 32, 32),
                    "roughness":     torch.rand(2, 1, 32, 32),
                    "metallic":      torch.rand(2, 1, 32, 32),
                    "category_idx":  torch.zeros(2, dtype=torch.long),
                    "name":          ["a", "b"],
                    "category":      ["unknown", "unknown"],
                }

    model = PBRUNet(encoder_name="resnet18", encoder_weights=None,
                    use_category=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    disc = PatchGANDiscriminator(in_channels=8)
    disc_opt = torch.optim.SGD(disc.parameters(), lr=1e-2)

    disc_before = {k: v.clone() for k, v in disc.state_dict().items()}

    train_one_epoch(
        model, FakeLoader(), opt, torch.device("cpu"), log_every=999,
        discriminator=disc, disc_optimizer=disc_opt, adversarial=0.01,
        adv_warmup_epochs=5, current_epoch=10,
    )

    changed = any(
        not torch.equal(disc_before[k], disc.state_dict()[k])
        for k in disc_before if disc.state_dict()[k].dtype.is_floating_point
    )
    assert changed, "Discriminator params should change after warmup ends"


def test_r1_penalty_is_nonneg_and_scales():
    """R1 penalty should be >= 0 and scale linearly with gamma."""
    from scripts.train import r1_gradient_penalty
    from src.discriminator import PatchGANDiscriminator

    torch.manual_seed(0)
    disc = PatchGANDiscriminator(in_channels=8)
    real = torch.rand(2, 8, 64, 64)

    gp_1 = r1_gradient_penalty(disc, real, gamma=1.0).item()
    gp_5 = r1_gradient_penalty(disc, real, gamma=5.0).item()

    assert gp_1 >= 0, f"Penalty should be non-negative, got {gp_1}"
    # gamma=5 should produce 5x the penalty of gamma=1
    assert abs(gp_5 - 5.0 * gp_1) < 1e-4, (
        f"Linear in gamma: gp_5={gp_5}, 5*gp_1={5*gp_1}"
    )


def test_r1_penalty_contributes_to_disc_loss():
    """With gamma > 0, the discriminator's parameters must move differently
    than without the penalty after one step."""
    from scripts.train import adversarial_step
    from src.discriminator import PatchGANDiscriminator

    def _run(gamma):
        torch.manual_seed(0)
        disc = PatchGANDiscriminator(in_channels=8)
        opt = torch.optim.SGD(disc.parameters(), lr=1e-2)
        base = torch.rand(2, 3, 32, 32)
        tgt = {
            "normal":    torch.rand(2, 3, 32, 32),
            "roughness": torch.rand(2, 1, 32, 32),
            "metallic":  torch.rand(2, 1, 32, 32),
        }
        pred = {
            "normal":    torch.rand(2, 3, 32, 32, requires_grad=True),
            "roughness": torch.rand(2, 1, 32, 32, requires_grad=True),
            "metallic":  torch.rand(2, 1, 32, 32, requires_grad=True),
        }
        adversarial_step(disc, opt, base, tgt, pred, adv_weight=0.01,
                         r1_gamma=gamma)
        return {k: v.clone() for k, v in disc.state_dict().items()}

    a = _run(0.0)
    b = _run(10.0)
    diffs = [not torch.equal(a[k], b[k]) for k in a
             if a[k].dtype.is_floating_point]
    assert any(diffs), "R1 penalty did not alter discriminator updates"
