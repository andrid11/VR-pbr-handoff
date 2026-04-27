"""PatchGAN discriminator for PBR map adversarial training.

Takes concatenated (basecolor + predicted maps) as input.
Outputs a grid of real/fake logits (one per receptive-field patch).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (from pix2pix).

    Input: (B, in_channels, H, W) — typically 8ch (3 base + 3 normal + 1 rough + 1 metal)
    Output: (B, 1, H', W') — grid of real/fake logits
    """

    def __init__(self, in_channels: int = 8, ndf: int = 64):
        super().__init__()

        # 4-layer PatchGAN: C64-C128-C256-C512 -> 1ch output
        # No batchnorm on first layer
        self.model = nn.Sequential(
            # Layer 1: no norm
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4: stride 1
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 1 channel of logits
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
