"""Loss functions for PBR map prediction.

FFT spectral loss for penalizing missing high-frequency detail in predicted maps.
"""

import torch
import torch.nn as nn


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral loss comparing FFT magnitudes of pred and target.

    Computes 2D FFT of each channel, compares log-magnitude spectra via L1.
    Flat predictions (zero high-frequency energy) are heavily penalized when
    GT has detail.

    Args:
        pred:   (B, C, H, W) predicted map
        target: (B, C, H, W) ground truth map

    Returns:
        Scalar loss (mean L1 of log-magnitude spectra).
    """
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")

    pred_mag = torch.log(pred_fft.abs() + 1e-8)
    target_mag = torch.log(target_fft.abs() + 1e-8)

    return nn.functional.l1_loss(pred_mag, target_mag)
