"""Exponential-moving-average shadow weights for the generator.

Used at training time to maintain a stabilized copy of the model that often
generalizes better than the raw SGD trajectory. Only floating-point tensors
are decayed; integer buffers (e.g. BatchNorm.num_batches_tracked) are copied
as-is.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}
