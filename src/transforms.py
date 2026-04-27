"""Shared image transforms for PBR map processing."""

import random
import torch
from torchvision import transforms


# Maps that use standard [0,1] range
STANDARD_MAPS = ("basecolor", "roughness", "metallic")

# Normal maps need special handling ([-1,1] range)
NORMAL_MAP = "normal"

MAP_NAMES = ("basecolor", "normal", "roughness", "metallic")


def get_resize_transform(size: int = 256):
    """Basic resize + to-tensor for all map types."""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),  # [0,255] -> [0,1], HWC -> CHW
    ])


def get_train_transform(size: int = 256):
    """Training transform with augmentation (shared crop/flip for all maps)."""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomCrop(size),
        transforms.ToTensor(),
    ])


def get_preview_transform(size: int = 512):
    """Larger resize for visual inspection."""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])


class PBRAugmentation:
    """Augmentation for PBR map samples with correct normal map handling.

    Applies identical spatial transforms to all maps, then corrects normal
    map X/Y channels for flips and rotations.

    Normal map convention (tangent space):
        R (ch 0) = X (right),  stored as [0,1], 0.5 = zero
        G (ch 1) = Y (up),    stored as [0,1], 0.5 = zero
        B (ch 2) = Z (out),   always positive

    Corrections after spatial transforms:
        H-flip:  negate X  ->  R = 1.0 - R
        V-flip:  negate Y  ->  G = 1.0 - G
        90° CW:  (X, Y) -> (Y, -X)  ->  R_new = G_old, G_new = 1.0 - R_old
    """

    def __init__(self, hflip=True, vflip=True, rot90=True):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90

    def __call__(self, sample: dict) -> dict:
        """Augment a sample dict in-place. All map tensors must be (C, H, W)."""
        do_hflip = self.hflip and random.random() < 0.5
        do_vflip = self.vflip and random.random() < 0.5
        n_rot90 = random.randint(0, 3) if self.rot90 else 0

        # Apply spatial transforms to all maps
        for key in MAP_NAMES:
            if key not in sample:
                continue
            t = sample[key]
            if do_hflip:
                t = t.flip(-1)  # flip W
            if do_vflip:
                t = t.flip(-2)  # flip H
            if n_rot90 > 0:
                t = torch.rot90(t, n_rot90, dims=(-2, -1))
            sample[key] = t

        # Correct normal map channels
        if "normal" in sample:
            n = sample["normal"]
            # Apply flip corrections
            if do_hflip:
                n[0] = 1.0 - n[0]  # negate X
            if do_vflip:
                n[1] = 1.0 - n[1]  # negate Y
            # Apply rotation corrections
            if n_rot90 % 4 == 1:  # 90° CW
                r_old, g_old = n[0].clone(), n[1].clone()
                n[0] = g_old            # X_new = Y_old
                n[1] = 1.0 - r_old      # Y_new = -X_old
            elif n_rot90 % 4 == 2:  # 180°
                n[0] = 1.0 - n[0]  # negate X
                n[1] = 1.0 - n[1]  # negate Y
            elif n_rot90 % 4 == 3:  # 270° CW (= 90° CCW)
                r_old, g_old = n[0].clone(), n[1].clone()
                n[0] = 1.0 - g_old  # X_new = -Y_old
                n[1] = r_old         # Y_new = X_old
            sample["normal"] = n

        return sample
