"""Differentiable height map to normal map conversion.

Computes normals from a height field using finite differences (Sobel-like),
producing tangent-space normals in [0, 1] range matching the PBR convention.
"""

import torch
import torch.nn.functional as F


def height_to_normal(height: torch.Tensor, intensity: float = 1.0) -> torch.Tensor:
    """Convert a height map to a tangent-space normal map.

    Uses central differences to compute surface gradients, then constructs
    the normal vector and converts to [0, 1] storage range.

    Args:
        height:    (B, 1, H, W) height map in [0, 1]
        intensity: Strength of the normal detail (higher = more pronounced bumps)

    Returns:
        (B, 3, H, W) normal map in [0, 1], tangent-space convention:
        R=X, G=Y, B=Z, where 0.5 means zero for X/Y.
    """
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=height.dtype, device=height.device).view(1, 1, 3, 3) / 4.0
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=height.dtype, device=height.device).view(1, 1, 3, 3) / 4.0

    padded = F.pad(height, (1, 1, 1, 1), mode="replicate")
    dh_dx = F.conv2d(padded, kernel_x) * intensity
    dh_dy = F.conv2d(padded, kernel_y) * intensity

    nx = -dh_dx
    ny = -dh_dy
    nz = torch.ones_like(nx)

    normal = torch.cat([nx, ny, nz], dim=1)
    normal = F.normalize(normal, dim=1, eps=1e-6)

    # Convert from [-1, 1] to [0, 1] storage
    normal = normal * 0.5 + 0.5
    return normal
