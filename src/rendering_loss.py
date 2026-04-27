"""Differentiable Cook-Torrance GGX rendering loss for PBR map supervision.

Renders predicted and ground-truth PBR maps under random lighting conditions
and computes L1 loss in log space. This provides physically-grounded supervision
that penalizes flat/average predictions — flat normals produce wrong shading
under varied lighting, creating a strong gradient signal.

Based on Deschaintre et al. 2018 "Single-Image SVBRDF Capture with a
Rendering-Aware Deep Network" and the PyTorch port by MellowMurphy.

Key conventions:
    - All PBR maps are in [0, 1] range
    - Normals are tangent-space: R=X, G=Y, B=Z, 0.5 = zero for X/Y
    - Internally converts normals to [-1, 1] signed range for dot products
    - Uses metallic workflow: converts to specular/diffuse for rendering
    - Loss computed in log space to handle HDR (specular highlights)
"""

import math
import torch
import torch.nn as nn


class GGXRenderingLoss(nn.Module):
    """Differentiable rendering loss using Cook-Torrance GGX BRDF.

    Renders under random lighting (3 diffuse + 6 near-specular by default)
    and compares rendered images in log space.

    No learned parameters — pure math, no VRAM overhead beyond intermediates
    (~50-100MB at batch=4, 256px).
    """

    def __init__(self, n_diffuse: int = 3, n_specular: int = 6, epsilon: float = 0.1):
        super().__init__()
        self.n_diffuse = n_diffuse
        self.n_specular = n_specular
        self.epsilon = epsilon

    @staticmethod
    def _normalize(v: torch.Tensor) -> torch.Tensor:
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1, keepdim=True)

    def _render_single(
        self,
        diffuse: torch.Tensor,   # (B, H, W, 3)
        specular: torch.Tensor,  # (B, H, W, 3) — F0
        roughness: torch.Tensor, # (B, H, W, 1)
        normal: torch.Tensor,    # (B, H, W, 3) — unit length, signed [-1,1]
        wi: torch.Tensor,        # light direction (B, 1, 1, 3) or (B, H, W, 3)
        wo: torch.Tensor,        # view direction  (B, 1, 1, 3) or (B, H, W, 3)
    ) -> torch.Tensor:
        """Render one image under given light/view directions."""
        wi = self._normalize(wi)
        wo = self._normalize(wo)
        h = self._normalize((wi + wo) / 2.0)

        # Clamp roughness to avoid division by zero in GGX
        roughness = roughness.clamp(min=0.001)

        NdotH = self._dot(normal, h).clamp(min=0.0)
        NdotL = self._dot(normal, wi).clamp(min=0.0)
        NdotV = self._dot(normal, wo).clamp(min=0.0)
        VdotH = self._dot(wo, h).clamp(min=0.0)

        # D: GGX Normal Distribution Function
        alpha = roughness ** 2
        alpha2 = alpha ** 2
        denom = NdotH ** 2 * (alpha2 - 1.0) + 1.0
        D = alpha2 / (math.pi * denom ** 2 + 1e-6)

        # G: Smith-Schlick Geometry Function
        k = (roughness ** 2) / 2.0
        G1_L = NdotL / (NdotL * (1.0 - k) + k + 1e-6)
        G1_V = NdotV / (NdotV * (1.0 - k) + k + 1e-6)
        G = G1_L * G1_V

        # F: Schlick Fresnel Approximation
        F = specular + (1.0 - specular) * (1.0 - VdotH) ** 5

        # Cook-Torrance specular: F*G*D / (4*NdotL*NdotV)
        # Multiply by NdotL later, so we include it in the denominator cancellation
        spec = F * G * D / (4.0 * NdotV.clamp(min=1e-6) + 1e-6)

        # Lambertian diffuse (energy-conserving: scaled by 1-F)
        diff = diffuse * (1.0 - F) / math.pi

        # Final: (diffuse + specular) * NdotL * pi (hemisphere integral factor)
        result = (diff + spec) * NdotL * math.pi
        return result.clamp(min=0.0)

    def _random_direction(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Cosine-weighted hemisphere sampling, avoiding grazing angles."""
        r1 = torch.rand(batch_size, 1, device=device) * 0.949 + 0.001  # [0.001, 0.95]
        r2 = torch.rand(batch_size, 1, device=device)
        r = torch.sqrt(r1)
        phi = 2 * math.pi * r2
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt((1.0 - r1).clamp(min=0.0))
        return torch.cat([x, y, z], dim=-1)  # (B, 3)

    def _surface_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Position grid [-1, 1] for position-dependent light/view directions."""
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, -yy, torch.zeros_like(xx)], dim=-1)  # (H, W, 3)
        return grid.unsqueeze(0)  # (1, H, W, 3)

    def _metallic_to_specular(
        self, basecolor: torch.Tensor, metallic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert metallic workflow to specular/diffuse for rendering.

        F0 = lerp(0.04, basecolor, metallic)
        diffuse = basecolor * (1 - metallic)
        """
        specular_f0 = 0.04 * (1.0 - metallic) + basecolor * metallic
        diffuse = basecolor * (1.0 - metallic)
        return diffuse, specular_f0

    def forward(
        self,
        pred_normal: torch.Tensor,    # (B, 3, H, W) [0, 1]
        pred_roughness: torch.Tensor,  # (B, 1, H, W) [0, 1]
        pred_metallic: torch.Tensor,   # (B, 1, H, W) [0, 1]
        gt_normal: torch.Tensor,       # (B, 3, H, W) [0, 1]
        gt_roughness: torch.Tensor,    # (B, 1, H, W) [0, 1]
        gt_metallic: torch.Tensor,     # (B, 1, H, W) [0, 1]
        basecolor: torch.Tensor,       # (B, 3, H, W) [0, 1]
    ) -> torch.Tensor:
        """Compute rendering loss between predicted and GT PBR maps.

        Returns scalar loss (log-space L1 over all renderings).
        """
        B, _, H, W = basecolor.shape
        device = basecolor.device

        # Convert BCHW -> BHWC for rendering math
        bc = basecolor.permute(0, 2, 3, 1)

        # Convert normals from [0,1] storage to [-1,1] signed, then normalize
        p_n = self._normalize(pred_normal.permute(0, 2, 3, 1) * 2.0 - 1.0)
        g_n = self._normalize(gt_normal.permute(0, 2, 3, 1) * 2.0 - 1.0)

        p_r = pred_roughness.permute(0, 2, 3, 1)
        p_m = pred_metallic.permute(0, 2, 3, 1)
        g_r = gt_roughness.permute(0, 2, 3, 1)
        g_m = gt_metallic.permute(0, 2, 3, 1)

        # Convert both pred and GT to specular workflow
        p_diff, p_spec = self._metallic_to_specular(bc, p_m)
        g_diff, g_spec = self._metallic_to_specular(bc, g_m)

        surface = self._surface_grid(H, W, device)

        all_pred = []
        all_gt = []

        # Diffuse renderings: random light + view
        for _ in range(self.n_diffuse):
            wi = self._random_direction(B, device).unsqueeze(1).unsqueeze(1)
            wo = self._random_direction(B, device).unsqueeze(1).unsqueeze(1)
            all_pred.append(self._render_single(p_diff, p_spec, p_r, p_n, wi, wo))
            all_gt.append(self._render_single(g_diff, g_spec, g_r, g_n, wi, wo))

        # Specular renderings: mirror config with random shift
        # Places light in mirror position relative to view so specular highlights
        # are always visible — provides strong gradient signal for roughness/normals
        for _ in range(self.n_specular):
            view_dir = self._random_direction(B, device)
            # Mirror: flip X and Y, keep Z (reflects across surface)
            light_dir = view_dir * torch.tensor([-1.0, -1.0, 1.0], device=device)

            # Random distance and lateral shift
            dist = torch.exp(torch.randn(B, 1, device=device) * 0.75 + 0.5)
            shift = torch.cat([
                torch.rand(B, 2, device=device) * 2.0 - 1.0,
                torch.zeros(B, 1, device=device),
            ], dim=-1)

            view_pos = view_dir * dist + shift
            light_pos = light_dir * dist + shift

            # Position-dependent directions (vary across the surface)
            wo = view_pos.unsqueeze(1).unsqueeze(1) - surface
            wi = light_pos.unsqueeze(1).unsqueeze(1) - surface

            all_pred.append(self._render_single(p_diff, p_spec, p_r, p_n, wi, wo))
            all_gt.append(self._render_single(g_diff, g_spec, g_r, g_n, wi, wo))

        # Stack all renderings and compute log-space L1
        pred_stack = torch.cat(all_pred, dim=-1)  # (B, H, W, N*3)
        gt_stack = torch.cat(all_gt, dim=-1)

        loss = nn.functional.l1_loss(
            torch.log(pred_stack + self.epsilon),
            torch.log(gt_stack + self.epsilon),
        )
        return loss
