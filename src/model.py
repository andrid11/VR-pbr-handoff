"""U-Net model for PBR map prediction: basecolor -> normal + roughness + metallic."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.height_to_normal import height_to_normal


# Canonical category list — order matters (maps category string -> index).
# "unknown" is index 0 and acts as fallback for unseen categories.
CATEGORIES = [
    "unknown",
    "Ceramic",
    "Concrete",
    "Fabric",
    "Ground",
    "Leather",
    "Marble",
    "Metal",
    "Misc",
    "Plaster",
    "Plastic",
    "Stone",
    "Terracotta",
    "Wood",
]
CATEGORY_TO_IDX = {name: i for i, name in enumerate(CATEGORIES)}


def category_to_index(category: str) -> int:
    """Convert category string to integer index, falling back to 0 ('unknown')."""
    return CATEGORY_TO_IDX.get(category, 0)


class PBRUNet(nn.Module):
    """Predicts normal, roughness, and metallic maps from a basecolor input.

    Optionally conditioned on material category via a learned embedding
    that is broadcast spatially and concatenated to the input image.

    Input:  basecolor  (B, 3, H, W)  float32 [0, 1]
            category   (B,)          int64 indices (optional)
    Output: dict with:
        "normal"    (B, 3, H, W)  float32 [0, 1]
        "roughness" (B, 1, H, W)  float32 [0, 1]
        "metallic"  (B, 1, H, W)  float32 [0, 1]
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        n_categories: int = len(CATEGORIES),
        category_embed_dim: int = 8,
        use_category: bool = False,
        normal_xy_only: bool = False,
        separate_normal_decoder: bool = False,
        predict_height: bool = False,
    ):
        super().__init__()
        self.use_category = use_category
        self.normal_xy_only = normal_xy_only
        self.separate_normal_decoder = separate_normal_decoder
        self.predict_height = predict_height

        in_channels = 3
        if use_category:
            self.category_embed = nn.Embedding(n_categories, category_embed_dim)
            in_channels = 3 + category_embed_dim

        normal_ch = 2 if normal_xy_only else 3
        _weights = encoder_weights if encoder_weights != "none" else None

        if separate_normal_decoder:
            normal_out_ch = 1 if predict_height else normal_ch
            self.normal_unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=_weights,
                in_channels=in_channels,
                classes=normal_out_ch,
            )
            self.material_unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=_weights,
                in_channels=in_channels,
                classes=2,
            )
        else:
            out_ch = (1 if predict_height else normal_ch) + 2
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=_weights,
                in_channels=in_channels,
                classes=out_ch,
            )

        self.sigmoid = nn.Sigmoid()

    def _prepare_input(self, basecolor, category):
        """Build network input: basecolor + optional category embedding."""
        x = basecolor
        if self.use_category:
            if category is None:
                category = torch.zeros(
                    basecolor.shape[0], dtype=torch.long, device=basecolor.device
                )
            emb = self.category_embed(category)
            emb = emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, emb], dim=1)
        return x

    def _decode_normal(self, raw_normal):
        """Decode raw normal predictions to [0,1] range."""
        if self.normal_xy_only:
            xy = torch.tanh(raw_normal)
            xy_sq = (xy ** 2).sum(dim=1, keepdim=True).clamp(max=1.0 - 1e-6)
            z = torch.sqrt(1.0 - xy_sq)
            normal = torch.cat([xy * 0.5 + 0.5, z * 0.5 + 0.5], dim=1)
        else:
            normal = self.sigmoid(raw_normal)
        return normal

    def forward(
        self,
        basecolor: torch.Tensor,
        category: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self._prepare_input(basecolor, category)

        if self.separate_normal_decoder:
            raw_normal = self.normal_unet(x)
            raw_material = self.material_unet(x)
            if self.predict_height:
                height = self.sigmoid(raw_normal)
                normal = height_to_normal(height, intensity=1.0)
            else:
                normal = self._decode_normal(raw_normal)
            roughness, metallic = self.sigmoid(raw_material).split([1, 1], dim=1)
        else:
            out = self.unet(x)
            if self.predict_height:
                height_raw, rest = out.split([1, 2], dim=1)
                height = self.sigmoid(height_raw)
                normal = height_to_normal(height, intensity=1.0)
                roughness, metallic = self.sigmoid(rest).split([1, 1], dim=1)
            elif self.normal_xy_only:
                normal_raw, rest = out.split([2, 2], dim=1)
                normal = self._decode_normal(normal_raw)
                roughness, metallic = self.sigmoid(rest).split([1, 1], dim=1)
            else:
                out = self.sigmoid(out)
                normal, roughness, metallic = out.split([3, 1, 1], dim=1)

        result = {"normal": normal, "roughness": roughness, "metallic": metallic}
        if self.predict_height:
            result["height"] = height
        return result
