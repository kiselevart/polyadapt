"""Video Swin Transformer (Kinetics-400 pretrained) with Polynomial Adapters.

Three variants are available: Tiny, Small, Base.  All pretrained on Kinetics-400.
Backbone is fully frozen; only adapter branches and the classification head train.

Kinetics-400 top-1 of the frozen backbones (from the Video Swin paper):
  swin3d_t: ~77.9%   vs R3D-18: ~63.6%
  swin3d_s: ~80.6%
  swin3d_b: ~82.7%

Channel widths per stage (four stages, indexed 1–4):
  swin3d_t / swin3d_s : [96, 192, 384, 768]
  swin3d_b            : [128, 256, 512, 1024]

Stage numbering matches R3DAdapted convention so existing --adapter_stages flags
transfer directly (1=earliest/finest, 4=deepest/coarsest).

Swin blocks operate in channel-last format [B, T, H, W, C].  SwinBlockAdapter
handles the permutation internally, so the adapter math is identical to R3D.

Call get_1x_lr_params() / get_10x_lr_params() for the optimizer parameter groups.
"""

import torch
import torch.nn as nn
import torchvision

from network.adapters.swin_adapter import inject_swin_adapters


def _is_swin_stage(module: nn.Module) -> bool:
    """True iff module is an nn.Sequential of Swin blocks (has .attn + .norm1)."""
    return (
        isinstance(module, nn.Sequential)
        and len(module) > 0
        and hasattr(module[0], "attn")
        and hasattr(module[0], "norm1")
    )


class Swin3DAdapted(nn.Module):
    """Video Swin Transformer (frozen) + polynomial adapter branches per block."""

    def __init__(
        self,
        num_classes: int,
        Q: int = 4,
        adapter_stages: tuple = (1, 2, 3, 4),
        bottleneck_rank: int | None = None,
        adapter_mode: str = "cross_poly",
        variant: str = "swin3d_t",
    ):
        super().__init__()

        _loaders = {
            "swin3d_t": torchvision.models.video.swin3d_t,
            "swin3d_s": torchvision.models.video.swin3d_s,
            "swin3d_b": torchvision.models.video.swin3d_b,
        }
        if variant not in _loaders:
            raise ValueError(f"variant must be one of {list(_loaders)}, got {variant!r}")

        bb = _loaders[variant](weights="DEFAULT")
        for p in bb.parameters():
            p.requires_grad_(False)

        self.patch_embed = bb.patch_embed
        self.pos_drop    = getattr(bb, "pos_drop", nn.Identity())

        swin_stage_idx = 0
        new_features: list[nn.Module] = []
        for module in bb.features:
            if _is_swin_stage(module):
                swin_stage_idx += 1
                if swin_stage_idx in adapter_stages:
                    assert isinstance(module, nn.Sequential)
                    module = inject_swin_adapters(module, Q, bottleneck_rank, adapter_mode)
            new_features.append(module)

        self.features = nn.Sequential(*new_features)
        self.norm     = bb.norm
        self.avgpool  = bb.avgpool

        self.head = nn.Linear(bb.head.in_features, num_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)        # [B, T', H', W', C]
        x = self.pos_drop(x)
        x = self.features(x)           # [B, T', H', W', C]
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T', H', W']
        x = self.avgpool(x)
        return self.head(x.flatten(1))

    def get_1x_lr_params(self):
        for name, p in self.named_parameters():
            if p.requires_grad and not name.startswith("head."):
                yield p

    def get_10x_lr_params(self):
        return iter(self.head.parameters())


class Swin3DBaseline(nn.Module):
    """Video Swin Transformer (frozen or fine-tuned) with a fresh classification head.

    Use freeze_backbone=True for a linear-probe baseline,
        freeze_backbone=False for full fine-tune.
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True, variant: str = "swin3d_t"):
        super().__init__()

        _loaders = {
            "swin3d_t": torchvision.models.video.swin3d_t,
            "swin3d_s": torchvision.models.video.swin3d_s,
            "swin3d_b": torchvision.models.video.swin3d_b,
        }
        if variant not in _loaders:
            raise ValueError(f"variant must be one of {list(_loaders)}, got {variant!r}")

        bb = _loaders[variant](weights="DEFAULT")
        if freeze_backbone:
            for p in bb.parameters():
                p.requires_grad_(False)

        self.patch_embed = bb.patch_embed
        self.pos_drop    = getattr(bb, "pos_drop", nn.Identity())
        self.features    = bb.features
        self.norm        = bb.norm
        self.avgpool     = bb.avgpool

        self.head = nn.Linear(bb.head.in_features, num_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.avgpool(x)
        return self.head(x.flatten(1))

    def get_1x_lr_params(self):
        for name, p in self.named_parameters():
            if p.requires_grad and not name.startswith("head."):
                yield p

    def get_10x_lr_params(self):
        return iter(self.head.parameters())
