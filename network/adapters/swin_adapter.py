"""Polynomial Adapter for Video Swin Transformer blocks.

Swin blocks use channel-last [B, T, H, W, C] tensors.  The adapter branch
permutes to channel-first [B, C, T, H, W] for Conv3d ops, applies the same
polynomial interaction as PolynomialAdapter (poly_adapter.py), then permutes
back before adding to the block output.

Architecture per block:
  y = frozen_swin_block(x) + gate * permute(BN(interaction(permute(x))))

All four adapter_mode values are supported (cross_poly / poly / relu / linear).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.primitives import volterra_quadratic, volterra_quadratic_cross


def _make_swin_conv(in_ch: int, out_ch: int, bottleneck_rank: int | None) -> nn.Module:
    """3×1×1 Conv3d (or bottlenecked pair) mapping in_ch → out_ch.  Stride always 1."""
    if bottleneck_rank is not None:
        r = bottleneck_rank
        return nn.Sequential(
            nn.Conv3d(in_ch, r,     kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.Conv3d(r,    out_ch, kernel_size=1,          bias=False),
        )
    return nn.Conv3d(in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)


class SwinBlockAdapter(nn.Module):
    """Parallel adapter branch attached to a frozen Swin video transformer block.

    Identical polynomial math to PolynomialAdapter; the only difference is the
    channel-last ↔ channel-first permutation around the Conv3d ops.
    """

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        Q: int = 4,
        bottleneck_rank: int | None = None,
        adapter_mode: str = "cross_poly",
    ):
        super().__init__()
        self.block        = block
        self.Q            = Q
        self.dim          = dim
        self.adapter_mode = adapter_mode

        if adapter_mode not in ("poly", "linear", "cross_poly", "relu"):
            raise ValueError(
                f"adapter_mode must be poly/linear/cross_poly/relu, got {adapter_mode!r}"
            )

        if adapter_mode == "poly":
            expand_ch = 2 * Q * dim
        elif adapter_mode == "cross_poly":
            expand_ch = 2 * Q
        elif adapter_mode == "relu":
            expand_ch = Q
        else:  # linear
            expand_ch = dim

        self.adapter_conv = _make_swin_conv(dim, expand_ch, bottleneck_rank)

        self.out_proj: nn.Conv3d | None = (
            nn.Conv3d(Q, dim, kernel_size=1, bias=False)
            if adapter_mode in ("cross_poly", "relu") else None
        )

        self.adapter_bn = nn.BatchNorm3d(dim)
        self.gate        = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, W, C]
        y  = self.block(x)                                    # [B, T, H, W, C]
        xc = x.permute(0, 4, 1, 2, 3).contiguous()           # [B, C, T, H, W]
        z  = self.adapter_conv(xc)

        if self.adapter_mode == "poly":
            z = volterra_quadratic(z, self.Q, self.dim)
        elif self.adapter_mode == "cross_poly":
            assert self.out_proj is not None
            z = self.out_proj(volterra_quadratic_cross(z, self.Q))
        elif self.adapter_mode == "relu":
            assert self.out_proj is not None
            z = self.out_proj(F.relu(z))
        # linear: z already [B, dim, T, H, W]

        z = self.adapter_bn(z)
        z = z.permute(0, 2, 3, 4, 1).contiguous()            # [B, T, H, W, C]
        return y + self.gate * z


def _swin_block_dim(block: nn.Module) -> int:
    """Extract channel dim from a SwinTransformerBlock via its first LayerNorm."""
    norm1 = getattr(block, "norm1")
    return int(norm1.normalized_shape[0])


def inject_swin_adapters(
    stage: nn.Sequential,
    Q: int,
    bottleneck_rank: int | None = None,
    adapter_mode: str = "cross_poly",
) -> nn.Sequential:
    """Wrap every block in *stage* with a SwinBlockAdapter.

    Channel dim is inferred from block.norm1.normalized_shape[0].
    """
    return nn.Sequential(*[
        SwinBlockAdapter(
            block,
            dim=_swin_block_dim(block),
            Q=Q,
            bottleneck_rank=bottleneck_rank,
            adapter_mode=adapter_mode,
        )
        for block in stage
    ])
