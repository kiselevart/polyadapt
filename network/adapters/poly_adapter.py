"""Polynomial Adapter — parallel Volterra branch for pretrained 3D CNN blocks.

Architecture per block (no bottleneck):
    y = frozen_block(x)  +  gate * BN(volterra_quadratic(adapter_conv(x), Q, out_ch))

With bottleneck rank r:
    adapter_conv = LaguerreConv3d(in_ch → r)  +  Conv3d(r → 2*Q*out_ch, 1×1×1)

    Cost: in_ch*r*N_lag + r*2*Q*out_ch   vs   in_ch*2*Q*out_ch*N_lag  (no bottleneck)
    ~10× fewer params at r=64, Q=4 on the 512-channel layers.

Gate initialises at 1e-2 so the model starts at pretrained baseline quality.
"""

import torch
import torch.nn as nn

from network.primitives import LaguerreConv3d, LaguerreConv3d_Full, volterra_quadratic


def _build_adapter_conv(
    in_ch: int,
    expand_ch: int,
    stride,
    conv_type: str,
    N_lag: int | None,
    bottleneck_rank: int | None,
) -> nn.Module:
    """Return the conv (or conv pair) that maps in_ch → expand_ch."""

    if bottleneck_rank is not None:
        r = bottleneck_rank
        if conv_type == "laguerre":
            compress = LaguerreConv3d(
                in_ch, r, kernel_size=(3, 1, 1), N_lag=N_lag,
                stride=stride, padding=(1, 0, 0), bias=False,
            )
        elif conv_type == "full":
            compress = LaguerreConv3d_Full(
                in_ch, r, kernel_size=(3, 3, 3), N_lag_T=N_lag,
                stride=stride, padding=(1, 1, 1), bias=False,
            )
        elif conv_type == "conv3d":
            compress = nn.Conv3d(
                in_ch, r, kernel_size=(3, 1, 1),
                stride=stride, padding=(1, 0, 0), bias=False,
            )
        else:
            raise ValueError(f"Unknown adapter conv_type: {conv_type!r}")

        expand = nn.Conv3d(r, expand_ch, kernel_size=1, bias=False)
        return nn.Sequential(compress, expand)

    # No bottleneck — single conv directly to expand_ch
    if conv_type == "laguerre":
        return LaguerreConv3d(
            in_ch, expand_ch, kernel_size=(3, 1, 1), N_lag=N_lag,
            stride=stride, padding=(1, 0, 0), bias=False,
        )
    elif conv_type == "full":
        return LaguerreConv3d_Full(
            in_ch, expand_ch, kernel_size=(3, 3, 3), N_lag_T=N_lag,
            stride=stride, padding=(1, 1, 1), bias=False,
        )
    elif conv_type == "conv3d":
        conv = nn.Conv3d(
            in_ch, expand_ch, kernel_size=(3, 1, 1),
            stride=stride, padding=(1, 0, 0), bias=False,
        )
        nn.init.zeros_(conv.weight)
        return conv
    else:
        raise ValueError(f"Unknown adapter conv_type: {conv_type!r}")


class PolynomialAdapter(nn.Module):
    """Wraps a pretrained block with a parallel polynomial (Volterra) branch."""

    def __init__(
        self,
        block: nn.Module,
        in_ch: int,
        out_ch: int,
        stride,
        Q: int = 4,
        conv_type: str = "laguerre",
        N_lag: int | None = None,
        bottleneck_rank: int | None = None,
    ):
        super().__init__()
        self.block  = block
        self.Q      = Q
        self.out_ch = out_ch

        self.adapter_conv = _build_adapter_conv(
            in_ch, 2 * Q * out_ch, stride, conv_type, N_lag, bottleneck_rank,
        )
        self.adapter_bn = nn.BatchNorm3d(out_ch)
        self.gate = nn.Parameter(torch.full((1,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        z = self.adapter_conv(x)
        z = volterra_quadratic(z, self.Q, self.out_ch)
        z = self.adapter_bn(z)
        return y + self.gate * z


def _block_dims(block: nn.Module):
    """Extract (in_ch, out_ch, stride) from an R3D BasicBlock."""
    return (
        block.conv1[0].in_channels,
        block.conv2[0].out_channels,
        block.conv1[0].stride,
    )


def inject_adapters(
    layer: nn.Sequential,
    Q: int,
    conv_type: str,
    N_lag: int | None,
    bottleneck_rank: int | None = None,
) -> nn.Sequential:
    """Wrap every block in *layer* with a PolynomialAdapter."""
    return nn.Sequential(*[
        PolynomialAdapter(
            block, *_block_dims(block),
            Q=Q, conv_type=conv_type, N_lag=N_lag, bottleneck_rank=bottleneck_rank,
        )
        for block in layer
    ])
