"""Polynomial Adapter — parallel Volterra branch for pretrained 3D CNN blocks.

Architecture per block:
    y = frozen_block(x)  +  gate * BN(volterra_quadratic(adapter_conv(x), Q, out_ch))

Gate initialises at 1e-2 so the model starts at pretrained baseline quality.
"""

import torch
import torch.nn as nn

from network.primitives import LaguerreConv3d, LaguerreConv3d_Full, volterra_quadratic


class PolynomialAdapter(nn.Module):
    """Wraps a pretrained block with a parallel polynomial (Volterra) branch."""

    def __init__(self, block: nn.Module, in_ch: int, out_ch: int, stride,
                 Q: int = 4, conv_type: str = "laguerre", N_lag: int | None = None):
        super().__init__()
        self.block  = block
        self.Q      = Q
        self.out_ch = out_ch

        if conv_type == "laguerre":
            self.adapter_conv = LaguerreConv3d(
                in_ch, 2 * Q * out_ch,
                kernel_size=(3, 1, 1),
                N_lag=N_lag,
                stride=stride,
                padding=(1, 0, 0),
                bias=False,
            )
        elif conv_type == "full":
            self.adapter_conv = LaguerreConv3d_Full(
                in_ch, 2 * Q * out_ch,
                kernel_size=(3, 3, 3),
                N_lag_T=N_lag,
                stride=stride,
                padding=(1, 1, 1),
                bias=False,
            )
        elif conv_type == "conv3d":
            self.adapter_conv = nn.Conv3d(
                in_ch, 2 * Q * out_ch,
                kernel_size=(3, 1, 1),
                stride=stride,
                padding=(1, 0, 0),
                bias=False,
            )
            nn.init.zeros_(self.adapter_conv.weight)
        else:
            raise ValueError(f"Unknown adapter conv_type: {conv_type!r}")

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
) -> nn.Sequential:
    """Wrap every block in *layer* with a PolynomialAdapter."""
    return nn.Sequential(*[
        PolynomialAdapter(block, *_block_dims(block), Q=Q, conv_type=conv_type, N_lag=N_lag)
        for block in layer
    ])
