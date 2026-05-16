"""Polynomial Adapter — parallel branch for pretrained 3D CNN blocks.

Four modes controlled by adapter_mode:
  'linear'     — standard LoRA (linear output)
  'cross_poly' — cross-channel Volterra quadratic: x → 2*Q shared interaction
                 channels, multiply, then project Q → out_ch via a pointwise
                 conv.  Output channel c = Σ_q W[c,q]·a_q(x)·b_q(x), so all
                 input channels mix polynomially into every output channel.
                 expand_ch = 2*Q (much smaller conv, separate out_proj).
  'relu'       — ablation: same bottleneck as cross_poly but ReLU instead of
                 left*right. x → conv(in_ch → Q) → ReLU → proj(Q → out_ch).
                 Tests whether the quadratic interaction is necessary or any
                 nonlinearity in a Q-dim bottleneck suffices.
  'poly'       — channel-parallel Volterra quadratic: each output channel c
                 gets Σ_q a_qc(x)·b_qc(x).  expand_ch = 2*Q*out_ch.

Conv is always 3×1×1 Conv3d (temporal only).

Architecture:
  no bottleneck:  conv(in_ch → expand_ch) → [interaction] → [out_proj] → BN → gate
  with bottleneck: compress(in_ch → r) → expand(r → expand_ch) → [interaction] → [out_proj] → BN → gate

Gate initialises at 0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.primitives import volterra_quadratic, volterra_quadratic_cross


def _make_conv(in_ch: int, out_ch: int, stride, bottleneck_rank: int | None) -> nn.Module:
    """Return a 3×1×1 Conv3d (or bottlenecked pair) mapping in_ch → out_ch."""
    if bottleneck_rank is not None:
        r = bottleneck_rank
        return nn.Sequential(
            nn.Conv3d(in_ch, r,      kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0), bias=False),
            nn.Conv3d(r,    out_ch,  kernel_size=1,         bias=False),
        )
    return nn.Conv3d(in_ch, out_ch, kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0), bias=False)


class PolynomialAdapter(nn.Module):
    """Parallel adapter branch attached to a frozen pretrained block."""

    def __init__(
        self,
        block: nn.Module,
        in_ch: int,
        out_ch: int,
        stride,
        Q: int = 4,
        bottleneck_rank: int | None = None,
        adapter_mode: str = "cross_poly",
    ):
        super().__init__()
        self.block        = block
        self.Q            = Q
        self.out_ch       = out_ch
        self.adapter_mode = adapter_mode

        if adapter_mode not in ("poly", "linear", "cross_poly", "relu"):
            raise ValueError(
                f"adapter_mode must be 'poly', 'linear', 'cross_poly', or 'relu', got {adapter_mode!r}"
            )

        if adapter_mode == "poly":
            expand_ch = 2 * Q * out_ch
        elif adapter_mode == "cross_poly":
            expand_ch = 2 * Q
        elif adapter_mode == "relu":
            expand_ch = Q
        else:  # linear
            expand_ch = out_ch

        self.adapter_conv = _make_conv(in_ch, expand_ch, stride, bottleneck_rank)

        self.out_proj: nn.Conv3d | None = (
            nn.Conv3d(Q, out_ch, kernel_size=1, bias=False)
            if adapter_mode in ("cross_poly", "relu") else None
        )

        self.adapter_bn = nn.BatchNorm3d(out_ch)
        self.gate = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        z = self.adapter_conv(x)
        if self.adapter_mode == "poly":
            z = volterra_quadratic(z, self.Q, self.out_ch)
        elif self.adapter_mode == "cross_poly":
            assert self.out_proj is not None
            z = self.out_proj(volterra_quadratic_cross(z, self.Q))
        elif self.adapter_mode == "relu":
            assert self.out_proj is not None
            z = self.out_proj(F.relu(z))
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
    bottleneck_rank: int | None = None,
    adapter_mode: str = "cross_poly",
) -> nn.Sequential:
    """Wrap every block in *layer* with an adapter."""
    return nn.Sequential(*[
        PolynomialAdapter(
            block, *_block_dims(block),
            Q=Q, bottleneck_rank=bottleneck_rank, adapter_mode=adapter_mode,
        )
        for block in layer
    ])
