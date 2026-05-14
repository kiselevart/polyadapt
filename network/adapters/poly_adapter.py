"""Polynomial Adapter — parallel branch for pretrained 3D CNN blocks.

Two modes controlled by adapter_mode:
  'poly'   (default) — Volterra quadratic output (our method)
  'linear'           — linear output, i.e. standard LoRA

Two conv types control the temporal parameterisation:
  'conv3d'   — standard 3×1×1 Conv3d
  'laguerre' — Laguerre temporal basis (3×1×1 kernel)
  'full'     — Tucker Laguerre basis (3×3×3 kernel)

This gives a clean 2×2 ablation:
  conv3d  + linear → pure LoRA
  laguerre + linear → LoRA with orthogonal temporal basis
  conv3d  + poly   → Volterra without Laguerre
  laguerre + poly  → full method

Architecture:
  no bottleneck:  adapter_conv(in_ch → expand_ch) → [volterra] → BN → gate
  with bottleneck: compress(in_ch → r) → expand(r → expand_ch) → [volterra] → BN → gate

  expand_ch = 2*Q*out_ch  (poly mode)
             = out_ch      (linear mode)

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
        elif conv_type in ("conv3d", "lora"):
            compress = nn.Conv3d(
                in_ch, r, kernel_size=(3, 1, 1),
                stride=stride, padding=(1, 0, 0), bias=False,
            )
        else:
            raise ValueError(f"Unknown adapter conv_type: {conv_type!r}")

        expand = nn.Conv3d(r, expand_ch, kernel_size=1, bias=False)
        return nn.Sequential(compress, expand)

    # No bottleneck
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
    elif conv_type in ("conv3d", "lora"):
        conv = nn.Conv3d(
            in_ch, expand_ch, kernel_size=(3, 1, 1),
            stride=stride, padding=(1, 0, 0), bias=False,
        )
        if expand_ch == in_ch:
            nn.init.zeros_(conv.weight)
        return conv
    else:
        raise ValueError(f"Unknown adapter conv_type: {conv_type!r}")


class PolynomialAdapter(nn.Module):
    """Parallel adapter branch attached to a frozen pretrained block."""

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
        adapter_mode: str = "poly",
    ):
        super().__init__()
        self.block       = block
        self.Q           = Q
        self.out_ch      = out_ch
        self.adapter_mode = adapter_mode

        if adapter_mode not in ("poly", "linear"):
            raise ValueError(f"adapter_mode must be 'poly' or 'linear', got {adapter_mode!r}")

        # poly: expand to 2*Q*out_ch → volterra_quadratic collapses back to out_ch
        # linear: expand directly to out_ch (standard LoRA)
        expand_ch = 2 * Q * out_ch if adapter_mode == "poly" else out_ch

        self.adapter_conv = _build_adapter_conv(
            in_ch, expand_ch, stride, conv_type, N_lag, bottleneck_rank,
        )
        self.adapter_bn = nn.BatchNorm3d(out_ch)
        self.gate = nn.Parameter(torch.full((1,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        z = self.adapter_conv(x)
        if self.adapter_mode == "poly":
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
    adapter_mode: str = "poly",
) -> nn.Sequential:
    """Wrap every block in *layer* with an adapter."""
    return nn.Sequential(*[
        PolynomialAdapter(
            block, *_block_dims(block),
            Q=Q, conv_type=conv_type, N_lag=N_lag,
            bottleneck_rank=bottleneck_rank, adapter_mode=adapter_mode,
        )
        for block in layer
    ])
