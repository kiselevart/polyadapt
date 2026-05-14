"""
Core polynomial primitives for polyadapt.

Ported from vnn/network/video_higher_order/{volterra_blocks,laguerre_conv}.py.
Only the math layers are included — VNN backbone/head classes are not.

Exports
-------
  volterra_quadratic          — CP-factorised 2nd-order Volterra interaction
  volterra_cubic_symmetric    — tied CP 3rd-order (a²·b)
  volterra_cubic_general      — independent CP 3rd-order (a·b·c)
  compute_laguerre_basis      — temporal Laguerre basis [N, T]
  LaguerreConv3d              — Conv3d with Laguerre temporal basis
  LaguerreConv3d_Full         — Conv3d with Tucker Laguerre basis on T, H, W
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Volterra interaction primitives
# ---------------------------------------------------------------------------

# Clamp values: chosen so a single un-summed product term is ≤ 50 in magnitude.
_QUAD_FC = 7.071   # √50
_CUB_FC  = 3.684   # ∛50


def volterra_quadratic(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """CP-factorised 2nd-order Volterra interaction.

    h2(i,j) ≈ Σ_q a_q(i)·b_q(j)

    Args:
        x_conv:  [B, 2*Q*C, T, H, W] — output of the quadratic expansion conv.
        Q:       Rank.
        nch_out: Output channels C.
    Returns:
        [B, C, T, H, W]
    """
    mid   = Q * nch_out
    left  = x_conv[:, :mid].clamp(-_QUAD_FC, _QUAD_FC)
    right = x_conv[:, mid:].clamp(-_QUAD_FC, _QUAD_FC)
    product = left * right  # [B, Q*C, T, H, W]
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1).clamp(-50.0, 50.0)


def volterra_cubic_symmetric(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """Symmetric 3rd-order Volterra interaction (a²·b, 2·Q·C channels).

    h3(i,j,k) ≈ Σ_q a_q(i)·a_q(j)·b_q(k)
    """
    mid = Q * nch_out
    a = x_conv[:, :mid].clamp(-_CUB_FC, _CUB_FC)
    b = x_conv[:, mid:].clamp(-_CUB_FC, _CUB_FC)
    product = (a * a) * b
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1).clamp(-50.0, 50.0)


def volterra_cubic_general(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """General 3rd-order Volterra interaction (a·b·c, 3·Q·C channels).

    h3(i,j,k) ≈ Σ_q a_q(i)·b_q(j)·c_q(k)
    """
    a, b, c = [t.clamp(-_CUB_FC, _CUB_FC) for t in torch.chunk(x_conv, 3, dim=1)]
    product = a * b * c
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1).clamp(-50.0, 50.0)


# ---------------------------------------------------------------------------
# Laguerre basis construction
# ---------------------------------------------------------------------------

def _laguerre_poly(n: int, t: torch.Tensor) -> torch.Tensor:
    """Evaluate L_n(t) via three-term recurrence."""
    if n == 0:
        return torch.ones_like(t)
    L_prev2 = torch.ones_like(t)
    L_prev1 = 1.0 - t
    if n == 1:
        return L_prev1
    for k in range(2, n + 1):
        L_curr = ((2 * k - 1 - t) * L_prev1 - (k - 1) * L_prev2) / k
        L_prev2 = L_prev1
        L_prev1 = L_curr
    return L_prev1


def compute_laguerre_basis(T: int, N: int, alpha: float = 1.0) -> torch.Tensor:
    """N L2-normalised Laguerre functions sampled at t = 0…T-1.

    φ_n(t) = L_n(αt) · exp(-αt/2)

    Returns: [N, T]
    """
    t = torch.arange(T, dtype=torch.float32) * alpha
    rows = []
    for n in range(N):
        phi = _laguerre_poly(n, t) * torch.exp(-t / 2.0)
        rows.append(phi / phi.norm().clamp(min=1e-8))
    return torch.stack(rows, dim=0)


def _compute_laguerre_basis_spatial(size: int, N: int, alpha: float = 1.0,
                                     center: bool = True) -> torch.Tensor:
    """Laguerre basis for a spatial dimension.

    center=True  — evaluate at |pos - size//2| * alpha (symmetric, rank ≤ (size+1)//2).
    center=False — positions {0,...,size-1} * alpha (same as temporal).

    Returns: [N, size]
    """
    pos = torch.arange(size, dtype=torch.float32)
    t = (pos - size // 2).abs() * alpha if center else pos * alpha
    rows = []
    for n in range(N):
        phi = _laguerre_poly(n, t) * torch.exp(-t / 2.0)
        rows.append(phi / phi.norm().clamp(min=1e-8))
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# LaguerreConv3d — temporal Laguerre basis
# ---------------------------------------------------------------------------

class LaguerreConv3d(nn.Module):
    """Conv3d whose temporal kernel is expressed in a Laguerre basis.

        W[o,i,t,h,w] = Σ_n  coeff[o,i,n,h,w] · basis[n,t]

    The basis is a fixed buffer; only ``coeff`` is learned.
    Laguerre coeff params are named ``.coeff`` so the optimizer can apply
    a multiplied LR with ``name.endswith('.coeff')``.

    Args:
        in_ch, out_ch:  Channel dims.
        kernel_size:    (T, H, W) or int. Only T is Laguerre-parameterised.
        N_lag:          Laguerre orders. None = T (full expressiveness).
        alpha:          Laguerre scale. Default 1.0.
        padding:        Passed to F.conv3d.
        bias:           Learnable bias.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size,
                 N_lag: int | None = None, alpha: float = 1.0,
                 stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        T, H, W = kernel_size
        if N_lag is None:
            N_lag = T
        self.in_ch       = in_ch
        self.out_ch      = out_ch
        self.kernel_size = kernel_size
        self.N_lag       = N_lag
        self.padding     = padding

        self.register_buffer("basis", compute_laguerre_basis(T, N_lag, alpha))

        self.coeff      = nn.Parameter(torch.empty(out_ch, in_ch, N_lag, H, W))
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride     = stride if isinstance(stride, tuple) else (stride,) * 3
        self._init_weights()

    def _init_weights(self):
        T, H, W = self.kernel_size
        nn.init.normal_(self.coeff, std=math.sqrt(2.0 / (self.in_ch * T * H * W)))

    def _get_kernel(self) -> torch.Tensor:
        # coeff: [O,I,N,H,W]  basis: [N,T]  →  kernel: [O,I,T,H,W]
        return torch.einsum("oinhw,nt->oithw", self.coeff, self.basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self._get_kernel(), self.bias_param,
                        stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# LaguerreConv3d_Full — Tucker Laguerre basis on T, H, W
# ---------------------------------------------------------------------------

class LaguerreConv3d_Full(nn.Module):
    """Conv3d with independent Laguerre bases on all three (T, H, W) dims.

        W[o,i,t,h,w] = Σ_{n,m,k} coeff[o,i,n,m,k] · φ_T[n,t] · φ_H[m,h] · φ_W[k,w]

    Parameter count per (out,in) pair: N_T×N_H×N_W vs T×H×W for Conv3d.
    ``coeff`` params end in `.coeff` for per-group LR scaling.

    Args:
        in_ch, out_ch:   Channel dims.
        kernel_size:     (T, H, W) or int.
        N_lag_T/H/W:     Laguerre orders per dim. None = full size (no compression).
        alpha_T/H/W:     Laguerre scales per dim.
        center_spatial:  Symmetric H/W basis (rank ≤ (size+1)//2 per dim). Default True.
        padding, bias:   Passed to F.conv3d.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size,
                 N_lag_T: int | None = None, N_lag_H: int | None = None,
                 N_lag_W: int | None = None,
                 alpha_T: float = 1.0, alpha_H: float = 1.0, alpha_W: float = 1.0,
                 center_spatial: bool = True, stride: int | tuple = 1,
                 padding: int | tuple = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        T, H, W = kernel_size
        N_T = N_lag_T if N_lag_T is not None else T
        N_H = N_lag_H if N_lag_H is not None else H
        N_W = N_lag_W if N_lag_W is not None else W

        self.in_ch       = in_ch
        self.out_ch      = out_ch
        self.kernel_size = kernel_size
        self.padding     = padding

        self.register_buffer("basis_T", compute_laguerre_basis(T, N_T, alpha_T))
        self.register_buffer("basis_H", _compute_laguerre_basis_spatial(H, N_H, alpha_H, center_spatial))
        self.register_buffer("basis_W", _compute_laguerre_basis_spatial(W, N_W, alpha_W, center_spatial))

        self.coeff      = nn.Parameter(torch.empty(out_ch, in_ch, N_T, N_H, N_W))
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride     = stride if isinstance(stride, tuple) else (stride,) * 3
        self._init_weights()

    def _init_weights(self):
        T, H, W = self.kernel_size
        nn.init.normal_(self.coeff, std=math.sqrt(2.0 / (self.in_ch * T * H * W)))

    def _get_kernel(self) -> torch.Tensor:
        k = torch.einsum("oinmk,nt->oitmk", self.coeff,  self.basis_T)
        k = torch.einsum("oitmk,mh->oithk", k,           self.basis_H)
        k = torch.einsum("oithk,kw->oithw", k,           self.basis_W)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self._get_kernel(), self.bias_param,
                        stride=self.stride, padding=self.padding)
