"""Triton kernels for the SLA2 router front-end.

The router needs, per (batch, head):
    Qbar = pool(Q)          : (B, H, Mb, D)  Mb = ceil(L/bq) query blocks
    Kbar = pool(K - mean K) : (B, H, Nb, D)  Nb = ceil(L/bk) key blocks

`mean_pool_smooth` fuses the smooth-K subtraction (SageAttention's K - colmean(K),
kept from SLA) into the pooling kernel, so the (B,H,L,D) centered copy of K is never
written to HBM.

Shapes: B = batch, H = heads, L = tokens, D = head dim.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _colmean(X, XM, L, D: tl.constexpr, BLOCK_L: tl.constexpr):
    """XM[b,h,d] = mean over tokens of X[b,h,:,d].  X: (B,H,L,D) -> XM: (B,H,D) fp32."""
    idx_bh = tl.program_id(0).to(tl.int64)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for start in tl.range(0, L, BLOCK_L):
        offs_l = start + tl.arange(0, BLOCK_L)
        x = tl.load(X + idx_bh * L * D + offs_l[:, None] * D + offs_d[None, :],
                    mask=offs_l[:, None] < L, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(XM + idx_bh * D + offs_d, acc / L)


@triton.jit
def _pool_sub(X, XM, XB, L, D: tl.constexpr, BLOCK_L: tl.constexpr, SUB_MEAN: tl.constexpr):
    """XB[b,h,j,d] = mean_{t in block j} (X[b,h,t,d] - XM[b,h,d]).

    X : (B,H,L,D); XM : (B,H,D) fp32 column means; XB : (B,H,ceil(L/BLOCK_L),D) fp32.
    """
    idx_l = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    n_blocks = tl.cdiv(L, BLOCK_L)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    x = tl.load(X + idx_bh * L * D + offs_l[:, None] * D + offs_d[None, :],
                mask=offs_l[:, None] < L, other=0.0).to(tl.float32)
    n = min(BLOCK_L, L - idx_l * BLOCK_L)
    xm = tl.sum(x, axis=0) / n
    if SUB_MEAN:
        xm -= tl.load(XM + idx_bh * D + offs_d)
    tl.store(XB + idx_bh * n_blocks * D + idx_l * D + offs_d, xm.to(XB.type.element_ty))


def mean_pool_smooth(x, BLK, sub_mean, out_dtype=torch.float32):
    """x: (B,H,L,D) -> (B,H,ceil(L/BLK),D) block means, optionally after centering x."""
    B, H, L, D = x.shape
    n_blocks = triton.cdiv(L, BLK)
    # column mean via torch's reduction (a single-CTA-per-head Triton loop would
    # leave the GPU idle); xm : (B,H,D) fp32
    xm = x.mean(dim=-2, dtype=torch.float32).contiguous() if sub_mean else \
        torch.empty((B, H, D), device=x.device, dtype=torch.float32)
    xb = torch.empty((B, H, n_blocks, D), device=x.device, dtype=out_dtype)
    _pool_sub[(n_blocks, B * H)](x, xm, xb, L, D, BLK, sub_mean, num_warps=4)
    return xb
