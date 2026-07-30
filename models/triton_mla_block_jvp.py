"""Triton forward-mode JVP (Jacobian-vector product) of the MLA
(Multi-head Latent Attention) transformer block of models/imf_dit_video.py,
MeanFlow-style: the tangent is taken w.r.t. the input x only, all parameter
tangents are zero.

Block (TransformerBlock in imf_dit_video.py):
    x -> RMSNorm1 -> MLAAttention (partial RoPE) -> * attn_scale -> +x
      -> RMSNorm2 -> SwiGLU MLP -> * mlp_scale -> +residual

MLAAttention:
    n1 (b,l,d) -> q_a (b,l,dq) -> q_a_layernorm -> q_b (b,l,H*(dn+dr))
    n1 (b,l,d) -> kv_a (b,l,dc+dr) -> [kv_latent (b,l,dc) | k_rope (b,l,dr)]
    kv_latent -> kv_a_layernorm -> kv_b (b,l,H*(dn+dv))
    q_nope/k_nope: per-head RMSNorm over dn;  q_rope/k_rope: RoPE rotation
    k_rope is ONE shared head broadcast over all H heads
    attention (head dim dn+dr = dv = 64) -> out_proj (H*dv -> d)

Shape symbols used in every comment below:
    b  : batch size (primal); the stacked primal+tangent batch is 2b
    l  : sequence length
    d  : hidden size (model width)
    H  : number of heads
    dq : q_lora_rank (query latent dim)
    dc : kv_lora_rank (key/value latent dim)
    dn : qk_nope_head_dim (position-free q/k channels per head)
    dr : qk_rope_head_dim (rotary q/k channels; dr2 = dr // 2 angle pairs)
    dv : v_head_dim
    F  : SwiGLU hidden width (padded to a multiple of 16 for fp8 GEMMs)
    M  : 2 * b * l stacked token rows (primal rows first, tangent rows after)

Because parameter tangents are zero, every linear applies the SAME weight to
primal and tangent, so each linear runs as ONE GEMM (general matrix multiply)
on the stacked batch. RoPE is linear in its input at fixed position, so the
tangent gets the same rotation as the primal.

fp32 io, two variants selected by `variant`:
  * "fp16" (default): fp16 weights + fp16 activations, cuBLAS fp16 GEMMs
    (fp32 accumulation), the fp16 flash-attention JVP core (fp16 operands,
    fp16-accumulate per-tile dots, fp32 softmax statistics and cross-tile
    accumulation). No fp8 anywhere; error is fp16-rounding-limited.
  * "fp8": all six linears in fp8 (float8_e4m3fn) tensor cores via
    torch._scaled_mm with per-tensor weight scales; activations quantized
    to fp8 inside the producing Triton kernels (unit scale, clamped).
In both variants the residual stream is fp16 with fp32 norm math in
registers, and both outputs are fp32.
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from models.triton_block_jvp import (
    _FP8,
    _flash_jvp,
    _fp8_mm,
    _fp8_weight,
    _stream_pair,
    _swiglu_jvp,
)

__all__ = ["triton_mla_block_jvp", "TritonMLABlockJVP", "mla_params_from_block"]


# --------------------------------------------------------------------------
# fp16 GEMM with fp16-accumulate tiles:  c[m, n] = sum_k a[m, k] * w[n, k]
# (einsum "mk,nk->mn", i.e. A @ W^T with W stored (N, K) row-major).
# Each BK-wide tl.dot accumulates in fp16 (2x MMA rate on GeForce, where
# cuBLAS fp16 GEMMs run fp32-accumulate at half rate); the cross-chunk
# accumulator is fp32, so per-element error stays at the fp16 BK-tile level.
# --------------------------------------------------------------------------
_MM16_CONFIGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=4, num_stages=3),
    triton.Config({"BM": 128, "BN": 64, "BK": 64}, num_warps=4, num_stages=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=4),
    triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 256, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN": 128, "BK": 128}, num_warps=8, num_stages=2),
    triton.Config({"BM": 64, "BN": 256, "BK": 64}, num_warps=4, num_stages=4),
    triton.Config({"BM": 256, "BN": 64, "BK": 64}, num_warps=4, num_stages=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 128}, num_warps=4, num_stages=3),
    triton.Config({"BM": 128, "BN": 64, "BK": 128}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_MM16_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mm16_kernel(A, W, C, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
                 GM: tl.constexpr = 8):
    # A (M, K) fp16 row-major;  W (N, K) fp16 row-major;  C (M, N) fp16
    # 1D grid with grouped block ordering (GM row-blocks per column sweep)
    # for L2 reuse of the W tiles.
    pid = tl.program_id(0)
    nbm = tl.cdiv(M, BM)
    nbn = tl.cdiv(N, BN)
    width = GM * nbn
    gid = pid // width
    first = gid * GM
    gsz = tl.minimum(nbm - first, GM)
    pid_m = first + (pid % width) % gsz
    pid_n = (pid % width) // gsz
    rm = pid_m * BM + tl.arange(0, BM)     # (BM,) output row index
    rn = pid_n * BN + tl.arange(0, BN)     # (BN,) output col index
    rk = tl.arange(0, BK)                  # (BK,) reduction index
    ap = A + rm[:, None] * K + rk[None, :]           # (BM, BK) A tile ptrs
    wp = W + rn[:, None] * K + rk[None, :]           # (BN, BK) W tile ptrs
    acc = tl.zeros([BM, BN], tl.float32)   # (BM, BN) cross-chunk accumulator
    for _ in range(0, K, BK):              # K is a multiple of BK for all uses
        if EVEN_M:
            a = tl.load(ap)
        else:
            a = tl.load(ap, mask=rm[:, None] < M, other=0.0)
        if EVEN_N:
            w = tl.load(wp)
        else:
            w = tl.load(wp, mask=rn[:, None] < N, other=0.0)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float16).to(tl.float32)
        ap += BK
        wp += BK
    cp = C + rm[:, None] * N + rn[None, :]           # (BM, BN) C tile ptrs
    if EVEN_M and EVEN_N:
        tl.store(cp, acc.to(C.dtype.element_ty))
    else:
        tl.store(cp, acc.to(C.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def _mm16(a, w_nk):
    """a (M, K) fp16 @ w_nk (N, K) fp16 -> (M, N) fp16 (A @ W^T).
    K must be a multiple of 128 (largest BK); M and N are masked."""
    M, K = a.shape
    N = w_nk.shape[0]
    c = torch.empty(M, N, device=a.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),
    )
    _mm16_kernel[grid](
        a, w_nk, c, M, N, K,
        EVEN_M=(M % 256 == 0), EVEN_N=(N % 256 == 0),
    )
    return c


# --------------------------------------------------------------------------
# Down-projection GEMM with fused gated-residual epilogue:
#   c[m, n] = sum_k act[m, k] * w_down[n, k]      (einsum "mk,nk->mn")
#   primal rows (m < half):  y[m, n]  = res[m, n]  + c[m, n] * g[n]
#   tangent rows (m >= half): dy[m', n] = dres[m', n] + c[m, n] * g[n]
# fp16-accumulate tiles like _mm16; outputs written fp32 directly, so the
# intermediate down-projection buffer, its re-read and the separate
# gate-add launch all disappear.
# --------------------------------------------------------------------------
@triton.autotune(configs=_MM16_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mm16_gate_add_kernel(A, W, RES, G, Y, DY, M, N, K, half,
                          BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr,
                          EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
                          GM: tl.constexpr = 8):
    # A (M, K) fp16;  W (N, K) fp16;  RES (M, N) fp16 stacked residual
    # G (N,) fp32 gate;  Y/DY (half, N) fp32 primal/tangent outputs
    # 1D grid, grouped ordering as in _mm16_kernel.
    pid = tl.program_id(0)
    nbm = tl.cdiv(M, BM)
    nbn = tl.cdiv(N, BN)
    width = GM * nbn
    gid = pid // width
    first = gid * GM
    gsz = tl.minimum(nbm - first, GM)
    pid_m = first + (pid % width) % gsz
    pid_n = (pid % width) // gsz
    rm = pid_m * BM + tl.arange(0, BM)     # (BM,) stacked row index
    rn = pid_n * BN + tl.arange(0, BN)     # (BN,) output col index
    rk = tl.arange(0, BK)                  # (BK,) reduction index
    ap = A + rm[:, None] * K + rk[None, :]           # (BM, BK) A tile ptrs
    wp = W + rn[:, None] * K + rk[None, :]           # (BN, BK) W tile ptrs
    acc = tl.zeros([BM, BN], tl.float32)   # (BM, BN) cross-chunk accumulator
    for _ in range(0, K, BK):
        if EVEN_M:
            a = tl.load(ap)
        else:
            a = tl.load(ap, mask=rm[:, None] < M, other=0.0)
        if EVEN_N:
            w = tl.load(wp)
        else:
            w = tl.load(wp, mask=rn[:, None] < N, other=0.0)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float16).to(tl.float32)
        ap += BK
        wp += BK
    nmask = rn[None, :] < N
    mmask = rm[:, None] < M
    g = tl.load(G + rn, mask=rn < N, other=0.0).to(tl.float32)   # (BN,)
    rp = RES + rm[:, None] * N + rn[None, :]         # (BM, BN) residual ptrs
    r = tl.load(rp, mask=mmask & nmask, other=0.0).to(tl.float32)
    outv = r + acc * g[None, :]            # (BM, BN) gated residual output
    # A BM block may straddle the primal/tangent boundary: two masked
    # stores, one per half, with per-row destination row indices.
    p_mask = (rm[:, None] < half) & nmask
    t_row = tl.maximum(rm - half, 0)       # (BM,) tangent-half row index
    t_mask = (rm[:, None] >= half) & mmask & nmask
    tl.store(Y + rm[:, None] * N + rn[None, :], outv, mask=p_mask)
    tl.store(DY + t_row[:, None] * N + rn[None, :], outv, mask=t_mask)


def _mm16_gate_add(a, w_nk, res, g, out_y, out_dy, half):
    """a (M, K) fp16 @ w_nk (N, K) fp16, fused y/dy = res + c*g epilogue.

    res (M, N) fp16 stacked residual; g (N,) fp32 gate;
    out_y/out_dy (half, N) fp32 output views; half = M // 2.
    """
    M, K = a.shape
    N = w_nk.shape[0]
    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),
    )
    _mm16_gate_add_kernel[grid](
        a, w_nk, res, g, out_y, out_dy, M, N, K, half,
        EVEN_M=(M % 256 == 0), EVEN_N=(N % 256 == 0),
    )


# --------------------------------------------------------------------------
# Strided RMSNorm JVP:  r = sqrt(mean(x^2)+eps);  y = w*x/r
#                       dy = w*(dx/r - x*mean(x*dx)/r^3)
# In/out row strides are free so the kernel can read a D-wide band out of a
# wider GEMM output (q_a_layernorm and kv_a_layernorm read slices of the
# fused q_a/kv_a projection) and write a contiguous fp8 buffer.
# --------------------------------------------------------------------------
@triton.jit
def _rmsnorm_jvp_strided_kernel(X, DX, W, Y, DY, D, sx, sy, eps,
                                BLOCK: tl.constexpr, CLAMP: tl.constexpr):
    # X, DX: primal/tangent input rows, row stride sx, D valid columns
    # Y, DY: primal/tangent output rows, row stride sy
    # W: (D,) norm gain
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)                     # (BLOCK,) channel index
    mask = cols < D
    x = tl.load(X + row * sx + cols, mask=mask, other=0.0).to(tl.float32)
    dx = tl.load(DX + row * sx + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    inv_r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)   # scalar 1/r
    mxdx = tl.sum(x * dx, axis=0) / D                        # scalar mean(x*dx)
    y = w * x * inv_r                                        # (BLOCK,)
    dy = w * (dx * inv_r - x * (mxdx * inv_r * inv_r * inv_r))
    if CLAMP:
        y = tl.clamp(y, -448.0, 448.0)
        dy = tl.clamp(dy, -448.0, 448.0)
    tl.store(Y + row * sy + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(DY + row * sy + cols, dy.to(DY.dtype.element_ty), mask=mask)


def _rmsnorm_jvp_strided(x, dx, w, y, dy, d, sx, sy, eps, n_rows):
    # x/dx: pointers to primal/tangent rows (n_rows each, row stride sx)
    # y/dy: output pointers (row stride sy); d: normalized width
    BLOCK = triton.next_power_of_2(d)
    num_warps = 4 if BLOCK <= 1024 else 8
    _rmsnorm_jvp_strided_kernel[(n_rows,)](
        x, dx, w, y, dy, d, sx, sy, eps, BLOCK=BLOCK, num_warps=num_warps,
        CLAMP=(y.dtype == _FP8),
    )


# --------------------------------------------------------------------------
# q head prep: per (token row, head) take the q_b output row
#   [q_nope (dn) | q_rope (dr)],
# apply per-head RMSNorm JVP to the nope band and the RoPE rotation JVP to
# the rope band, and store the assembled (dn+dr)-wide head into the packed
# fp8 q slot of the (2b, l, 3, H, 64) attention buffer.
# RoPE pairs adjacent channels: (x[2i], x[2i+1]) rotated by angle[pos, i].
# --------------------------------------------------------------------------
@triton.jit
def _q_prep_kernel(QB, WQN, COS, SIN, OUT,
                   sq_row,           # row stride of QB (= H * (dn+dr))
                   so_b, so_s,       # batch/seq strides of packed out buffer
                   B, S, half,       # batch, seq len, half = B*S primal rows
                   eps,
                   NH: tl.constexpr,     # number of heads H
                   DN: tl.constexpr,     # nope band width dn
                   DR2: tl.constexpr,    # rope angle pairs dr // 2
                   HD: tl.constexpr,     # head dim dn + dr (= 64)
                   BN: tl.constexpr,     # next_power_of_2(DN)
                   CLAMP: tl.constexpr): # clamp outputs into the fp8 range
    # One program per token row; all NH heads processed as a 2D tile.
    pid = tl.program_id(0)             # token row in [0, B*S)
    b = pid // S                       # batch index
    s = pid % S                        # sequence position
    src_p = pid * sq_row               # primal row offset in QB
    src_t = (pid + half) * sq_row      # tangent row offset in QB
    dst_p = b * so_b + s * so_s        # primal row offset in OUT q slot
    dst_t = (b + B) * so_b + s * so_s

    h2 = tl.arange(0, NH)[:, None]     # (NH, 1) head index
    cn = tl.arange(0, BN)[None, :]     # (1, BN) nope channel index
    nm = cn < DN
    off_n = h2 * HD + cn               # (NH, BN) nope offsets within row
    xn = tl.load(QB + src_p + off_n, mask=nm, other=0.0).to(tl.float32)
    dn_ = tl.load(QB + src_t + off_n, mask=nm, other=0.0).to(tl.float32)
    w = tl.load(WQN + cn, mask=nm, other=0.0).to(tl.float32)   # (1, BN)
    inv_r = 1.0 / tl.sqrt(tl.sum(xn * xn, axis=1) / DN + eps)  # (NH,)
    mxdx = tl.sum(xn * dn_, axis=1) / DN                       # (NH,)
    ir = inv_r[:, None]                # (NH, 1)
    yn = w * xn * ir
    dyn = w * (dn_ * ir - xn * (mxdx[:, None] * ir * ir * ir))
    if CLAMP:
        yn = tl.clamp(yn, -448.0, 448.0)
        dyn = tl.clamp(dyn, -448.0, 448.0)
    tl.store(OUT + dst_p + off_n, yn.to(OUT.dtype.element_ty), mask=nm)
    tl.store(OUT + dst_t + off_n, dyn.to(OUT.dtype.element_ty), mask=nm)

    # ---- rope band: rotation JVP (same rotation on primal and tangent) ----
    i = tl.arange(0, DR2)[None, :]     # (1, DR2) angle-pair index
    c = tl.load(COS + s * DR2 + i).to(tl.float32)   # (1, DR2) cos(angle)
    sn = tl.load(SIN + s * DR2 + i).to(tl.float32)  # (1, DR2) sin(angle)
    re = h2 * HD + DN + 2 * i          # (NH, DR2) real-part offsets
    im = re + 1                        # (NH, DR2) imag-part offsets
    xr = tl.load(QB + src_p + re).to(tl.float32)
    xi = tl.load(QB + src_p + im).to(tl.float32)
    dxr = tl.load(QB + src_t + re).to(tl.float32)
    dxi = tl.load(QB + src_t + im).to(tl.float32)
    yr = xr * c - xi * sn
    yi = xr * sn + xi * c
    dyr = dxr * c - dxi * sn
    dyi = dxr * sn + dxi * c
    if CLAMP:
        yr = tl.clamp(yr, -448.0, 448.0)
        yi = tl.clamp(yi, -448.0, 448.0)
        dyr = tl.clamp(dyr, -448.0, 448.0)
        dyi = tl.clamp(dyi, -448.0, 448.0)
    tl.store(OUT + dst_p + re, yr.to(OUT.dtype.element_ty))
    tl.store(OUT + dst_p + im, yi.to(OUT.dtype.element_ty))
    tl.store(OUT + dst_t + re, dyr.to(OUT.dtype.element_ty))
    tl.store(OUT + dst_t + im, dyi.to(OUT.dtype.element_ty))


# --------------------------------------------------------------------------
# k/v head prep: per (token row, head) take the kv_b output row
#   [k_nope (dn) | v (dv)],
# apply per-head RMSNorm JVP to k_nope, copy v, rotate the SHARED k_rope
# band (read from the fused q_a/kv_a GEMM output), and store the assembled
# k = [k_nope | k_rope] and v heads into the packed fp8 k/v slots.
# --------------------------------------------------------------------------
@triton.jit
def _kv_prep_kernel(KVB, KR, WKN, COS, SIN, OUTK, OUTV,
                    skv_row,          # row stride of KVB (= H * (dn+dv))
                    skr_row,          # row stride of KR (fused GEMM width)
                    so_b, so_s,       # batch/seq strides of packed out buffer
                    B, S, half,       # batch, seq len, half = B*S primal rows
                    eps,
                    NH: tl.constexpr,     # number of heads H
                    DN: tl.constexpr,     # nope band width dn
                    DR2: tl.constexpr,    # rope angle pairs dr // 2
                    DV: tl.constexpr,     # value head dim dv (= 64)
                    HD: tl.constexpr,     # qk head dim dn + dr (= 64)
                    BN: tl.constexpr,     # next_power_of_2(DN)
                    KVW: tl.constexpr,    # kv_b per-head width dn + dv
                    CLAMP: tl.constexpr): # clamp outputs into the fp8 range
    # One program per token row; all NH heads processed as a 2D tile.
    pid = tl.program_id(0)             # token row in [0, B*S)
    b = pid // S
    s = pid % S
    src_p = pid * skv_row              # primal row offset in KVB
    src_t = (pid + half) * skv_row     # tangent row offset in KVB
    dst_p = b * so_b + s * so_s        # primal row offset in OUT k/v slots
    dst_t = (b + B) * so_b + s * so_s

    h2 = tl.arange(0, NH)[:, None]     # (NH, 1) head index
    cn = tl.arange(0, BN)[None, :]     # (1, BN) nope channel index
    nm = cn < DN
    off_in = h2 * KVW + cn             # (NH, BN) k_nope offsets in KVB row
    off_out = h2 * HD + cn             # (NH, BN) k_nope offsets in OUT row
    xn = tl.load(KVB + src_p + off_in, mask=nm, other=0.0).to(tl.float32)
    dn_ = tl.load(KVB + src_t + off_in, mask=nm, other=0.0).to(tl.float32)
    w = tl.load(WKN + cn, mask=nm, other=0.0).to(tl.float32)   # (1, BN)
    inv_r = 1.0 / tl.sqrt(tl.sum(xn * xn, axis=1) / DN + eps)  # (NH,)
    mxdx = tl.sum(xn * dn_, axis=1) / DN                       # (NH,)
    ir = inv_r[:, None]                # (NH, 1)
    yn = w * xn * ir
    dyn = w * (dn_ * ir - xn * (mxdx[:, None] * ir * ir * ir))
    if CLAMP:
        yn = tl.clamp(yn, -448.0, 448.0)
        dyn = tl.clamp(dyn, -448.0, 448.0)
    tl.store(OUTK + dst_p + off_out, yn.to(OUTK.dtype.element_ty), mask=nm)
    tl.store(OUTK + dst_t + off_out, dyn.to(OUTK.dtype.element_ty), mask=nm)

    # ---- shared k_rope: RoPE rotation JVP, broadcast over heads ----
    i = tl.arange(0, DR2)[None, :]     # (1, DR2) angle-pair index
    c = tl.load(COS + s * DR2 + i).to(tl.float32)   # (1, DR2)
    sn = tl.load(SIN + s * DR2 + i).to(tl.float32)  # (1, DR2)
    xr = tl.load(KR + pid * skr_row + 2 * i).to(tl.float32)          # (1, DR2)
    xi = tl.load(KR + pid * skr_row + 2 * i + 1).to(tl.float32)
    dxr = tl.load(KR + (pid + half) * skr_row + 2 * i).to(tl.float32)
    dxi = tl.load(KR + (pid + half) * skr_row + 2 * i + 1).to(tl.float32)
    yr = (xr * c - xi * sn) + tl.zeros([NH, DR2], tl.float32)   # broadcast
    yi = (xr * sn + xi * c) + tl.zeros([NH, DR2], tl.float32)
    dyr = (dxr * c - dxi * sn) + tl.zeros([NH, DR2], tl.float32)
    dyi = (dxr * sn + dxi * c) + tl.zeros([NH, DR2], tl.float32)
    # ---- v: straight copy of the dv band (JVP of a copy is a copy) ----
    cv = tl.arange(0, DV)[None, :]     # (1, DV) value channel index
    off_v_in = h2 * KVW + DN + cv      # (NH, DV) v offsets in KVB row
    off_v_out = h2 * DV + cv           # (NH, DV) v offsets in OUT v slot
    v = tl.load(KVB + src_p + off_v_in).to(tl.float32)
    dv_ = tl.load(KVB + src_t + off_v_in).to(tl.float32)
    if CLAMP:
        yr = tl.clamp(yr, -448.0, 448.0)
        yi = tl.clamp(yi, -448.0, 448.0)
        dyr = tl.clamp(dyr, -448.0, 448.0)
        dyi = tl.clamp(dyi, -448.0, 448.0)
        v = tl.clamp(v, -448.0, 448.0)
        dv_ = tl.clamp(dv_, -448.0, 448.0)
    re = h2 * HD + DN + 2 * i          # (NH, DR2) real-part offsets in OUT
    im = re + 1
    tl.store(OUTK + dst_p + re, yr.to(OUTK.dtype.element_ty))
    tl.store(OUTK + dst_p + im, yi.to(OUTK.dtype.element_ty))
    tl.store(OUTK + dst_t + re, dyr.to(OUTK.dtype.element_ty))
    tl.store(OUTK + dst_t + im, dyi.to(OUTK.dtype.element_ty))
    tl.store(OUTV + dst_p + off_v_out, v.to(OUTV.dtype.element_ty))
    tl.store(OUTV + dst_t + off_v_out, dv_.to(OUTV.dtype.element_ty))


# --------------------------------------------------------------------------
# Row-based SituAndMul JVP (Kimi-K3 bounded-SiLU gate):
#   a(g)  = beta * tanh(g/beta) * sigmoid(g)
#   a'(g) = sech^2(g/beta) * sigmoid(g)
#           + beta * tanh(g/beta) * sigmoid(g) * (1 - sigmoid(g))
#   h = a(g)*u,  dh = a'(g)*dg*u + a(g)*du
# GU rows are [gate | up] of width 2F; one program per (row, F-chunk) with
# plain vector loads (no per-element div/mod as in the flat kernel).
# --------------------------------------------------------------------------
@triton.jit
def _swiglu_row_jvp_kernel(GU, DGU, Hout, DHout, F, beta,
                           BLOCK: tl.constexpr, CLAMP: tl.constexpr):
    # GU/DGU: primal/tangent rows of width 2F ([gate | up]); Hout/DHout:
    # output rows of width F;  beta: SituAndMul gate cap
    row = tl.program_id(0)             # token row index
    cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)   # (BLOCK,) col
    mask = cb < F
    base = row * (2 * F) + cb
    g = tl.load(GU + base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + base + F, mask=mask, other=0.0).to(tl.float32)
    dg = tl.load(DGU + base, mask=mask, other=0.0).to(tl.float32)
    du = tl.load(DGU + base + F, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    # tanh(x) = 2*sigmoid(2x) - 1 (tl.math.tanh unavailable in this Triton)
    th = 2.0 * tl.sigmoid(2.0 * g / beta) - 1.0
    act = beta * th * sig                        # a(g)
    dact = (1.0 - th * th) * sig + beta * th * sig * (1.0 - sig)   # a'(g)
    h = act * u
    dh = dact * dg * u + act * du
    if CLAMP:
        h = tl.clamp(h, -448.0, 448.0)
        dh = tl.clamp(dh, -448.0, 448.0)
    op = row * F + cb
    tl.store(Hout + op, h.to(Hout.dtype.element_ty), mask=mask)
    tl.store(DHout + op, dh.to(DHout.dtype.element_ty), mask=mask)


def _swiglu_row_jvp(gu, dgu, h, dh, n_rows, f, beta=1.0):
    # gu/dgu: (n_rows, 2F) fp16 primal/tangent; h/dh: (n_rows, F) outputs
    BLOCK = 1024
    _swiglu_row_jvp_kernel[(n_rows, triton.cdiv(f, BLOCK))](
        gu, dgu, h, dh, f, beta, BLOCK=BLOCK, num_warps=4,
        CLAMP=(h.dtype == _FP8),
    )


# --------------------------------------------------------------------------
# MLA output-gate JVP (Kimi-K3 mla_use_output_gate), in place on the
# attention output buffer:
#   a_new  = a * sigmoid(z)
#   da_new = da * sigmoid(z) + a * sigmoid(z) * (1 - sigmoid(z)) * dz
# --------------------------------------------------------------------------
@triton.jit
def _out_gate_jvp_kernel(A, DA, Z, DZ, total, BLOCK: tl.constexpr,
                         CLAMP: tl.constexpr):
    # A/DA: (total,) primal/tangent attention output (modified in place)
    # Z/DZ: (total,) primal/tangent gate pre-activations
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # flat element index
    mask = i < total
    a = tl.load(A + i, mask=mask, other=0.0).to(tl.float32)
    da = tl.load(DA + i, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + i, mask=mask, other=0.0).to(tl.float32)
    dz = tl.load(DZ + i, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(z)
    out = a * sig
    dout = da * sig + a * sig * (1.0 - sig) * dz
    if CLAMP:
        out = tl.clamp(out, -448.0, 448.0)
        dout = tl.clamp(dout, -448.0, 448.0)
    tl.store(A + i, out.to(A.dtype.element_ty), mask=mask)
    tl.store(DA + i, dout.to(DA.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------
# Fused gated final add:  y = res + dwn * g   (fp32 out, both halves in one
# launch; avoids the fp32 temporary that `res + dwn * g` in torch would
# materialize for the product).
# --------------------------------------------------------------------------
@triton.jit
def _gate_add_kernel(RES, DRES, DWN, DDWN, G, Y, DY, D, total,
                     BLOCK: tl.constexpr):
    # RES/DRES: (total,) fp16 primal/tangent residual rows (flattened)
    # DWN/DDWN: (total,) fp16 primal/tangent down-projection rows
    # G: (D,) fp32 vector gate;  Y/DY: (total,) fp32 outputs
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # flat element index
    mask = i < total
    col = i % D                                          # channel index
    g = tl.load(G + col, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + i, mask=mask, other=0.0).to(tl.float32)
    dr_ = tl.load(DRES + i, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(DWN + i, mask=mask, other=0.0).to(tl.float32)
    dw = tl.load(DDWN + i, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + i, r + w * g, mask=mask)
    tl.store(DY + i, dr_ + dw * g, mask=mask)


# --------------------------------------------------------------------------
# Fused gated-residual + RMSNorm JVP:
#   res = x + p * g          (vector gate g, zero parameter tangent)
#   y   = RMSNorm(res) JVP   (fp8 out, feeding the next GEMM)
# res/dres stored fp16; the norm reads the pre-rounding fp32 sum.
# --------------------------------------------------------------------------
@triton.jit
def _res_gate_norm_jvp_kernel(P, DP, G, X, DX, W, RES, DRES, Y, DY, D, eps,
                              BLOCK: tl.constexpr, CLAMP: tl.constexpr):
    # P, DP: primal/tangent branch output rows (D,)
    # G: (D,) residual vector gate;  X, DX: primal/tangent block input rows
    # W: (D,) norm gain;  RES, DRES: fp16 residual out;  Y, DY: fp8 norm out
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    p = tl.load(P + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    dp = tl.load(DP + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    dx = tl.load(DX + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + p * g                       # (BLOCK,) gated residual, primal
    dx = dx + dp * g                    # (BLOCK,) gated residual, tangent
    tl.store(RES + row * D + cols, x.to(RES.dtype.element_ty), mask=mask)
    tl.store(DRES + row * D + cols, dx.to(DRES.dtype.element_ty), mask=mask)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    inv_r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    mxdx = tl.sum(x * dx, axis=0) / D
    y = w * x * inv_r
    dy = w * (dx * inv_r - x * (mxdx * inv_r * inv_r * inv_r))
    if CLAMP:
        y = tl.clamp(y, -448.0, 448.0)
        dy = tl.clamp(dy, -448.0, 448.0)
    tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(DY + row * D + cols, dy.to(DY.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------
# Full MLA block JVP (fp32 io fast path)
# --------------------------------------------------------------------------
def _pad_rows_128(w):
    """Zero-pad the leading (output) dim of w (N, K) to a multiple of 128
    (the largest _mm16 K-chunk; also covers the fp8 16-alignment).
    Padded gate/up rows give silu(0)*0 = 0 through SwiGLU, and padded down
    columns then multiply those exact zeros, so the block output is
    unchanged."""
    n = w.shape[0]
    n_pad = (n + 127) // 128 * 128
    if n_pad == n:
        return w
    return torch.cat([w, w.new_zeros(n_pad - n, w.shape[1])], dim=0)


def _pad_cols_128(w):
    """Zero-pad the trailing (input) dim of w (N, K) to a multiple of 128."""
    k = w.shape[1]
    k_pad = (k + 127) // 128 * 128
    if k_pad == k:
        return w
    return torch.cat([w, w.new_zeros(w.shape[0], k_pad - k)], dim=1)


def _mla_fused_weights(p):
    """Fuse/pad the six GEMM weights (fp32, (out_features, in_features)):
        w_a       (dq + dc + dr, d)  fused q_a_proj + kv_a_proj
        w_qb      (H * (dn+dr), dq)
        w_kvb     (H * (dn+dv), dc)
        w_out     (d, H * dv)
        w_gate_up (2 * F, d)         F padded to a multiple of 16
        w_down    (d, F)
    """
    out = {
        "w_a": torch.cat([p["w_qa"], p["w_kva"]], dim=0),
        "w_qb": p["w_qb"],
        "w_kvb": p["w_kvb"],
        "w_out": p["w_out"],
        "w_gate_up": torch.cat(
            [_pad_rows_128(p["w_gate"]), _pad_rows_128(p["w_up"])], dim=0
        ),
        "w_down": _pad_cols_128(p["w_down"]),
    }
    if p.get("w_og") is not None:
        out["w_og"] = p["w_og"]            # (H*dv, d) MLA output gate
    return out


def _mla_fp8_cache(p):
    """fp8-quantize the fused weights with per-tensor amax scales.
    Values are (w8 (N, K) fp8, descale (1,) fp32) pairs."""
    return {k: _fp8_weight(w) for k, w in _mla_fused_weights(p).items()}


def _mla_fp16_cache(p):
    """fp16 copies of the fused weights, each (N, K) fp16."""
    return {k: w.detach().half() for k, w in _mla_fused_weights(p).items()}


def triton_mla_block_jvp(x, dx, params, rope_cos, rope_sin, eps=1e-6,
                         variant="fp16"):
    """Forward-mode JVP of the MLA transformer block w.r.t. the input only.

    Args:
        x, dx: (b, l, d) fp32 CUDA contiguous primal input and tangent.
        params: dict with keys (shapes as in _mla_fused_weights, plus)
            w_norm1 (d,), w_qa (dq, d), w_kva (dc+dr, d), w_qa_ln (dq,),
            w_kva_ln (dc,), w_qb (H*(dn+dr), dq), w_kvb (H*(dn+dv), dc),
            w_qnorm (dn,), w_knorm (dn,), w_out (d, H*dv), g_attn (d,),
            w_norm2 (d,), w_gate (F0, d), w_up (F0, d), w_down (d, F0),
            g_mlp (d,), and ints H, dq, dc, dn, dr, dv.
            Optional "_fp8" / "_fp16": cached weight dicts.
        rope_cos, rope_sin: (l, dr // 2) fp32 rotary angle tables.
        variant: "fp16" (default; fp16 weights/activations, cuBLAS fp16
            GEMMs, no fp8 anywhere) or "fp8" (fp8 GEMMs + fp8 activations).

    Returns:
        (y, dy): each (b, l, d) fp32.
    """
    assert x.is_cuda and x.is_contiguous() and dx.is_contiguous()
    assert x.dtype == torch.float32
    assert variant in ("fp16", "fp8")
    B = x.shape[0]                         # b
    y = torch.empty_like(x)                # (b, l, d) fp32 primal out
    dy = torch.empty_like(x)               # (b, l, d) fp32 tangent out
    if B >= 2 and B % 2 == 0:
        # Two batch chunks on two CUDA streams: attention is per-batch
        # independent, so chunking is exact; the memory-bound stages of one
        # chunk overlap the tensor-core stages of the other.
        h_b = B // 2                       # chunk batch size
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)
        capturing = torch.cuda.is_current_stream_capturing()
        for st, sl in zip(_stream_pair(x.device),
                          (slice(0, h_b), slice(h_b, B))):
            st.wait_event(ev)
            with torch.cuda.stream(st):
                if not capturing:
                    # allocator cross-stream safety; disallowed (and
                    # unnecessary, graphs own their pool) during capture
                    x.record_stream(st)
                    dx.record_stream(st)
                _mla_chunk(x[sl], dx[sl], params, rope_cos, rope_sin, eps,
                           variant, y[sl], dy[sl])
            cur.wait_stream(st)
    else:
        _mla_chunk(x, dx, params, rope_cos, rope_sin, eps, variant, y, dy)
    return y, dy


def _mla_chunk(x, dx, params, rope_cos, rope_sin, eps, variant, out_y, out_dy):
    """One batch chunk of the MLA block JVP pipeline.

    x, dx: (b, l, d) fp32 contiguous-in-row views (batch slices of the full
    input are fine: slicing dim 0 keeps rows contiguous).
    out_y, out_dy: (b, l, d) fp32 views to write the outputs into.
    """
    B, S, D = x.shape                      # b, l, d
    H = params["H"]
    dq, dc = params["dq"], params["dc"]
    dn, dr, dv = params["dn"], params["dr"], params["dv"]
    hd = dn + dr                           # qk head dim
    assert hd == 64 and dv == 64, "flash JVP core is compiled for head dim 64"
    scale = 1.0 / math.sqrt(hd)            # softmax scale 1/sqrt(dn+dr)
    dev = x.device
    half = B * S                           # primal token rows
    M = 2 * half                           # stacked rows

    if variant == "fp8":
        w = params.get("_fp8") or _mla_fp8_cache(params)
        act_dt = _FP8                      # activation buffer dtype
        f = w["w_gate_up"][0].shape[0] // 2  # padded SwiGLU width F

        def mm(a2d, key):
            # a2d (M, K) fp8 @ cached fp8 weight (N, K) -> (M, N) fp16
            return _fp8_mm(a2d, w[key], torch.float16)
    else:
        w = params.get("_fp16") or _mla_fp16_cache(params)
        act_dt = torch.float16
        f = w["w_gate_up"].shape[0] // 2   # padded SwiGLU width F

        def mm(a2d, key):
            # a2d (M, K) fp16 @ fp16 weight (N, K) -> (M, N) fp16 via the
            # Triton fp16-accumulate GEMM (2x the cuBLAS fp32-acc rate)
            return _mm16(a2d, w[key])

    clamp = variant == "fp8"               # fp8 range clamp in prep kernels

    # ---- RMSNorm1 JVP: (b,l,d) fp32 -> stacked (2b,l,d) act_dt ----
    xs = torch.empty(2 * B, S, D, device=dev, dtype=act_dt)
    _rmsnorm_jvp_strided(x, dx, params["w_norm1"], xs[:B], xs[B:],
                         D, D, D, eps, half)

    # ---- fused q_a + kv_a projection: one stacked GEMM ----
    # a[m, j] = sum_k n1[m, k] * w_a[j, k]   (einsum "mk,jk->mj")
    # columns: [q_latent (dq) | kv_latent (dc) | k_rope (dr)]
    wa = dq + dc + dr                      # fused output width
    a = mm(xs.view(M, D), "w_a")           # (M, wa) fp16

    # ---- latent RMSNorms (strided reads out of `a`, act_dt out) ----
    q_lat = torch.empty(M, dq, device=dev, dtype=act_dt)
    _rmsnorm_jvp_strided(a, a[half:], params["w_qa_ln"],
                         q_lat, q_lat[half:], dq, wa, dq, eps, half)
    kv_lat = torch.empty(M, dc, device=dev, dtype=act_dt)
    _rmsnorm_jvp_strided(a[:, dq:], a[half:, dq:], params["w_kva_ln"],
                         kv_lat, kv_lat[half:], dc, wa, dc, eps, half)

    # ---- up-projections: q_b and kv_b stacked GEMMs ----
    # qb[m, j] = sum_k q_lat[m, k] * w_qb[j, k]      (einsum "mk,jk->mj")
    # kvb[m, j] = sum_k kv_lat[m, k] * w_kvb[j, k]   (einsum "mk,jk->mj")
    qb = mm(q_lat, "w_qb")                 # (M, H*(dn+dr)) fp16
    kvb = mm(kv_lat, "w_kvb")              # (M, H*(dn+dv)) fp16

    # ---- assemble packed q/k/v: (2b, l, 3, H, 64) act_dt ----
    packed = torch.empty(2 * B, S, 3, H, hd, device=dev, dtype=act_dt)
    so_b, so_s = packed.stride(0), packed.stride(1)
    qv = packed[:, :, 0]                  # (2b, l, H, 64) q slot view
    kv_ = packed[:, :, 1]                 # (2b, l, H, 64) k slot view
    vv = packed[:, :, 2]                  # (2b, l, H, 64) v slot view
    BN = triton.next_power_of_2(dn)
    _q_prep_kernel[(half,)](
        qb, params["w_qnorm"], rope_cos, rope_sin, qv,
        H * hd, so_b, so_s, B, S, half, eps,
        NH=H, DN=dn, DR2=dr // 2, HD=hd, BN=BN, CLAMP=clamp, num_warps=4,
    )
    _kv_prep_kernel[(half,)](
        kvb, a[:, dq + dc:], params["w_knorm"], rope_cos, rope_sin, kv_, vv,
        H * (dn + dv), wa, so_b, so_s, B, S, half, eps,
        NH=H, DN=dn, DR2=dr // 2, DV=dv, HD=hd, BN=BN, KVW=dn + dv,
        CLAMP=clamp, num_warps=4,
    )

    # ---- fused flash-attention JVP (fp16/fp8 io, fp32 softmax stats) ----
    # o[m, h, :] = softmax_s(q[m,h,:] . k[s,h,:] * scale) @ v[s,h,:]
    attn = torch.empty(2 * B, S, H * dv, device=dev, dtype=act_dt)
    _flash_jvp(qv[:B], kv_[:B], vv[:B], qv[B:], kv_[B:], vv[B:],
               attn[:B].view(B, S, H, dv),
               attn[B:].view(B, S, H, dv), scale)

    # ---- Kimi MLA output gate: attn *= sigmoid(n1 @ w_og^T) ----
    # z[m, j] = sum_k n1[m, k] * w_og[j, k]          (einsum "mk,jk->mj")
    if "w_og" in w:
        z = mm(xs.view(M, D), "w_og")                # (M, H*dv) fp16
        tot = half * H * dv
        BLKG = 1024
        _out_gate_jvp_kernel[(triton.cdiv(tot, BLKG),)](
            attn[:B], attn[B:], z, z[half:], tot,
            BLOCK=BLKG, CLAMP=variant == "fp8", num_warps=4,
        )

    # ---- out projection + gated residual + RMSNorm2 (fused kernel) ----
    # proj[m, j] = sum_k attn[m, k] * w_out[j, k]    (einsum "mk,jk->mj")
    proj = mm(attn.view(M, H * dv), "w_out").view(2 * B, S, D)
    res = torch.empty(2 * B, S, D, device=dev, dtype=torch.float16)
    BLOCK = triton.next_power_of_2(D)
    _res_gate_norm_jvp_kernel[(half,)](
        proj[:B], proj[B:], params["g_attn"], x, dx, params["w_norm2"],
        res[:B], res[B:], xs[:B], xs[B:], D, eps,
        BLOCK=BLOCK, num_warps=4 if BLOCK <= 1024 else 8,
        CLAMP=clamp,
    )

    # ---- gate/up projection + fused SwiGLU JVP ----
    # gu[m, j] = sum_k n2[m, k] * w_gate_up[j, k]    (einsum "mk,jk->mj")
    gu = mm(xs.view(M, D), "w_gate_up").view(2 * B, S, 2 * f)
    act = torch.empty(2 * B, S, f, device=dev, dtype=act_dt)
    _swiglu_row_jvp(gu[:B], gu[B:], act[:B], act[B:], half, f,
                    beta=params.get("situ_beta", 1.0))

    # ---- down projection with fused gated final residual add (fp32 out) ----
    # dwn[m, j] = sum_k act[m, k] * w_down[j, k]     (einsum "mk,jk->mj")
    if variant == "fp16":
        _mm16_gate_add(act.view(M, f), w["w_down"], res.view(M, D),
                       params["g_mlp"], out_y, out_dy, half)
    else:
        dwn = mm(act.view(M, f), "w_down").view(2 * B, S, D)
        total = half * D                  # elements per half
        BLK = 1024
        _gate_add_kernel[(triton.cdiv(total, BLK),)](
            res[:B], res[B:], dwn[:B], dwn[B:], params["g_mlp"],
            out_y, out_dy, D, total, BLOCK=BLK, num_warps=4,
        )


def mla_params_from_block(block):
    """Extract the parameter dict from an imf_dit_video.TransformerBlock.

    Args:
        block: TransformerBlock with MLAAttention (fp32 CUDA).

    Returns:
        params dict for triton_mla_block_jvp (weights detached, fp32).
    """
    attn = block.attn
    p = {
        "w_norm1": block.norm1.weight.detach(),        # (d,)
        "w_qa": attn.q_a_proj.weight.detach(),         # (dq, d)
        "w_kva": attn.kv_a_proj.weight.detach(),       # (dc+dr, d)
        "w_qa_ln": attn.q_a_layernorm.weight.detach(), # (dq,)
        "w_kva_ln": attn.kv_a_layernorm.weight.detach(),  # (dc,)
        "w_qb": attn.q_b_proj.weight.detach(),         # (H*(dn+dr), dq)
        "w_kvb": attn.kv_b_proj.weight.detach(),       # (H*(dn+dv), dc)
        "w_qnorm": attn.q_norm.weight.detach(),        # (dn,)
        "w_knorm": attn.k_norm.weight.detach(),        # (dn,)
        "w_out": attn.out_proj.weight.detach(),        # (d, H*dv)
        "g_attn": block.attn_scale.detach(),           # (d,)
        "w_norm2": block.norm2.weight.detach(),        # (d,)
        "w_gate": block.mlp.w1.weight.detach(),        # (F0, d)
        "w_up": block.mlp.w3.weight.detach(),          # (F0, d)
        "w_down": block.mlp.w2.weight.detach(),        # (d, F0)
        "g_mlp": block.mlp_scale.detach(),             # (d,)
        "situ_beta": getattr(getattr(block.mlp, "act", None), "beta", 1.0),
        "w_og": (attn.g_proj.weight.detach()
                 if getattr(attn, "use_output_gate", False) else None),
        "H": attn.num_heads,
        "dq": attn.q_lora_rank,
        "dc": attn.kv_lora_rank,
        "dn": attn.qk_nope_head_dim,
        "dr": attn.qk_rope_head_dim,
        "dv": attn.v_head_dim,
    }
    return p


class TritonMLABlockJVP(nn.Module):
    """Module wrapper around triton_mla_block_jvp.

    Holds the parameter dict extracted from a TransformerBlock and caches
    the fused fp16 (default) or fp8 weights across calls.
    forward(x, dx, cos, sin) -> (y, dy); x/dx (b, l, d) fp32,
    cos/sin (l, dr//2) fp32.
    """

    def __init__(self, block, eps=1e-6, variant="fp16", use_graph=True):
        super().__init__()
        self.eps = eps
        self.variant = variant
        self.use_graph = use_graph
        self.params = mla_params_from_block(block)
        if variant == "fp8":
            self.params["_fp8"] = _mla_fp8_cache(self.params)
        else:
            self.params["_fp16"] = _mla_fp16_cache(self.params)
        # CUDA-graph cache: (b, l) -> (graph, static x/dx/y/dy buffers).
        # Replay writes the outputs into the SAME static y/dy tensors each
        # call; consume them before the next forward (true in a training
        # step, where the loss is computed immediately).
        self._graphs = {}

    def _run(self, x, dx, rope_cos, rope_sin):
        return triton_mla_block_jvp(
            x, dx, self.params, rope_cos, rope_sin, eps=self.eps,
            variant=self.variant,
        )

    def forward(self, x, dx, rope_cos, rope_sin):
        if not self.use_graph:
            return self._run(x, dx, rope_cos, rope_sin)
        key = (x.shape[0], x.shape[1])     # (b, l) static-shape key
        entry = self._graphs.get(key)
        if entry is None:
            # warm up allocator/autotuner outside capture, then capture
            sx = x.clone()                 # (b, l, d) static input primal
            sdx = dx.clone()               # (b, l, d) static input tangent
            for _ in range(2):
                self._run(sx, sdx, rope_cos, rope_sin)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sy, sdy = self._run(sx, sdx, rope_cos, rope_sin)
            entry = (graph, sx, sdx, sy, sdy)
            self._graphs[key] = entry
        graph, sx, sdx, sy, sdy = entry
        sx.copy_(x)
        sdx.copy_(dx)
        graph.replay()
        return sy, sdy
