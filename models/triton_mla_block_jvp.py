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
    _swiglu_jvp,
)

__all__ = ["triton_mla_block_jvp", "TritonMLABlockJVP", "mla_params_from_block"]


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
                   DN: tl.constexpr,     # nope band width dn
                   DR2: tl.constexpr,    # rope angle pairs dr // 2
                   HD: tl.constexpr,     # head dim dn + dr (= 64)
                   BN: tl.constexpr,     # next_power_of_2(DN)
                   CLAMP: tl.constexpr): # clamp outputs into the fp8 range
    pid = tl.program_id(0)             # token row in [0, B*S)
    h = tl.program_id(1)               # head index in [0, H)
    b = pid // S                       # batch index
    s = pid % S                        # sequence position
    src_p = pid * sq_row + h * HD      # primal head offset in QB
    src_t = (pid + half) * sq_row + h * HD   # tangent head offset in QB
    dst_p = b * so_b + s * so_s + h * HD     # primal head offset in OUT q slot
    dst_t = (b + B) * so_b + s * so_s + h * HD

    # ---- nope band: per-head RMSNorm JVP over dn channels ----
    cn = tl.arange(0, BN)              # (BN,) nope channel index
    nm = cn < DN
    xn = tl.load(QB + src_p + cn, mask=nm, other=0.0).to(tl.float32)
    dn_ = tl.load(QB + src_t + cn, mask=nm, other=0.0).to(tl.float32)
    w = tl.load(WQN + cn, mask=nm, other=0.0).to(tl.float32)
    inv_r = 1.0 / tl.sqrt(tl.sum(xn * xn, axis=0) / DN + eps)
    mxdx = tl.sum(xn * dn_, axis=0) / DN
    yn = w * xn * inv_r
    dyn = w * (dn_ * inv_r - xn * (mxdx * inv_r * inv_r * inv_r))
    if CLAMP:
        yn = tl.clamp(yn, -448.0, 448.0)
        dyn = tl.clamp(dyn, -448.0, 448.0)
    tl.store(OUT + dst_p + cn, yn.to(OUT.dtype.element_ty), mask=nm)
    tl.store(OUT + dst_t + cn, dyn.to(OUT.dtype.element_ty), mask=nm)

    # ---- rope band: rotation JVP (same rotation on primal and tangent) ----
    i = tl.arange(0, DR2)              # (DR2,) angle-pair index
    c = tl.load(COS + s * DR2 + i).to(tl.float32)   # (DR2,) cos(angle)
    sn = tl.load(SIN + s * DR2 + i).to(tl.float32)  # (DR2,) sin(angle)
    re = DN + 2 * i                    # channel of the pair's real part
    im = DN + 2 * i + 1                # channel of the pair's imag part
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
                    DN: tl.constexpr,     # nope band width dn
                    DR2: tl.constexpr,    # rope angle pairs dr // 2
                    DV: tl.constexpr,     # value head dim dv (= 64)
                    HD: tl.constexpr,     # qk head dim dn + dr (= 64)
                    BN: tl.constexpr,     # next_power_of_2(DN)
                    KVW: tl.constexpr,    # kv_b per-head width dn + dv
                    CLAMP: tl.constexpr): # clamp outputs into the fp8 range
    pid = tl.program_id(0)             # token row in [0, B*S)
    h = tl.program_id(1)               # head index in [0, H)
    b = pid // S
    s = pid % S
    src_p = pid * skv_row + h * KVW    # primal head offset in KVB
    src_t = (pid + half) * skv_row + h * KVW
    dk_p = b * so_b + s * so_s + h * HD    # primal head offset in OUT k slot
    dk_t = (b + B) * so_b + s * so_s + h * HD
    dv_p = b * so_b + s * so_s + h * DV    # primal head offset in OUT v slot
    dv_t = (b + B) * so_b + s * so_s + h * DV

    # ---- k_nope: per-head RMSNorm JVP over dn channels ----
    cn = tl.arange(0, BN)
    nm = cn < DN
    xn = tl.load(KVB + src_p + cn, mask=nm, other=0.0).to(tl.float32)
    dn_ = tl.load(KVB + src_t + cn, mask=nm, other=0.0).to(tl.float32)
    w = tl.load(WKN + cn, mask=nm, other=0.0).to(tl.float32)
    inv_r = 1.0 / tl.sqrt(tl.sum(xn * xn, axis=0) / DN + eps)
    mxdx = tl.sum(xn * dn_, axis=0) / DN
    yn = w * xn * inv_r
    dyn = w * (dn_ * inv_r - xn * (mxdx * inv_r * inv_r * inv_r))
    if CLAMP:
        yn = tl.clamp(yn, -448.0, 448.0)
        dyn = tl.clamp(dyn, -448.0, 448.0)
    tl.store(OUTK + dk_p + cn, yn.to(OUTK.dtype.element_ty), mask=nm)
    tl.store(OUTK + dk_t + cn, dyn.to(OUTK.dtype.element_ty), mask=nm)

    # ---- shared k_rope: RoPE rotation JVP, identical for every head ----
    i = tl.arange(0, DR2)
    c = tl.load(COS + s * DR2 + i).to(tl.float32)
    sn = tl.load(SIN + s * DR2 + i).to(tl.float32)
    xr = tl.load(KR + pid * skr_row + 2 * i).to(tl.float32)
    xi = tl.load(KR + pid * skr_row + 2 * i + 1).to(tl.float32)
    dxr = tl.load(KR + (pid + half) * skr_row + 2 * i).to(tl.float32)
    dxi = tl.load(KR + (pid + half) * skr_row + 2 * i + 1).to(tl.float32)
    re = DN + 2 * i
    im = DN + 2 * i + 1
    yr = xr * c - xi * sn
    yi = xr * sn + xi * c
    dyr = dxr * c - dxi * sn
    dyi = dxr * sn + dxi * c
    # ---- v: straight copy of the dv band (JVP of a copy is a copy) ----
    cv = tl.arange(0, DV)              # (DV,) value channel index
    v = tl.load(KVB + src_p + DN + cv).to(tl.float32)
    dv_ = tl.load(KVB + src_t + DN + cv).to(tl.float32)
    if CLAMP:
        yr = tl.clamp(yr, -448.0, 448.0)
        yi = tl.clamp(yi, -448.0, 448.0)
        dyr = tl.clamp(dyr, -448.0, 448.0)
        dyi = tl.clamp(dyi, -448.0, 448.0)
        v = tl.clamp(v, -448.0, 448.0)
        dv_ = tl.clamp(dv_, -448.0, 448.0)
    tl.store(OUTK + dk_p + re, yr.to(OUTK.dtype.element_ty))
    tl.store(OUTK + dk_p + im, yi.to(OUTK.dtype.element_ty))
    tl.store(OUTK + dk_t + re, dyr.to(OUTK.dtype.element_ty))
    tl.store(OUTK + dk_t + im, dyi.to(OUTK.dtype.element_ty))
    tl.store(OUTV + dv_p + cv, v.to(OUTV.dtype.element_ty))
    tl.store(OUTV + dv_t + cv, dv_.to(OUTV.dtype.element_ty))


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
def _pad_rows_16(w):
    """Zero-pad the leading (output) dim of w (N, K) to a multiple of 16.
    Padded gate/up rows give silu(0)*0 = 0 through SwiGLU, and padded down
    columns then multiply those exact zeros, so the block output is
    unchanged."""
    n = w.shape[0]
    n16 = (n + 15) // 16 * 16
    if n16 == n:
        return w
    return torch.cat([w, w.new_zeros(n16 - n, w.shape[1])], dim=0)


def _pad_cols_16(w):
    """Zero-pad the trailing (input) dim of w (N, K) to a multiple of 16."""
    k = w.shape[1]
    k16 = (k + 15) // 16 * 16
    if k16 == k:
        return w
    return torch.cat([w, w.new_zeros(w.shape[0], k16 - k)], dim=1)


def _mla_fused_weights(p):
    """Fuse/pad the six GEMM weights (fp32, (out_features, in_features)):
        w_a       (dq + dc + dr, d)  fused q_a_proj + kv_a_proj
        w_qb      (H * (dn+dr), dq)
        w_kvb     (H * (dn+dv), dc)
        w_out     (d, H * dv)
        w_gate_up (2 * F, d)         F padded to a multiple of 16
        w_down    (d, F)
    """
    return {
        "w_a": torch.cat([p["w_qa"], p["w_kva"]], dim=0),
        "w_qb": p["w_qb"],
        "w_kvb": p["w_kvb"],
        "w_out": p["w_out"],
        "w_gate_up": torch.cat(
            [_pad_rows_16(p["w_gate"]), _pad_rows_16(p["w_up"])], dim=0
        ),
        "w_down": _pad_cols_16(p["w_down"]),
    }


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
            # a2d (M, K) fp16 @ fp16 weight (N, K) -> (M, N) fp16, cuBLAS
            # fp16 tensor cores with fp32 accumulation
            return a2d @ w[key].t()

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
    _q_prep_kernel[(half, H)](
        qb, params["w_qnorm"], rope_cos, rope_sin, qv,
        H * hd, so_b, so_s, B, S, half, eps,
        DN=dn, DR2=dr // 2, HD=hd, BN=BN, CLAMP=clamp, num_warps=1,
    )
    _kv_prep_kernel[(half, H)](
        kvb, a[:, dq + dc:], params["w_knorm"], rope_cos, rope_sin, kv_, vv,
        H * (dn + dv), wa, so_b, so_s, B, S, half, eps,
        DN=dn, DR2=dr // 2, DV=dv, HD=hd, BN=BN, KVW=dn + dv, CLAMP=clamp,
        num_warps=1,
    )

    # ---- fused flash-attention JVP (fp16/fp8 io, fp32 softmax stats) ----
    # o[m, h, :] = softmax_s(q[m,h,:] . k[s,h,:] * scale) @ v[s,h,:]
    attn = torch.empty(2 * B, S, H * dv, device=dev, dtype=act_dt)
    _flash_jvp(qv[:B], kv_[:B], vv[:B], qv[B:], kv_[B:], vv[B:],
               attn[:B].view(B, S, H, dv),
               attn[B:].view(B, S, H, dv), scale)

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
    _swiglu_jvp(gu[:B], gu[B:], act[:B], act[B:])

    # ---- down projection + gated final residual add (fp32 out) ----
    # dwn[m, j] = sum_k act[m, k] * w_down[j, k]     (einsum "mk,jk->mj")
    dwn = mm(act.view(M, f), "w_down").view(2 * B, S, D)
    g_mlp = params["g_mlp"]               # (d,) fp32 vector gate
    y = res[:B] + dwn[:B] * g_mlp         # (b, l, d) fp32 (promotion)
    dy = res[B:] + dwn[B:] * g_mlp        # (b, l, d) fp32
    return y, dy


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

    def __init__(self, block, eps=1e-6, variant="fp16"):
        super().__init__()
        self.eps = eps
        self.variant = variant
        self.params = mla_params_from_block(block)
        if variant == "fp8":
            self.params["_fp8"] = _mla_fp8_cache(self.params)
        else:
            self.params["_fp16"] = _mla_fp16_cache(self.params)

    def forward(self, x, dx, rope_cos, rope_sin):
        return triton_mla_block_jvp(
            x, dx, self.params, rope_cos, rope_sin, eps=self.eps,
            variant=self.variant,
        )
