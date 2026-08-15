"""
Block-sparse softmax attention for SLA2 with INT8 quantization-aware forward.

SLA2 Section 5: the forward pass runs low-bit attention
    S = dequant(quant(Q) quant(K)^T) / sqrt(d),  P = softmax(S .* M)
    O_s = dequant(quant(P) quant(V))
while the backward pass stays fully FP16 (it reuses the FP16 kernels from
`sparse_linear_attention.kernel`).  Quantization is per query/key block
(scale = max|.| / 127), matching the per-block scheme of SageAttention2.

Shapes:
    Q, K, V, OS : (B, H, L, D)      B=batch, H=heads, L=tokens, D=head dim
    LUT         : (B, H, M_BLOCKS, topk) int32 -- selected key-block ids per query block
    LSE         : (B, H, L) fp32    -- log2-sum-exp, saved for the FP16 backward
"""

import torch
import triton
import triton.language as tl

from .sla_kernel import (
    _attn_bwd_preprocess,
    _attn_bwd_dq,
    _attn_bwd_dkdv,
)


@triton.jit
def _attn_fwd_qat(
    Q, K, V, LUT, LSE, OS, QPHI,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_INT8: tl.constexpr,
    PHI_OUT: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L, other=0.0)
    if PHI_OUT:
        # phi(Q) = softmax over the D head channels, written as a by-product for
        # the linear branch -- the query tile is already in registers here
        qf = q.to(tl.float32)
        qe = tl.exp(qf - tl.max(qf, axis=1)[:, None])
        qphi = qe / tl.sum(qe, axis=1)[:, None]
        tl.store(QPHI + qkv_offset + offs_m[:, None] * D + offs_d[None, :],
                 qphi.to(QPHI.type.element_ty), mask=offs_m[:, None] < L)
    if USE_INT8:
        # per-query-block symmetric INT8 quantization
        sq = tl.max(tl.abs(q.to(tl.float32))) / 127.0 + 1e-12
        q_i8 = (q.to(tl.float32) / sq + 0.5 * tl.where(q.to(tl.float32) >= 0, 1.0, -1.0)).to(tl.int8)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT + lut_offset + block_idx).to(tl.int64)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)

        if USE_INT8:
            sk = tl.max(tl.abs(k.to(tl.float32))) / 127.0 + 1e-12
            k_i8 = (k.to(tl.float32) / sk + 0.5 * tl.where(k.to(tl.float32) >= 0, 1.0, -1.0)).to(tl.int8)
            qk = tl.dot(q_i8, tl.trans(k_i8)).to(tl.float32) * (sq * sk)
        else:
            qk = tl.dot(q, tl.trans(k)).to(tl.float32)

        qk = qk * (qk_scale * 1.4426950408889634)  # 1/ln(2), so exp2 can be used
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        p = tl.exp2(qk - new_m[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]

        if USE_INT8 and BLOCK_N >= 32:
            # P in [0, 1]; V quantized per key block. int8 MMA needs the
            # inner dim (BLOCK_N) >= 32; for smaller tiles (e.g. E = 16
            # cube tiles) the PV dot falls back to fp16 while the QK score
            # dot stays int8 -- matching the JVP kernel's semantics.
            p_i8 = (p * 127.0 + 0.5).to(tl.int8)
            sv = tl.max(tl.abs(v.to(tl.float32))) / 127.0 + 1e-12
            v_i8 = (v.to(tl.float32) / sv + 0.5 * tl.where(v.to(tl.float32) >= 0, 1.0, -1.0)).to(tl.int8)
            o_s += tl.dot(p_i8, v_i8).to(tl.float32) * (sv / 127.0)
        else:
            o_s += tl.dot(p.to(v.dtype), v).to(tl.float32)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
    m_i += tl.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)


@triton.jit
def _quant_blocks(X, XQ, XS, L, D: tl.constexpr, BLOCK_L: tl.constexpr):
    """Per-block symmetric INT8 quantization.

    X  : (B,H,L,D) fp16/bf16          -- source tensor (Q, K or V)
    XQ : (B,H,L,D) int8               -- quantized values, round-to-nearest
    XS : (B,H,ceil(L/BLOCK_L)) fp32   -- one scale per token block
    """
    idx_l = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    n_blocks = tl.cdiv(L, BLOCK_L)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    ptrs = X + idx_bh * L * D + offs_l[:, None] * D + offs_d[None, :]
    mask = offs_l[:, None] < L

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    s = tl.max(tl.abs(x)) / 127.0 + 1e-12
    xq = tl.extra.cuda.libdevice.round(x / s).to(tl.int8)
    tl.store(XQ + idx_bh * L * D + offs_l[:, None] * D + offs_d[None, :], xq, mask=mask)
    tl.store(XS + idx_bh * n_blocks + idx_l, s)


def quantize_qkv(x, BLOCK_L):
    """x: (B,H,L,D) -> (xq int8 (B,H,L,D), xs fp32 (B,H,ceil(L/BLOCK_L)))."""
    B, H, L, D = x.shape
    n_blocks = triton.cdiv(L, BLOCK_L)
    xq = torch.empty_like(x, dtype=torch.int8)
    xs = torch.empty((B, H, n_blocks), device=x.device, dtype=torch.float32)
    _quant_blocks[(n_blocks, B * H)](x, xq, xs, L, D, BLOCK_L, num_warps=4)
    return xq, xs


@triton.jit
def _attn_fwd_int8(
    QQ, QS, KQ, KS, VQ, VS, LUT, LSE, OS, QPHI,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PHI_OUT: tl.constexpr,
):
    """Block-sparse attention on pre-quantized INT8 Q, K, V.

    With PHI_OUT the kernel also writes QPHI = softmax(Q, dim=D) : (B,H,L,D) --
    the feature map of the linear branch -- for free, since the raw query tile
    is already resident in registers.

    QQ, KQ, VQ : (B,H,L,D) int8         -- quantized Q, K, V
    QS         : (B,H,M_BLOCKS) fp32    -- per query-block scale
    KS, VS     : (B,H,N_BLOCKS) fp32    -- per key-block scale
    LSE        : (B,H,L) fp32           -- log2-sum-exp for the FP16 backward
    OS         : (B,H,L,D)              -- sparse attention output
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Q is quantized here, in registers: each query block is read by exactly one
    # program, so a separate pre-pass over Q would only add HBM traffic.  K and V
    # are pre-quantized because their blocks are re-read by many query blocks.
    q_f = tl.load(QQ + qkv_offset + offs_m[:, None] * D + offs_d[None, :],
                  mask=offs_m[:, None] < L, other=0.0).to(tl.float32)
    sq = tl.max(tl.abs(q_f)) / 127.0 + 1e-12
    q = tl.extra.cuda.libdevice.round(q_f / sq).to(tl.int8)
    if PHI_OUT:
        qe = tl.exp(q_f - tl.max(q_f, axis=1)[:, None])
        qphi = qe / tl.sum(qe, axis=1)[:, None]
        tl.store(QPHI + qkv_offset + offs_m[:, None] * D + offs_d[None, :],
                 qphi.to(QPHI.type.element_ty), mask=offs_m[:, None] < L)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    K_ptrs = KQ + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = VQ + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT + lut_offset + block_idx).to(tl.int64)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0)
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0)
        sk = tl.load(KS + idx_bh * N_BLOCKS + idx_n)
        sv = tl.load(VS + idx_bh * N_BLOCKS + idx_n)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * (sq * sk * qk_scale * 1.4426950408889634)
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - new_m[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        # P in [0, 1] -> fixed-point INT8 with scale 1/127
        p_i8 = (p * 127.0 + 0.5).to(tl.int8)
        o_s += tl.dot(p_i8, v).to(tl.float32) * (sv / 127.0)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :],
             o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
    m_i += tl.log2(l_i)
    tl.store(LSE + lse_offset + offs_m, m_i, mask=offs_m < L)


def sparse_attention_int8(q, k, v, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None,
                          phi_out=False):
    """Inference-only INT8 block-sparse attention.

    q, k, v : (B,H,L,D) fp16/bf16 -> o_s : (B,H,L,D), quantization done once up-front.
    With phi_out, also returns qphi = softmax(q, dim=-1) : (B,H,L,D), produced in
    the kernel's prologue at the cost of one extra store.
    """
    B, H, L, D = q.shape
    if qk_scale is None:
        qk_scale = D ** -0.5
    M_BLOCKS, N_BLOCKS = triton.cdiv(L, BLOCK_M), triton.cdiv(L, BLOCK_N)
    kq, ks = quantize_qkv(k, BLOCK_N)
    vq, vs = quantize_qkv(v, BLOCK_N)
    o_s = torch.empty_like(v)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    qphi = torch.empty_like(q) if phi_out else o_s
    _attn_fwd_int8[(M_BLOCKS, B * H)](
        q, ks, kq, ks, vq, vs, lut, lse, o_s, qphi,
        qk_scale, topk, L, M_BLOCKS, N_BLOCKS, D, BLOCK_M, BLOCK_N, phi_out,
        num_warps=4 if D == 64 else 8, num_stages=3,
    )
    return (o_s, qphi) if phi_out else o_s


class _sparse_attention_qat(torch.autograd.Function):
    """Forward: INT8 (or FP16) block-sparse attention.  Backward: FP16 (SLA kernels)."""

    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, use_int8=True,
                qk_scale=None, phi_out=None):
        # phi_out : optional preallocated (B,H,L,D) tensor; when given, the kernel
        # also writes softmax(q, dim=-1) into it (feature map of the linear branch)
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        B, H, L, D = q.shape
        if qk_scale is None:
            qk_scale = D ** -0.5
        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        o_s = torch.empty_like(v)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        _attn_fwd_qat[(M_BLOCKS, B * H)](
            q, k, v, lut, lse, o_s, o_s if phi_out is None else phi_out,
            qk_scale, topk, L, M_BLOCKS, D, BLOCK_M, BLOCK_N, use_int8,
            phi_out is not None,
            num_warps=4 if D == 64 else 8, num_stages=3,
        )

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale, ctx.topk = qk_scale, topk
        ctx.BLOCK_M, ctx.BLOCK_N = BLOCK_M, BLOCK_N
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()
        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape
        M_BLOCKS, N_BLOCKS = triton.cdiv(L, BLOCK_M), triton.cdiv(L, BLOCK_N)

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        _attn_bwd_preprocess[(M_BLOCKS, B * H)](o_s, do_s, delta_s, L, D, BLOCK_M)
        _attn_bwd_dq[(M_BLOCKS, B * H)](
            q, k, v, lse, delta_s, do_s, dq, lut, ctx.qk_scale, ctx.topk,
            L, M_BLOCKS, D, BLOCK_M, BLOCK_N,
            num_warps=4 if D == 64 else 8, num_stages=4 if D == 64 else 5,
        )
        _attn_bwd_dkdv[(N_BLOCKS, B * H)](
            q, k, v, do_s, dk, dv, ctx.qk_scale, k_block_id, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS, D, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if D == 64 else 8, num_stages=4 if D == 64 else 5,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None
