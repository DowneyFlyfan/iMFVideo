"""Single fused SLA2 forward kernel (Algorithm 2 of the SLA2 paper).

Algorithm 2 walks the selected key blocks of a query block ONCE and updates both
branches inside that loop.  This kernel does the same, so K_j / V_j are read once
instead of twice and O_s never round-trips through HBM:

    for each query block i:
        O_s_i : online-softmax attention over the selected blocks (M = 1)
        O_l_i : linear attention over the complement, obtained as
                  num_i = phi(Q_i) H_tot - sum_{j in sel(i)} (phi(Q_i) phi(K_j)^T) V_j
                  den_i = phi(Q_i) z_tot - sum_{j in sel(i)} rowsum(phi(Q_i) phi(K_j)^T)
        O_i   = alpha_i * O_s_i + (1 - alpha_i) * O_l_i          (Eq. 13)

Shapes (B = batch, H = heads, L = tokens, D = head dim,
        Mb = ceil(L/BLOCK_M) query blocks, Nb = ceil(L/BLOCK_N) key blocks):
    Q, K, V, O : (B, H, L, D)
    HTOT       : (B, H, D, D)  phi(K)^T V, [key-channel, value-channel], V's dtype
    ZTOT       : (B, H, D)     colsum phi(K), fp32
    LUT        : (B, H, Mb, topk) int32   selected key-block ids
    ALPHA      : (Mb,) fp32 in (0, 1)
    QS         : (B, H, Mb) fp32   per query-block INT8 scale   (INT8 path only)
    KS, VS     : (B, H, Nb) fp32   per key-block INT8 scales    (INT8 path only)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sla2_fwd_fused(
    Q, K, V, QQ, QS, KQ, KS, VQ, VS, HTOT, ZTOT, LUT, ALPHA, O,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    LUT_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
    USE_INT8: tl.constexpr,
):
    """SPLIT = number of BLOCK_M tiles per router query block (lut row)."""
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    idx_lut = idx_m // SPLIT

    qkv_off = idx_bh * L * D
    lut_off = (idx_bh * LUT_BLOCKS + idx_lut) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)
    m_mask = offs_m[:, None] < L

    # ---- query tile: raw values for the sparse branch, phi(.) for the linear branch
    q = tl.load(Q + qkv_off + offs_m[:, None] * D + offs_d[None, :], mask=m_mask, other=0.0)
    qf = q.to(tl.float32)
    qf = tl.exp(qf - tl.max(qf, axis=1)[:, None])
    qp = (qf / tl.sum(qf, axis=1)[:, None]).to(q.dtype)          # phi(Q_i), (BLOCK_M, D)
    if USE_INT8:
        q_i8 = tl.load(QQ + qkv_off + offs_m[:, None] * D + offs_d[None, :], mask=m_mask, other=0)
        sq = tl.load(QS + idx_bh * M_BLOCKS + idx_m)

    # ---- linear branch: start from the all-block totals, subtract the selected blocks
    h = tl.load(HTOT + idx_bh * D * D + offs_d[:, None] * D + offs_e[None, :])
    z = tl.load(ZTOT + idx_bh * D + offs_d).to(tl.float32)
    num = tl.dot(qp, h).to(tl.float32)                            # (BLOCK_M, D)
    den = tl.sum(qp.to(tl.float32) * z[None, :], axis=1)          # (BLOCK_M,)

    # ---- sparse branch accumulators (online softmax)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    K_ptrs = K + qkv_off + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_off + offs_n[:, None] * D + offs_e[None, :]
    for b in tl.range(topk):
        idx_n = tl.load(LUT + lut_off + b).to(tl.int64)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)

        # --- sparse (softmax) branch
        if USE_INT8:
            k_i8 = tl.load(KQ + qkv_off + (idx_n * BLOCK_N + offs_n)[:, None] * D + offs_d[None, :],
                           mask=n_mask[:, None], other=0)
            sk = tl.load(KS + idx_bh * N_BLOCKS + idx_n)
            qk = tl.dot(q_i8, tl.trans(k_i8)).to(tl.float32) * (sq * sk * qk_scale * 1.4426950408889634)
        else:
            qk = tl.dot(q, tl.trans(k)).to(tl.float32) * (qk_scale * 1.4426950408889634)
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - new_m[:, None])
        corr = tl.exp2(m_i - new_m)
        l_i = l_i * corr + tl.sum(p, 1)
        o_s = o_s * corr[:, None]
        if USE_INT8:
            v_i8 = tl.load(VQ + qkv_off + (idx_n * BLOCK_N + offs_n)[:, None] * D + offs_e[None, :],
                           mask=n_mask[:, None], other=0)
            sv = tl.load(VS + idx_bh * N_BLOCKS + idx_n)
            p_i8 = (p * 127.0 + 0.5).to(tl.int8)
            o_s += tl.dot(p_i8, v_i8).to(tl.float32) * (sv / 127.0)
        else:
            o_s += tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = new_m

        # --- linear branch correction, reusing the same k / v tiles
        kf = k.to(tl.float32)
        kf = tl.exp(kf - tl.max(kf, axis=1)[:, None])
        kp = (kf / tl.sum(kf, axis=1)[:, None]).to(k.dtype)       # phi(K_j)
        w = tl.dot(qp, tl.trans(kp))                              # (BLOCK_M, BLOCK_N)
        w = tl.where(n_mask[None, :], w, 0.0)
        num -= tl.dot(w.to(v.dtype), v).to(tl.float32)
        den -= tl.sum(w.to(tl.float32), axis=1)

    o_s = o_s / l_i[:, None]
    den = tl.where(den >= 0, den + 1e-5, den - 1e-5)
    o_l = num / den[:, None]

    a = tl.load(ALPHA + idx_lut).to(tl.float32)
    o = a * o_s + (1.0 - a) * o_l
    tl.store(O + qkv_off + offs_m[:, None] * D + offs_e[None, :],
             o.to(O.type.element_ty), mask=m_mask)


def sla2_forward_fused(q, k, v, kphi, lut, topk, bq, bk, alpha,
                       block_m=None, use_int8=False, quantized=None, qk_scale=None):
    """Fused SLA2 forward (inference only).

    q, k, v, kphi : (B,H,L,D)   -- kphi = phi(K), consumed only by the H_tot GEMM
    lut           : (B,H,Mb,topk) int32, Mb = ceil(L/bq)
    alpha         : (Mb,) fp32 in (0,1)
    block_m       : query tile of the kernel; must divide bq (default: bq)
    quantized     : optional ((qq,qs),(kq,ks),(vq,vs)) from `quantize_qkv`
    returns o     : (B,H,L,D)
    """
    B, H, L, D = q.shape
    if qk_scale is None:
        qk_scale = D ** -0.5
    block_m = block_m or bq
    assert bq % block_m == 0
    split = bq // block_m
    lut_blocks = triton.cdiv(L, bq)
    m_blocks = triton.cdiv(L, block_m)
    n_blocks = triton.cdiv(L, bk)

    htot = (kphi.transpose(-1, -2) @ v).contiguous()               # (B,H,D,D) V dtype
    ztot = kphi.sum(dim=-2, dtype=torch.float32).contiguous()      # (B,H,D)

    if use_int8:
        (qq, qs), (kq, ks), (vq, vs) = quantized
    else:
        qq = kq = vq = q
        qs = ks = vs = ztot

    o = torch.empty_like(v)
    # The kernel keeps two (BLOCK_M, D) fp32 accumulators (O_s and num), so large
    # tiles can exceed the shared-memory / register budget; fall back to smaller
    # tiles and fewer pipeline stages instead of failing.
    key = (D, bq, bk, block_m, use_int8)
    cfgs = _CONFIG_CACHE.get(key) or _configs(block_m, bq, use_int8)
    last_err = None
    for bm, stages in cfgs:
        try:
            _sla2_fwd_fused[(triton.cdiv(L, bm), B * H)](
                q, k, v, qq, qs, kq, ks, vq, vs, htot, ztot, lut, alpha, o,
                qk_scale, topk, L, triton.cdiv(L, bm), n_blocks, lut_blocks, D, bm, bk,
                bq // bm, use_int8,
                num_warps=8, num_stages=stages,
            )
            _CONFIG_CACHE[key] = [(bm, stages)]
            return o
        except triton.runtime.errors.OutOfResources as e:  # noqa: PERF203
            last_err = e
    raise last_err


_CONFIG_CACHE = {}


def _configs(block_m, bq, use_int8):
    """Candidate (BLOCK_M, num_stages) pairs, most aggressive first.

    With INT8 the query scales were computed for `block_m`-sized blocks, so the
    query tile may not shrink; only the pipeline depth can be reduced.
    """
    if use_int8:
        return [(block_m, 2), (block_m, 1)]
    cfgs, bm = [], block_m
    while bm >= 16:
        cfgs += [(bm, 2), (bm, 1)]
        if bm % 2 or bq % (bm // 2):
            break
        bm //= 2
    return cfgs
