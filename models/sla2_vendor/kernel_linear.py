"""
Triton kernels for the *complement* linear-attention branch of SLA2.

SLA2 (Zhang et al., 2026, arXiv:2602.12675) Eq. 14:

    O_l = norm( phi(Q) ( phi(K)^T .* (1 - M) ) ) V

i.e. for query block i, the linear branch only aggregates the key/value blocks
that were NOT selected by the sparse branch.  Writing

    H_tot = phi(K)^T V            (D, D)      sum over ALL key blocks
    z_tot = colsum(phi(K))        (D,)
    H_i   = H_tot - sum_{j in sel(i)} phi(K_j)^T V_j
    z_i   = z_tot - sum_{j in sel(i)} colsum(phi(K_j))
    O_l_i = (phi(Q_i) H_i) / (phi(Q_i) z_i)

the complement sums are obtained by subtracting only the `topk` selected blocks,
so the cost of this branch scales with the sparse budget, not with L.

Tensor shapes (consistent across this file):
    B  = batch, H = heads, L = sequence length, D = head dim
    QPHI, KPHI, V, OL : (B, H, L, D)   -- D indexes head channels
    HTOT              : (B, H, D, D)   -- [key-channel, value-channel]
    ZTOT              : (B, H, D)      -- key-channel
    LUT               : (B, H, M_BLOCKS, topk) int32/int64 -- selected key-block ids
    KBID              : (B, H, M_BLOCKS, N_BLOCKS) int8 -- block mask M_c (1 = sparse)
    DEN               : (B, H, L)      -- denominator phi(Q_i) z_i
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lin_fwd(
    QPHI, KPHI, V, HTOT, ZTOT, LUT, OL, DEN,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_off = idx_bh * L * D
    h_off = idx_bh * D * D
    z_off = idx_bh * D
    lut_off = (idx_bh * M_BLOCKS + idx_m) * topk
    den_off = idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)

    # H_i accumulator: (D key-channel, D value-channel)
    h = tl.load(HTOT + h_off + offs_d[:, None] * D + offs_e[None, :]).to(tl.float32)
    z = tl.load(ZTOT + z_off + offs_d).to(tl.float32)

    KPHI_ptrs = KPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_off + offs_n[:, None] * D + offs_e[None, :]
    for b in tl.range(topk):
        idx_n = tl.load(LUT + lut_off + b).to(tl.int64)
        n_mask = offs_n < L - idx_n * BLOCK_N
        kp = tl.load(KPHI_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
        vv = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
        h -= tl.dot(tl.trans(kp), vv).to(tl.float32)
        z -= tl.sum(kp.to(tl.float32), axis=0)

    qp = tl.load(QPHI + qkv_off + offs_m[:, None] * D + offs_d[None, :],
                 mask=offs_m[:, None] < L, other=0.0)
    num = tl.dot(qp, h.to(qp.dtype))                       # (BLOCK_M, D)
    den = tl.sum(qp.to(tl.float32) * z[None, :], axis=1)   # (BLOCK_M,)
    den = tl.where(den >= 0, den + 1e-5, den - 1e-5)
    ol = num / den[:, None]

    tl.store(OL + qkv_off + offs_m[:, None] * D + offs_e[None, :], ol.to(OL.type.element_ty),
             mask=offs_m[:, None] < L)
    tl.store(DEN + den_off + offs_m, den, mask=offs_m < L)


@triton.jit
def _lin_bwd_dq(
    QPHI, KPHI, V, HTOT, ZTOT, LUT, OL, DEN, DOL, DQPHI,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dQphi_i = G_i H_i^T - s_i z_i^T, with G_i = dOl_i / den_i, s_i = <dOl_i, Ol_i>/den_i."""
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_off = idx_bh * L * D
    h_off = idx_bh * D * D
    z_off = idx_bh * D
    lut_off = (idx_bh * M_BLOCKS + idx_m) * topk
    den_off = idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)

    h = tl.load(HTOT + h_off + offs_d[:, None] * D + offs_e[None, :]).to(tl.float32)
    z = tl.load(ZTOT + z_off + offs_d).to(tl.float32)

    KPHI_ptrs = KPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_off + offs_n[:, None] * D + offs_e[None, :]
    for b in tl.range(topk):
        idx_n = tl.load(LUT + lut_off + b).to(tl.int64)
        n_mask = offs_n < L - idx_n * BLOCK_N
        kp = tl.load(KPHI_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
        vv = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
        h -= tl.dot(tl.trans(kp), vv).to(tl.float32)
        z -= tl.sum(kp.to(tl.float32), axis=0)

    m_mask = offs_m[:, None] < L
    dol = tl.load(DOL + qkv_off + offs_m[:, None] * D + offs_e[None, :], mask=m_mask, other=0.0)
    ol = tl.load(OL + qkv_off + offs_m[:, None] * D + offs_e[None, :], mask=m_mask, other=0.0)
    den = tl.load(DEN + den_off + offs_m, mask=offs_m < L, other=1.0)

    g = dol.to(tl.float32) / den[:, None]                                    # (BLOCK_M, D)
    s = tl.sum(dol.to(tl.float32) * ol.to(tl.float32), axis=1) / den         # (BLOCK_M,)
    # den bottoms out at eps_l when the phi channel a query selects has no
    # global mass (one-hot phi at large logits); g then reaches ~1/eps_l and
    # the fp16 cast below turns it into inf. Saturate: healthy g is O(10).
    g = tl.clamp(g, -60000.0, 60000.0)

    dq = tl.dot(g.to(dol.dtype), tl.trans(h).to(dol.dtype)).to(tl.float32)
    dq -= s[:, None] * z[None, :]
    tl.store(DQPHI + qkv_off + offs_m[:, None] * D + offs_d[None, :],
             dq.to(DQPHI.type.element_ty), mask=m_mask)


@triton.jit
def _lin_bwd_dkdv(
    QPHI, KPHI, V, DHTOT, DZTOT, OL, DEN, DOL, KBID, DKPHI, DV,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dH_j = dH_tot - sum_{i: M_ij=1} Qphi_i^T G_i ; dKphi_j = V_j dH_j^T + dz_j ; dV_j = Kphi_j dH_j."""
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_off = idx_bh * L * D
    h_off = idx_bh * D * D
    z_off = idx_bh * D
    den_off = idx_bh * L
    kbid_off = idx_bh * M_BLOCKS * N_BLOCKS

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)

    dh = tl.load(DHTOT + h_off + offs_d[:, None] * D + offs_e[None, :]).to(tl.float32)
    dz = tl.load(DZTOT + z_off + offs_d).to(tl.float32)

    QPHI_ptrs = QPHI + qkv_off + offs_m[:, None] * D + offs_d[None, :]
    OL_ptrs = OL + qkv_off + offs_m[:, None] * D + offs_e[None, :]
    DOL_ptrs = DOL + qkv_off + offs_m[:, None] * D + offs_e[None, :]
    DEN_ptrs = DEN + den_off + offs_m
    KBID_ptr = KBID + kbid_off + idx_n

    for idx_m in tl.range(0, M_BLOCKS):
        if tl.load(KBID_ptr) == 1:
            m_mask = offs_m < L - idx_m * BLOCK_M
            qp = tl.load(QPHI_ptrs, mask=m_mask[:, None], other=0.0)
            dol = tl.load(DOL_ptrs, mask=m_mask[:, None], other=0.0)
            ol = tl.load(OL_ptrs, mask=m_mask[:, None], other=0.0)
            den = tl.load(DEN_ptrs, mask=m_mask, other=1.0)
            g = dol.to(tl.float32) / den[:, None]
            s = tl.sum(dol.to(tl.float32) * ol.to(tl.float32), axis=1) / den
            dh -= tl.dot(tl.trans(qp), g.to(qp.dtype)).to(tl.float32)
            dz += tl.sum(qp.to(tl.float32) * s[:, None], axis=0)
        QPHI_ptrs += BLOCK_M * D
        OL_ptrs += BLOCK_M * D
        DOL_ptrs += BLOCK_M * D
        DEN_ptrs += BLOCK_M
        KBID_ptr += N_BLOCKS

    n_mask = offs_n[:, None] < L
    kp = tl.load(KPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :], mask=n_mask, other=0.0)
    vv = tl.load(V + qkv_off + offs_n[:, None] * D + offs_e[None, :], mask=n_mask, other=0.0)

    dkp = tl.dot(vv, tl.trans(dh).to(vv.dtype)).to(tl.float32) + dz[None, :]
    dv = tl.dot(kp, dh.to(kp.dtype)).to(tl.float32)

    tl.store(DKPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :],
             dkp.to(DKPHI.type.element_ty), mask=n_mask)
    tl.store(DV + qkv_off + offs_n[:, None] * D + offs_e[None, :],
             dv.to(DV.type.element_ty), mask=n_mask)


@triton.jit
def _lin_fwd2(
    QPHI, KPHI, V, HTOT, ZTOT, LUT, OL, DEN, OS, ALPHA, O,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FUSE_ALPHA: tl.constexpr,
    FUSE_PHI: tl.constexpr,
    SCORE_FORM: tl.constexpr,
):
    """Correction form of the complement linear branch.

    Instead of materializing H_i = H_tot - sum_sel phi(K_j)^T V_j (a D x D matrix per
    query block, i.e. 64 KB of loads per program), subtract the selected blocks in the
    (BLOCK_M x BLOCK_N) score domain:

        num_i = phi(Q_i) H_tot - sum_{j in sel(i)} (phi(Q_i) phi(K_j)^T) V_j
        den_i = phi(Q_i) z_tot - sum_{j in sel(i)} rowsum(phi(Q_i) phi(K_j)^T)

    HTOT = phi(K)^T V : (B,H,D,D) fp32  -- key-channel x value-channel; L2-resident
    ZTOT = colsum phi(K) : (B,H,D) fp32
    OS   = O_s of the sparse branch : (B,H,L,D)   (only read when FUSE_ALPHA)
    ALPHA: (M_BLOCKS,) fp32 in (0,1)              (only read when FUSE_ALPHA)
    O    = alpha * O_s + (1 - alpha) * O_l : (B,H,L,D)  (only written when FUSE_ALPHA)
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_off = idx_bh * L * D
    lut_off = (idx_bh * M_BLOCKS + idx_m) * topk
    den_off = idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)
    m_mask = offs_m[:, None] < L

    qp = tl.load(QPHI + qkv_off + offs_m[:, None] * D + offs_d[None, :], mask=m_mask, other=0.0)
    if FUSE_PHI:
        # phi(.) = row softmax over the D head channels, computed in registers so that
        # phi(Q) never has to be materialized in HBM
        qf = qp.to(tl.float32)
        qf = tl.exp(qf - tl.max(qf, axis=1)[:, None])
        qp = (qf / tl.sum(qf, axis=1)[:, None]).to(qp.dtype)
    # HTOT is stored in the same dtype as QPHI for the forward kernel, so no
    # per-program fp32->fp16 conversion of the D x D matrix is needed
    h = tl.load(HTOT + idx_bh * D * D + offs_d[:, None] * D + offs_e[None, :])
    z = tl.load(ZTOT + idx_bh * D + offs_d).to(tl.float32)
    KPHI_ptrs = KPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_off + offs_n[:, None] * D + offs_d[None, :]

    if SCORE_FORM:
        # cost per selected block: 2 * BLOCK_M * BLOCK_N * D MACs, no D x D accumulator
        num = tl.dot(qp, h).to(tl.float32)                   # (BLOCK_M, D)
        den = tl.sum(qp.to(tl.float32) * z[None, :], axis=1)  # (BLOCK_M,)
        for b in tl.range(topk):
            idx_n = tl.load(LUT + lut_off + b).to(tl.int64)
            n_mask = offs_n < L - idx_n * BLOCK_N
            kp = tl.load(KPHI_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
            vv = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
            w = tl.dot(qp, tl.trans(kp))                     # (BLOCK_M, BLOCK_N)
            w = tl.where(n_mask[None, :], w, 0.0)
            num -= tl.dot(w.to(vv.dtype), vv).to(tl.float32)
            den -= tl.sum(w.to(tl.float32), axis=1)
    else:
        # cost per selected block: BLOCK_N * D * D MACs (half of the score form when
        # BLOCK_M = 2 * BLOCK_N), plus one BLOCK_M x D x D matmul in the epilogue;
        # cheaper once several blocks are selected (low sparsity)
        hi = h.to(tl.float32)                                # (D key-ch, D value-ch)
        zi = z
        for b in tl.range(topk):
            idx_n = tl.load(LUT + lut_off + b).to(tl.int64)
            n_mask = offs_n < L - idx_n * BLOCK_N
            kp = tl.load(KPHI_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
            vv = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None], other=0.0)
            hi -= tl.dot(tl.trans(kp), vv).to(tl.float32)
            zi -= tl.sum(kp.to(tl.float32), axis=0)
        num = tl.dot(qp, hi.to(qp.dtype)).to(tl.float32)     # (BLOCK_M, D)
        den = tl.sum(qp.to(tl.float32) * zi[None, :], axis=1)  # (BLOCK_M,)

    den = tl.where(den >= 0, den + 1e-5, den - 1e-5)
    ol = num / den[:, None]

    tl.store(DEN + den_off + offs_m, den, mask=offs_m < L)
    if FUSE_ALPHA:
        a = tl.load(ALPHA + (idx_bh % H) * M_BLOCKS + idx_m).to(tl.float32)
        os_ = tl.load(OS + qkv_off + offs_m[:, None] * D + offs_d[None, :],
                      mask=m_mask, other=0.0).to(tl.float32)
        o = a * os_ + (1.0 - a) * ol
        tl.store(O + qkv_off + offs_m[:, None] * D + offs_d[None, :],
                 o.to(O.type.element_ty), mask=m_mask)
    else:
        tl.store(OL + qkv_off + offs_m[:, None] * D + offs_d[None, :],
                 ol.to(OL.type.element_ty), mask=m_mask)


def _precompute_global(qphi, kphi, v):
    """Dense parts shared by every query block.

    qphi, kphi, v : (B,H,L,D) fp16/bf16
    returns htot (B,H,D,D) fp32 [key-channel, value-channel], ztot (B,H,D) fp32
    """
    htot = (kphi.transpose(-1, -2) @ v).float().contiguous()      # (B,H,D,D)
    ztot = kphi.sum(dim=-2, dtype=torch.float32).contiguous()     # (B,H,D)
    return htot, ztot


def complement_linear_fused(q_raw, kphi, v, lut, topk, BLOCK_M, BLOCK_N, o_s, alpha,
                            score_form_max=1 << 30):
    """Inference-only path: complement linear branch fused with the alpha combination.

    q_raw : (B,H,L,D) raw queries -- phi(Q) is applied inside the kernel
    kphi  : (B,H,L,D) phi(K), materialized because the H_tot GEMM consumes it
    v, o_s: (B,H,L,D) ; alpha : (M_BLOCKS,) fp32 in (0,1)
    returns o = alpha * O_s + (1 - alpha) * O_l : (B,H,L,D)
    """
    qphi = q_raw
    B, H, L, D = qphi.shape
    M_BLOCKS = triton.cdiv(L, BLOCK_M)
    htot = (kphi.transpose(-1, -2) @ v).contiguous()                 # (B,H,D,D) same dtype as V
    ztot = kphi.sum(dim=-2, dtype=torch.float32).contiguous()        # (B,H,D)
    o = torch.empty_like(v)                                          # (B,H,L,D)
    den = torch.empty(qphi.shape[:-1], device=qphi.device, dtype=torch.float32)  # (B,H,L)
    _lin_fwd2[(M_BLOCKS, B * H)](
        qphi, kphi, v, htot, ztot, lut, o, den, o_s, alpha, o,
        topk, L, M_BLOCKS, D, BLOCK_M, BLOCK_N, True, True, topk <= score_form_max,
        num_warps=8, num_stages=3,
    )
    return o


class _complement_linear_attention(torch.autograd.Function):
    """O_l of SLA2 Eq. 14, restricted to the (1 - M) complement blocks."""

    @staticmethod
    def forward(ctx, qphi, kphi, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N):
        # qphi/kphi/v: (B, H, L, D) contiguous, fp16/bf16
        assert qphi.is_contiguous() and kphi.is_contiguous() and v.is_contiguous()
        B, H, L, D = qphi.shape
        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        htot, ztot = _precompute_global(qphi, kphi, v)

        ol = torch.empty_like(v)                                           # (B,H,L,D)
        den = torch.empty(qphi.shape[:-1], device=qphi.device, dtype=torch.float32)  # (B,H,L)

        _lin_fwd2[(M_BLOCKS, B * H)](
            qphi, kphi, v, htot.to(qphi.dtype).contiguous(), ztot, lut, ol, den, ol, ztot, ol,
            topk, L, M_BLOCKS, D, BLOCK_M, BLOCK_N, False, False, topk <= 4,
            num_warps=8, num_stages=3,
        )

        ctx.save_for_backward(qphi, kphi, v, htot, ztot, lut, k_block_id, ol, den)
        ctx.topk, ctx.BLOCK_M, ctx.BLOCK_N = topk, BLOCK_M, BLOCK_N
        return ol

    @staticmethod
    def backward(ctx, dol):
        qphi, kphi, v, htot, ztot, lut, k_block_id, ol, den = ctx.saved_tensors
        dol = dol.contiguous()
        B, H, L, D = qphi.shape
        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        M_BLOCKS, N_BLOCKS = triton.cdiv(L, BLOCK_M), triton.cdiv(L, BLOCK_N)

        # global (all-block) gradients of H_tot / z_tot
        g = dol.float() / den[..., None]                                   # (B,H,L,D)
        s = (dol.float() * ol.float()).sum(-1) / den                       # (B,H,L)
        dhtot = (qphi.float().transpose(-1, -2) @ g).contiguous()          # (B,H,D,D)
        dztot = -(qphi.float() * s[..., None]).sum(-2).contiguous()        # (B,H,D)

        dqphi = torch.empty_like(qphi)
        dkphi = torch.empty_like(kphi)
        dv = torch.empty_like(v)

        _lin_bwd_dq[(M_BLOCKS, B * H)](
            qphi, kphi, v, htot, ztot, lut, ol, den, dol, dqphi,
            ctx.topk, L, M_BLOCKS, D, BLOCK_M, BLOCK_N,
            num_warps=8, num_stages=2,
        )
        _lin_bwd_dkdv[(N_BLOCKS, B * H)](
            qphi, kphi, v, dhtot, dztot, ol, den, dol, k_block_id, dkphi, dv,
            L, M_BLOCKS, N_BLOCKS, D, BLOCK_M, BLOCK_N,
            num_warps=8, num_stages=2,
        )
        return dqphi, dkphi, dv, None, None, None, None, None


@triton.jit
def _hblocks(KPHI, V, HALL, ZALL, L, D: tl.constexpr, BLOCK_N: tl.constexpr,
             TRANSPOSE: tl.constexpr = False):
    """Per-key-block linear-attention state.

    HALL[b,h,j] = phi(K_j)^T V_j : (D key-channel, D value-channel), V's dtype
    (with TRANSPOSE the two channel axes are swapped, which is the layout the
    CuTeDSL kernel wants for its MMA B operand)
    ZALL[b,h,j] = colsum phi(K_j) : (D,) fp32
    Computed once per forward; afterwards a query block only has to ADD the
    D x D states of its selected blocks instead of re-running two matmuls each.
    """
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    n_blocks = tl.cdiv(L, BLOCK_N)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)
    n_mask = offs_n[:, None] < L

    kp = tl.load(KPHI + idx_bh * L * D + offs_n[:, None] * D + offs_d[None, :],
                 mask=n_mask, other=0.0)
    vv = tl.load(V + idx_bh * L * D + offs_n[:, None] * D + offs_e[None, :],
                 mask=n_mask, other=0.0)
    h = tl.dot(tl.trans(kp), vv)                                   # (D, D)
    z = tl.sum(kp.to(tl.float32), axis=0)                          # (D,)
    base = (idx_bh * n_blocks + idx_n)
    if TRANSPOSE:
        tl.store(HALL + base * D * D + offs_e[:, None] * D + offs_d[None, :],
                 tl.trans(h).to(HALL.type.element_ty))
    else:
        tl.store(HALL + base * D * D + offs_d[:, None] * D + offs_e[None, :],
                 h.to(HALL.type.element_ty))
    tl.store(ZALL + base * D + offs_d, z)


@triton.jit
def _lin_fwd3(
    Q, HALL, ZALL, HTOT, ZTOT, LUT, OS, ALPHA, O,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    H: tl.constexpr,
):
    """State-subtraction form: H_i = H_tot - sum_{j in sel(i)} HALL[j] (no matmul in the loop).

    Q     : (B,H,L,D) raw queries, phi(.) applied in registers
    HALL  : (B,H,Nb,D,D), ZALL : (B,H,Nb,D) fp32   per-key-block states
    HTOT  : (B,H,D,D), ZTOT : (B,H,D) fp32         all-block totals
    OS    : (B,H,L,D) sparse-branch output
    ALPHA : (H, M_BLOCKS) fp32 -- per-head, per-query-block mixing ratio (Eq. 7's
            alpha is per query row; heads differ enough that sharing alpha across
            them destroys the output when alpha is calibrated rather than trained)
    O     : (B,H,L,D) = alpha * O_s + (1 - alpha) * O_l
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_off = idx_bh * L * D
    lut_off = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)
    m_mask = offs_m[:, None] < L

    q = tl.load(Q + qkv_off + offs_m[:, None] * D + offs_d[None, :], mask=m_mask, other=0.0)
    qf = q.to(tl.float32)
    qf = tl.exp(qf - tl.max(qf, axis=1)[:, None])
    qp = (qf / tl.sum(qf, axis=1)[:, None]).to(q.dtype)            # phi(Q_i)

    h = tl.load(HTOT + idx_bh * D * D + offs_d[:, None] * D + offs_e[None, :]).to(tl.float32)
    z = tl.load(ZTOT + idx_bh * D + offs_d).to(tl.float32)
    for b in tl.range(topk):
        idx_n = tl.load(LUT + lut_off + b).to(tl.int64)
        base = idx_bh * N_BLOCKS + idx_n
        h -= tl.load(HALL + base * D * D + offs_d[:, None] * D + offs_e[None, :]).to(tl.float32)
        z -= tl.load(ZALL + base * D + offs_d)

    num = tl.dot(qp, h.to(qp.dtype)).to(tl.float32)                # (BLOCK_M, D)
    den = tl.sum(qp.to(tl.float32) * z[None, :], axis=1)           # (BLOCK_M,)
    den = tl.where(den >= 0, den + 1e-5, den - 1e-5)
    o_l = num / den[:, None]

    a = tl.load(ALPHA + (idx_bh % H) * M_BLOCKS + idx_m).to(tl.float32)
    os_ = tl.load(OS + qkv_off + offs_m[:, None] * D + offs_e[None, :], mask=m_mask, other=0.0)
    o = a * os_.to(tl.float32) + (1.0 - a) * o_l
    tl.store(O + qkv_off + offs_m[:, None] * D + offs_e[None, :],
             o.to(O.type.element_ty), mask=m_mask)


def complement_linear_states(q_raw, kphi, v, lut, topk, BLOCK_M, BLOCK_N, o_s, alpha,
                             num_warps=8, num_stages=3):
    """Inference path using precomputed per-key-block states (see `_lin_fwd3`).

    alpha : (H, M_BLOCKS) fp32, per-head per-query-block mixing ratio.
    """
    B, H, L, D = q_raw.shape
    M_BLOCKS, N_BLOCKS = triton.cdiv(L, BLOCK_M), triton.cdiv(L, BLOCK_N)

    hall = torch.empty((B, H, N_BLOCKS, D, D), device=v.device, dtype=v.dtype)
    zall = torch.empty((B, H, N_BLOCKS, D), device=v.device, dtype=torch.float32)
    _hblocks[(N_BLOCKS, B * H)](kphi, v, hall, zall, L, D, BLOCK_N, num_warps=4)

    htot = hall.sum(dim=2, dtype=torch.float32).contiguous()       # (B,H,D,D)
    ztot = zall.sum(dim=2).contiguous()                            # (B,H,D)

    o = torch.empty_like(v)
    _lin_fwd3[(M_BLOCKS, B * H)](
        q_raw, hall, zall, htot, ztot, lut, o_s, alpha, o,
        topk, L, M_BLOCKS, N_BLOCKS, D, BLOCK_M, H,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o
