"""JVP (forward-mode Jacobian-vector product) of SLA2 sparse-linear attention,
for fusing SLA2 (SLA/SLA2/core.py) with the MLA (multi-head latent attention)
JVP pipeline in mla_jvp_fast.py.

SLA2 attention (SLA2 Eq. 12-14), per head:

    O   = a .* O_s + (1 - a) .* O_l
    O_s = softmax(Q K^T * scale  restricted to routed blocks M) V
    O_l = (Qphi Kphi^T .* (1 - M)) V / ((Qphi Kphi^T .* (1 - M)) 1 + eps_l)
    M   = hard top-k block mask from the router;  a = sigmoid(alpha_logit)

JVP structure (tangents dq/dk/dv w.r.t. the network input only; router mask M,
LUT and mixing ratio a are piecewise constant in the input, so their tangent
is zero almost everywhere and they are reused from the primal forward):

    sparse branch: flash-JVP recurrence (same math as triton_block_jvp.
        _flash_jvp_kernel_impl) but the key-tile loop walks only the routed
        key blocks listed in the LUT.
    linear branch: per-key-block states
        Hb[n] = sum_{j in block n} kphi_j v_j^T   (D, Dv)
        zb[n] = sum_{j in block n} kphi_j         (D,)
        complement state of query block m: H_c[m] = H_all - sum_{n in LUT[m]} Hb[n]
        O_l = (qphi^T H_c) / (qphi^T z_c + eps_l)
        which is bilinear in (qphi, states), so the tangent is a sum of two
        einsum terms per numerator/denominator plus a quotient rule.
    phi = softmax over the head channel:  dphi = phi .* (dx - <phi, dx>).

Shape symbols (consistent with SLA2/core.py):
    B  : batch;  H : heads;  L : tokens;  D : qk head dim;  Dv : value head dim
    Mb : number of query blocks  = ceil(L / bq)
    Nb : number of key blocks    = ceil(L / bk)
    T  : topk, routed key blocks per query block
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["route_blocks", "sla2_mla_jvp", "sla2_dense_ref"]

_EPS_L = 1e-5  # linear-branch normalizer epsilon (matches SLA2 sla2_dense)


# ---------------------------------------------------------------------------
# Router (identity-projection version of SLA2/router.py, torch ops only).
# The learnable proj_q/proj_k start at identity, so this reproduces the
# router's init-time behaviour; learnable projections can be added by
# passing proj_q/proj_k.
# ---------------------------------------------------------------------------
def _block_pool(x, blk):
    """Mean-pool tokens into blocks.

    Args:
        x: (B, H, L, D) fp32/fp16 tokens.
        blk: block length (int).

    Returns:
        (B, H, ceil(L/blk), D) fp32 per-block means (tail block zero-padded
        mean over the real tokens only).
    """
    B, H, L, D = x.shape
    nb = (L + blk - 1) // blk
    pad = nb * blk - L
    xf = x.float()
    if pad:
        xf = F.pad(xf, (0, 0, 0, pad))                      # (B,H,nb*blk,D)
    xb = xf.view(B, H, nb, blk, D)
    cnt = torch.full((nb,), blk, device=x.device, dtype=torch.float32)
    if pad:
        cnt[-1] = blk - pad
    return xb.sum(3) / cnt.view(1, 1, nb, 1)                # (B,H,nb,D)


def route_blocks(q, k, topk_frac, bq, bk, proj_q=None, proj_k=None):
    """SLA2 router: pooled-score hard top-k over key blocks.

    Args:
        q, k: (B, H, L, D) primal queries / keys.
        topk_frac: fraction of key blocks routed to the sparse branch.
        bq, bk: query / key block lengths.
        proj_q, proj_k: optional (D, D) router projection weights.

    Returns:
        lut: (B, H, Mb, T) int32 routed key-block ids (unsorted).
        T:   topk count (int).
    """
    qb = _block_pool(q, bq)                                  # (B,H,Mb,D)
    kb = _block_pool(k, bk)                                  # (B,H,Nb,D)
    # smooth-k (SageAttention): subtract the per-(B,H) mean key before scoring
    kb = kb - kb.mean(2, keepdim=True)
    if proj_q is not None:
        qb = qb @ proj_q.t().to(qb.dtype)
    if proj_k is not None:
        kb = kb @ proj_k.t().to(kb.dtype)
    pc = (qb @ kb.transpose(-1, -2)) * (q.shape[-1] ** -0.5)  # (B,H,Mb,Nb)
    nb = pc.shape[-1]
    T = max(1, min(nb, int(topk_frac * nb)))
    lut = torch.topk(pc, T, dim=-1, sorted=False).indices.to(torch.int32)
    return lut.contiguous(), T


# ---------------------------------------------------------------------------
# Sparse branch: routed-block flash JVP.  Body copied from triton_block_jvp.
# _flash_jvp_kernel_impl; the only structural change is that the key-tile
# loop walks LUT entries (BLOCK_N == bk) instead of the whole sequence.
# ---------------------------------------------------------------------------
@triton.jit
def _sla2_sparse_jvp_kernel(
    Q, K, V, DQ, DK, DV, O, DO, LUT,
    s_qb, s_qh, s_qs,          # strides of the (B, H, L, D) qkv tensors
    s_ob, s_oh, s_os,          # strides of the (B, H, L, Dv) outputs
    s_lb, s_lh, s_lm,          # strides of the (B, H, Mb, T) LUT
    H, L, T, scale,
    HEAD_DIM: tl.constexpr,    # D = Dv
    BLOCK_M: tl.constexpr,     # query tile, divides bq
    BLOCK_N: tl.constexpr,     # key tile == bk
    TILES_PER_BQ: tl.constexpr,  # bq // BLOCK_M
    FP16_MMA: tl.constexpr,
):
    pid_m = tl.program_id(0)               # global query-tile index
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    mb = pid_m // TILES_PER_BQ             # query block id (LUT row)

    base = b * s_qb + h * s_qh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # token rows
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < L

    qp = base + offs_m[:, None] * s_qs + offs_d[None, :]
    q = tl.load(Q + qp, mask=mask_m[:, None], other=0.0)
    dq = tl.load(DQ + qp, mask=mask_m[:, None], other=0.0)
    # interleaved [dq|q] so one dot of width 2*D gives dQ K^T + Q dK^T
    qj = tl.reshape(tl.join(dq, q), (BLOCK_M, 2 * HEAD_DIM))

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    mu_i = tl.zeros([BLOCK_M], tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    acc_do = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E
    lut_base = LUT + b * s_lb + h * s_lh + mb * s_lm

    for j in range(0, T):
        nb_id = tl.load(lut_base + j)
        offs = nb_id * BLOCK_N + offs_n                    # key token rows
        mask_n = offs < L
        kp = base + offs[:, None] * s_qs + offs_d[None, :]
        k = tl.load(K + kp, mask=mask_n[:, None], other=0.0)
        dk = tl.load(DK + kp, mask=mask_n[:, None], other=0.0)
        kj = tl.reshape(tl.join(k, dk), (BLOCK_N, 2 * HEAD_DIM))

        if FP16_MMA:
            s_f = tl.dot(q, tl.trans(k), out_dtype=tl.float16).to(tl.float32)
            ds = tl.dot(qj, tl.trans(kj),
                        out_dtype=tl.float16).to(tl.float32) * scale
        else:
            # fp32 io is the reference/testing path: IEEE dots, not tf32
            s_f = tl.dot(q, tl.trans(k), input_precision="ieee")
            ds = tl.dot(qj, tl.trans(kj), input_precision="ieee") * scale
        s2 = tl.where(mask_n[None, :], s_f * qk_scale, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s2, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s2 - m_new[:, None])
        t = p * ds

        l_i = l_i * alpha + tl.sum(p, 1)
        mu_i = mu_i * alpha + tl.sum(t, 1)

        v = tl.load(V + kp, mask=mask_n[:, None], other=0.0)
        dv = tl.load(DV + kp, mask=mask_n[:, None], other=0.0)
        if FP16_MMA:
            pc = p.to(V.dtype.element_ty)
            tc = t.to(V.dtype.element_ty)
            o_t = tl.dot(pc, v, out_dtype=tl.float16).to(tl.float32)
            do_t = (tl.dot(tc, v, out_dtype=tl.float16)
                    + tl.dot(pc, dv, out_dtype=tl.float16)).to(tl.float32)
        else:
            o_t = tl.dot(p, v, input_precision="ieee")
            do_t = (tl.dot(t, v, input_precision="ieee")
                    + tl.dot(p, dv, input_precision="ieee"))
        acc_o = acc_o * alpha[:, None] + o_t
        acc_do = acc_do * alpha[:, None] + do_t
        m_i = m_new

    o = acc_o / l_i[:, None]
    do = acc_do / l_i[:, None] - (mu_i / l_i)[:, None] * o

    op = b * s_ob + h * s_oh + offs_m[:, None] * s_os + offs_d[None, :]
    tl.store(O + op, o.to(O.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DO + op, do.to(DO.dtype.element_ty), mask=mask_m[:, None])


def _sparse_jvp(q, k, v, dq, dk, dv, lut, T, bq, bk, scale, block_m=64):
    """Routed-block flash JVP wrapper.

    Args:
        q, k, v, dq, dk, dv: (B, H, L, D) contiguous, fp16 or fp32.
        lut: (B, H, Mb, T) int32 routed key-block ids.
        T: topk count;  bq, bk: block lengths (BLOCK_N == bk).
        scale: softmax scale 1/sqrt(D).
        block_m: query tile (must divide bq).

    Returns:
        (o_s, do_s): (B, H, L, D) fp32 sparse-branch output and tangent.
    """
    B, H, L, D = q.shape
    if q.dtype == torch.float32:
        # IEEE fp32 tiles need ~2x the shared memory of fp16; halve the
        # query tile to stay under the 100 KB smem limit.
        block_m = min(block_m, 32)
    assert bq % block_m == 0, (bq, block_m)
    Mb = lut.shape[2]
    o = torch.empty(B, H, L, D, device=q.device, dtype=torch.float32)
    do = torch.empty_like(o)
    grid = (Mb * (bq // block_m), B * H)
    _sla2_sparse_jvp_kernel[grid](
        q, k, v, dq, dk, dv, o, do, lut,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lut.stride(0), lut.stride(1), lut.stride(2),
        H, L, T, scale,
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=bk,
        TILES_PER_BQ=bq // block_m,
        FP16_MMA=(q.dtype != torch.float32),
        num_warps=4,
        # fp32 operands double every K/V tile in shared memory; stages=1
        # keeps the kernel under the 100 KB smem limit for all dtypes.
        num_stages=1 if q.dtype == torch.float32 else 2,
    )
    return o, do


# ---------------------------------------------------------------------------
# Linear branch: complement per-key-block states, JVP by bilinearity.
# ---------------------------------------------------------------------------
def _phi_jvp(x, dx):
    """phi = softmax over the head channel D, with its JVP.

    Args:
        x, dx: (B, H, L, D) primal / tangent.

    Returns:
        (phi, dphi): (B, H, L, D) fp32.
    """
    phi = F.softmax(x.float(), dim=-1)                       # (B,H,L,D)
    dxf = dx.float()
    dphi = phi * (dxf - (phi * dxf).sum(-1, keepdim=True))   # (B,H,L,D)
    return phi, dphi


def _linear_jvp(qphi, dqphi, kphi, dkphi, v, dv, lut, bq, bk):
    """Complement linear branch O_l with tangent, via per-key-block states.

    Args:
        qphi, dqphi: (B, H, L, D) fp32 phi(q) and tangent.
        kphi, dkphi: (B, H, L, D) fp32 phi(k) and tangent.
        v, dv: (B, H, L, Dv) values and tangent.
        lut: (B, H, Mb, T) int32 routed key-block ids (excluded blocks).
        bq, bk: query / key block lengths.

    Returns:
        (o_l, do_l): (B, H, L, Dv) fp32 linear-branch output and tangent.
    """
    B, H, L, D = qphi.shape
    Dv = v.shape[-1]
    Nb = (L + bk - 1) // bk
    Mb = (L + bq - 1) // bq
    padk = Nb * bk - L
    padq = Mb * bq - L

    def kblocks(x, width):
        # (B,H,L,width) -> (B,H,Nb,bk,width), tail zero-padded
        xf = x.float()
        if padk:
            xf = F.pad(xf, (0, 0, 0, padk))
        return xf.view(B, H, Nb, bk, width)

    kb, dkb = kblocks(kphi, D), kblocks(dkphi, D)            # (B,H,Nb,bk,D)
    vb, dvb = kblocks(v, Dv), kblocks(dv, Dv)                # (B,H,Nb,bk,Dv)

    # per-key-block states and their tangents
    Hb = torch.einsum("bhntd,bhntv->bhndv", kb, vb)          # (B,H,Nb,D,Dv)
    dHb = (torch.einsum("bhntd,bhntv->bhndv", dkb, vb)
           + torch.einsum("bhntd,bhntv->bhndv", kb, dvb))    # (B,H,Nb,D,Dv)
    zb = kb.sum(3)                                           # (B,H,Nb,D)
    dzb = dkb.sum(3)                                         # (B,H,Nb,D)

    H_all = Hb.sum(2)                                        # (B,H,D,Dv)
    dH_all = dHb.sum(2)
    z_all = zb.sum(2)                                        # (B,H,D)
    dz_all = dzb.sum(2)

    # routed (excluded) sums per query block, gathered by LUT
    T = lut.shape[-1]
    idx = lut.long()                                         # (B,H,Mb,T)
    gH = torch.gather(
        Hb, 2, idx.view(B, H, Mb * T, 1, 1).expand(-1, -1, -1, D, Dv)
    ).view(B, H, Mb, T, D, Dv).sum(3)                        # (B,H,Mb,D,Dv)
    gdH = torch.gather(
        dHb, 2, idx.view(B, H, Mb * T, 1, 1).expand(-1, -1, -1, D, Dv)
    ).view(B, H, Mb, T, D, Dv).sum(3)
    gz = torch.gather(
        zb, 2, idx.view(B, H, Mb * T, 1).expand(-1, -1, -1, D)
    ).view(B, H, Mb, T, D).sum(3)                            # (B,H,Mb,D)
    gdz = torch.gather(
        dzb, 2, idx.view(B, H, Mb * T, 1).expand(-1, -1, -1, D)
    ).view(B, H, Mb, T, D).sum(3)

    H_c = H_all.unsqueeze(2) - gH                            # (B,H,Mb,D,Dv)
    dH_c = dH_all.unsqueeze(2) - gdH
    z_c = z_all.unsqueeze(2) - gz                            # (B,H,Mb,D)
    dz_c = dz_all.unsqueeze(2) - gdz

    def qblocks(x):
        # (B,H,L,D) -> (B,H,Mb,bq,D), tail zero-padded
        xf = x.float()
        if padq:
            xf = F.pad(xf, (0, 0, 0, padq))
        return xf.view(B, H, Mb, bq, D)

    qg, dqg = qblocks(qphi), qblocks(dqphi)                  # (B,H,Mb,bq,D)

    num = torch.einsum("bhmtd,bhmdv->bhmtv", qg, H_c)        # (B,H,Mb,bq,Dv)
    dnum = (torch.einsum("bhmtd,bhmdv->bhmtv", dqg, H_c)
            + torch.einsum("bhmtd,bhmdv->bhmtv", qg, dH_c))
    den = torch.einsum("bhmtd,bhmd->bhmt", qg, z_c) + _EPS_L  # (B,H,Mb,bq)
    dden = (torch.einsum("bhmtd,bhmd->bhmt", dqg, z_c)
            + torch.einsum("bhmtd,bhmd->bhmt", qg, dz_c))

    o_l = num / den.unsqueeze(-1)                            # (B,H,Mb,bq,Dv)
    do_l = (dnum - o_l * dden.unsqueeze(-1)) / den.unsqueeze(-1)

    o_l = o_l.view(B, H, Mb * bq, Dv)[:, :, :L]              # (B,H,L,Dv)
    do_l = do_l.view(B, H, Mb * bq, Dv)[:, :, :L]
    return o_l, do_l


# ---------------------------------------------------------------------------
# Full op and dense reference.
# ---------------------------------------------------------------------------
def sla2_mla_jvp(q, k, v, dq, dk, dv, alpha, topk_frac, bq, bk,
                 lut=None, T=None, block_m=64):
    """SLA2 attention (primal, tangent) from qkv dual pairs.

    Args:
        q, k, v: (B, H, L, D) primal (D == Dv assumed, head_dim 64 in MLA).
        dq, dk, dv: (B, H, L, D) tangents.
        alpha: (H, Mb) mixing ratio in (0,1) (already sigmoided), or scalar.
        topk_frac: routed fraction of key blocks (ignored when lut given).
        bq, bk: query / key block lengths.
        lut, T: optional precomputed routing (from the primal training pass).
        block_m: query tile of the sparse kernel.

    Returns:
        (o, do): (B, H, L, D) fp32 SLA2 output and tangent.
    """
    B, H, L, D = q.shape
    scale = D ** -0.5
    if lut is None:
        lut, T = route_blocks(q, k, topk_frac, bq, bk)

    qc = [t.contiguous() for t in (q, k, v, dq, dk, dv)]
    o_s, do_s = _sparse_jvp(*qc, lut, T, bq, bk, scale, block_m)

    qphi, dqphi = _phi_jvp(q, dq)
    kphi, dkphi = _phi_jvp(k, dk)
    o_l, do_l = _linear_jvp(qphi, dqphi, kphi, dkphi, v, dv, lut, bq, bk)

    Mb = lut.shape[2]
    if torch.is_tensor(alpha) and alpha.dim() == 2:
        # (H, Mb) -> (1, H, L, 1) per-token ratio
        a = alpha.float().repeat_interleave(bq, dim=-1)[:, :L]
        a = a.view(1, H, L, 1)
    else:
        a = torch.as_tensor(alpha, device=q.device, dtype=torch.float32)
    o = a * o_s + (1.0 - a) * o_l
    do = a * do_s + (1.0 - a) * do_l
    return o, do


def sla2_dense_ref(q, k, v, lut, alpha, bq, bk):
    """Dense fp32 SLA2 reference (mask from LUT), differentiable, for tests.

    Args:
        q, k, v: (B, H, L, D) fp32.
        lut: (B, H, Mb, T) int32 routed key-block ids.
        alpha: (H, Mb) mixing ratio in (0,1).
        bq, bk: block lengths.

    Returns:
        o: (B, H, L, D) fp32.
    """
    B, H, L, D = q.shape
    Mb, T = lut.shape[2], lut.shape[3]
    Nb = (L + bk - 1) // bk
    mc = torch.zeros(B, H, Mb, Nb, device=q.device)
    mc.scatter_(-1, lut.long(), 1.0)                          # (B,H,Mb,Nb)
    m = mc.repeat_interleave(bq, dim=-2).repeat_interleave(bk, dim=-1)
    m = m[..., :L, :L]                                        # (B,H,L,L)

    s = (q @ k.transpose(-1, -2)) * D ** -0.5                 # (B,H,L,L)
    p = torch.softmax(s.masked_fill(m <= 0, float("-inf")), dim=-1)
    o_s = p @ v                                               # (B,H,L,D)

    qphi = F.softmax(q, dim=-1)
    kphi = F.softmax(k, dim=-1)
    w = (qphi @ kphi.transpose(-1, -2)) * (1.0 - m)           # (B,H,L,L)
    o_l = (w @ v) / (w.sum(-1, keepdim=True) + _EPS_L)

    a = alpha.repeat_interleave(bq, dim=-1)[:, :L].view(1, H, L, 1)
    return a * o_s + (1.0 - a) * o_l
