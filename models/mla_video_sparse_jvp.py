"""MLA_Video_Sparse_JVP: VSA (video sparse attention, fastvideo) x SLA2
(sparse-linear attention) fused for MLA, with forward-mode JVP.

Composition per head (tokens in TILE-MAJOR order: 3D video tiles of
`tile_elems` tokens each come first, the `prefix_len` conditioning tokens
ride at the end as one always-attended ragged tail block). NO coarse
branch: VSA contributes the 3D-cube blocks, SLA2 contributes the router
and the two-branch structure:

    O   = a .* O_s + (1 - a) .* O_l
    O_s = softmax(Q K^T / sqrt(D)  on {topk tiles from the router}
                                       + {prefix tail block}) V
    O_l = complement linear branch (SLA2) over the NON-selected tiles
          (prefix block excluded from the linear branch)
    router (SLA2): mean-pool tiles, smooth-k (subtract mean key),
        learnable proj_q / proj_k (D x D, identity init), scores
        pool(Q) W_q (pool(K) W_k)^T / sqrt(D), hard top-k -> LUT

JVP: the hard top-k LUT is piecewise constant (zero tangent a.e.,
reused from the primal); the constant a mixes linearly; the sparse
branch is the routed-block flash-JVP recurrence in the Triton kernel
`_mla_video_sparse_jvp_kernel`; the linear branch is bilinear in
per-tile states, tangents by einsum.

Shape symbols:
    B : batch;  H : heads;  L : tokens = Np * E + P;  D : head dim (= Dv)
    Np : number of video tiles;  E : tile_elems (tokens per 3D tile)
    P : prefix_len (conditioning tokens, P < E);  T : topk tiles per tile
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["tile_permutation", "route_tiles", "mla_video_sparse_jvp",
           "mvs_dense_ref"]

_EPS_L = 1e-5  # linear-branch normalizer epsilon (matches SLA2)


# ---------------------------------------------------------------------------
# 3D tile permutation: t-major token order -> tile-major order.
# ---------------------------------------------------------------------------
def tile_permutation(grid, tile, prefix_len, device):
    """Index maps between model order and tile-major order.

    Model order: [prefix tokens (P)] + [patch tokens, t-major over grid].
    Tile order:  [patch tokens grouped by 3D tile] + [prefix tokens].

    Args:
        grid: (T, Hh, Ww) patch-grid sizes; each must divide by tile.
        tile: (tt, th, tw) tile sizes (E = tt*th*tw tokens per tile).
        prefix_len: P, number of conditioning tokens.
        device: torch device.

    Returns:
        perm: (L,) int64, tile_order_tokens = model_order_tokens[perm]
        inv:  (L,) int64, model_order_tokens = tile_order_tokens[inv]
        n_tiles: Np (int)
    """
    T, Hh, Ww = grid
    tt, th, tw = tile
    assert T % tt == 0 and Hh % th == 0 and Ww % tw == 0, (grid, tile)
    idx = torch.arange(T * Hh * Ww, device=device).view(T, Hh, Ww)
    # (nt, nh, nw, tt, th, tw) tile blocks, then flatten tile-major
    idx = (idx.view(T // tt, tt, Hh // th, th, Ww // tw, tw)
              .permute(0, 2, 4, 1, 3, 5).reshape(-1))
    patch_perm = idx + prefix_len          # patch tokens follow the prefix
    prefix_ids = torch.arange(prefix_len, device=device)
    perm = torch.cat([patch_perm, prefix_ids])            # (L,)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)
    n_tiles = (T // tt) * (Hh // th) * (Ww // tw)
    return perm, inv, n_tiles


# ---------------------------------------------------------------------------
# Sparse branch: routed-tile flash JVP with an always-attended prefix tail.
# Same dual-accumulator recurrence as models/sla2_mla_jvp.py; the key-tile
# loop walks T LUT entries plus (when HAS_PREFIX) the ragged tail block that
# starts at token Np*E.
# ---------------------------------------------------------------------------
@triton.jit
def _mla_video_sparse_jvp_kernel(
    Q, K, V, DQ, DK, DV, O, DO, LUT,
    s_qb, s_qh, s_qs,          # strides of the (B, H, L, D) qkv tensors
    s_ob, s_oh, s_os,          # strides of the (B, H, L, D) outputs
    s_lb, s_lh, s_lm,          # strides of the (B, H, Mb, T) LUT
    H, L, T, NPE, scale,       # NPE = Np * E, start of the prefix tail
    HEAD_DIM: tl.constexpr,    # D = Dv
    BLOCK_M: tl.constexpr,     # query tile rows per program (divides E)
    BLOCK_N: tl.constexpr,     # key tile length == E
    TILES_PER_BQ: tl.constexpr,  # E // BLOCK_M
    HAS_PREFIX: tl.constexpr,  # attend the ragged tail block after the LUT
    FP16_MMA: tl.constexpr,
):
    pid_m = tl.program_id(0)               # global query-row-tile index
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    mb = pid_m // TILES_PER_BQ             # query block id (LUT row)

    base = b * s_qb + h * s_qh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query token rows
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < L

    qp = base + offs_m[:, None] * s_qs + offs_d[None, :]
    q = tl.load(Q + qp, mask=mask_m[:, None], other=0.0)
    dq = tl.load(DQ + qp, mask=mask_m[:, None], other=0.0)
    # interleaved [dq|q]: one dot of width 2*D gives dQ K^T + Q dK^T
    qj = tl.reshape(tl.join(dq, q), (BLOCK_M, 2 * HEAD_DIM))

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    mu_i = tl.zeros([BLOCK_M], tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    acc_do = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E
    lut_base = LUT + b * s_lb + h * s_lh + mb * s_lm

    n_iters = T + 1 if HAS_PREFIX else T
    for j in range(0, n_iters):
        if HAS_PREFIX and j == T:
            n0 = NPE                        # ragged prefix tail block
        else:
            n0 = tl.load(lut_base + j) * BLOCK_N
        offs = n0 + offs_n                  # key token rows
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


def _sparse_jvp(q, k, v, dq, dk, dv, lut, T, E, scale, has_prefix,
                block_m=64):
    """Routed-tile flash JVP wrapper.

    Args:
        q..dv: (B, H, L, D) contiguous fp16/fp32, tile-major token order.
        lut: (B, H, Mb, T) int32 selected tile ids (Mb = number of query
            blocks incl. the prefix tail block when present).
        T: topk;  E: tile length (BLOCK_N);  scale: 1/sqrt(D).
        has_prefix: append the ragged tail block (tokens Np*E..L) to every
            query block's key list.
        block_m: query rows per program (must divide E).

    Returns:
        (o_s, do_s): (B, H, L, D) fp32.
    """
    B, H, L, D = q.shape
    if q.dtype == torch.float32:
        block_m = min(block_m, 32)  # IEEE fp32 tiles: stay under smem limit
    assert E % block_m == 0, (E, block_m)
    Mb = lut.shape[2]
    NPE = (L // E) * E if has_prefix else L  # start of the prefix tail
    o = torch.empty(B, H, L, D, device=q.device, dtype=torch.float32)
    do = torch.empty_like(o)
    grid = (Mb * (E // block_m), B * H)
    _mla_video_sparse_jvp_kernel[grid](
        q, k, v, dq, dk, dv, o, do, lut,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lut.stride(0), lut.stride(1), lut.stride(2),
        H, L, T, NPE, scale,
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=E,
        TILES_PER_BQ=E // block_m,
        HAS_PREFIX=has_prefix,
        FP16_MMA=(q.dtype != torch.float32),
        num_warps=4,
        num_stages=1 if q.dtype == torch.float32 else 2,
    )
    return o, do


# ---------------------------------------------------------------------------
# Router (SLA2): pooled 3D tiles, smooth-k, learnable projections, top-k.
# ---------------------------------------------------------------------------
def _pool_tiles(x, Np, E, prefix_cnt):
    """Mean-pool tokens into tiles (primal only; the hard router needs no
    tangent).

    Args:
        x: (B, H, L, D) tile-major; tokens Np*E..L are the ragged prefix
            tail when prefix_cnt > 0.
        Np: number of video tiles;  E: tile length.
        prefix_cnt: P tokens in the tail block (0 = no tail).

    Returns:
        xc: (B, H, Nb, D) pooled tiles, Nb = Np (+1 with prefix).
    """
    B, H, L, D = x.shape
    xf = x.float()
    xc = xf[:, :, :Np * E].view(B, H, Np, E, D).mean(3)      # (B,H,Np,D)
    if prefix_cnt:
        xt = xf[:, :, Np * E:].mean(2, keepdim=True)         # (B,H,1,D)
        xc = torch.cat([xc, xt], dim=2)                      # (B,H,Np+1,D)
    return xc


def route_tiles(q, k, topk_frac, Np, E, prefix_len,
                proj_q=None, proj_k=None):
    """SLA2 router on VSA 3D-cube tiles: hard top-k over pooled scores.

    Args:
        q, k: (B, H, L, D) primal queries / keys, tile-major order.
        topk_frac: fraction of the Np video tiles routed sparse.
        Np, E, prefix_len: tiling geometry.
        proj_q, proj_k: optional learnable (D, D) router projections
            (identity-initialized in the SLA2 module); None = identity.

    Returns:
        lut: (B, H, Mb, T) int32 selected video-tile ids (Mb = Np + 1 with
            a prefix tail, else Np).
        T: topk count (int).
    """
    qb = _pool_tiles(q, Np, E, prefix_len)                   # (B,H,Mb,D)
    kb = _pool_tiles(k, Np, E, prefix_len)[:, :, :Np]        # (B,H,Np,D)
    # smooth-k (SageAttention, kept from SLA/SLA2): subtract the mean key
    kb = kb - kb.mean(2, keepdim=True)
    if proj_q is not None:
        qb = qb @ proj_q.t().to(qb.dtype)
    if proj_k is not None:
        kb = kb @ proj_k.t().to(kb.dtype)
    pc = (qb @ kb.transpose(-1, -2)) * (q.shape[-1] ** -0.5)  # (B,H,Mb,Np)
    T = max(1, min(Np, int(topk_frac * Np)))
    lut = torch.topk(pc, T, dim=-1, sorted=False).indices.to(torch.int32)
    return lut.contiguous(), T


# ---------------------------------------------------------------------------
# Linear complement branch: SLA2 per-tile states, JVP by bilinearity.
# ---------------------------------------------------------------------------
def _phi_jvp(x, dx):
    """phi = softmax over the head channel D, with its JVP.

    Args: x, dx: (B, H, L, D).  Returns (phi, dphi): (B, H, L, D) fp32."""
    phi = F.softmax(x.float(), dim=-1)
    dxf = dx.float()
    dphi = phi * (dxf - (phi * dxf).sum(-1, keepdim=True))
    return phi, dphi


def _linear_jvp(qphi, dqphi, kphi, dkphi, v, dv, lut, Np, E):
    """Complement linear branch over NON-selected video tiles.

    The prefix tail block is excluded entirely (it is always in the sparse
    branch). Query side covers all L tokens (video + prefix queries).

    Args:
        qphi..dv: (B, H, L, D) fp32/fp16, tile-major order.
        lut: (B, H, Mb, T) int32 selected tile ids in [0, Np).
        Np: number of video tiles;  E: tile length.

    Returns:
        (o_l, do_l): (B, H, L, D) fp32.
    """
    B, H, L, D = qphi.shape
    Dv = v.shape[-1]
    Mb = lut.shape[2]

    kb = kphi.float()[:, :, :Np * E].view(B, H, Np, E, D)    # (B,H,Np,E,D)
    dkb = dkphi.float()[:, :, :Np * E].view(B, H, Np, E, D)
    vb = v.float()[:, :, :Np * E].view(B, H, Np, E, Dv)
    dvb = dv.float()[:, :, :Np * E].view(B, H, Np, E, Dv)

    Hb = torch.einsum("bhntd,bhntv->bhndv", kb, vb)          # (B,H,Np,D,Dv)
    dHb = (torch.einsum("bhntd,bhntv->bhndv", dkb, vb)
           + torch.einsum("bhntd,bhntv->bhndv", kb, dvb))
    zb = kb.sum(3)                                           # (B,H,Np,D)
    dzb = dkb.sum(3)

    H_all, dH_all = Hb.sum(2), dHb.sum(2)                    # (B,H,D,Dv)
    z_all, dz_all = zb.sum(2), dzb.sum(2)                    # (B,H,D)

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

    # query tokens grouped by block: Mb blocks of E rows (tail zero-padded)
    padq = Mb * E - L
    qg = F.pad(qphi.float(), (0, 0, 0, padq)).view(B, H, Mb, E, D)
    dqg = F.pad(dqphi.float(), (0, 0, 0, padq)).view(B, H, Mb, E, D)

    num = torch.einsum("bhmtd,bhmdv->bhmtv", qg, H_c)        # (B,H,Mb,E,Dv)
    dnum = (torch.einsum("bhmtd,bhmdv->bhmtv", dqg, H_c)
            + torch.einsum("bhmtd,bhmdv->bhmtv", qg, dH_c))
    den = torch.einsum("bhmtd,bhmd->bhmt", qg, z_c) + _EPS_L  # (B,H,Mb,E)
    dden = (torch.einsum("bhmtd,bhmd->bhmt", dqg, z_c)
            + torch.einsum("bhmtd,bhmd->bhmt", qg, dz_c))

    o_l = num / den.unsqueeze(-1)                            # (B,H,Mb,E,Dv)
    do_l = (dnum - o_l * dden.unsqueeze(-1)) / den.unsqueeze(-1)
    o_l = o_l.view(B, H, Mb * E, Dv)[:, :, :L]               # (B,H,L,Dv)
    do_l = do_l.view(B, H, Mb * E, Dv)[:, :, :L]
    return o_l, do_l


# ---------------------------------------------------------------------------
# Full op and dense reference.
# ---------------------------------------------------------------------------
def mla_video_sparse_jvp(q, k, v, dq, dk, dv, alpha,
                         topk_frac, Np, E, prefix_len,
                         proj_q=None, proj_k=None,
                         lut=None, T=None, block_m=64):
    """VSA-tiled SLA2 attention (primal, tangent) from qkv dual pairs.

    Inputs are in TILE-MAJOR order (use tile_permutation): Np 3D tiles of
    E tokens, then prefix_len conditioning tokens as a ragged tail.

    Args:
        q, k, v / dq, dk, dv: (B, H, L, D) primal / tangent, L = Np*E + P.
        alpha: (H, Mb) sparse/linear mix in (0,1), or scalar.
        topk_frac: fraction of video tiles routed (ignored when lut given).
        Np, E, prefix_len: tiling geometry.
        proj_q, proj_k: optional learnable (D, D) SLA2 router projections.
        lut, T: optional precomputed routing (from a primal training pass).
        block_m: query rows per kernel program.

    Returns:
        (o, do): (B, H, L, D) fp32.
    """
    B, H, L, D = q.shape
    scale = D ** -0.5
    P = prefix_len
    has_prefix = P > 0

    # ---- SLA2 router on VSA 3D tiles (hard: no tangent) ----
    if lut is None:
        lut, T = route_tiles(q, k, topk_frac, Np, E, P,
                             proj_q=proj_q, proj_k=proj_k)
        #   lut: (B, H, Mb, T) selected video-tile ids per query block

    # ---- sparse branch over routed tiles + prefix tail ----
    qs = [t.contiguous() for t in (q, k, v, dq, dk, dv)]
    o_s, do_s = _sparse_jvp(*qs, lut, T, E, scale, has_prefix, block_m)

    # ---- linear complement branch over non-selected video tiles ----
    qphi, dqphi = _phi_jvp(q, dq)
    kphi, dkphi = _phi_jvp(k, dk)
    o_l, do_l = _linear_jvp(qphi, dqphi, kphi, dkphi, v, dv, lut, Np, E)

    # ---- constant-a mix ----
    if torch.is_tensor(alpha) and alpha.dim() == 2:
        a = alpha.float().repeat_interleave(
            torch.tensor([E] * Np + ([P] if has_prefix else []),
                         device=q.device), dim=-1)
        a = a.view(1, H, L, 1)
    else:
        a = torch.as_tensor(alpha, device=q.device, dtype=torch.float32)
    o = a * o_s + (1.0 - a) * o_l
    do = a * do_s + (1.0 - a) * do_l
    return o, do


def mvs_dense_ref(q, k, v, alpha, lut, Np, E, prefix_len):
    """Dense fp32 reference of the fused op, differentiable, for tests.

    Args:
        q, k, v: (B, H, L, D) fp32, tile-major order.
        alpha: (H, Mb) sparse/linear mix.
        lut: (B, H, Mb, T) selected video-tile ids.
        Np, E, prefix_len: tiling geometry.

    Returns:
        o: (B, H, L, D) fp32.
    """
    B, H, L, D = q.shape
    P = prefix_len
    Mb, T = lut.shape[2], lut.shape[3]
    scale = D ** -0.5
    reps = torch.tensor([E] * Np + ([P] if P else []), device=q.device)

    # token mask from LUT (+ always-on prefix tail, ragged)
    mc = torch.zeros(B, H, Mb, Np, device=q.device)
    mc.scatter_(-1, lut.long(), 1.0)                          # video tiles
    m = mc.repeat_interleave(E, dim=-2)[:, :, :L]             # rows
    m = m.repeat_interleave(E, dim=-1)                        # (B,H,L,Np*E)
    if P:
        m = torch.cat([m, torch.ones(B, H, L, P, device=q.device)], -1)

    sf = (q @ k.transpose(-1, -2)) * scale                    # (B,H,L,L)
    p = torch.softmax(sf.masked_fill(m <= 0, float("-inf")), dim=-1)
    o_s = p @ v

    qphi = F.softmax(q, dim=-1)
    kphi = F.softmax(k, dim=-1)
    w = (qphi @ kphi.transpose(-1, -2))
    # complement = non-selected VIDEO tiles only (prefix cols excluded)
    comp = 1.0 - m
    if P:
        comp[..., Np * E:] = 0.0
    w = w * comp
    o_l = (w @ v) / (w.sum(-1, keepdim=True) + _EPS_L)

    a = alpha.repeat_interleave(reps, dim=-1).view(1, H, L, 1)
    return a * o_s + (1.0 - a) * o_l
