"""SLA2-Cube-JVP-QAT: cube-block sparse-linear attention with INT8 QAT
(quantization-aware training) and a forward-mode JVP kernel.

Two halves:

1. Training path `SLA2CubeQATAttentionImpl` (autograd): tile-permute tokens
   to cube-major order, route with the SLA2 learnable router on VSA 3D
   tiles (models/mla_video_sparse_jvp.route_tiles), append the ragged
   prefix tail block as an always-selected LUT entry, then run the
   VENDORED SLA2 kernels (models/sla2_vendor/kernel_sparse_qat.py int8
   sparse forward + fp16 backward, kernel_linear.py complement linear)
   with bq = bk = E = 64, and mix by the learnable per-(head, block)
   alpha. Identical two-branch math to the 1D SLA2 module; only the block
   geometry (3D cubes), router pooling and prefix handling differ.

2. du/dt path `sla2_cube_qat_jvp` with Triton kernel
   `_sla2_cube_jvp_qat_kernel`: the routed-tile flash-JVP recurrence of
   models/mla_video_sparse_jvp.py, with the PRIMAL score dot running on
   INT8 operands (per-64-token-block amax scales, matching the QAT
   training forward numerics) while the tangent dots and the PV dots stay
   fp16. The hard LUT and alpha are piecewise constant (zero tangent).

Shape symbols:
    B : batch;  H : heads;  L : tokens = Np * E + P;  D : head dim (= Dv)
    Np : video tiles;  E : tile length (64);  P : prefix tokens (< E)
    T : routed video tiles per query block;  Mb = Np + 1 query blocks
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from models.mla_video_sparse_jvp import (
    _linear_jvp,
    _phi_jvp,
    _plenty_of_vram,
    route_tiles,
    route_tiles_fast,
    tile_permutation,
)
from models.sla2_vendor.kernel_linear import _complement_linear_attention
from models.sla2_vendor.kernel_sparse_qat import (
    _sparse_attention_qat,
    quantize_qkv,
)

__all__ = ["SLA2CubeQATAttentionImpl", "sla2_cube_qat_jvp"]

_EPS_L = 1e-5  # linear-branch normalizer epsilon (matches SLA2)


# ---------------------------------------------------------------------------
# JVP kernel: int8 primal score dot, fp16 tangent/PV dots, routed tiles +
# always-attended ragged prefix tail.
# ---------------------------------------------------------------------------
@triton.jit
def _sla2_cube_jvp_qat_kernel(
    Q, K, V, DQ, DK, DV,       # (B, H, L, D) fp16
    Q8, K8,                    # (B, H, L, D) int8, per-64-block amax quant
    QS, KS,                    # (B, H, NB) fp32 per-block scales
    O, DO, LUT,
    s_qb, s_qh, s_qs,          # strides of the (B, H, L, D) fp16 tensors
    s_ob, s_oh, s_os,          # strides of the (B, H, L, D) outputs
    s_lb, s_lh, s_lm,          # strides of the (B, H, Mb, T) LUT
    s_sb, s_sh,                # strides of the (B, H, NB) scale tensors
    H, L, T, NPE, scale,       # NPE = Np * E, start of the prefix tail
    HEAD_DIM: tl.constexpr,    # D = Dv
    BLOCK: tl.constexpr,       # E = 64: query tile = key tile = quant block
    FP16_MMA: tl.constexpr,
    WITH_TANGENT: tl.constexpr,  # False: primal only (guidance no-grad path)
):
    pid_m = tl.program_id(0)               # query block id (= LUT row mb)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    base = b * s_qb + h * s_qh
    sbase = b * s_sb + h * s_sh
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)   # query token rows
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < L

    qp = base + offs_m[:, None] * s_qs + offs_d[None, :]
    q8 = tl.load(Q8 + qp, mask=mask_m[:, None], other=0)     # int8
    sq = tl.load(QS + sbase + pid_m)                          # fp32 scalar
    qf = tl.load(Q + qp, mask=mask_m[:, None], other=0.0)    # fp16
    if WITH_TANGENT:
        dq = tl.load(DQ + qp, mask=mask_m[:, None], other=0.0)  # fp16
    else:
        dq = qf  # unused; keeps the type checker happy

    m_i = tl.full([BLOCK], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK], tl.float32)
    mu_i = tl.zeros([BLOCK], tl.float32)
    acc_o = tl.zeros([BLOCK, HEAD_DIM], tl.float32)
    acc_do = tl.zeros([BLOCK, HEAD_DIM], tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    lut_base = LUT + b * s_lb + h * s_lh + pid_m * s_lm

    for j in range(0, T + 1):
        if j == T:
            nb_id = NPE // BLOCK           # ragged prefix tail block id
        else:
            nb_id = tl.load(lut_base + j)
        offs = nb_id * BLOCK + offs_n      # key token rows
        mask_n = offs < L
        kp = base + offs[:, None] * s_qs + offs_d[None, :]
        k8 = tl.load(K8 + kp, mask=mask_n[:, None], other=0)  # int8
        sk = tl.load(KS + sbase + nb_id)                      # fp32 scalar
        kf = tl.load(K + kp, mask=mask_n[:, None], other=0.0)
        if WITH_TANGENT:
            dk = tl.load(DK + kp, mask=mask_n[:, None], other=0.0)

        # primal scores on INT8 operands (int32 accumulate), dequantized
        s_f = tl.dot(q8, tl.trans(k8), out_dtype=tl.int32).to(tl.float32)
        s_f = s_f * (sq * sk)
        # tangent scores fp16: dQ K^T + Q dK^T
        if WITH_TANGENT:
            if FP16_MMA:
                ds = (tl.dot(dq, tl.trans(kf),
                             out_dtype=tl.float16).to(tl.float32)
                      + tl.dot(qf, tl.trans(dk),
                               out_dtype=tl.float16).to(tl.float32)) * scale
            else:
                ds = (tl.dot(dq, tl.trans(kf))
                      + tl.dot(qf, tl.trans(dk))) * scale
        s2 = tl.where(mask_n[None, :], s_f * (scale * LOG2E), float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s2, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s2 - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kp, mask=mask_n[:, None], other=0.0)
        if WITH_TANGENT:
            t = p * ds
            mu_i = mu_i * alpha + tl.sum(t, 1)
            dv = tl.load(DV + kp, mask=mask_n[:, None], other=0.0)
            if FP16_MMA:
                pc = p.to(tl.float16)
                tc = t.to(tl.float16)
                o_t = tl.dot(pc, v, out_dtype=tl.float16).to(tl.float32)
                do_t = (tl.dot(tc, v, out_dtype=tl.float16)
                        + tl.dot(pc, dv,
                                 out_dtype=tl.float16)).to(tl.float32)
            else:
                o_t = tl.dot(p, v)
                do_t = tl.dot(t, v) + tl.dot(p, dv)
            acc_do = acc_do * alpha[:, None] + do_t
        else:
            if FP16_MMA:
                o_t = tl.dot(p.to(tl.float16), v,
                             out_dtype=tl.float16).to(tl.float32)
            else:
                o_t = tl.dot(p, v)
        acc_o = acc_o * alpha[:, None] + o_t
        m_i = m_new

    o = acc_o / l_i[:, None]
    op = b * s_ob + h * s_oh + offs_m[:, None] * s_os + offs_d[None, :]
    tl.store(O + op, o.to(O.dtype.element_ty), mask=mask_m[:, None])
    if WITH_TANGENT:
        do = acc_do / l_i[:, None] - (mu_i / l_i)[:, None] * o
        tl.store(DO + op, do.to(DO.dtype.element_ty), mask=mask_m[:, None])


@torch.no_grad()
def _sparse_primal_only(q, k, v, lut, T, Np, E, prefix_len):
    """Primal-only sparse branch: int8 scores, no tangent loads/dots.

    Serves the guidance no-grad forwards, where the vendored autograd
    kernels' backward bookkeeping is wasted work.

    Args:
        q, k, v: (B, H, L, D) fp16, tile-major token order.
        lut: (B, H, Mb, T) int32 routed VIDEO-tile ids (prefix appended
            in-kernel via the T+1-th iteration).
        T: routed tile count;  Np, E, prefix_len: tiling geometry.

    Returns:
        o_s: (B, H, L, D) fp32 sparse-branch output.
    """
    B, H, L, D = q.shape
    scale = D ** -0.5
    qc = [t.contiguous().half() for t in (q, k, v)]
    q8, qs = quantize_qkv(qc[0], E)     # int8 (B,H,L,D) + (B,H,NB) scales
    k8, ks = quantize_qkv(qc[1], E)
    Mb = lut.shape[2]
    o = torch.empty(B, H, L, D, device=q.device, dtype=torch.float32)
    grid = (Mb, B * H)
    _sla2_cube_jvp_qat_kernel[grid](
        qc[0], qc[1], qc[2], qc[0], qc[0], qc[0],  # tangent ptrs unused
        q8, k8, qs, ks, o, o, lut,                 # DO ptr unused
        qc[0].stride(0), qc[0].stride(1), qc[0].stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lut.stride(0), lut.stride(1), lut.stride(2),
        qs.stride(0), qs.stride(1),
        H, L, T, Np * E, scale,
        HEAD_DIM=D, BLOCK=E, FP16_MMA=True, WITH_TANGENT=False,
        num_warps=4, num_stages=2,
    )
    return o


def sla2_cube_qat_jvp(q, k, v, dq, dk, dv, alpha, topk_frac, Np, E,
                      prefix_len, proj_q=None, proj_k=None,
                      lut=None, T=None):
    """Cube-block QAT attention (primal, tangent): int8 primal scores.

    Inputs in TILE-MAJOR order; see module docstring for the composition.

    Args:
        q..dv: (B, H, L, D) fp16 primal / tangent, L = Np*E + P.
        alpha: (H, Mb) sparse/linear mix in (0,1), or scalar.
        topk_frac: routed fraction of video tiles (ignored when lut given).
        Np, E, prefix_len: tiling geometry (E must be 64-like, = quant block).
        proj_q, proj_k: optional (D, D) router projections.
        lut, T: optional precomputed routing.

    Returns:
        (o, do): (B, H, L, D) fp32.
    """
    B, H, L, D = q.shape
    scale = D ** -0.5
    P = prefix_len
    if lut is None:
        lut, T = route_tiles(q, k, topk_frac, Np, E, P,
                             proj_q=proj_q, proj_k=proj_k)

    qc = [t.contiguous().half() for t in (q, k, v, dq, dk, dv)]
    q8, qs = quantize_qkv(qc[0], E)        # int8 (B,H,L,D) + (B,H,NB) scales
    k8, ks = quantize_qkv(qc[1], E)
    Mb = lut.shape[2]
    # fp32 outputs: the linear-branch tangent (dnum - o_l*dden)/den can
    # exceed the fp16 range when den is small; an fp16 cast here produced
    # inf -> NaN mid-training. The caller processes one batch row at a
    # time, so the fp32 footprint is small.
    o = torch.empty(B, H, L, D, device=q.device, dtype=torch.float32)
    do = torch.empty_like(o)
    NPE = Np * E
    grid = (Mb, B * H)
    _sla2_cube_jvp_qat_kernel[grid](
        qc[0], qc[1], qc[2], qc[3], qc[4], qc[5],
        q8, k8, qs, ks, o, do, lut,
        qc[0].stride(0), qc[0].stride(1), qc[0].stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lut.stride(0), lut.stride(1), lut.stride(2),
        qs.stride(0), qs.stride(1),
        H, L, T, NPE, scale,
        HEAD_DIM=D, BLOCK=E, FP16_MMA=True, WITH_TANGENT=True,
        num_warps=4, num_stages=2,
    )

    # linear complement branch. Wide path: all heads in one shot (the fp32
    # phi/state/einsum transients are ~1.2 GB at 29k tokens -- fine on
    # data-center GPUs, and avoids H serial launch chains). Low-memory
    # path: chunked per head, which keeps a 16 GB card alive beside the
    # training footprint.
    if _plenty_of_vram(q.device):
        qphi, dqphi = _phi_jvp(q, dq)                        # (B,H,L,D)
        kphi, dkphi = _phi_jvp(k, dk)
        o_l, do_l = _linear_jvp(qphi, dqphi, kphi, dkphi, v, dv,
                                lut, Np, E)                  # (B,H,L,D)
        del qphi, dqphi, kphi, dkphi
    else:
        o_l = torch.empty_like(o)
        do_l = torch.empty_like(do)
        for h0 in range(H):
            sl = slice(h0, h0 + 1)
            qphi, dqphi = _phi_jvp(q[:, sl], dq[:, sl])
            kphi, dkphi = _phi_jvp(k[:, sl], dk[:, sl])
            ol_h, dol_h = _linear_jvp(qphi, dqphi, kphi, dkphi,
                                      v[:, sl], dv[:, sl], lut[:, sl],
                                      Np, E)
            o_l[:, sl] = ol_h
            do_l[:, sl] = dol_h
            del qphi, dqphi, kphi, dkphi, ol_h, dol_h

    if torch.is_tensor(alpha) and alpha.dim() == 2:
        reps = torch.tensor([E] * Np + ([P] if P else []), device=q.device)
        a = alpha.float().repeat_interleave(reps, dim=-1).view(1, H, L, 1)
    else:
        a = torch.as_tensor(alpha, device=q.device, dtype=torch.float32)
    return a * o + (1.0 - a) * o_l, a * do + (1.0 - a) * do_l


# ---------------------------------------------------------------------------
# Training module: vendored SLA2 QAT kernels on cube geometry.
# ---------------------------------------------------------------------------
class SLA2CubeQATAttentionImpl(nn.Module):
    """attn_impl: cube-block sparse-linear attention with INT8 QAT training.

    Args:
        head_dim: qk head dim D (= v head dim, 64).
        seq_len: L = prefix + patches (bound by IMFDiTVideo).
        num_heads: H (bound by IMFDiTVideo).
        grid: (T, Hh, Ww) patch grid (bound by train.py from config).
        tile: (tt, th, tw) 3D tile, prod = E = 64.
        topk: routed fraction of video tiles.
        alpha_init: initial sparse/linear mixing ratio in (0, 1).
        use_int8: run the sparse training forward in INT8 (QAT).
    """

    TILE_MAJOR = True  # supports model-level tile-major residency

    def __init__(self, head_dim, seq_len, num_heads, grid, tile=(4, 4, 4),
                 topk=0.03, alpha_init=0.9, use_int8=True,
                 pre_permuted=False):
        # pre_permuted: the MODEL keeps the whole sequence in tile-major
        # token order (IMFDiTVideo.tile_major); skip per-call permutes.
        super().__init__()
        self.pre_permuted = pre_permuted
        self.E = tile[0] * tile[1] * tile[2]
        self.topk_frac = topk
        self.use_int8 = use_int8
        n_patch = grid[0] * grid[1] * grid[2]
        self.prefix_len = seq_len - n_patch
        perm, inv, self.Np = tile_permutation(
            grid, tile, self.prefix_len, torch.device("cpu"))
        self.register_buffer("perm", perm, persistent=False)   # (L,)
        self.register_buffer("inv", inv, persistent=False)     # (L,)
        self.Mb = self.Np + 1                                  # + prefix tail
        # a_index: (L,) int64, tile-major token row -> query-block id
        # (E-sized video blocks, then the prefix tail block Mb-1); used by
        # the fused alpha mix in _CubeQATFunction.
        rows = torch.arange(seq_len)
        self.register_buffer(
            "a_index", torch.clamp(rows // self.E, max=self.Mb - 1),
            persistent=False)
        self.Nb = math.ceil(seq_len / self.E)                  # key blocks
        # learnable router projections (identity init, SLA2 style): (D, D)
        self.proj_q = nn.Parameter(torch.eye(head_dim))
        self.proj_k = nn.Parameter(torch.eye(head_dim))
        # (H, Mb) sparse/linear mixing logits, sigmoid -> (0, 1)
        init = torch.logit(torch.tensor(float(alpha_init)))
        self.alpha_logit = nn.Parameter(
            torch.full((num_heads, self.Mb), init.item()))

    def forward(self, q, k, v):
        """q, k, v: (b, l, H, hd) -> (b, l, H, hd) attention output."""
        B, L, H, D = q.shape
        if self.pre_permuted:
            # tokens already tile-major: only the (B,L,H,D)->(B,H,L,D)
            # layout transpose remains
            qh = q.permute(0, 2, 1, 3).half().contiguous()
            kh = k.permute(0, 2, 1, 3).half().contiguous()
            vh = v.permute(0, 2, 1, 3).half().contiguous()
        else:
            qh = q.permute(0, 2, 1, 3)[:, :, self.perm].half().contiguous()
            kh = k.permute(0, 2, 1, 3)[:, :, self.perm].half().contiguous()
            vh = v.permute(0, 2, 1, 3)[:, :, self.perm].half().contiguous()
        #   (B, H, L, D) fp16, tile-major token order

        # router: hard top-k video tiles (detached; trains via proj grads
        # inside route_tiles? -- hard topk has no gradient; proj_q/proj_k
        # receive gradients only through the LINEAR branch complement
        # selection changing is non-differentiable, matching vendored SLA2
        # stage-2 behaviour where the router is frozen-hard per step)
        with torch.no_grad():
            lut, T = route_tiles_fast(qh, kh, self.topk_frac, self.Np,
                                      self.E, self.prefix_len,
                                      proj_q=self.proj_q,
                                      proj_k=self.proj_k)
        # append the ragged prefix tail block as an always-selected entry
        tail = torch.full_like(lut[..., :1], self.Nb - 1)      # (B,H,Mb,1)
        lut_aug = torch.cat([lut, tail], dim=-1).contiguous()  # (B,H,Mb,T+1)
        # dense {0,1} block mask for the backward kernels: (B,H,Mb,Nb)
        mask = torch.zeros(B, H, self.Mb, self.Nb, device=q.device,
                           dtype=torch.int8)
        mask.scatter_(-1, lut_aug.long(), 1)

        needs_grad = torch.is_grad_enabled() and (
            q.requires_grad or k.requires_grad or v.requires_grad
            or self.alpha_logit.requires_grad)
        if needs_grad:
            o = _CubeQATFunction.apply(qh, kh, vh, self.alpha_logit, lut,
                                       lut_aug, mask, T, self.E,
                                       self.use_int8, self.a_index)
            if self.pre_permuted:
                return o.permute(0, 2, 1, 3).to(q.dtype)
            return o[:, :, self.inv].permute(0, 2, 1, 3).to(q.dtype)
        else:
            # guidance path (no_grad): primal-only JVP kernel (int8 primal
            # scores, no tangent loads/dots) + states-based linear branch;
            # ~2x leaner than the autograd kernels.
            o_s = _sparse_primal_only(qh, kh, vh, lut, T, self.Np, self.E,
                                      self.prefix_len)        # (B,H,L,D)
            qphi = torch.softmax(qh.float(), -1)              # (B,H,L,D)
            kphi = torch.softmax(kh.float(), -1)
            o_l, _ = _linear_jvp(qphi, None, kphi, None, vh, None, lut,
                                 self.Np, self.E,
                                 primal_only=True)            # (B,H,L,D)

        reps = [self.E] * self.Np + [self.prefix_len]
        a = torch.sigmoid(self.alpha_logit).repeat_interleave(
            torch.tensor(reps, device=q.device), dim=-1)       # (H, L)
        a = a.view(1, H, L, 1).to(o_s.dtype)
        o = a * o_s + (1.0 - a) * o_l                          # (B,H,L,D)
        if self.pre_permuted:
            return o.permute(0, 2, 1, 3).to(q.dtype)
        return o[:, :, self.inv].permute(0, 2, 1, 3).to(q.dtype)


# ---------------------------------------------------------------------------
# FUSED JVP kernel: sparse recurrence + in-kernel phi + complement linear
# branch + alpha mix, one program per query tile. Replaces the separate
# phi softmax, states gathers, quotient einsums and mix kernels of the
# unfused path (~70% of the jvp op time at d=1024 on A100).
# ---------------------------------------------------------------------------
@triton.jit
def _cube_fused_jvp_kernel(
    Q, K, V, DQ, DK, DV,       # (B, H, L, D) fp16 primal/tangent
    Q8, K8,                    # (B, H, L, D) int8 per-tile amax quant
    QS, KS,                    # (B, H, NB) fp32 per-tile quant scales
    HB, DHB,                   # (B, H, Np, D, Dv) fp32 phi-key/value states
    ZB, DZB,                   # (B, H, Np, D) fp32 phi-key sums
    HALL, DHALL,               # (B, H, D, Dv) fp32 global states
    ZALL, DZALL,               # (B, H, D) fp32 global sums
    ALPHA,                     # (H, Mb) fp32 sparse/linear mix in (0,1)
    O, DO, LUT,
    s_qb, s_qh, s_qs,          # strides of the (B, H, L, D) fp16 tensors
    s_ob, s_oh, s_os,          # strides of the (B, H, L, D) outputs
    s_lb, s_lh, s_lm,          # strides of the (B, H, Mb, T) LUT
    s_sb, s_sh,                # strides of the (B, H, NB) scale tensors
    s_hb, s_hh, s_hn,          # strides of the (B, H, Np, D, Dv) states
    s_zbb, s_zbh, s_zbn,       # strides of the (B, H, Np, D) sums
    s_gb, s_gh,                # strides of the (B, H, D, Dv) globals
    s_ab,                      # stride of the (H, Mb) alpha rows
    H, L, T, NPE, scale, eps_l,
    HEAD_DIM: tl.constexpr,    # D = Dv = 64
    BLOCK: tl.constexpr,       # E = 64: query tile = key tile = quant block
):
    pid_m = tl.program_id(0)               # query block id (= LUT row mb)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    base = b * s_qb + h * s_qh
    sbase = b * s_sb + h * s_sh
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)   # query token rows
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_e = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < L

    qp = base + offs_m[:, None] * s_qs + offs_d[None, :]
    q8 = tl.load(Q8 + qp, mask=mask_m[:, None], other=0)     # int8
    sq = tl.load(QS + sbase + pid_m)                          # fp32 scalar
    qf = tl.load(Q + qp, mask=mask_m[:, None], other=0.0)    # fp16
    dq = tl.load(DQ + qp, mask=mask_m[:, None], other=0.0)   # fp16

    # ---- in-kernel phi = softmax over the D channel, with tangent ----
    qf32 = qf.to(tl.float32)
    dq32 = dq.to(tl.float32)
    qmx = tl.max(qf32, 1)                                    # (BLOCK,)
    qex = tl.exp(qf32 - qmx[:, None])                        # (BLOCK, D)
    qden = tl.sum(qex, 1)                                    # (BLOCK,)
    qphi = qex / qden[:, None]                               # (BLOCK, D)
    qdot = tl.sum(qphi * dq32, 1)                            # (BLOCK,)
    dqphi = qphi * (dq32 - qdot[:, None])                    # (BLOCK, D)

    # ---- sparse-branch flash-JVP recurrence over T routed + prefix ----
    m_i = tl.full([BLOCK], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK], tl.float32)
    mu_i = tl.zeros([BLOCK], tl.float32)
    acc_o = tl.zeros([BLOCK, HEAD_DIM], tl.float32)
    acc_do = tl.zeros([BLOCK, HEAD_DIM], tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    lut_base = LUT + b * s_lb + h * s_lh + pid_m * s_lm

    # ---- linear-branch complement states, subtracted while looping ----
    gb = b * s_gb + h * s_gh
    hc = tl.load(HALL + gb + offs_d[:, None] * HEAD_DIM + offs_e[None, :]
                 ).to(tl.float32)                            # (D, Dv)
    dhc = tl.load(DHALL + gb + offs_d[:, None] * HEAD_DIM + offs_e[None, :]
                  ).to(tl.float32)                           # (D, Dv)
    # z_all / dz_all are contiguous (B, H, D): row offset = (b*H + h) * D
    zc = tl.load(ZALL + (b * H + h) * HEAD_DIM + offs_d).to(tl.float32)
    dzc = tl.load(DZALL + (b * H + h) * HEAD_DIM + offs_d).to(tl.float32)

    hb_base = HB + b * s_hb + h * s_hh
    dhb_base = DHB + b * s_hb + h * s_hh
    zb_base = ZB + b * s_zbb + h * s_zbh
    dzb_base = DZB + b * s_zbb + h * s_zbh

    for j in range(0, T + 1):
        if j == T:
            nb_id = NPE // BLOCK           # ragged prefix tail block id
        else:
            nb_id = tl.load(lut_base + j)
        offs = nb_id * BLOCK + offs_n      # key token rows
        mask_n = offs < L
        kp = base + offs[:, None] * s_qs + offs_d[None, :]
        k8 = tl.load(K8 + kp, mask=mask_n[:, None], other=0)  # int8
        sk = tl.load(KS + sbase + nb_id)                      # fp32 scalar
        kf = tl.load(K + kp, mask=mask_n[:, None], other=0.0)
        dk = tl.load(DK + kp, mask=mask_n[:, None], other=0.0)

        s_f = tl.dot(q8, tl.trans(k8), out_dtype=tl.int32).to(tl.float32)
        s_f = s_f * (sq * sk)
        ds = (tl.dot(dq, tl.trans(kf), out_dtype=tl.float16).to(tl.float32)
              + tl.dot(qf, tl.trans(dk),
                       out_dtype=tl.float16).to(tl.float32)) * scale
        s2 = tl.where(mask_n[None, :], s_f * (scale * LOG2E), float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s2, 1))
        alpha_c = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s2 - m_new[:, None])
        t = p * ds
        l_i = l_i * alpha_c + tl.sum(p, 1)
        mu_i = mu_i * alpha_c + tl.sum(t, 1)

        v = tl.load(V + kp, mask=mask_n[:, None], other=0.0)
        dv = tl.load(DV + kp, mask=mask_n[:, None], other=0.0)
        pc = p.to(tl.float16)
        tc = t.to(tl.float16)
        o_t = tl.dot(pc, v, out_dtype=tl.float16).to(tl.float32)
        do_t = (tl.dot(tc, v, out_dtype=tl.float16)
                + tl.dot(pc, dv, out_dtype=tl.float16)).to(tl.float32)
        acc_o = acc_o * alpha_c[:, None] + o_t
        acc_do = acc_do * alpha_c[:, None] + do_t
        m_i = m_new

        # subtract this routed VIDEO tile's states from the complement
        # (the prefix tail j == T is never part of the linear branch)
        if j < T:
            hp = nb_id * s_hn + offs_d[:, None] * HEAD_DIM + offs_e[None, :]
            hc -= tl.load(hb_base + hp).to(tl.float32)
            dhc -= tl.load(dhb_base + hp).to(tl.float32)
            zp = nb_id * s_zbn + offs_d
            zc -= tl.load(zb_base + zp).to(tl.float32)
            dzc -= tl.load(dzb_base + zp).to(tl.float32)

    o_s = acc_o / l_i[:, None]
    do_s = acc_do / l_i[:, None] - (mu_i / l_i)[:, None] * o_s

    # ---- linear branch quotient (per query row) ----
    num = tl.dot(qphi, hc)                                   # (BLOCK, Dv)
    dnum = tl.dot(dqphi, hc) + tl.dot(qphi, dhc)             # (BLOCK, Dv)
    den = tl.sum(qphi * zc[None, :], 1) + eps_l              # (BLOCK,)
    dden = tl.sum(dqphi * zc[None, :], 1) + tl.sum(qphi * dzc[None, :], 1)
    o_l = num / den[:, None]                                 # (BLOCK, Dv)
    do_l = (dnum - o_l * dden[:, None]) / den[:, None]

    # ---- alpha mix (scalar per (h, mb)) and store ----
    a = tl.load(ALPHA + h * s_ab + pid_m).to(tl.float32)
    o = a * o_s + (1.0 - a) * o_l
    do = a * do_s + (1.0 - a) * do_l
    op = b * s_ob + h * s_oh + offs_m[:, None] * s_os + offs_d[None, :]
    tl.store(O + op, o.to(O.dtype.element_ty), mask=mask_m[:, None])
    tl.store(DO + op, do.to(DO.dtype.element_ty), mask=mask_m[:, None])


def sla2_cube_qat_jvp_fused(q, k, v, dq, dk, dv, alpha, topk_frac, Np, E,
                            prefix_len, proj_q=None, proj_k=None,
                            lut=None, T=None):
    """Fused cube-QAT JVP: one kernel for both branches + mix.

    Same math as sla2_cube_qat_jvp (int8 primal scores, fp16 tangent dots,
    fp32 linear branch), with phi / complement states / quotient / alpha
    folded into the Triton program. Host precomputes only the quantization
    and the per-tile phi-states.

    Args:
        q..dv: (B, H, L, D) fp16 primal / tangent, tile-major order.
        alpha: (H, Mb) mixing ratio in (0,1) (tensor required here).
        topk_frac, Np, E, prefix_len, proj_q, proj_k, lut, T: as unfused.

    Returns:
        (o, do): (B, H, L, D) fp32.
    """
    B, H, L, D = q.shape
    scale = D ** -0.5
    P = prefix_len
    if lut is None:
        lut, T = route_tiles_fast(q, k, topk_frac, Np, E, P,
                                  proj_q=proj_q, proj_k=proj_k)
    qc = [t.contiguous().half() for t in (q, k, v, dq, dk, dv)]
    q8, qs = quantize_qkv(qc[0], E)     # int8 (B,H,L,D) + (B,H,NB) scales
    k8, ks = quantize_qkv(qc[1], E)
    Mb = lut.shape[2]

    # host-side per-tile phi-states (the only remaining torch math):
    kphi, dkphi = _phi_jvp(k, dk)                            # (B,H,L,D) fp32
    kb = kphi[:, :, :Np * E].view(B, H, Np, E, D)            # (B,H,Np,E,D)
    dkb = dkphi[:, :, :Np * E].view(B, H, Np, E, D)
    vb = v.float()[:, :, :Np * E].view(B, H, Np, E, D)
    dvb = dv.float()[:, :, :Np * E].view(B, H, Np, E, D)
    Hb = torch.einsum("bhntd,bhntv->bhndv", kb, vb).contiguous()
    dHb = (torch.einsum("bhntd,bhntv->bhndv", dkb, vb)
           + torch.einsum("bhntd,bhntv->bhndv", kb, dvb)).contiguous()
    zb = kb.sum(3).contiguous()                              # (B,H,Np,D)
    dzb = dkb.sum(3).contiguous()
    H_all = Hb.sum(2).contiguous()                           # (B,H,D,D)
    dH_all = dHb.sum(2).contiguous()
    z_all = zb.sum(2).contiguous()                           # (B,H,D)
    dz_all = dzb.sum(2).contiguous()

    o = torch.empty(B, H, L, D, device=q.device, dtype=torch.float32)
    do = torch.empty_like(o)
    af = alpha.float().contiguous()                          # (H, Mb)
    grid = (Mb, B * H)
    _cube_fused_jvp_kernel[grid](
        qc[0], qc[1], qc[2], qc[3], qc[4], qc[5],
        q8, k8, qs, ks,
        Hb, dHb, zb, dzb, H_all, dH_all, z_all, dz_all, af,
        o, do, lut,
        qc[0].stride(0), qc[0].stride(1), qc[0].stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lut.stride(0), lut.stride(1), lut.stride(2),
        qs.stride(0), qs.stride(1),
        Hb.stride(0), Hb.stride(1), Hb.stride(2),
        zb.stride(0), zb.stride(1), zb.stride(2),
        H_all.stride(0), H_all.stride(1),
        af.stride(0),
        H, L, T, Np * E, scale, _EPS_L,
        HEAD_DIM=D, BLOCK=E,
        num_warps=8, num_stages=1,
    )
    return o, do


# ---------------------------------------------------------------------------
# Whole-op autograd Function: direct kernel calls for both branches, phi
# fused into the sparse forward (phi_out), analytic backward for the phi
# chains and the alpha mix. Removes the python-autograd glue (separate phi
# softmax fwd/bwd, alpha interleave fwd/bwd, double contexts) that
# dominated the training fwd+bwd of the module path.
# ---------------------------------------------------------------------------
from models.sla2_vendor.kernel_sparse_qat import _attn_fwd_qat  # noqa: E402
from models.sla2_vendor.sla_kernel import (  # noqa: E402
    _attn_bwd_dkdv,
    _attn_bwd_dq,
    _attn_bwd_preprocess,
)
from models.sla2_vendor.kernel_linear import (  # noqa: E402
    _lin_bwd_dkdv,
    _lin_bwd_dq,
    _lin_fwd2,
    _precompute_global,
)


class _CubeQATFunction(torch.autograd.Function):
    """o = a .* O_s(q,k,v) + (1-a) .* O_l(phi(q),phi(k),v), tile-major.

    Saved tensors and their shapes are annotated inline; all inputs are
    (B, H, L, D) fp16 contiguous in tile-major token order.
    """

    @staticmethod
    def forward(ctx, qh, kh, vh, alpha_logit, lut, lut_aug, mask, T, E,
                use_int8, a_index):
        # qh/kh/vh: (B,H,L,D) fp16;  alpha_logit: (H,Mb) fp32 logits
        # lut: (B,H,Mb,T) video tiles; lut_aug: (B,H,Mb,T+1) with prefix
        # mask: (B,H,Mb,Nb) int8; a_index: (L,) int64 row->block map
        B, H, L, D = qh.shape
        M_BLOCKS = triton.cdiv(L, E)
        scale = D ** -0.5

        o_s = torch.empty_like(vh)                          # (B,H,L,D)
        lse = torch.empty(B, H, L, device=qh.device, dtype=torch.float32)
        qphi = torch.empty_like(qh)                         # (B,H,L,D)
        _attn_fwd_qat[(M_BLOCKS, B * H)](
            qh, kh, vh, lut_aug, lse, o_s, qphi,
            scale, T + 1, L, M_BLOCKS, D, E, E, use_int8, True,
            num_warps=4, num_stages=3,
        )
        kphi = torch.softmax(kh.float(), -1).to(qh.dtype).contiguous()

        htot, ztot = _precompute_global(qphi, kphi, vh)     # (B,H,D,D),(B,H,D)
        o_l = torch.empty_like(vh)                          # (B,H,L,D)
        den = torch.empty(B, H, L, device=qh.device, dtype=torch.float32)
        # complement must exclude the prefix tail too (the globals span all
        # L tokens): use the augmented LUT with T+1 entries.
        _lin_fwd2[(M_BLOCKS, B * H)](
            qphi, kphi, vh, htot.to(qh.dtype).contiguous(), ztot, lut_aug,
            o_l, den, o_l, ztot, o_l,
            T + 1, L, M_BLOCKS, D, E, E, False, False, T + 1 <= 4,
            num_warps=8, num_stages=3,
        )

        a_row = torch.sigmoid(alpha_logit.float())          # (H, Mb)
        a = a_row[:, a_index].view(1, H, L, 1).to(o_s.dtype)  # (1,H,L,1)
        o = a * o_s + (1.0 - a) * o_l                       # (B,H,L,D)

        ctx.save_for_backward(qh, kh, vh, alpha_logit, lut, lut_aug, mask,
                              lse, o_s, o_l, den, qphi, kphi, htot, ztot,
                              a_index)
        ctx.T, ctx.E, ctx.scale = T, E, scale
        return o

    @staticmethod
    def backward(ctx, do):
        (qh, kh, vh, alpha_logit, lut, lut_aug, mask, lse, o_s, o_l, den,
         qphi, kphi, htot, ztot, a_index) = ctx.saved_tensors
        T, E, scale = ctx.T, ctx.E, ctx.scale
        B, H, L, D = qh.shape
        M_BLOCKS = triton.cdiv(L, E)
        N_BLOCKS = triton.cdiv(L, E)
        do = do.contiguous()

        a_row = torch.sigmoid(alpha_logit.float()).contiguous()  # (H, Mb)
        Mb = a_row.shape[1]
        do_s = torch.empty_like(do)                         # (B,H,L,D)
        do_l = torch.empty_like(do)
        seg = torch.zeros(H, Mb, device=do.device)          # (H, Mb)
        _bwd_split_kernel[(Mb, B * H)](
            do, o_s, o_l, a_row, do_s, do_l, seg,
            do.stride(0), do.stride(1), do.stride(2),
            a_row.stride(0), seg.stride(0),
            H, L, E=E, D=D, num_warps=4)
        d_alpha = seg * a_row * (1.0 - a_row)               # (H, Mb)

        # ---- sparse branch backward (vendored SLA kernels) ----
        dq_s = torch.empty_like(qh)
        dk_s = torch.empty_like(kh)
        dv_s = torch.empty_like(vh)
        delta_s = torch.empty_like(lse)
        _attn_bwd_preprocess[(M_BLOCKS, B * H)](o_s, do_s, delta_s, L, D, E)
        _attn_bwd_dq[(M_BLOCKS, B * H)](
            qh, kh, vh, lse, delta_s, do_s, dq_s, lut_aug, scale, T + 1,
            L, M_BLOCKS, D, E, E, num_warps=4, num_stages=4,
        )
        _attn_bwd_dkdv[(N_BLOCKS, B * H)](
            qh, kh, vh, do_s, dk_s, dv_s, scale, mask, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS, D, E, E, BLOCK_SLICE_FACTOR=E // 64,
            num_warps=4, num_stages=4,
        )

        # ---- linear branch backward (vendored kernels + globals) ----
        g = do_l.float() / den[..., None]                   # (B,H,L,D)
        s = (do_l.float() * o_l.float()).sum(-1) / den      # (B,H,L)
        dhtot = (qphi.float().transpose(-1, -2) @ g).contiguous()
        dztot = -(qphi.float() * s[..., None]).sum(-2).contiguous()
        dqphi = torch.empty_like(qphi)
        dkphi = torch.empty_like(kphi)
        dv_l = torch.empty_like(vh)
        _lin_bwd_dq[(M_BLOCKS, B * H)](
            qphi, kphi, vh, htot, ztot, lut_aug, o_l, den, do_l, dqphi,
            T + 1, L, M_BLOCKS, D, E, E, num_warps=8, num_stages=2,
        )
        _lin_bwd_dkdv[(N_BLOCKS, B * H)](
            qphi, kphi, vh, dhtot, dztot, o_l, den, do_l, mask, dkphi,
            dv_l, L, M_BLOCKS, N_BLOCKS, D, E, E,
            num_warps=8, num_stages=2,
        )

        # ---- phi (channel softmax) jacobian chains, fused with the add ----
        Mrows = triton.cdiv(L, E)
        dq = torch.empty_like(dq_s)
        dk = torch.empty_like(dk_s)
        _phi_chain_add_kernel[(Mrows, B * H)](
            qphi, dqphi, dq_s, dq,
            qphi.stride(0), qphi.stride(1), qphi.stride(2),
            H, L, E=E, D=D, num_warps=4)
        _phi_chain_add_kernel[(Mrows, B * H)](
            kphi, dkphi, dk_s, dk,
            kphi.stride(0), kphi.stride(1), kphi.stride(2),
            H, L, E=E, D=D, num_warps=4)
        dv = dv_s + dv_l
        return (dq, dk, dv, d_alpha, None, None, None, None, None, None,
                None)


# ---------------------------------------------------------------------------
# Backward micro-fusion kernels.
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_split_kernel(
    DO, OS, OL, AROW, DOS, DOL, DSEG,
    s_b, s_h, s_s,             # strides of the (B, H, L, D) tensors
    s_ab,                      # stride of the (H, Mb) alpha rows
    s_gb,                      # stride of the (H, Mb) d_alpha seg rows
    H, L,
    E: tl.constexpr,           # tile length (query block size)
    D: tl.constexpr,
):
    """Per query tile: do_s = do*a, do_l = do*(1-a), and the alpha-grad
    segment sum dseg[h, mb] += sum(do * (o_s - o_l)) in one pass.

    DO/OS/OL: (B, H, L, D); AROW: (H, Mb) sigmoid(alpha); DSEG: (H, Mb)
    fp32 accumulated via atomics (grid covers (Mb, B*H))."""
    pid_m = tl.program_id(0)               # query tile id (= alpha col mb)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_e = tl.arange(0, E)
    offs_d = tl.arange(0, D)
    rows = pid_m * E + offs_e
    mask = rows < L
    base = b * s_b + h * s_h
    ptr = base + rows[:, None] * s_s + offs_d[None, :]
    do = tl.load(DO + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    os_ = tl.load(OS + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    ol_ = tl.load(OL + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    a = tl.load(AROW + h * s_ab + pid_m).to(tl.float32)
    tl.store(DOS + ptr, (do * a).to(DOS.dtype.element_ty),
             mask=mask[:, None])
    tl.store(DOL + ptr, (do * (1.0 - a)).to(DOL.dtype.element_ty),
             mask=mask[:, None])
    seg = tl.sum(do * (os_ - ol_))
    tl.atomic_add(DSEG + h * s_gb + pid_m, seg)


@triton.jit
def _phi_chain_add_kernel(
    PHI, DPHI, DXS, DX,
    s_b, s_h, s_s,             # strides of the (B, H, L, D) tensors
    H, L,
    E: tl.constexpr,
    D: tl.constexpr,
):
    """dx = dx_sparse + phi * (dphi - <phi, dphi>): softmax-jacobian chain
    fused with the sparse-branch gradient add. All (B, H, L, D)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_e = tl.arange(0, E)
    offs_d = tl.arange(0, D)
    rows = pid_m * E + offs_e
    mask = rows < L
    base = b * s_b + h * s_h
    ptr = base + rows[:, None] * s_s + offs_d[None, :]
    phi = tl.load(PHI + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    dphi = tl.load(DPHI + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    dxs = tl.load(DXS + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
    dot = tl.sum(phi * dphi, 1)                              # (E,)
    dx = dxs + phi * (dphi - dot[:, None])
    tl.store(DX + ptr, dx.to(DX.dtype.element_ty), mask=mask[:, None])
