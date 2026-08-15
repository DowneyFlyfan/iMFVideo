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

    def __init__(self, head_dim, seq_len, num_heads, grid, tile=(4, 4, 4),
                 topk=0.03, alpha_init=0.9, use_int8=True):
        super().__init__()
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
            lut, T = route_tiles(qh, kh, self.topk_frac, self.Np, self.E,
                                 self.prefix_len,
                                 proj_q=self.proj_q, proj_k=self.proj_k)
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
            o_s = _sparse_attention_qat.apply(qh, kh, vh, mask, lut_aug,
                                              T + 1, self.E, self.E,
                                              self.use_int8)
            qphi = torch.softmax(qh.float(), -1).half().contiguous()
            kphi = torch.softmax(kh.float(), -1).half().contiguous()
            o_l = _complement_linear_attention.apply(
                qphi, kphi, vh, mask, lut_aug, T + 1, self.E, self.E)
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
        return o[:, :, self.inv].permute(0, 2, 1, 3).to(q.dtype)
