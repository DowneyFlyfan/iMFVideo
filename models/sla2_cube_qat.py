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
        self.n_tail = -(-self.prefix_len // self.E)  # prefix tail blocks
        self.Mb = self.Np + self.n_tail
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
        # append the ragged prefix tail block(s) as always-selected entries
        tail_ids = torch.arange(self.Np, self.Np + self.n_tail,
                                device=q.device, dtype=lut.dtype)
        tail = tail_ids.view(1, 1, 1, -1).expand(
            lut.shape[0], lut.shape[1], lut.shape[2], -1)  # (B,H,Mb,n_tail)
        lut_aug = torch.cat([lut, tail], dim=-1).contiguous()
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
                                       self.use_int8, self.a_index,
                                       self.n_tail)
            if self.pre_permuted:
                return o.permute(0, 2, 1, 3).to(q.dtype)
            return o[:, :, self.inv].permute(0, 2, 1, 3).to(q.dtype)
        else:
            # guidance path (no_grad): same direct kernels as the training
            # Function (vendored int8 sparse fwd with fused phi + states
            # linear fwd), no context saving.
            B_, H_, L_, D_ = qh.shape
            M_BLOCKS = triton.cdiv(L_, self.E)
            o_s = torch.empty_like(vh)                        # (B,H,L,D)
            lse = torch.empty(B_, H_, L_, device=qh.device,
                              dtype=torch.float32)
            qphi = torch.empty_like(qh)                       # (B,H,L,D)
            _attn_fwd_qat[(M_BLOCKS, B_ * H_)](
                qh, kh, vh, lut_aug, lse, o_s, qphi,
                D_ ** -0.5, T + self.n_tail, L_, M_BLOCKS, D_, self.E, self.E,
                self.use_int8, True, num_warps=4, num_stages=3,
            )
            kphi = torch.softmax(kh.float(), -1).to(qh.dtype).contiguous()
            htot, ztot = _precompute_global(qphi, kphi, vh)
            o_l = torch.empty_like(vh)                        # (B,H,L,D)
            den = torch.empty(B_, H_, L_, device=qh.device,
                              dtype=torch.float32)
            _lin_fwd2[(M_BLOCKS, B_ * H_)](
                qphi, kphi, vh, htot.to(qh.dtype).contiguous(), ztot,
                lut_aug, o_l, den, o_l, ztot, o_l,
                T + self.n_tail, L_, M_BLOCKS, D_, self.E, self.E,
                False, False, T + self.n_tail <= 4,
                num_warps=8, num_stages=3,
            )

        # token counts per query block: Np full video tiles, then the
        # prefix split into n_tail blocks (last one ragged)
        tail_sizes = [self.E] * (self.n_tail - 1) + [
            self.prefix_len - (self.n_tail - 1) * self.E]
        reps = [self.E] * self.Np + tail_sizes
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
    BLOCK: tl.constexpr,       # E: query tile = key tile = quant block
    N_TAIL: tl.constexpr,      # ceil(P / E) always-attended prefix blocks
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

    for j in range(0, T + N_TAIL):
        if j >= T:
            # ragged prefix tail block(s), always attended
            nb_id = NPE // BLOCK + (j - T)
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
        # fp32 accumulate: the fp16-accumulated dot overflowed at 65504 when
        # primals and tangents are jointly large (score tangents reach 1e4+),
        # the same class as the fp16 output-cast NaN fixed in 66a4e5b. Same
        # tensor-core rate on A100 (HMMA fp16-in/fp32-acc).
        ds = (tl.dot(dq, tl.trans(kf), out_dtype=tl.float32)
              + tl.dot(qf, tl.trans(dk), out_dtype=tl.float32)) * scale
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
        # t = p * ds carries the score tangent (reaches 1e4+ when primals and
        # tangents are jointly large); saturate its fp16 MMA operand and
        # accumulate in fp32 so the products stay finite.
        tc = tl.clamp(t, -60000.0, 60000.0).to(tl.float16)
        o_t = tl.dot(pc, v, out_dtype=tl.float32)
        do_t = (tl.dot(tc, v, out_dtype=tl.float32)
                + tl.dot(pc, dv, out_dtype=tl.float32))
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
    # zc is a complement state (global minus routed); fp32 cancellation can
    # push it negative, and a plain "+ eps_l" then lets den cross zero and
    # nan the quotient. Signed epsilon (|den| >= eps_l, sign preserved)
    # matches the vendored _lin_fwd2 exactly.
    den = tl.sum(qphi * zc[None, :], 1)                      # (BLOCK,)
    den = tl.where(den >= 0, den + eps_l, den - eps_l)
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

    # per-tile phi-states in one kernel pass (video tiles only)
    Hb = torch.empty(B, H, Np, D, D, device=q.device, dtype=torch.float32)
    dHb = torch.empty_like(Hb)
    zb = torch.empty(B, H, Np, D, device=q.device, dtype=torch.float32)
    dzb = torch.empty_like(zb)
    _phi_states_kernel[(Np, B * H)](
        qc[1], qc[4], qc[2], qc[5], Hb, dHb, zb, dzb,
        qc[1].stride(0), qc[1].stride(1), qc[1].stride(2),
        Hb.stride(0), Hb.stride(1), Hb.stride(2),
        zb.stride(0), zb.stride(1), zb.stride(2),
        H, L, E=E, D=D, num_warps=4)
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
        N_TAIL=-(-prefix_len // E),
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
                use_int8, a_index, n_tail=1):
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
            scale, T + n_tail, L, M_BLOCKS, D, E, E, use_int8, True,
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
            T + n_tail, L, M_BLOCKS, D, E, E, False, False,
            T + n_tail <= 4, num_warps=8, num_stages=3,
        )

        a_row = torch.sigmoid(alpha_logit.float())          # (H, Mb)
        a = a_row[:, a_index].view(1, H, L, 1).to(o_s.dtype)  # (1,H,L,1)
        o = a * o_s + (1.0 - a) * o_l                       # (B,H,L,D)

        ctx.save_for_backward(qh, kh, vh, alpha_logit, lut, lut_aug, mask,
                              lse, o_s, o_l, den, qphi, kphi, htot, ztot,
                              a_index)
        ctx.T, ctx.E, ctx.scale = T, E, scale
        ctx.n_tail = n_tail
        return o

    @staticmethod
    def backward(ctx, do):
        (qh, kh, vh, alpha_logit, lut, lut_aug, mask, lse, o_s, o_l, den,
         qphi, kphi, htot, ztot, a_index) = ctx.saved_tensors
        T, E, scale = ctx.T, ctx.E, ctx.scale
        n_tail = ctx.n_tail
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
            qh, kh, vh, lse, delta_s, do_s, dq_s, lut_aug, scale,
            T + n_tail, L, M_BLOCKS, D, E, E, num_warps=4, num_stages=4,
        )
        kv2q, qcnt = _invert_lut(mask)   # (B,H,Nb,Mb) int32, (B,H,Nb)
        _sparse_bwd_dkdv_lut[(N_BLOCKS, B * H)](
            qh, kh, vh, do_s, dk_s, dv_s, scale, kv2q, qcnt, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS, D=D, BLOCK=E,
            num_warps=4, num_stages=3,
        )

        # ---- linear branch backward (vendored kernels + globals) ----
        g = do_l.float() / den[..., None]                   # (B,H,L,D)
        s = (do_l.float() * o_l.float()).sum(-1) / den      # (B,H,L)
        dhtot = (qphi.float().transpose(-1, -2) @ g).contiguous()
        dztot = -(qphi.float() * s[..., None]).sum(-2).contiguous()
        # fp32 grad buffers: dqphi = g @ h^T - s z^T reaches ~1/eps_l * |h|
        # when a one-hot phi channel has no global mass (large-logit regime);
        # fp16 buffers here overflowed to inf and poisoned Wan2.2 run 1.
        # The kernels' .to(element_ty) stores adapt to the buffer dtype.
        dqphi = torch.empty_like(qphi, dtype=torch.float32)
        dkphi = torch.empty_like(kphi, dtype=torch.float32)
        dv_l = torch.empty_like(vh, dtype=torch.float32)
        _lin_bwd_dq[(M_BLOCKS, B * H)](
            qphi, kphi, vh, htot, ztot, lut_aug, o_l, den, do_l, dqphi,
            T + n_tail, L, M_BLOCKS, D, E, E, num_warps=8, num_stages=2,
        )
        _lin_bwd_dkdv_lut[(N_BLOCKS, B * H)](
            qphi, kphi, vh, dhtot, dztot, o_l, den, do_l, kv2q, qcnt,
            dkphi, dv_l, L, M_BLOCKS, N_BLOCKS, D=D, BLOCK=E,
            num_warps=4, num_stages=3,
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
        # dv_l is fp32 (see buffer note above); saturate the sum before the
        # fp16 grad cast autograd requires — one-hot-phi rows can carry
        # ~1/eps_l magnitudes that clip_grad_norm will bound after.
        dv = (dv_s.float() + dv_l).clamp_(-60000.0, 60000.0).to(vh.dtype)
        return (dq, dk, dv, d_alpha, None, None, None, None, None, None,
                None, None)


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
    # dphi can legitimately reach ~1/eps_l in the empty-phi-channel regime;
    # saturate so the fp16 store stays finite (grad clip then bounds the
    # step). Healthy dx is O(1).
    dx = tl.clamp(dx, -60000.0, 60000.0)
    tl.store(DX + ptr, dx.to(DX.dtype.element_ty), mask=mask[:, None])


# ---------------------------------------------------------------------------
# States-build kernel for the fused JVP: per-tile phi-states in one pass.
# Replaces the host-side phi softmax pair + einsums + sums (~6 ms/call).
# ---------------------------------------------------------------------------
@triton.jit
def _phi_states_kernel(
    K, DK, V, DV,              # (B, H, L, D) fp16 primal/tangent
    HB, DHB,                   # (B, H, Np, D, Dv) fp32 out states
    ZB, DZB,                   # (B, H, Np, D) fp32 out sums
    s_kb, s_kh, s_ks,          # strides of the (B, H, L, D) inputs
    s_hb, s_hh, s_hn,          # strides of the (B, H, Np, D, Dv) states
    s_zb, s_zh, s_zn,          # strides of the (B, H, Np, D) sums
    H, L,
    E: tl.constexpr,           # tile length
    D: tl.constexpr,           # head dim (= Dv)
):
    pid_n = tl.program_id(0)                    # video tile id in [0, Np)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_e = tl.arange(0, E)
    offs_d = tl.arange(0, D)
    rows = pid_n * E + offs_e
    base = b * s_kb + h * s_kh
    ptr = base + rows[:, None] * s_ks + offs_d[None, :]
    k = tl.load(K + ptr).to(tl.float32)         # (E, D)
    dk = tl.load(DK + ptr).to(tl.float32)
    v = tl.load(V + ptr).to(tl.float32)
    dv = tl.load(DV + ptr).to(tl.float32)

    # phi = channel softmax with tangent (rows of length D)
    mx = tl.max(k, 1)
    ex = tl.exp(k - mx[:, None])
    den = tl.sum(ex, 1)
    kphi = ex / den[:, None]                    # (E, D)
    dot = tl.sum(kphi * dk, 1)
    dkphi = kphi * (dk - dot[:, None])          # (E, D)

    # tf32 dots: the states feed the linear-branch quotient, whose
    # overall tolerance is ~1e-3; ieee fp32 dots measured 3x slower here.
    hb = tl.dot(tl.trans(kphi), v)                           # (D, Dv)
    dhb = tl.dot(tl.trans(dkphi), v) + tl.dot(tl.trans(kphi), dv)
    zb = tl.sum(kphi, 0)                        # (D,)
    dzb = tl.sum(dkphi, 0)

    hp = (b * s_hb + h * s_hh + pid_n * s_hn
          + offs_d[:, None] * D + tl.arange(0, D)[None, :])
    tl.store(HB + hp, hb)
    tl.store(DHB + hp, dhb)
    zp = b * s_zb + h * s_zh + pid_n * s_zn + offs_d
    tl.store(ZB + zp, zb)
    tl.store(DZB + zp, dzb)


# ---------------------------------------------------------------------------
# LUT-native dkdv backwards: instead of scanning all Mb query blocks per
# key block with a data-dependent branch (no prefetch), walk the inverted
# index kv2q directly -- qcnt[n] iterations of straight-line compute.
# ---------------------------------------------------------------------------
def _invert_lut(mask):
    """Invert the block mask into per-key-block query lists.

    Args:
        mask: (B, H, Mb, Nb) int8 selection mask (1 = query block m attends
            key block n), from the forward.

    Returns:
        kv2q: (B, H, Nb, Mb) int32; for each key block n the query-block
            ids with mask = 1 packed at the front (stable order).
        qcnt: (B, H, Nb) int32 number of valid entries per key block.
    """
    mt = mask.permute(0, 1, 3, 2).bool()                     # (B,H,Nb,Mb)
    qcnt = mt.sum(-1, dtype=torch.int32)                     # (B,H,Nb)
    # stable argsort of (not selected): selected blocks sort first, in
    # ascending query-block order
    kv2q = torch.argsort(~mt, dim=-1, stable=True).to(torch.int32)
    return kv2q.contiguous(), qcnt.contiguous()


@triton.jit
def _sparse_bwd_dkdv_lut(
    Q, K, V, DOS, DK, DV, qk_scale, KV2Q, QCNT, LSE, DELTAS,
    L, M_BLOCKS, N_BLOCKS,
    D: tl.constexpr,
    BLOCK: tl.constexpr,       # BLOCK_M == BLOCK_N == E
):
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_off = idx_bh * L * D
    lse_off = idx_bh * L
    inv_off = (idx_bh * N_BLOCKS + idx_n) * M_BLOCKS

    offs_n = idx_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    n_mask = offs_n[:, None] < L
    k = tl.load(K + qkv_off + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask, other=0.0)
    v = tl.load(V + qkv_off + offs_n[:, None] * D + offs_d[None, :],
                mask=n_mask, other=0.0)
    dk = tl.zeros([BLOCK, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK, D], dtype=tl.float32)
    cnt = tl.load(QCNT + idx_bh * N_BLOCKS + idx_n)

    LOG2E: tl.constexpr = 1.4426950408889634
    for j in range(0, cnt):
        m_id = tl.load(KV2Q + inv_off + j).to(tl.int64)
        rows = m_id * BLOCK + offs_m
        m_mask = rows < L
        q = tl.load(Q + qkv_off + rows[:, None] * D + offs_d[None, :],
                    mask=m_mask[:, None], other=0.0)
        lse = tl.load(LSE + lse_off + rows, mask=m_mask,
                      other=float("inf"))
        qkT = tl.dot(k, tl.trans(q)) * (qk_scale * LOG2E)
        # lse was computed by the INT8 forward while qkT here is the fp16
        # recompute; at large logits the quantization mismatch makes the
        # exponent spuriously positive and pT overflows the fp16 cast below
        # (inf rows in dk/dv poisoned Wan2.2 run 1 at step 3950). True
        # p <= 1, so clamping the exponent at +4 (pT <= 16) is inert in the
        # healthy regime and bounds the mismatch regime.
        pT = tl.exp2(tl.minimum(qkT - lse[None, :], 4.0))
        pT = tl.where(offs_n[:, None] < L, pT, 0.0)
        do = tl.load(DOS + qkv_off + rows[:, None] * D + offs_d[None, :],
                     mask=m_mask[:, None], other=0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        delta = tl.load(DELTAS + lse_off + rows, mask=m_mask, other=0.0)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        # Saturate before the fp16 cast (see the pT clamp above).
        dsT = tl.clamp(dsT, -60000.0, 60000.0)
        dk += tl.dot(dsT.to(q.dtype), q)

    tl.store(DK + qkv_off + offs_n[:, None] * D + offs_d[None, :],
             (dk * qk_scale).to(DK.dtype.element_ty), mask=n_mask)
    tl.store(DV + qkv_off + offs_n[:, None] * D + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=n_mask)


@triton.jit
def _lin_bwd_dkdv_lut(
    QPHI, KPHI, V, DHTOT, DZTOT, OL, DEN, DOL, KV2Q, QCNT, DKPHI, DV,
    L, M_BLOCKS, N_BLOCKS,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complement version: dh starts from the GLOBAL dhtot and subtracts
    the SELECTED query blocks' contributions (the complement's sum equals
    global minus selected), then dKphi/dV as in the vendored kernel."""
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_off = idx_bh * L * D
    h_off = idx_bh * D * D
    z_off = idx_bh * D
    den_off = idx_bh * L
    inv_off = (idx_bh * N_BLOCKS + idx_n) * M_BLOCKS

    offs_n = idx_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, D)
    dh = tl.load(DHTOT + h_off + offs_d[:, None] * D + offs_e[None, :]
                 ).to(tl.float32)
    dz = tl.load(DZTOT + z_off + offs_d).to(tl.float32)
    cnt = tl.load(QCNT + idx_bh * N_BLOCKS + idx_n)

    for j in range(0, cnt):
        m_id = tl.load(KV2Q + inv_off + j).to(tl.int64)
        rows = m_id * BLOCK + offs_m
        m_mask = rows < L
        qp = tl.load(QPHI + qkv_off + rows[:, None] * D + offs_d[None, :],
                     mask=m_mask[:, None], other=0.0)
        dol = tl.load(DOL + qkv_off + rows[:, None] * D + offs_e[None, :],
                      mask=m_mask[:, None], other=0.0)
        ol = tl.load(OL + qkv_off + rows[:, None] * D + offs_e[None, :],
                     mask=m_mask[:, None], other=0.0)
        den = tl.load(DEN + den_off + rows, mask=m_mask, other=1.0)
        g = dol.to(tl.float32) / den[:, None]
        # den ~ eps_l when a one-hot phi channel has no global mass; the
        # fp16 cast of g below would inf. Healthy g is O(10) (see
        # kernel_linear._lin_bwd_dq).
        g = tl.clamp(g, -60000.0, 60000.0)
        s_r = tl.sum(dol.to(tl.float32) * ol.to(tl.float32), 1) / den
        dh -= tl.dot(tl.trans(qp), g.to(qp.dtype)).to(tl.float32)
        dz += tl.sum(qp.to(tl.float32) * s_r[:, None], 0)

    n_mask = offs_n[:, None] < L
    kp = tl.load(KPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :],
                 mask=n_mask, other=0.0)
    vv = tl.load(V + qkv_off + offs_n[:, None] * D + offs_e[None, :],
                 mask=n_mask, other=0.0)
    # dh accumulates g-scaled dots; saturate before its fp16 casts.
    dh = tl.clamp(dh, -60000.0, 60000.0)
    dkp = tl.dot(vv, tl.trans(dh).to(vv.dtype)).to(tl.float32) + dz[None, :]
    dv = tl.dot(kp, dh.to(kp.dtype)).to(tl.float32)
    tl.store(DKPHI + qkv_off + offs_n[:, None] * D + offs_d[None, :],
             dkp.to(DKPHI.dtype.element_ty), mask=n_mask)
    tl.store(DV + qkv_off + offs_n[:, None] * D + offs_e[None, :],
             dv.to(DV.dtype.element_ty), mask=n_mask)
