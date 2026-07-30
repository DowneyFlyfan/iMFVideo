"""Triton kernel for the Kimi-K3 attention residual (fused forward + JVP).

The op (per token, see modeling_kimi_linear._apply_attn_res and
models/imf_dit_video.attn_res_apply): J candidate vectors (residual-stream
snapshots + current prefix sum) are RMS-normalized, scored against one
learned direction, softmaxed, and mixed:

    r_j = sqrt(mean(v_j^2) + eps)          per-candidate RMS
    s_j = <v_j, w> / r_j                   w = norm_gain * proj_weight
    p   = softmax(s)                       over the J candidates
    y   = sum_j p_j * v_j

Forward-mode JVP (parameter tangents zero, dv_j the input tangents):

    ds_j = <dv_j, w> / r_j - <v_j, w> * mean(v_j * dv_j) / r_j^3
    dp   = p * (ds - <p, ds>)
    dy   = sum_j dp_j * v_j + p_j * dv_j

Eager torch runs ~10 elementwise/reduction kernels and materializes the
(n, J, d) fp32 stack twice per pass (primal + tangent under torch.func.jvp);
this kernel does the whole thing in one launch, two register-resident passes
over the J candidates per token row.

Shape symbols:
    n : token rows (batch * sequence)
    J : candidates = snapshots + 1 (small: <= JP <= 32)
    d : hidden size
"""

import torch
import triton
import triton.language as tl

__all__ = ["triton_attn_res_jvp", "triton_attn_res_fwd", "attn_res_op"]


def _unwrap_functorch(t):
    """Strip functorch wrappers (jvp duals) down to the plain tensor."""
    while torch._C._functorch.is_functorch_wrapped_tensor(t):
        t = torch._C._functorch.get_unwrapped(t)
    return t.contiguous()


@triton.jit
def _attn_res_jvp_kernel(V, DV, W, TW, Y, DY, J, D, eps,
                         BLOCK: tl.constexpr,     # next_power_of_2(D)
                         JP: tl.constexpr,        # next_power_of_2(J)
                         HAS_T: tl.constexpr,     # tangent path on/off
                         HAS_TW: tl.constexpr):   # score-direction tangent
    # V, DV: (n, J, D) primal / tangent candidate stacks (row-major)
    # W: (D,) fused score direction (norm gain * proj weight)
    # TW: (D,) tangent of W (zeros tensor when absent)
    # Y, DY: (n, D) outputs
    row = tl.program_id(0)                 # token row index
    cols = tl.arange(0, BLOCK)             # (BLOCK,) channel index
    cm = cols < D
    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)   # (BLOCK,)
    if HAS_TW:
        tw = tl.load(TW + cols, mask=cm, other=0.0).to(tl.float32)
    base = row * J * D

    jidx = tl.arange(0, JP)                # (JP,) candidate slot index
    s_vec = tl.full([JP], float("-inf"), tl.float32)   # (JP,) scores
    ds_vec = tl.zeros([JP], tl.float32)    # (JP,) score tangents

    # ---- pass 1: per-candidate RMS statistics and scores ----
    for j in range(0, J):
        v = tl.load(V + base + j * D + cols, mask=cm, other=0.0).to(tl.float32)
        m2 = tl.sum(v * v, axis=0) / D     # scalar mean(v^2)
        a = tl.sum(v * w, axis=0)          # scalar <v, w>
        inv_r = 1.0 / tl.sqrt(m2 + eps)    # scalar 1/r
        s = a * inv_r                      # scalar score
        sel = jidx == j                    # (JP,) one-hot slot mask
        s_vec = tl.where(sel, s, s_vec)
        if HAS_T:
            dv = tl.load(DV + base + j * D + cols, mask=cm,
                         other=0.0).to(tl.float32)
            ip = tl.sum(v * dv, axis=0) / D            # scalar mean(v*dv)
            b = tl.sum(dv * w, axis=0)                 # scalar <dv, w>
            if HAS_TW:
                b += tl.sum(v * tw, axis=0)            # + <v, tw>
            ds = b * inv_r - a * ip * (inv_r * inv_r * inv_r)
            ds_vec = tl.where(sel, ds, ds_vec)

    # ---- softmax over the J valid slots (invalid slots stay -inf -> 0) ----
    mx = tl.max(s_vec, axis=0)             # scalar running max
    e = tl.exp(s_vec - mx)                 # (JP,) unnormalized probs
    p_vec = e / tl.sum(e, axis=0)          # (JP,) probabilities
    if HAS_T:
        dsm = tl.sum(p_vec * ds_vec, axis=0)           # scalar <p, ds>
        dp_vec = p_vec * (ds_vec - dsm)    # (JP,) probability tangents

    # ---- pass 2: probability-weighted mix (V/DV re-read, L2-resident) ----
    y = tl.zeros([BLOCK], tl.float32)      # (BLOCK,) output accumulator
    dy = tl.zeros([BLOCK], tl.float32)     # (BLOCK,) tangent accumulator
    for j in range(0, J):
        v = tl.load(V + base + j * D + cols, mask=cm, other=0.0).to(tl.float32)
        pj = tl.sum(tl.where(jidx == j, p_vec, 0.0), axis=0)   # scalar p_j
        y += pj * v
        if HAS_T:
            dv = tl.load(DV + base + j * D + cols, mask=cm,
                         other=0.0).to(tl.float32)
            dpj = tl.sum(tl.where(jidx == j, dp_vec, 0.0), axis=0)
            dy += dpj * v + pj * dv

    tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask=cm)
    if HAS_T:
        tl.store(DY + row * D + cols, dy.to(DY.dtype.element_ty), mask=cm)


def _launch(v, dv, w, eps, tw=None):
    # v/dv: (n, J, d) contiguous;  w/tw: (d,);  returns (y, dy) each (n, d)
    n, J, d = v.shape
    assert v.is_cuda and v.is_contiguous()
    assert J <= 32, "attention-residual kernel supports at most 32 candidates"
    y = torch.empty(n, d, device=v.device, dtype=v.dtype)
    dy = torch.empty(n, d, device=v.device, dtype=v.dtype) if dv is not None else y
    BLOCK = triton.next_power_of_2(d)
    _attn_res_jvp_kernel[(n,)](
        v, dv if dv is not None else v, w, tw if tw is not None else w,
        y, dy, J, d, eps,
        BLOCK=BLOCK, JP=triton.next_power_of_2(J),
        HAS_T=dv is not None, HAS_TW=tw is not None,
        num_warps=4 if BLOCK <= 1024 else 8,
    )
    return y, dy


def triton_attn_res_jvp(v, dv, w, eps=1e-6, tw=None):
    """Fused primal + JVP of the attention residual.

    Args:
        v: (n, J, d) fp32/fp16 candidate stack ([snapshots..., prefix]).
        dv: (n, J, d) input tangents (same dtype/layout).
        w: (d,) fused score direction = norm_gain * proj_weight.
        eps: RMSNorm epsilon.
        tw: optional (d,) tangent of w (None = zero tangent).

    Returns:
        (y, dy): each (n, d), the mixed stream and its tangent.
    """
    assert dv is not None and dv.is_contiguous()
    return _launch(v, dv, w, eps, tw)


def triton_attn_res_fwd(v, w, eps=1e-6):
    """Primal-only attention residual.

    Args:
        v: (n, J, d) candidate stack.  w: (d,) fused score direction.

    Returns:
        y: (n, d) mixed stream.
    """
    y, _ = _launch(v, None, w, eps)
    return y


@triton.jit
def _attn_res_bwd_kernel(V, W, GY, GV, GSR, J, D, eps,
                         BLOCK: tl.constexpr,     # next_power_of_2(D)
                         JP: tl.constexpr):       # next_power_of_2(J)
    # V: (n, J, D) saved primal candidates;  W: (D,) fused score direction
    # GY: (n, D) upstream gradient;  GV: (n, J, D) input gradient out
    # GSR: (n, J) gs_j / r_j workspace (host reduces it to the w gradient)
    row = tl.program_id(0)                 # token row index
    cols = tl.arange(0, BLOCK)             # (BLOCK,) channel index
    cm = cols < D
    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)   # (BLOCK,)
    gy = tl.load(GY + row * D + cols, mask=cm, other=0.0).to(tl.float32)
    base = row * J * D

    jidx = tl.arange(0, JP)                # (JP,) candidate slot index
    s_vec = tl.full([JP], float("-inf"), tl.float32)   # (JP,) scores
    gp_vec = tl.zeros([JP], tl.float32)    # (JP,) <gy, v_j>
    a_vec = tl.zeros([JP], tl.float32)     # (JP,) <v_j, w>
    ir_vec = tl.zeros([JP], tl.float32)    # (JP,) 1 / r_j

    # ---- pass 1: per-candidate statistics ----
    for j in range(0, J):
        v = tl.load(V + base + j * D + cols, mask=cm, other=0.0).to(tl.float32)
        m2 = tl.sum(v * v, axis=0) / D     # scalar mean(v^2)
        a = tl.sum(v * w, axis=0)          # scalar <v, w>
        inv_r = 1.0 / tl.sqrt(m2 + eps)    # scalar 1/r
        gp = tl.sum(gy * v, axis=0)        # scalar <gy, v>
        sel = jidx == j
        s_vec = tl.where(sel, a * inv_r, s_vec)
        gp_vec = tl.where(sel, gp, gp_vec)
        a_vec = tl.where(sel, a, a_vec)
        ir_vec = tl.where(sel, inv_r, ir_vec)

    # ---- softmax probabilities and their gradient ----
    mx = tl.max(s_vec, axis=0)
    e = tl.exp(s_vec - mx)                 # (JP,) invalid slots exp(-inf)=0
    p_vec = e / tl.sum(e, axis=0)          # (JP,)
    gph = tl.sum(p_vec * gp_vec, axis=0)   # scalar <p, gp>
    gs_vec = p_vec * (gp_vec - gph)        # (JP,) score gradient

    # ---- pass 2: input gradient;  gv_j = p_j gy + gs_j ds_j/dv_j ----
    for j in range(0, J):
        v = tl.load(V + base + j * D + cols, mask=cm, other=0.0).to(tl.float32)
        pj = tl.sum(tl.where(jidx == j, p_vec, 0.0), axis=0)
        gsj = tl.sum(tl.where(jidx == j, gs_vec, 0.0), axis=0)
        aj = tl.sum(tl.where(jidx == j, a_vec, 0.0), axis=0)
        irj = tl.sum(tl.where(jidx == j, ir_vec, 0.0), axis=0)
        gv = pj * gy + gsj * (w * irj - v * (aj * irj * irj * irj / D))
        tl.store(GV + base + j * D + cols, gv.to(GV.dtype.element_ty),
                 mask=cm)
        tl.store(GSR + row * J + j, gsj * irj)


class _AttnResFn(torch.autograd.Function):
    """Triton attention residual with full autograd + forward-AD support.

    forward: fused Triton primal kernel.
    backward: fused Triton gradient kernel for gv, plus one matmul reduction
        for gw (gw = sum_{n,j} (gs/r)_{nj} * v_{nj}).
    jvp (torch.func.jvp): fused primal+tangent kernel; the tangent of the
        score direction w is required to be None (parameter tangents are
        zero in the MeanFlow JVP).
    """

    @staticmethod
    def forward(v, w, eps):
        # v: (n, J, d) candidate stack;  w: (d,) fused score direction
        return triton_attn_res_fwd(v, w, eps)      # (n, d)

    @staticmethod
    def setup_context(ctx, inputs, output):
        v, w, eps = inputs
        ctx.save_for_backward(v, w)
        ctx.save_for_forward(v, w)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, gy):
        # gy: (n, d) upstream gradient
        v, w = ctx.saved_tensors
        n, J, d = v.shape
        gy = gy.contiguous()
        gv = torch.empty_like(v)                   # (n, J, d)
        gsr = torch.empty(n, J, device=v.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(d)
        _attn_res_bwd_kernel[(n,)](
            v, w, gy, gv, gsr, J, d, ctx.eps,
            BLOCK=BLOCK, JP=triton.next_power_of_2(J),
            num_warps=4 if BLOCK <= 1024 else 8,
        )
        # gw[d'] = sum_{n,j} gsr[n,j] * v[n,j,d']   (einsum "x,xd->d")
        gw = (gsr.reshape(1, n * J) @ v.reshape(n * J, d).float()).reshape(d)
        return gv, gw.to(w.dtype), None

    @staticmethod
    def jvp(ctx, tv, tw, _teps):
        from torch._functorch.pyfunctorch import (
            temporarily_clear_interpreter_stack,
        )

        v, w = ctx.saved_for_forward
        v = _unwrap_functorch(v)
        w = _unwrap_functorch(w)
        tv = None if tv is None else _unwrap_functorch(tv.to(v.dtype))
        tw = None if tw is None else _unwrap_functorch(tw.to(w.dtype))
        with temporarily_clear_interpreter_stack():
            if tv is None:
                tv = torch.zeros_like(v)
            _, dy = triton_attn_res_jvp(v, tv, w, ctx.eps, tw)   # (n, d)
        return dy


def attn_res_op(v, w, eps=1e-6):
    """Differentiable attention residual (Triton; autograd + forward AD).

    Args:
        v: (n, J, d) fp32 CUDA candidate stack ([snapshots..., prefix]).
        w: (d,) fused score direction (norm gain * proj weight; may carry
            grad -- the backward returns its gradient).
        eps: RMSNorm epsilon.

    Returns:
        (n, d) mixed stream.
    """
    return _AttnResFn.apply(v, w, eps)
