"""Fast fully-detached forward-mode JVP of IMFDiTVideo using the Triton MLA
kernels, for the iMF du/dt pass.

In iMF the tangent output enters the loss only inside stop-gradient
(V = u + (t - r) * sg(du/dt)), so when the primal for the loss comes from a
separate plain forward, the whole JVP pass needs NO autograd: this module
propagates (primal, tangent) dual pairs by hand -- no torch.func.jvp over
the network, no functorch wrappers, no backward graph.

Structure (mirrors IMFDiTVideo.forward with attn_res_block_size > 0, but the
v-heads are skipped: only the u-branch tangent du/dt is needed):
    sequence embed (small; torch.func.jvp over _build_sequence)
    -> per block: attn-res remix (fused Triton jvp) -> attn sublayer
       (Triton kernel stages) -> prefix update -> attn-res remix -> MLP
       sublayer (Triton kernel stages) -> prefix update
    -> u output attn-res remix -> u FinalLayer jvp -> unpatchify.

Shape symbols:
    b : batch;  l : sequence length (prefix tokens + patch tokens)
    d : hidden size;  H : heads;  dq/dc : q/kv latent ranks
    dn/dr/dv : nope / rope / value head dims;  F : padded SwiGLU width
    M : 2*b*l stacked token rows (primal rows first, tangent rows after)
    C, T, Hh, Ww : video latent channels / frames / height / width

Weight caches (fp16) must be rebuilt after every optimizer step:
build_fast_jvp_state(net) does one pass over the blocks.
"""

import gc

import torch
import triton

from models.triton_block_jvp import _flash_jvp
from models.triton_mla_block_jvp import (
    _mla_fp16_cache,
    _mm16,
    _out_gate_jvp_kernel,
    _q_prep_kernel,
    _kv_prep_kernel,
    _rmsnorm_jvp_strided,
    _swiglu_row_jvp,
    mla_params_from_block,
)
from models.triton_attn_res_jvp import triton_attn_res_jvp
from models.sla2_mla_jvp import route_blocks, sla2_mla_jvp

__all__ = ["build_fast_jvp_state", "model_du_dt_fast"]

_FP16 = torch.float16


def cube_attention_jvp_buffers(batch_size, sequence_length, attention_width, device):
    """Allocate cube-attention primal/tangent buffers without fp16 T2 loss."""
    primal = torch.empty(batch_size, sequence_length, attention_width,
                          device=device, dtype=_FP16)
    tangent = torch.empty(batch_size, sequence_length, attention_width,
                          device=device, dtype=torch.float32)
    return primal, tangent


def _attn_sublayer_jvp(h, dh, p, w16, cos, sin, eps):
    """MLA attention sublayer JVP: a = out_proj(gate(attn(norm1(h)))) * g_attn.

    Args:
        h, dh: (b, l, d) fp32 primal / tangent sublayer input.
        p: block param dict (mla_params_from_block).
        w16: fp16 fused-weight cache (_mla_fp16_cache).
        cos, sin: (l, dr//2) fp32 rotary tables.
        eps: RMSNorm epsilon.

    Returns:
        (a, da): (b, l, d) fp32 gated attention output and its tangent.
    """
    B, S, D = h.shape
    H, dq, dc = p["H"], p["dq"], p["dc"]
    dn, dr, dv = p["dn"], p["dr"], p["dv"]
    hd = dn + dr                           # qk head dim (= 64)
    dev = h.device
    half = B * S                           # primal token rows
    M = 2 * half                           # stacked rows
    scale = 1.0 / (hd ** 0.5)

    # ---- RMSNorm1 JVP: fp32 -> stacked fp16 (2b, l, d) ----
    xs = torch.empty(2 * B, S, D, device=dev, dtype=_FP16)
    _rmsnorm_jvp_strided(h.contiguous(), dh.contiguous(), p["w_norm1"],
                         xs[:B], xs[B:], D, D, D, eps, half)

    # a[m, j] = sum_k n1[m, k] * w_a[j, k]           (einsum "mk,jk->mj")
    wa = dq + dc + dr
    a_ = _mm16(xs.view(M, D), w16["w_a"])            # (M, dq+dc+dr) fp16

    q_lat = torch.empty(M, dq, device=dev, dtype=_FP16)
    _rmsnorm_jvp_strided(a_, a_[half:], p["w_qa_ln"],
                         q_lat, q_lat[half:], dq, wa, dq, eps, half)
    kv_lat = torch.empty(M, dc, device=dev, dtype=_FP16)
    _rmsnorm_jvp_strided(a_[:, dq:], a_[half:, dq:], p["w_kva_ln"],
                         kv_lat, kv_lat[half:], dc, wa, dc, eps, half)

    # qb[m, j] = sum_k q_lat[m, k] * w_qb[j, k]      (einsum "mk,jk->mj")
    qb = _mm16(q_lat, w16["w_qb"])                   # (M, H*hd)
    kvb = _mm16(kv_lat, w16["w_kvb"])                # (M, H*(dn+dv))

    packed = torch.empty(2 * B, S, 3, H, hd, device=dev, dtype=_FP16)
    so_b, so_s = packed.stride(0), packed.stride(1)
    qv, kv_, vv = packed[:, :, 0], packed[:, :, 1], packed[:, :, 2]
    BN = triton.next_power_of_2(dn)
    _q_prep_kernel[(half,)](
        qb, p["w_qnorm"], cos, sin, qv, H * hd, so_b, so_s, B, S, half, eps,
        NH=H, DN=dn, DR2=dr // 2, HD=hd, BN=BN, CLAMP=False, num_warps=4,
    )
    _kv_prep_kernel[(half,)](
        kvb, a_[:, dq + dc:], p["w_knorm"], cos, sin, kv_, vv,
        H * (dn + dv), wa, so_b, so_s, B, S, half, eps,
        NH=H, DN=dn, DR2=dr // 2, DV=dv, HD=hd, BN=BN, KVW=dn + dv,
        CLAMP=False, num_warps=4,
    )

    if "cube" in p:
        attn, dattn = cube_attention_jvp_buffers(B, S, H * dv, dev)
        # SLA2-Cube-QAT JVP (models/sla2_cube_qat.py): int8 primal scores
        # over routed 3D tiles + prefix tail, linear complement, alpha mix.
        cb = p["cube"]
        perm, inv = cb["perm"], cb["inv"]
        from models.sla2_cube_qat import sla2_cube_qat_jvp_fused as sla2_cube_qat_jvp
        # one batch row at a time: the six tile-major copies plus the
        # linear-branch fp32 states scale with B, and the probe runs this
        # at B=8 -- per-row processing keeps the peak at the B=1 level on
        # a 16 GB card. On large-VRAM GPUs the row loop only adds serial
        # launch latency, so batch rows are grouped there.
        from models.mla_video_sparse_jvp import _plenty_of_vram
        row_group = B if _plenty_of_vram(attn.device) else 1
        pre_perm = cb.get("pre_permuted", False)
        for bi in range(0, B, row_group):
            g = min(row_group, B - bi)  # rows in this group
            # (g, S, H, hd) fp16 views -> (g, H, L, hd) tile-major copies
            # (when the model runs tile-major residency the tokens are
            # already in tile order: layout transpose only)
            if pre_perm:
                qb, kb, vb, dqb, dkb, dvb = (
                    t[bi:bi + g].permute(0, 2, 1, 3).contiguous()
                    for t in (qv[:B], kv_[:B], vv[:B],
                              qv[B:], kv_[B:], vv[B:])
                )
            else:
                qb, kb, vb, dqb, dkb, dvb = (
                    t[bi:bi + g].permute(0, 2, 1, 3)[:, :, perm].contiguous()
                    for t in (qv[:B], kv_[:B], vv[:B],
                              qv[B:], kv_[B:], vv[B:])
                )
            o_a, do_a = sla2_cube_qat_jvp(
                qb, kb, vb, dqb, dkb, dvb, cb["alpha"], cb["topk"],
                cb["Np"], cb["E"], cb["P"],
                proj_q=cb["proj_q"], proj_k=cb["proj_k"])
            # o_a is fp16-safe. do_a contains linear quotient T2 and must
            # remain fp32 through the output projection.
            if pre_perm:
                attn[bi:bi + g] = o_a.permute(0, 2, 1, 3).reshape(
                    g, S, H * dv)
                dattn[bi:bi + g] = do_a.permute(
                    0, 2, 1, 3).reshape(g, S, H * dv)
            else:
                attn[bi:bi + g] = o_a[:, :, inv].permute(
                    0, 2, 1, 3).reshape(g, S, H * dv)
                dattn[bi:bi + g] = do_a[:, :, inv].permute(
                    0, 2, 1, 3).reshape(g, S, H * dv)
            del qb, kb, vb, dqb, dkb, dvb, o_a, do_a
    elif "sla2" in p:
        attn = torch.empty(2 * B, S, H * dv, device=dev, dtype=_FP16)
        # SLA2 sparse-linear attention JVP (models/sla2_mla_jvp.py):
        # sparse branch over routed blocks + complement linear states.
        sl = p["sla2"]  # dict: alpha (H,Mb) fp32, proj_q/proj_k (hd,hd), knobs
        # (B, S, H, hd) fp16 packed views -> (B, H, S, hd) contiguous
        qb, kb, vb, dqb, dkb, dvb = (
            t.permute(0, 2, 1, 3).contiguous()
            for t in (qv[:B], kv_[:B], vv[:B], qv[B:], kv_[B:], vv[B:])
        )
        lut, T = route_blocks(qb, kb, sl["topk"], sl["bq"], sl["bk"],
                              proj_q=sl["proj_q"], proj_k=sl["proj_k"])
        o_a, do_a = sla2_mla_jvp(qb, kb, vb, dqb, dkb, dvb, sl["alpha"],
                                 sl["topk"], sl["bq"], sl["bk"], lut=lut, T=T)
        # (B, H, S, dv) fp32 -> (B, S, H*dv) fp16 attn buffer
        attn[:B] = o_a.permute(0, 2, 1, 3).reshape(B, S, H * dv).to(_FP16)
        attn[B:] = do_a.permute(0, 2, 1, 3).reshape(B, S, H * dv).to(_FP16)
    else:
        attn = torch.empty(2 * B, S, H * dv, device=dev, dtype=_FP16)
        # o[m, h, :] = softmax_s(q . k * scale) @ v  (fused dense flash JVP)
        _flash_jvp(qv[:B], kv_[:B], vv[:B], qv[B:], kv_[B:], vv[B:],
                   attn[:B].view(B, S, H, dv), attn[B:].view(B, S, H, dv),
                   scale)

    # Kimi MLA output gate: attn *= sigmoid(n1 @ w_og^T), JVP fused
    if "w_og" in w16:
        z = _mm16(xs.view(M, D), w16["w_og"])        # (M, H*dv) fp16
        tot = half * H * dv
        if "cube" in p:
            _out_gate_jvp_kernel[(triton.cdiv(tot, 1024),)](
                attn, dattn, z, z[half:], tot,
                BLOCK=1024, CLAMP=False, num_warps=4,
            )
        else:
            _out_gate_jvp_kernel[(triton.cdiv(tot, 1024),)](
                attn[:B], attn[B:], z, z[half:], tot,
                BLOCK=1024, CLAMP=False, num_warps=4,
            )

    # proj[m, j] = sum_k attn[m, k] * w_out[j, k]    (einsum "mk,jk->mj")
    if "cube" in p:
        proj = _mm16(attn.view(half, H * dv), w16["w_out"],
                     out_dtype=torch.float32).view(B, S, D)
        dproj = torch.matmul(
            dattn.view(half, H * dv), p["w_out"].float().t()
        ).view(B, S, D)
        g = p["g_attn"]                              # (d,) fp32 vector gate
        return proj * g, dproj * g
    proj = _mm16(attn.view(M, H * dv), w16["w_out"],
                 out_dtype=torch.float32).view(2 * B, S, D)
    g = p["g_attn"]                                  # (d,) fp32 vector gate
    return proj[:B] * g, proj[B:] * g                # (b, l, d) fp32 each


def _mlp_sublayer_jvp(h, dh, p, w16, eps):
    """MLP sublayer JVP: m = down(situ(gate_up(norm2(h)))) * g_mlp.

    Args:
        h, dh: (b, l, d) fp32 primal / tangent sublayer input.
        p, w16, eps: as in _attn_sublayer_jvp.

    Returns:
        (m, dm): (b, l, d) fp32 gated MLP output and its tangent.
    """
    B, S, D = h.shape
    dev = h.device
    half = B * S
    M = 2 * half
    f = w16["w_gate_up"].shape[0] // 2               # padded SwiGLU width F

    xs = torch.empty(2 * B, S, D, device=dev, dtype=_FP16)
    _rmsnorm_jvp_strided(h.contiguous(), dh.contiguous(), p["w_norm2"],
                         xs[:B], xs[B:], D, D, D, eps, half)

    # gu[m, j] = sum_k n2[m, k] * w_gate_up[j, k]    (einsum "mk,jk->mj")
    gu = _mm16(xs.view(M, D), w16["w_gate_up"]).view(2 * B, S, 2 * f)
    act = torch.empty(2 * B, S, f, device=dev, dtype=_FP16)
    _swiglu_row_jvp(gu[:B], gu[B:], act[:B], act[B:], half, f,
                    beta=p.get("situ_beta", 1.0))

    # dwn[m, j] = sum_k act[m, k] * w_down[j, k]     (einsum "mk,jk->mj")
    dwn = _mm16(act.view(M, f), w16["w_down"],
                out_dtype=torch.float32).view(2 * B, S, D)
    g = p["g_mlp"]                                   # (d,) fp32 vector gate
    return dwn[:B] * g, dwn[B:] * g                  # (b, l, d) fp32 each


def _attn_res_pair(prefix, dprefix, snaps, dsnaps, gain, proj_w, eps):
    """Attention-residual remix of a dual pair via the fused Triton kernel.

    Args:
        prefix, dprefix: (b, l, d) primal / tangent prefix sum.
        snaps, dsnaps: lists of (b, l, d) snapshots and their tangents.
        gain: (d,) res-norm gain;  proj_w: (1, d) score projection.

    Returns:
        (h, dh): (b, l, d) remixed stream and tangent.
    """
    b, l, d = prefix.shape
    J = len(snaps) + 1
    v = torch.stack(list(snaps) + [prefix], dim=2)       # (b, l, J, d)
    dv = torch.stack(list(dsnaps) + [dprefix], dim=2)    # (b, l, J, d)
    w = gain.float() * proj_w.reshape(-1).float()        # (d,)
    y, dy = triton_attn_res_jvp(v.reshape(b * l, J, d),
                                dv.reshape(b * l, J, d), w, eps)
    return y.view(b, l, d), dy.view(b, l, d)


def build_fast_jvp_state(net):
    """Extract per-block params + fp16 weight caches from an IMFDiTVideo.

    Must be called after every optimizer step (weights change).

    Returns:
        dict with per-block (params, fp16 cache) for shared trunk + u-heads,
        u FinalLayer tensors, output-res tensors, rope tables, eps.
    """
    def pack(block):
        # p: block param dict; extended with the attention-residual heads
        # (attn_res_norm/proj (d,)/(1,d), mlp_res_norm/proj same shapes)
        p = mla_params_from_block(block)
        p["attn_res_norm"] = block.attn_res_norm.weight.detach()   # (d,)
        p["attn_res_proj"] = block.attn_res_proj.weight.detach()   # (1, d)
        p["mlp_res_norm"] = block.mlp_res_norm.weight.detach()     # (d,)
        p["mlp_res_proj"] = block.mlp_res_proj.weight.detach()     # (1, d)
        impl = getattr(block.attn, "attn_impl", None)
        if isinstance(impl, torch.nn.Module) and hasattr(impl, "perm"):
            # SLA2-Cube-QAT attention: snapshot routing/mixing/tiling params
            # so the fast JVP path reproduces the training-forward function.
            p["cube"] = {
                "pre_permuted": getattr(net, "tile_major", False),
                "alpha": torch.sigmoid(
                    impl.alpha_logit.detach()).float(),        # (H, Mb)
                "proj_q": impl.proj_q.detach(),                # (hd, hd)
                "proj_k": impl.proj_k.detach(),                # (hd, hd)
                "topk": impl.topk_frac, "Np": impl.Np, "E": impl.E,
                "P": impl.prefix_len,
                "perm": impl.perm, "inv": impl.inv,            # (L,) each
            }
        elif isinstance(impl, torch.nn.Module) and hasattr(impl, "sla2"):
            # SLA2 attention: snapshot the routing/mixing params so the fast
            # JVP path reproduces the training-forward attention function.
            p["sla2"] = {
                "alpha": torch.sigmoid(
                    impl.sla2.alpha_logit.detach()).float(),       # (H, Mb)
                "proj_q": impl.sla2.router.proj_q.weight.detach(), # (hd, hd)
                "proj_k": impl.sla2.router.proj_k.weight.detach(), # (hd, hd)
                "topk": impl.topk, "bq": impl.bq, "bk": impl.bk,
            }
        return p, _mla_fp16_cache(p)

    return {
        "shared": [pack(b) for b in net.shared_blocks],
        "u_heads": [pack(b) for b in net.u_heads],
        "u_out_res": (net.u_out_res_norm.weight.detach(),
                      net.u_out_res_proj.weight.detach()),
        "u_final": (net.u_final_layer.norm.weight.detach(),
                    net.u_final_layer.linear.weight.detach(),
                    net.u_final_layer.linear.bias.detach()),
        "rope": (net.rope_cos, net.rope_sin),
        "eps": net.rmsnorm_eps,
        "bs": net.attn_res_block_size,
        "prefix_tokens": net.prefix_tokens,
    }


@torch.no_grad()
def model_du_dt_fast(net, state, x, t, h, w_cfg, t_min, t_max, y, v_c):
    """du/dt of the u-branch via hand-rolled forward-mode with Triton kernels.

    Tangent seed: d/dt of the inputs (z_t, t, r) along (v_c, 1, 0) --
    x carries tangent v_c, the h = t - r conditioning carries tangent 1
    (dt=1, dr=0), all other conditionings carry zero tangent.

    Args:
        net: IMFDiTVideo (attn_res_block_size > 0).
        state: build_fast_jvp_state(net) output.
        x: (b, C, T, Hh, Ww) noisy latents z_t;  v_c: same shape, x-tangent.
        t, h, w_cfg, t_min, t_max: (b,) conditioning scalars; y: (b,) labels.

    Returns:
        du_dt: (b, C, T, Hh, Ww) fp32, detached.
    """
    eps = state["eps"]
    bs = state["bs"]
    cos, sin = state["rope"]

    # ---- sequence embed + tangent (small tensors: functorch is fine) ----
    seq, dseq = torch.func.jvp(
        lambda xx, hh: net._build_sequence(xx, hh, w_cfg, t_min, t_max, y),
        (x, h), (v_c, torch.ones_like(h)),
    )                                                # (b, l, d) each

    if getattr(net, "tile_major", False):
        # tile-major residency: one gather here replaces per-block gathers
        seq = seq[:, net.tm_perm]
        dseq = dseq[:, net.tm_perm]
    prefix, dprefix = seq, dseq
    snaps, dsnaps = [], []
    idx = 0
    for group in ("shared", "u_heads"):
        for p, w16 in state[group]:
            if snaps:
                h1, dh1 = _attn_res_pair(
                    prefix, dprefix, snaps, dsnaps,
                    p["attn_res_norm"], p["attn_res_proj"], eps,
                )
            else:
                h1, dh1 = prefix, dprefix
            fresh = idx % bs == 0
            if fresh:
                snaps.append(prefix)
                dsnaps.append(dprefix)
            a, da = _attn_sublayer_jvp(h1, dh1, p, w16, cos, sin, eps)
            if fresh:
                prefix, dprefix = a, da
            else:
                prefix, dprefix = prefix + a, dprefix + da
            h2, dh2 = _attn_res_pair(
                prefix, dprefix, snaps, dsnaps,
                p["mlp_res_norm"], p["mlp_res_proj"], eps,
            )
            m, dm = _mlp_sublayer_jvp(h2, dh2, p, w16, eps)
            prefix = prefix + m
            dprefix = dprefix + dm
            idx += 1

    # ---- u output attn-res + FinalLayer jvp + unpatchify (tangent only) ----
    gain, proj_w = state["u_out_res"]
    useq, duseq = _attn_res_pair(prefix, dprefix, snaps, dsnaps,
                                 gain, proj_w, eps)
    pt = state["prefix_tokens"]
    if getattr(net, "tile_major", False):
        # back to model order before the position-dependent slice
        useq = useq[:, net.tm_inv]
        duseq = duseq[:, net.tm_inv]
    ut, dut = useq[:, pt:], duseq[:, pt:]            # (b, n_patch, d)
    nw, lw, lb = state["u_final"]                    # norm gain, W, bias
    # FinalLayer jvp: y = W @ rmsnorm(u) + b -> dy = W @ d(rmsnorm)(du)
    b_, n_, d_ = ut.shape
    dn_ = torch.empty(2 * b_, n_, d_, device=ut.device, dtype=torch.float32)
    _rmsnorm_jvp_strided(ut.contiguous(), dut.contiguous(), nw,
                         dn_[:b_], dn_[b_:], d_, d_, d_, eps, b_ * n_)
    # tangent of the linear: einsum("bnd,od->bno", dnorm_t, W)
    du_seq = torch.einsum("bnd,od->bno", dn_[b_:], lw.float())
    out = net.unpatchify(du_seq)                     # (b, C, T, Hh, Ww)
    # torch.func.jvp closures form reference cycles that pin the whole
    # (b, l, d) dual chain (~5 GiB at 29k tokens) until Python GC runs;
    # collect eagerly so the backward pass has the memory.
    gc.collect()
    return out
