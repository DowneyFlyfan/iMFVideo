"""Regression tests for models/mla_jvp_fast.py at small model geometry.

hidden 128 / kv_lora_rank 32 (the local-tuning geometry) makes the latent
up-projection GEMMs run at K = 32 (kv_b) and K = 64 (q_b) -- smaller than
every _mm16 K-tile (BK >= 64). Without K masking the tile dot folded the
next row's data into each product and read uninitialized memory past the
last row of the latent buffer (possibly NaN fp16 bit patterns): du/dt was
wrong at best and one full 224-wide kv_b row of NaN at worst.

A surgically degenerate kv-latent row additionally covers the RMSNorm-JVP
tangent clamp: dy = w*(dx/r - x*mean(x*dx)/r^3) with r = sqrt(mean(x^2)+eps)
amplifies the tangent by up to 1/sqrt(eps) (1000x at eps=1e-6) on a row
whose primal norm is near zero, so a large-but-healthy tangent used to
overflow the fp16 store (inf) and NaN every downstream stage.

Reference = torch.func.jvp through the real net (fp32, sdpa math attention).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.imf_dit_video import IMFDiTVideo, sdpa_math_attention
from models.mla_jvp_fast import build_fast_jvp_state, model_du_dt_fast

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast JVP path needs CUDA/Triton"
)

B = 2                        # batch size of the NaN repro


def make_net(perturb=0.02):
    """Small-geometry net from the NaN repro: hidden 128, kv_lora_rank 32.

    Gates and final layers are zero-init, which makes du/dt identically
    zero; a small parameter perturbation makes the tests nontrivial.
    """
    torch.manual_seed(0)
    net = IMFDiTVideo(
        input_size=(16, 16), num_frames=4, patch_size=(1, 2, 2),
        in_channels=16, hidden_size=128, depth=3, aux_head_depth=1,
        num_heads=2, num_classes=10, kv_lora_rank=32,
        attn_res_block_size=2, grad_checkpoint=False,
        attn_impl=sdpa_math_attention,
    ).cuda()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(perturb * torch.randn_like(p))
    return net


def make_inputs(tangent_scale=1.0):
    """Repro conditioning: t=0.5, h=0.3, omega=2, t_min=0, t_max=1."""
    torch.manual_seed(1)
    x = torch.randn(B, 16, 4, 16, 16, device="cuda")   # (b, C, T, Hh, Ww)
    v_c = tangent_scale * torch.randn_like(x)          # x-tangent
    t = torch.full((B,), 0.5, device="cuda")
    h = torch.full((B,), 0.3, device="cuda")
    w = torch.full((B,), 2.0, device="cuda")
    t_min = torch.zeros(B, device="cuda")
    t_max = torch.ones(B, device="cuda")
    y = torch.randint(0, 10, (B,), device="cuda")
    return x, v_c, t, h, w, t_min, t_max, y


def fast_du_dt(net, x, v_c, t, h, w, t_min, t_max, y):
    state = build_fast_jvp_state(net)
    return model_du_dt_fast(net, state, x, t, h, w, t_min, t_max, y, v_c)


def test_small_geometry_matches_reference():
    """K=32/64 latent GEMMs: du/dt finite and close to the fp32 reference."""
    net = make_net()
    x, v_c, t, h, w, t_min, t_max, y = make_inputs()
    du = fast_du_dt(net, x, v_c, t, h, w, t_min, t_max, y)
    assert du.shape == x.shape
    assert torch.isfinite(du).all(), (
        f"{(~torch.isfinite(du)).sum().item()} non-finite du/dt values"
    )

    ref = torch.func.jvp(
        lambda z, hh: net(z, t, hh, w, t_min, t_max, y)[0],
        (x, h), (v_c, torch.ones_like(h)),
    )[1]
    rel = ((du.double() - ref.double()).norm() / ref.double().norm()).item()
    # fp16 pipeline over 3 blocks + attn-res + final layer; a single block
    # sits at ~3e-3 (test_triton_mla_block_jvp), compounding stays well
    # under 3e-2. The pre-fix K-tile bug gave O(1) error / NaN here.
    assert rel < 3e-2, f"fast du/dt off the fp32 reference: rel err {rel:.3e}"


def test_degenerate_kv_latent_row_stays_finite():
    """A near-zero-norm kv-latent row + large tangent must not NaN du/dt.

    kv_a_proj gets a rank-1 correction that annihilates the norm1 primal of
    the first patch token, so that row's kv latent has mean-square
    ~ fp16-rounding^2 (~1e-8, well under eps) while its tangent (scaled
    v_c) stays large: the unclamped RMSNorm-JVP tangent, ~2.6e5 here,
    overflowed the fp16 store and turned everything downstream NaN.
    """
    net = make_net()
    x, v_c, t, h, w, t_min, t_max, y = make_inputs(tangent_scale=300.0)
    blk = net.shared_blocks[0]
    dc = blk.attn.kv_lora_rank
    pt = net.prefix_tokens
    with torch.no_grad():
        seq = net._build_sequence(x, h, w, t_min, t_max, y)   # (b, l, d)
        n1 = blk.norm1(seq)
        row = n1[0, pt].float()
        nhat = row / row.norm()                               # (d,)
        W = blk.attn.kv_a_proj.weight                         # (dc+dr, d)
        W[:dc] -= (W[:dc] @ nhat)[:, None] * nhat[None, :]
        # the construction really is degenerate: primal mean-sq << eps
        ms = blk.attn.kv_a_proj(n1)[0, pt, :dc].pow(2).mean()
        assert ms < 1e-5, f"degenerate row not degenerate: mean-sq {ms:.2e}"

    du = fast_du_dt(net, x, v_c, t, h, w, t_min, t_max, y)
    assert torch.isfinite(du).all(), (
        f"{(~torch.isfinite(du)).sum().item()} non-finite du/dt values "
        "from the degenerate kv-latent row"
    )
