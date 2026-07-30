"""Tests for the Kimi-K3 attention residual: eager model integration and
the fused Triton forward+JVP kernel.

Shape symbols:
    n : token rows;  J : candidates (snapshots + 1);  d : hidden size
    b : batch;  l : sequence length
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.imf_dit_video import attn_res_apply
from models.triton_attn_res_jvp import triton_attn_res_fwd, triton_attn_res_jvp

torch.manual_seed(0)

D = 1024
EPS = 1e-6


def eager_apply(v, w, eps=EPS):
    """Reference: identical math to imf_dit_video.attn_res_apply on a
    pre-stacked candidate tensor.

    Args:
        v: (n, J, d) candidate stack.  w: (d,) fused score direction.

    Returns:
        (n, d) mixed stream.
    """
    vf = v.float()
    var = vf.pow(2).mean(-1, keepdim=True)             # (n, J, 1)
    k = vf * torch.rsqrt(var + eps)                    # (n, J, d)
    scores = torch.einsum("njd,d->nj", k, w.float())   # (n, J)
    probs = scores.softmax(-1)                         # (n, J)
    return torch.einsum("nj,njd->nd", probs, vf).to(v.dtype)   # (n, d)


def rel_err(a, b):
    """Relative Frobenius error of a w.r.t. reference b."""
    return (a.double() - b.double()).norm().item() / b.double().norm().item()


@pytest.mark.parametrize("n,J", [(512, 6), (333, 2), (256, 1), (128, 20)])
def test_matches_fp64_reference(n, J):
    """Triton kernel vs fp64 eager jvp at several (n, J)."""
    torch.manual_seed(1)
    v = torch.randn(n, J, D, device="cuda")            # (n, J, d) primal
    dv = torch.randn(n, J, D, device="cuda")           # (n, J, d) tangent
    w = torch.randn(D, device="cuda")                  # (d,) score direction

    y64, dy64 = torch.func.jvp(
        lambda t: eager_apply(t, w.double()), (v.double(),), (dv.double(),)
    )
    y, dy = triton_attn_res_jvp(v, dv, w)
    assert y.shape == (n, D) and dy.shape == (n, D)
    assert rel_err(y, y64) < 1e-5, f"primal rel err {rel_err(y, y64):.2e}"
    assert rel_err(dy, dy64) < 1e-5, f"tangent rel err {rel_err(dy, dy64):.2e}"

    y_f = triton_attn_res_fwd(v, w)
    assert torch.equal(y_f, y)


def test_matches_model_helper():
    """Kernel == imf_dit_video.attn_res_apply (list-of-snapshots API)."""
    torch.manual_seed(2)
    b, l, J = 2, 210, 4
    snaps = [torch.randn(b, l, D, device="cuda") for _ in range(J - 1)]
    prefix = torch.randn(b, l, D, device="cuda")       # (b, l, d)
    gain = torch.rand(D, device="cuda") + 0.5          # (d,) norm gain
    proj = torch.randn(1, D, device="cuda")            # (1, d) projection
    ref = attn_res_apply(prefix, snaps, gain, proj, EPS)   # (b, l, d)
    v = torch.stack(snaps + [prefix], dim=2).reshape(b * l, J, D).contiguous()
    y = triton_attn_res_fwd(v, gain * proj.reshape(-1), EPS)
    assert rel_err(y.view(b, l, D), ref) < 1e-6


def _bench(fn, iters=30, warmup=5):
    """Median CUDA-event milliseconds of fn()."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def test_speedup_vs_eager_jvp():
    """Fused kernel vs eager fp32 torch.func.jvp of the op, model-scale
    (n = 2*4096 tokens, J = 6, d = 1024)."""
    n, J = 8192, 6
    torch.manual_seed(3)
    v = torch.randn(n, J, D, device="cuda")            # (n, J, d)
    dv = torch.randn(n, J, D, device="cuda")           # (n, J, d)
    w = torch.randn(D, device="cuda")                  # (d,)
    t_eager = _bench(
        lambda: torch.func.jvp(lambda t: eager_apply(t, w), (v,), (dv,))
    )
    t_triton = _bench(lambda: triton_attn_res_jvp(v, dv, w))
    speedup = t_eager / t_triton
    print(f"\nattn-res jvp (n={n}, J={J}, d={D}): eager {t_eager:.3f} ms, "
          f"triton {t_triton:.3f} ms, speedup {speedup:.2f}x")
    assert speedup >= 3.0, f"only {speedup:.2f}x"


if __name__ == "__main__":
    for n, J in [(8192, 6), (8192, 3), (16384, 6), (1680, 6)]:
        torch.manual_seed(3)
        v = torch.randn(n, J, D, device="cuda")
        dv = torch.randn(n, J, D, device="cuda")
        w = torch.randn(D, device="cuda")
        t_eg = _bench(
            lambda: torch.func.jvp(lambda t: eager_apply(t, w), (v, ), (dv,))
        )
        t_tr = _bench(lambda: triton_attn_res_jvp(v, dv, w))
        t_fw = _bench(lambda: triton_attn_res_fwd(v, w))
        print(f"(n={n:5d}, J={J}, d={D}): eager jvp {t_eg:7.3f} ms  "
              f"triton jvp {t_tr:6.3f} ms ({t_eg/t_tr:5.2f}x)  "
              f"triton fwd {t_fw:6.3f} ms")
