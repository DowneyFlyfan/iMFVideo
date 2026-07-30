"""Correctness + speed tests for models/triton_mla_block_jvp.py.

Reference = torch.func.jvp through the actual imf_dit_video.TransformerBlock
(MLA attention, sdpa math backend) in fp64 / fp32.

Shape symbols:
    b : batch;  l : sequence length;  d : hidden size (1024)
    H : heads (16);  dq : q_lora_rank (512);  dc : kv_lora_rank (256)
    dn : qk_nope_head_dim (48);  dr : qk_rope_head_dim (16);  dv : 64
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.imf_dit_video import TransformerBlock, sdpa_math_attention
from models.triton_mla_block_jvp import (
    TritonMLABlockJVP,
    mla_params_from_block,
    triton_mla_block_jvp,
)

torch.manual_seed(0)

D, H = 1024, 16          # hidden size d, heads H
DQ, DC, DN, DR, DV = 512, 256, 48, 16, 64


def make_block(device="cuda", dtype=torch.float32):
    """Build a model-config MLA TransformerBlock with random (nonzero) gates.

    Returns:
        block: TransformerBlock on device/dtype; attn_scale/mlp_scale are
        randomized (model zero-init would make the test trivial).
    """
    torch.manual_seed(0)
    block = TransformerBlock(
        hidden_size=D, num_heads=H, q_lora_rank=DQ, kv_lora_rank=DC,
        qk_nope_head_dim=DN, qk_rope_head_dim=DR, v_head_dim=DV,
        mlp_ratio=8 / 3, weight_init_constant=0.32, norm_eps=1e-6,
        attn_impl=sdpa_math_attention,
    )
    with torch.no_grad():
        block.attn_scale.normal_(0.0, 0.5)   # (d,) attention residual gate
        block.mlp_scale.normal_(0.0, 0.5)    # (d,) MLP residual gate
    return block.to(device=device, dtype=dtype)


def make_rope(l, device="cuda", dtype=torch.float32):
    """Random rotary tables. Returns (cos, sin): each (l, dr // 2)."""
    torch.manual_seed(1)
    ang = torch.rand(l, DR // 2, device=device) * (2 * math.pi)  # (l, dr/2)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def eager_jvp(block, x, dx, cos, sin):
    """(y, dy) via torch.func.jvp through the eager block.

    x, dx: (b, l, d);  cos/sin: (l, dr//2). Returns (b, l, d) each.
    """
    return torch.func.jvp(lambda t: block(t, cos, sin), (x,), (dx,))


def rel_err(a, b):
    """Relative Frobenius error of a (any shape) w.r.t. reference b."""
    return (a.double() - b.double()).norm().item() / b.double().norm().item()


@pytest.fixture(scope="module")
def block32():
    return make_block(dtype=torch.float32)


@pytest.mark.parametrize("b,l", [(2, 512), (1, 210), (3, 384)])
def test_matches_fp64_reference(block32, b, l):
    """Triton MLA block JVP vs fp64 eager reference (primal and tangent)."""
    torch.manual_seed(2)
    x = torch.randn(b, l, D, device="cuda")            # (b, l, d) primal
    dx = torch.randn(b, l, D, device="cuda")           # (b, l, d) tangent
    cos, sin = make_rope(l)

    block64 = make_block(dtype=torch.float64)
    y64, dy64 = eager_jvp(block64, x.double(), dx.double(),
                          cos.double(), sin.double())

    # sanity: fp32 eager itself sits at ~1e-6 of fp64
    y32, dy32 = eager_jvp(block32, x, dx, cos, sin)
    assert rel_err(y32, y64) < 1e-4
    assert rel_err(dy32, dy64) < 1e-4

    params = mla_params_from_block(block32)
    y, dy = triton_mla_block_jvp(x.contiguous(), dx.contiguous(), params,
                                 cos, sin)
    assert y.shape == (b, l, D) and dy.shape == (b, l, D)
    assert torch.isfinite(y).all() and torch.isfinite(dy).all()
    # fp8 activations + fp16-accumulate attention: ~1e-2 rel Fro class
    assert rel_err(y, y64) < 4e-2, f"primal rel err {rel_err(y, y64):.3e}"
    assert rel_err(dy, dy64) < 4e-2, f"tangent rel err {rel_err(dy, dy64):.3e}"


def test_module_wrapper_matches_functional(block32):
    """TritonMLABlockJVP (cached fp8 weights) == functional call."""
    torch.manual_seed(3)
    b, l = 2, 512
    x = torch.randn(b, l, D, device="cuda")            # (b, l, d)
    dx = torch.randn(b, l, D, device="cuda")           # (b, l, d)
    cos, sin = make_rope(l)
    params = mla_params_from_block(block32)
    y_f, dy_f = triton_mla_block_jvp(x, dx, params, cos, sin)
    mod = TritonMLABlockJVP(block32)
    y_m, dy_m = mod(x, dx, cos, sin)
    assert torch.equal(y_f, y_m) and torch.equal(dy_f, dy_m)


def _bench(fn, iters=20, warmup=5):
    """Median CUDA-event milliseconds of fn() over `iters` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def test_speedup_at_least_10x(block32):
    """>= 10x vs pure-PyTorch eager fp32 jvp at (2, 2048, 1024).

    (2, 4096) is infeasible for the eager baseline on 16 GB: sdpa math
    materializes four (b, H, l, l) fp32 score/prob buffers under jvp.
    Attention is quadratic in l, so 2048 is the conservative shape for
    the speedup claim.
    """
    b, l = 2, 2048
    torch.manual_seed(4)
    x = torch.randn(b, l, D, device="cuda")            # (b, l, d)
    dx = torch.randn(b, l, D, device="cuda")           # (b, l, d)
    cos, sin = make_rope(l)
    mod = TritonMLABlockJVP(block32)

    t_triton = _bench(lambda: mod(x, dx, cos, sin))
    t_eager = _bench(lambda: eager_jvp(block32, x, dx, cos, sin), iters=10)
    speedup = t_eager / t_triton
    print(f"\n(2, {l}, {D}): eager fp32 {t_eager:.2f} ms, "
          f"triton {t_triton:.3f} ms, speedup {speedup:.2f}x")
    assert speedup >= 10.0, f"only {speedup:.2f}x"


if __name__ == "__main__":
    block = make_block(dtype=torch.float32)
    mod = TritonMLABlockJVP(block)
    print(f"MLA block d={D} H={H} dq={DQ} dc={DC} dn={DN} dr={DR} dv={DV}")
    for b, l in [(2, 2048), (2, 4096), (8, 210)]:
        torch.manual_seed(5)
        x = torch.randn(b, l, D, device="cuda")
        dx = torch.randn(b, l, D, device="cuda")
        cos, sin = make_rope(l)
        t_tr = _bench(lambda: mod(x, dx, cos, sin), iters=30)
        try:
            t_eg = _bench(lambda: eager_jvp(block, x, dx, cos, sin), iters=10)
            note = f"eager fp32 {t_eg:8.2f} ms  speedup {t_eg / t_tr:6.2f}x"
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            note = "eager fp32 OOM"
        print(f"({b}, {l:4d}, {D}): triton {t_tr:7.3f} ms   {note}")
