"""Tests for SLA2-Cube-JVP-QAT.

1. JVP kernel vs torch.func.jvp over the fp32 dense reference (int8 primal
   scores -> looser primal tolerance than the fp16 kernel).
2. Training module: forward/backward finite on (b, l, H, hd) inputs, grads
   reach alpha_logit; int8 forward agrees with fp16 forward loosely.

Geometry: grid (4, 8, 8), tile (4, 4, 4) -> E = 64, Np = 4 video tiles,
prefix 18 -> L = 274 (ragged tail), B = 2, H = 2, D = 64.

Run: python tests/test_sla2_cube_qat.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.mla_video_sparse_jvp import mvs_dense_ref, route_tiles
from models.sla2_cube_qat import SLA2CubeQATAttentionImpl, sla2_cube_qat_jvp

torch.manual_seed(0)
dev = "cuda"

B, H, D = 2, 2, 64
Np, E, P = 4, 64, 18
L = Np * E + P
Mb = Np + 1
q, k, v = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
dq, dk, dv = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
alpha = torch.rand(H, Mb, device=dev) * 0.8 + 0.1  # (H, Mb)

lut, T = route_tiles(q, k, 0.5, Np, E, P)          # (B,H,Mb,T), T = 2

o_ref, do_ref = torch.func.jvp(
    lambda a_, b_, c_: mvs_dense_ref(a_, b_, c_, alpha, lut, Np, E, P),
    (q, k, v), (dq, dk, dv))

o, do = sla2_cube_qat_jvp(q, k, v, dq, dk, dv, alpha, 0.5, Np, E, P,
                          lut=lut, T=T)
rel_o = ((o - o_ref).norm() / o_ref.norm()).item()
rel_do = ((do - do_ref).norm() / do_ref.norm()).item()
print(f"qat jvp kernel: o {rel_o:.2e} do {rel_do:.2e}")
assert rel_o < 3e-2 and rel_do < 8e-2, (rel_o, rel_do)

# ---- training module ----
for int8 in (True, False):
    torch.manual_seed(1)
    mod = SLA2CubeQATAttentionImpl(
        head_dim=D, seq_len=L, num_heads=H, grid=(4, 8, 8), tile=(4, 4, 4),
        topk=0.5, alpha_init=0.9, use_int8=int8).to(dev)
    x = torch.randn(B, L, H, D, device=dev, requires_grad=True)  # (b,l,H,hd)
    o_m = mod(x, x * 0.9, x * 1.1)                               # (b,l,H,hd)
    assert torch.isfinite(o_m).all()
    o_m.square().mean().backward()
    ga = mod.alpha_logit.grad
    assert ga is not None and torch.isfinite(ga).all() and ga.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    print(f"module int8={int8}: out rms {o_m.float().pow(2).mean().sqrt():.4f} "
          f"alpha-grad absmax {ga.abs().max():.2e}")
    if int8:
        o_int8 = o_m.detach()
    else:
        rel = ((o_int8 - o_m.detach()).norm() / o_m.detach().norm()).item()
        print(f"int8 vs fp16 forward rel diff {rel:.2e}")
        assert rel < 5e-2
print("PASS")
