"""Correctness test: sla2_mla_jvp vs torch.func.jvp over the dense reference.

Shapes: B=2 batch, H=2 heads, L tokens (one non-divisible case), D=64 head dim.
Router LUT computed once on the primal and shared by both paths, so the only
difference is kernel/states math vs dense autograd-jvp math.
"""
import sys, torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sla2_mla_jvp import route_blocks, sla2_mla_jvp, sla2_dense_ref

torch.manual_seed(0)
dev = "cuda"

for L in (512, 640, 583):          # divisible, divisible, ragged tail
    B, H, D, bq, bk = 2, 2, 64, 128, 64
    q, k, v = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
    dq, dk, dv = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
    Mb = (L + bq - 1) // bq
    alpha = torch.rand(H, Mb, device=dev) * 0.8 + 0.1        # (H, Mb)

    lut, T = route_blocks(q, k, 0.25, bq, bk)                # (B,H,Mb,T)

    # reference primal + tangent via functorch over the dense implementation
    o_ref, do_ref = torch.func.jvp(
        lambda a, b, c: sla2_dense_ref(a, b, c, lut, alpha, bq, bk),
        (q, k, v), (dq, dk, dv))

    # fp32 path (tf32 dots) and fp16 path
    o32, do32 = sla2_mla_jvp(q, k, v, dq, dk, dv, alpha, 0.25, bq, bk,
                             lut=lut, T=T)
    o16, do16 = sla2_mla_jvp(q.half(), k.half(), v.half(),
                             dq.half(), dk.half(), dv.half(),
                             alpha, 0.25, bq, bk, lut=lut, T=T)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    print(f"L={L}: fp32 o {rel(o32, o_ref):.2e} do {rel(do32, do_ref):.2e} | "
          f"fp16 o {rel(o16, o_ref):.2e} do {rel(do16, do_ref):.2e}")
    assert rel(o32, o_ref) < 2e-4 and rel(do32, do_ref) < 2e-3
    assert rel(o16, o_ref) < 2e-2 and rel(do16, do_ref) < 5e-2
print("PASS")
