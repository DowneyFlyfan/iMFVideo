"""Correctness test: mla_video_sparse_jvp vs torch.func.jvp over the dense
reference, plus the tile permutation round trip.

Geometry: grid (4, 8, 8) = 256 patch tokens, tile (2, 4, 4) -> E = 32
tokens per tile, Np = 8 video tiles, prefix 18 -> L = 274 (ragged tail),
B = 2, H = 2, D = 64.  Routing LUT from the SLA2-style tile router (with
non-identity projections) computed once on the primal and shared by both
paths.

Run: python tests/test_mla_video_sparse_jvp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.mla_video_sparse_jvp import (
    mla_video_sparse_jvp,
    mvs_dense_ref,
    route_tiles,
    tile_permutation,
)

torch.manual_seed(0)
dev = "cuda"

# ---- tile permutation round trip ----
perm, inv, n_tiles = tile_permutation((4, 8, 8), (2, 4, 4), 18, dev)
x = torch.randn(274, 5, device=dev)              # (L, feat) model order
assert torch.equal(x[perm][inv], x) and n_tiles == 8
print("tile permutation round trip ok, n_tiles", n_tiles)

B, H, D = 2, 2, 64
Np, E, P = 8, 32, 18
L = Np * E + P
Mb = Np + 1
q, k, v = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
dq, dk, dv = (torch.randn(B, H, L, D, device=dev) for _ in range(3))
alpha = torch.rand(H, Mb, device=dev) * 0.8 + 0.1  # (H, Mb)
# router projections: identity + noise, to exercise the learnable path
proj_q = torch.eye(D, device=dev) + 0.05 * torch.randn(D, D, device=dev)
proj_k = torch.eye(D, device=dev) + 0.05 * torch.randn(D, D, device=dev)

lut, T = route_tiles(q, k, 0.25, Np, E, P, proj_q=proj_q, proj_k=proj_k)
print("lut", tuple(lut.shape), "T", T)

# reference primal + tangent: functorch over the dense implementation
o_ref, do_ref = torch.func.jvp(
    lambda a_, b_, c_: mvs_dense_ref(a_, b_, c_, alpha, lut, Np, E, P),
    (q, k, v), (dq, dk, dv))

for dtype, tol_o, tol_do in ((torch.float32, 2e-4, 2e-3),
                             (torch.float16, 2e-2, 5e-2)):
    cast = lambda t: t.to(dtype)
    o, do = mla_video_sparse_jvp(
        cast(q), cast(k), cast(v), cast(dq), cast(dk), cast(dv),
        alpha, 0.25, Np, E, P, lut=lut, T=T, block_m=32)
    rel_o = ((o - o_ref).norm() / o_ref.norm()).item()
    rel_do = ((do - do_ref).norm() / do_ref.norm()).item()
    print(f"{str(dtype):14s}: o {rel_o:.2e} do {rel_do:.2e}")
    assert rel_o < tol_o and rel_do < tol_do, (dtype, rel_o, rel_do)
print("PASS")
