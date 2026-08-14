"""Correctness test: mla_video_sparse_jvp vs torch.func.jvp over the dense
reference, plus the tile permutation round trip.

Geometry: grid (4, 8, 8) = 256 patch tokens, tile (2, 4, 4) -> E = 32
tokens per tile, Np = 8 video tiles, prefix 18 -> L = 274 (ragged tail),
B = 2, H = 2, D = 64.  Routing LUT computed once from the coarse scores of
the primal and shared by both paths.

Run: python tests/test_mla_video_sparse_jvp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.mla_video_sparse_jvp import (
    mla_video_sparse_jvp,
    mvs_dense_ref,
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
gate = torch.rand(B, H, L, 1, device=dev)        # (B,H,L,1) coarse gate
dgate = torch.randn(B, H, L, 1, device=dev) * 0.1
alpha = torch.rand(H, Mb, device=dev) * 0.8 + 0.1  # (H, Mb)

# routing LUT from the primal coarse scores (fixed for both paths)
_, _ = None, None
o_probe, _ = mla_video_sparse_jvp(q, k, v, dq, dk, dv, gate, dgate, alpha,
                                  0.25, Np, E, P, block_m=32)
# recompute lut exactly as the op does, to pin it for the reference
from models.mla_video_sparse_jvp import _pool_pair, _coarse_jvp
qc, dqc = _pool_pair(q, dq, Np, E, P)
kc, dkc = _pool_pair(k, dk, Np, E, P)
vc, dvc = _pool_pair(v, dv, Np, E, P)
_, _, s = _coarse_jvp(qc, dqc, kc, dkc, vc, dvc, D ** -0.5)
T = max(1, int(0.25 * Np))
lut = torch.topk(s[..., :Np], T, dim=-1,
                 sorted=False).indices.to(torch.int32).contiguous()

# reference primal + tangent: functorch over the dense implementation,
# with gate carrying its own tangent
o_ref, do_ref = torch.func.jvp(
    lambda a_, b_, c_, g_: mvs_dense_ref(a_, b_, c_, g_, alpha, lut,
                                         Np, E, P),
    (q, k, v, gate), (dq, dk, dv, dgate))

for dtype, tol_o, tol_do in ((torch.float32, 2e-4, 2e-3),
                             (torch.float16, 2e-2, 5e-2)):
    cast = lambda t: t.to(dtype)
    o, do = mla_video_sparse_jvp(
        cast(q), cast(k), cast(v), cast(dq), cast(dk), cast(dv),
        gate, dgate, alpha, 0.25, Np, E, P, lut=lut, T=T, block_m=32)
    rel_o = ((o - o_ref).norm() / o_ref.norm()).item()
    rel_do = ((do - do_ref).norm() / do_ref.norm()).item()
    print(f"{str(dtype):14s}: o {rel_o:.2e} do {rel_do:.2e}")
    assert rel_o < tol_o and rel_do < tol_do, (dtype, rel_o, rel_do)
print("PASS")
