"""CPU regression tests for the bounded linear-attention feature map.

Run: .venv/bin/python tests/test_linear_t2_stability.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.sla2_cube_qat import (
    SLA2CubeQATAttentionImpl,
    _linear_feature_map_jvp,
    sla2_cube_qat_jvp_fused,
)


torch.manual_seed(0)
D, EPSILON, N_COMPLEMENT = 64, 2.5e-4, 26_464
x = torch.randn(2, 3, 5, D, dtype=torch.float64)
dx = torch.randn_like(x)


def feature_map(x_):
    return _linear_feature_map_jvp(x_, None, EPSILON)[0]


phi, dphi = _linear_feature_map_jvp(x, dx, EPSILON)
_, dphi_ref = torch.func.jvp(feature_map, (x,), (dx,))
torch.testing.assert_close(dphi, dphi_ref, atol=2e-12, rtol=2e-12)
torch.testing.assert_close(phi.sum(-1), torch.ones_like(phi[..., 0]))
assert phi.amin().item() >= EPSILON / D - 1e-14

# Adversarial channel collapse: every ordinary softmax key is exactly one-hot
# in fp64.  The smoothed map keeps every complement denominator above the
# analytically required N * epsilon / D bound.
q = torch.zeros(1, 1, 1, D, dtype=torch.float64)
k = torch.full((1, 1, N_COMPLEMENT, D), -1_000.0, dtype=torch.float64)
k[..., 0] = 0.0
dq = torch.randn_like(q)
dk = torch.randn_like(k)
v = torch.randn(1, 1, N_COMPLEMENT, D, dtype=torch.float64)
dv = torch.randn_like(v)
qphi, dqphi = _linear_feature_map_jvp(q, dq, EPSILON)
kphi, dkphi = _linear_feature_map_jvp(k, dk, EPSILON)
h = kphi.transpose(-1, -2) @ v
dh = dkphi.transpose(-1, -2) @ v + kphi.transpose(-1, -2) @ dv
z, dz = kphi.sum(-2), dkphi.sum(-2)
den = (qphi * z.unsqueeze(-2)).sum(-1)
num = qphi @ h
dnum = dqphi @ h + qphi @ dh
dden = (dqphi * z.unsqueeze(-2)).sum(-1) + (qphi * dz.unsqueeze(-2)).sum(-1)
o = num / den.unsqueeze(-1)
t2 = -o * dden.unsqueeze(-1) / den.unsqueeze(-1)

bound = N_COMPLEMENT * EPSILON / D
assert den.amin().item() >= bound * (1.0 - 1e-12)
assert torch.isfinite(t2).all()
print(f"PASS den_min={den.amin().item():.6e} bound={bound:.6e} "
      f"t2_absmax={t2.abs().amax().item():.6e}")

if torch.cuda.is_available():
    # CUDA integration: exercises the custom autograd backward and the fused
    # JVP implementation with a nonzero smoothing mass.
    device = "cuda"
    B, H, Np, E, P = 1, 1, 4, 64, 18
    L = Np * E + P
    module = SLA2CubeQATAttentionImpl(
        head_dim=D, seq_len=L, num_heads=H, grid=(4, 8, 8), tile=(4, 4, 4),
        topk=0.5, alpha_init=0.5, use_int8=False,
        linear_den_floor=1e-2,
    ).to(device)
    q = torch.randn(B, L, H, D, device=device, requires_grad=True)
    output = module(q, q.detach() * 0.7, q.detach() * 1.1)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(q.grad).all()

    qh, kh, vh = (torch.randn(B, H, L, D, device=device) for _ in range(3))
    dqh, dkh, dvh = (torch.randn_like(qh) for _ in range(3))
    alpha = torch.sigmoid(module.alpha_logit.detach())
    _, tangent = sla2_cube_qat_jvp_fused(
        qh, kh, vh, dqh, dkh, dvh, alpha, 0.5, Np, E, P,
        phi_epsilon=module.linear_phi_epsilon,
    )
    assert torch.isfinite(tangent).all()
    print(f"CUDA PASS epsilon={module.linear_phi_epsilon:.6e} "
          f"grad_absmax={q.grad.abs().amax().item():.6e}")
