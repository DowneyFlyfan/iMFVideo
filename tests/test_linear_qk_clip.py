"""Regression test for linear-attention query/key clipping.

Run: .venv/bin/python tests/test_linear_qk_clip.py
"""

import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.imf_dit_video import MLAAttention, sdpa_math_attention
from models.sla2_cube_qat import SLA2CubeQATAttentionImpl
from train import phi_clip


class Net(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn


torch.manual_seed(0)
device = "cuda"
H, DN, DR, DV = 2, 48, 16, 64
N_COMP, DEN_FLOOR = 28_224, 1e-2
attn = MLAAttention(
    hidden_size=128,
    num_heads=H,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=DN,
    qk_rope_head_dim=DR,
    v_head_dim=DV,
    attn_impl=sdpa_math_attention,
).to(device)
net = Net(attn)

# Full channel ranges, not max-minus-mean.  The largest q+k range is 18.
attn.attn_impl = SimpleNamespace(
    phi_range_q=torch.tensor([10.0, 4.0], device=device),
    phi_range_k=torch.tensor([8.0, 3.0], device=device),
    linear_complement_tokens=N_COMP,
)

before = {
    "q_norm": attn.q_norm.weight.detach().clone(),
    "k_norm": attn.k_norm.weight.detach().clone(),
    "q_b": attn.q_b_proj.weight.detach().clone(),
    "kv_a": attn.kv_a_proj.weight.detach().clone(),
}

n_clipped, range_sum, gamma = phi_clip(net, DEN_FLOOR, world=1)
rho = math.log(N_COMP / (DV * DEN_FLOOR))
expected_gamma = rho / 18.0

assert n_clipped == 1
assert abs(range_sum - 18.0) < 1e-6
assert abs(gamma - expected_gamma) < 1e-6

# Scaling only q_b/kv_a rope rows is insufficient: RMS normalization makes
# their NoPE projection rows scale-invariant.  The full q/k output producers
# must share the same global factor.
torch.testing.assert_close(attn.q_norm.weight, before["q_norm"] * gamma)
torch.testing.assert_close(attn.k_norm.weight, before["k_norm"] * gamma)
q_b = attn.q_b_proj.weight.view(H, DN + DR, -1)
q_b_before = before["q_b"].view(H, DN + DR, -1)
torch.testing.assert_close(q_b[:, :DN], q_b_before[:, :DN])
torch.testing.assert_close(q_b[:, DN:], q_b_before[:, DN:] * gamma)
torch.testing.assert_close(attn.kv_a_proj.weight[:-DR], before["kv_a"][:-DR])
torch.testing.assert_close(attn.kv_a_proj.weight[-DR:], before["kv_a"][-DR:] * gamma)

# The bound is the feature-softmax denominator lower bound used to cap T2.
assert N_COMP * math.exp(-(range_sum * gamma)) / DV >= DEN_FLOOR

# A single arbitrarily negative channel must be visible to the signal.  The
# old max-minus-mean statistic reports 12 here, although feature softmax is
# effectively zero on that channel; full range is 768.
signal = SLA2CubeQATAttentionImpl(
    head_dim=DV, seq_len=274, num_heads=H, grid=(4, 8, 8), tile=(4, 4, 4),
    topk=0.5, alpha_init=0.9, use_int8=False,
).to(device)
q_signal = torch.zeros(1, 274, H, DV, device=device)
k_signal = torch.zeros_like(q_signal)
q_signal[..., -1] = -768.0
k_signal[..., -1] = -768.0
signal(q_signal.requires_grad_(), k_signal, torch.ones_like(q_signal))
torch.testing.assert_close(signal.phi_range_q, torch.full((H,), 768.0, device=device))
torch.testing.assert_close(signal.phi_range_k, torch.full((H,), 768.0, device=device))
assert signal.linear_complement_tokens == 128
print(f"PASS gamma={gamma:.6f} range_sum={range_sum:.6f} rho={rho:.6f}")
