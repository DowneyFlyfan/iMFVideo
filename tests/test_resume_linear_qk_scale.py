"""Regression: a resumed checkpoint can be Q/K-preconditioned pre-forward."""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.imf_dit_video import MLAAttention
from train import (
    needs_linear_qk_preconditioner,
    rescale_linear_qk_producers,
    rescale_resumed_linear_qk,
)


def test_preconditioner_is_not_reapplied_to_a_saved_preconditioned_checkpoint():
    assert needs_linear_qk_preconditioner({}, 0.3)
    assert not needs_linear_qk_preconditioner(
        {"linear_qk_preconditioned": True}, 0.3
    )
    assert not needs_linear_qk_preconditioner({}, 1.0)


def test_resume_scale_updates_online_ema_and_optimizer_momentum():
    """All states must stay in the same Q/K parameterisation after resume."""
    online = MLAAttention(
        hidden_size=64,
        num_heads=1,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=48,
        qk_rope_head_dim=16,
        v_head_dim=64,
        attn_impl=None,
    )
    with torch.no_grad():
        online.q_norm.weight.fill_(1.0)
        online.k_norm.weight.fill_(1.0)
        online.q_b_proj.weight.fill_(1.0)
        online.kv_a_proj.weight.fill_(1.0)
    ema = copy.deepcopy(online)
    optimizer = torch.optim.SGD(online.parameters(), lr=0.1, momentum=0.9)
    for parameter in online.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    rescale_resumed_linear_qk(online, ema, optimizer, 0.3)

    assert torch.allclose(online.q_norm.weight, torch.full_like(online.q_norm.weight, 0.27))
    assert torch.allclose(ema.q_norm.weight, online.q_norm.weight)
    q_momentum = optimizer.state[online.q_b_proj.weight]["momentum_buffer"]
    kv_momentum = optimizer.state[online.kv_a_proj.weight]["momentum_buffer"]
    assert torch.allclose(q_momentum[:48], torch.ones_like(q_momentum[:48]))
    assert torch.allclose(q_momentum[48:], torch.full_like(q_momentum[48:], 0.3))
    assert torch.allclose(kv_momentum[:-16], torch.ones_like(kv_momentum[:-16]))
    assert torch.allclose(kv_momentum[-16:], torch.full_like(kv_momentum[-16:], 0.3))


def main():
    attn = MLAAttention(
        hidden_size=64,
        num_heads=1,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=48,
        qk_rope_head_dim=16,
        v_head_dim=64,
        attn_impl=None,
    )
    with torch.no_grad():
        attn.q_norm.weight.fill_(1.0)
        attn.k_norm.weight.fill_(1.0)
        attn.q_b_proj.weight.fill_(1.0)
        attn.kv_a_proj.weight.fill_(1.0)

    scaled = rescale_linear_qk_producers(attn, 0.3)
    assert scaled == 1
    assert torch.allclose(attn.q_norm.weight, torch.full_like(attn.q_norm.weight, 0.3))
    assert torch.allclose(attn.k_norm.weight, torch.full_like(attn.k_norm.weight, 0.3))
    assert torch.allclose(attn.q_b_proj.weight[:48], torch.ones_like(attn.q_b_proj.weight[:48]))
    assert torch.allclose(attn.q_b_proj.weight[48:], torch.full_like(attn.q_b_proj.weight[48:], 0.3))
    assert torch.allclose(attn.kv_a_proj.weight[:-16], torch.ones_like(attn.kv_a_proj.weight[:-16]))
    assert torch.allclose(attn.kv_a_proj.weight[-16:], torch.full_like(attn.kv_a_proj.weight[-16:], 0.3))
    print("PASS")


if __name__ == "__main__":
    main()
