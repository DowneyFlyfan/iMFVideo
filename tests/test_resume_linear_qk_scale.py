"""Regression: a resumed checkpoint can be Q/K-preconditioned pre-forward."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.imf_dit_video import MLAAttention
from train import rescale_linear_qk_producers


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
