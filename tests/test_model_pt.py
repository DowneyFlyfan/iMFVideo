"""Smoke test for the PyTorch video imfDiT model and improved MeanFlow loss.

Covers the MLA (multi-head latent attention) attention path with decoupled
("partial") rotary position embedding, the torch.func.jvp forward-mode pass,
the loss backward, and the sampler. Runs on CUDA, MPS (Apple Metal) or CPU.

Run: python tests/test_model_pt.py [cuda|mps|cpu]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from imf_video import IMFVideoLoss
from models.imf_dit_video import (
    eager_math_attention,
    imf_dit_video_S,
    sdpa_math_attention,
)


def pick_device():
    """Honour an explicit argv override, else prefer CUDA, then MPS, then CPU."""
    if len(sys.argv) > 1:
        return torch.device(sys.argv[1])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_attention_kernels():
    """eager_math_attention must agree with the SDPA math backend on CPU.

    Shape symbols: b batch, l seq len, H heads, dk q/k head dim, dv v head dim.
    """
    torch.manual_seed(0)
    b, l, H, dk, dv = 2, 7, 4, 64, 48
    q = torch.randn(b, l, H, dk, dtype=torch.float64)  # (b, l, H, dk)
    k = torch.randn(b, l, H, dk, dtype=torch.float64)  # (b, l, H, dk)
    v = torch.randn(b, l, H, dv, dtype=torch.float64)  # (b, l, H, dv)
    diff = (eager_math_attention(q, k, v) - sdpa_math_attention(q, k, v)).abs().max()
    assert diff < 1e-5, f"eager vs sdpa mismatch: {diff}"
    print(f"attention kernels agree (max abs diff {diff.item():.2e}), dv != dk OK")


def check_mla_geometry(net):
    """The rotary band must be a single shared key head and an even width."""
    attn = net.shared_blocks[0].attn
    assert attn.qk_head_dim == attn.qk_nope_head_dim + attn.qk_rope_head_dim
    assert attn.qk_rope_head_dim % 2 == 0 and attn.qk_nope_head_dim % 2 == 0
    # kv_a_proj emits the dc latent plus exactly ONE dr-wide rotary key head.
    assert attn.kv_a_proj.weight.shape[0] == attn.kv_lora_rank + attn.qk_rope_head_dim
    # The rope table covers only the rotary band, not the whole head vector.
    assert net.rope_cos.shape[-1] == attn.qk_rope_head_dim // 2
    mla = sum(p.numel() for p in attn.parameters())
    mha = 4 * attn.hidden_size * attn.hidden_size
    assert mla < mha, f"MLA ({mla}) should be cheaper than full-rank MHA ({mha})"
    print(f"MLA geometry OK: dq={attn.q_lora_rank} dc={attn.kv_lora_rank} "
          f"dn={attn.qk_nope_head_dim} dr={attn.qk_rope_head_dim} "
          f"dv={attn.v_head_dim}; attn params {mla/mha:.3f}x MHA")


def main():
    device = pick_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    check_attention_kernels()

    num_classes = 10
    batch, channels, frames, height, width = 2, 16, 3, 16, 16

    net = imf_dit_video_S(num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"model params: {num_params / 1e6:.2f}M")

    check_mla_geometry(net)

    # latents: (B, C, T, H, W) video latents;  labels: (B,) class ids
    latents = torch.randn(batch, channels, frames, height, width, device=device)
    labels = torch.randint(0, num_classes, (batch,), device=device)

    # --- plain forward shape check ---
    t = torch.rand(batch, device=device)
    u, v = net(latents, t, t, 1 + t, 0 * t, 1 + 0 * t, labels)
    assert u.shape == latents.shape, u.shape
    assert v.shape == latents.shape, v.shape
    print("forward shapes OK:", tuple(u.shape))

    # --- torch.func.jvp path with default SDPA-math attention ---
    def u_fn(z, t_in, r_in):
        u_out, v_out = net(
            z, t_in, t_in - r_in, 1 + 0 * t_in, 0 * t_in, 1 + 0 * t_in, labels
        )
        return u_out, v_out

    tangent_z = torch.randn_like(latents)
    r = 0.5 * t
    u_p, du_dt, v_aux = torch.func.jvp(
        u_fn, (latents, t, r), (tangent_z, torch.ones_like(t), torch.zeros_like(t)),
        has_aux=True,
    )
    assert du_dt.shape == latents.shape
    assert torch.isfinite(u_p).all() and torch.isfinite(du_dt).all()
    assert torch.isfinite(v_aux).all()
    print("torch.func.jvp path OK, |du_dt| mean:", du_dt.abs().mean().item())

    # At init the zero-init gates/final layers make the output (and du_dt)
    # exactly zero; perturb parameters to verify the jvp tangent is nontrivial.
    with torch.no_grad():
        for param in net.parameters():
            param.add_(0.02 * torch.randn_like(param))
    _, du_dt_perturbed, _ = torch.func.jvp(
        u_fn, (latents, t, r), (tangent_z, torch.ones_like(t), torch.zeros_like(t)),
        has_aux=True,
    )
    assert torch.isfinite(du_dt_perturbed).all()
    assert du_dt_perturbed.abs().mean() > 0, "jvp tangent is identically zero"
    print(
        "perturbed jvp |du_dt| mean:", du_dt_perturbed.abs().mean().item()
    )

    # --- full loss forward + backward ---
    loss_module = IMFVideoLoss(net, num_classes=num_classes)
    loss, dict_losses = loss_module(latents, labels)

    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    grad_params = 0
    for name, param in net.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
            grad_params += 1
    assert grad_params > 0, "no gradients produced"

    print(f"loss: {loss.item():.6f}")
    for key, value in dict_losses.items():
        print(f"  {key}: {value.item():.6f}")
    print(f"finite grads on {grad_params} parameter tensors")

    # --- sampler: one-step and multi-step generation ---
    for num_steps in (1, 4):
        z_1 = torch.randn(  # (B, C, T, H, W) initial noise at t = 1
            batch, channels, frames, height, width, device=device
        )
        z_0 = loss_module.sample(z_1, labels, num_steps=num_steps, omega=2.0)
        assert z_0.shape == z_1.shape, z_0.shape
        assert torch.isfinite(z_0).all(), f"non-finite sample at {num_steps} steps"
        print(f"sample({num_steps} steps) OK: {tuple(z_0.shape)} "
              f"std={z_0.std().item():.4f}")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
