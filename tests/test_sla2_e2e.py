"""E2E at production-like geometry: d=256, H=4, dc=64, SLA2 attention.

Latents (16, 4, 16, 16) keep the test small; hidden/head geometry matches
the trained config (hidden 256, heads 4, head_dim 64).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import partial
import torch
from imf_video import IMFVideoLoss
from models.imf_dit_video import IMFDiTVideo
from models.sla2_attention import SLA2AttentionImpl

dev = "cuda"
torch.manual_seed(0)
net = IMFDiTVideo(
    input_size=(16, 16), num_frames=4, patch_size=(1, 2, 2), in_channels=16,
    hidden_size=256, depth=4, aux_head_depth=1, num_heads=4, num_classes=10,
    attn_impl=partial(SLA2AttentionImpl, topk=0.4, bq=128, bk=64,
                      alpha_init=0.9),
    attn_res_block_size=2, grad_checkpoint=False,
).to(dev)
loss_mod = IMFVideoLoss(net, 10, jvp_impl="fast")
x = torch.randn(2, 16, 4, 16, 16, device=dev)
y = torch.randint(0, 10, (2,), device=dev)
ok = 0
for s in range(8):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    loss, d = loss_mod(x, y)
    lu = float(d["loss_u"])
    ok += lu == lu
    if s == 0:
        loss.backward()
        gn = sum(p.grad.norm() for p in net.parameters()
                 if p.grad is not None)
        print(f"bwd grad-norm-sum {gn.item():.2f} finite {torch.isfinite(gn).item()}")
print(f"finite loss_u in {ok}/8 seeds")
assert ok == 8
print("PASS")
