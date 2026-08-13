"""SLA2 (sparse-linear attention) as an MLA attn_impl for training forward.

The MLA module calls `attn_impl(q, k, v)` with (b, l, H, hd) tensors; this
wrapper permutes to SLA2's (B, H, L, D) layout, runs the vendored SLA2
module (triton backend: block-sparse softmax branch + complement linear
branch, learnable router, per-(head, query-block) alpha mix), and permutes
back.  The fast du/dt JVP path (mla_jvp_fast.py) reads this module's
parameters and reproduces the same function with models/sla2_mla_jvp.py.

Shape symbols:
    b : batch;  l/L : tokens;  H : heads;  hd/D : qk head dim (= 64)
    dv : value head dim (= hd here);  Mb : query blocks = ceil(L / bq)
"""

import math

import torch
import torch.nn as nn

from models.sla2_vendor import SLA2


class SLA2AttentionImpl(nn.Module):
    """Stateful attn_impl: one SLA2 (router + alpha) per attention module.

    Args:
        head_dim: qk head dim D (must equal v head dim; 64 in this repo).
        seq_len: full sequence length L (prefix + patch tokens), known at
            model build time; used to create alpha_logit (H, Mb) eagerly so
            the optimizer split sees every parameter before the first step.
        num_heads: H, for the eager alpha creation.
        topk: routed fraction of key blocks (SLA2 paper default 0.03).
        bq, bk: query / key block lengths (paper: 128 / 64).
        alpha_init: initial mixing ratio in (0, 1) before logit transform.
    """

    def __init__(self, head_dim, seq_len, num_heads, topk=0.03,
                 bq=128, bk=64, alpha_init=0.9):
        super().__init__()
        self.seq_len = seq_len
        self.bq, self.bk, self.topk = bq, bk, topk
        # use_int8=False: QAT is an inference-deployment feature; training
        # here stays fp16 compute / fp32 master.
        self.sla2 = SLA2(head_dim, topk=topk, bq=bq, bk=bk, use_bf16=False,
                         use_int8=False, learnable_router=True,
                         alpha_init=alpha_init, backend="triton")
        # eager alpha: (H, Mb) mixing logits, sigmoid(alpha_logit) in (0,1);
        # SLA2 would lazily create this at first forward, too late for the
        # Moonlight parameter split.
        mb = math.ceil(seq_len / bq)
        init = torch.logit(torch.tensor(float(alpha_init)))
        self.sla2.alpha_logit = nn.Parameter(
            torch.full((num_heads, mb), init.item()))

    def forward(self, q, k, v):
        """q, k, v: (b, l, H, hd) -> (b, l, H, hd) attention output."""
        qh = q.permute(0, 2, 1, 3)                       # (B, H, L, D)
        kh = k.permute(0, 2, 1, 3)                       # (B, H, L, D)
        vh = v.permute(0, 2, 1, 3)                       # (B, H, L, D)
        o = self.sla2(qh, kh, vh)                        # (B, H, L, D)
        return o.permute(0, 2, 1, 3).to(q.dtype)         # (b, l, H, hd)
