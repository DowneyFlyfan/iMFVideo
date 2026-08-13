"""
Learnable router R of SLA2 (Section 4) and the SoftTop-k operator (Eq. 17).

    Qbar = pool(Q) in R^{N/bq x d},  Kbar = pool(K) in R^{N/bk x d}     (Eq. 15)
    Pc   = proj_q(Qbar) proj_k(Kbar)^T in R^{N/bq x N/bk}               (Eq. 16)
    Mc   = Top-k(k%, Pc)            (inference, hard)
         = sigmoid(Pc/tau + lambda) (stage-1 training, differentiable)  (Eq. 17)

Shapes:
    q, k          : (B, H, L, D)  B=batch, H=heads, L=tokens, D=head dim
    q_pool/k_pool : (B, H, L/bq, D) / (B, H, L/bk, D)
    pc            : (B, H, L/bq, L/bk) -- compressed score, [query block, key block]
    mask/mc       : (B, H, L/bq, L/bk) in {0,1} (hard) or [0,1] (soft)
"""

import torch
import torch.nn as nn

from .kernel_router import mean_pool_smooth


def soft_topk(pc, k_frac, tau=0.1, n_iters=30):
    """SoftTop-k (Ding et al., 2024): sigmoid(pc/tau + lambda_i) with row sums = k_frac * n_kblocks.

    pc     : (B, H, Mb, Nb) fp32 compressed scores
    k_frac : fraction of key blocks routed to the sparse branch
    returns: (B, H, Mb, Nb) soft mask in [0, 1]
    """
    pc = pc.float()
    target = k_frac * pc.shape[-1]
    x = pc / tau
    # bisection on lambda per row so that sum_j sigmoid(x_ij + lambda_i) = target
    lo = torch.full(pc.shape[:-1], -1e4, device=pc.device)
    hi = torch.full(pc.shape[:-1], 1e4, device=pc.device)
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        s = torch.sigmoid(x + mid[..., None]).sum(-1)
        too_big = s > target
        hi = torch.where(too_big, mid, hi)
        lo = torch.where(too_big, lo, mid)
    lam = (0.5 * (lo + hi)).detach()
    return torch.sigmoid(x + lam[..., None])


class Router(nn.Module):
    """R(Q, K) -> compressed block scores Pc and the block mask."""

    def __init__(self, head_dim, bq=128, bk=64, tau=0.1, learnable=True):
        super().__init__()
        self.bq, self.bk, self.tau = bq, bk, tau
        self.learnable = learnable
        # proj_q, proj_k in R^{d x d}, identity-initialized so the router starts
        # at the heuristic pool(Q)pool(K)^T routing used by SLA.
        self.proj_q = nn.Linear(head_dim, head_dim, bias=False, dtype=torch.float32)
        self.proj_k = nn.Linear(head_dim, head_dim, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.proj_q.weight.copy_(torch.eye(head_dim))
            self.proj_k.weight.copy_(torch.eye(head_dim))
        if not learnable:
            for p in self.parameters():
                p.requires_grad_(False)

    def scores(self, q, k):
        """q, k: (B, H, L, D) -> pc: (B, H, Mb, Nb) fp32, Mb = L/bq, Nb = L/bk."""
        # smooth-k (SageAttention, kept from SLA) is fused into the pooling kernel
        qb = mean_pool_smooth(q.contiguous(), self.bq, False)   # (B,H,Mb,D) fp32
        kb = mean_pool_smooth(k.contiguous(), self.bk, True)    # (B,H,Nb,D) fp32
        if self.learnable:
            qb = self.proj_q(qb)
            kb = self.proj_k(kb)
        return (qb @ kb.transpose(-1, -2)) * (q.shape[-1] ** -0.5)

    def forward(self, q, k, k_frac, soft=False, need_mask=True):
        """Returns (mask, lut, topk, pc).

        mask : (B,H,Mb,Nb) int8 hard mask (always hard, used by the kernels)
        lut  : (B,H,Mb,topk) int32 selected key-block ids
        soft_mask (if soft=True) is returned in place of the hard mask for training R.
        """
        pc = self.scores(q, k)                                   # (B,H,Mb,Nb) fp32
        nb = pc.shape[-1]
        topk = max(1, min(nb, int(k_frac * nb)))
        lut = torch.topk(pc, topk, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
        # the dense {0,1} mask is only needed by the backward kernels; at inference
        # the LUT alone drives the forward kernels
        hard = torch.zeros_like(pc, dtype=torch.int8).scatter_(-1, lut.long(), 1) \
            if need_mask else pc
        if soft:
            return soft_topk(pc, k_frac, self.tau), lut, topk, pc
        return hard, lut, topk, pc
