"""
SLA2: Sparse-Linear Attention with Learnable Routing and QAT.

    O = alpha .* O_s + (1 - alpha) .* O_l                                 (Eq. 13)
    O_s = softmax(QK^T/sqrt(d) .* M) V
    O_l = norm(phi(Q) phi(K)^T .* (1 - M)) V
    M   = R(Q, K)                                                         (Eq. 14)

Differences vs SLA (`sparse_linear_attention/core.py`):
  * routing comes from a learnable router R (proj_q / proj_k) instead of the
    fixed pool(Q)pool(K)^T heuristic;
  * the linear branch only sees the complement blocks (1 - M), not all blocks;
  * the two branches are combined by a learnable per-query-block ratio alpha,
    so the extra `proj_l` projection of SLA is dropped;
  * the sparse forward may run in INT8 (QAT).

Shapes:
    q, k, v : (B, H, L, D)  B=batch, H=heads, L=tokens, D=head dim
    alpha   : (Mb,) learnable, sigmoid-squashed, one value per query block
              (paper: alpha in R^{N/bq x 1}); lazily created at first forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import Router
from .kernel_linear import (_complement_linear_attention, complement_linear_fused,
                            complement_linear_states)
from .kernel_sparse_qat import _sparse_attention_qat, sparse_attention_int8, quantize_qkv
from .kernel_fused import sla2_forward_fused


def softmax_feature_map(x):
    """phi(.) of the linear branch: row softmax over head channels. x: (B,H,L,D)."""
    return F.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype) if x.dtype != torch.float32 else F.softmax(x, dim=-1)


class SLA2(nn.Module):
    def __init__(self, head_dim, topk=0.03, bq=128, bk=64, tau=0.1,
                 use_bf16=False, use_int8=True, learnable_router=True, alpha_init=0.9,
                 block_m=None, fused=False, backend='cutedsl'):
        """
        head_dim        : D, per-head channel count
        topk            : k%, fraction of key blocks routed to the sparse branch
        bq, bk          : query / key block sizes (paper: 128 / 64)
        use_int8        : run the sparse forward in INT8 (QAT); backward stays FP16
        alpha_init      : initial value of alpha (before the sigmoid inverse)
        block_m         : query tile of the fused inference kernel (must divide bq);
                          None keeps it equal to bq
        fused           : use the single-kernel Algorithm-2 forward (kernel_fused.py)
        backend         : 'cutedsl' (default) or 'triton' for the linear branch;
                          the CuTeDSL kernel measured faster at every sparsity
        """
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk, self.bq, self.bk = topk, bq, bk
        self.use_int8 = use_int8
        self.block_m = block_m
        self.fused = fused
        self.backend = backend
        self.router = Router(head_dim, bq=bq, bk=bk, tau=tau, learnable=learnable_router)
        self._alpha_init = alpha_init
        self.alpha_logit = None  # (Mb,) lazily created

    def _get_alpha(self, n_heads, n_qblocks, device):
        """alpha in (0,1), one entry per (head, query block); shape (H, Mb).

        Heads carry very different softmax mass on their routed blocks, so a
        head-shared alpha is systematically wrong for most of them (measured to
        collapse video sampling when alpha is calibrated instead of trained)."""
        if self.alpha_logit is None or self.alpha_logit.shape != (n_heads, n_qblocks):
            init = torch.logit(torch.tensor(self._alpha_init))
            self.alpha_logit = nn.Parameter(
                torch.full((n_heads, n_qblocks), init.item(), device=device))
        return torch.sigmoid(self.alpha_logit)

    def forward(self, q, k, v, return_sparsity=False):
        """q, k, v: (B, H, L, D). Returns o: (B, H, L, D)."""
        out_dtype = q.dtype
        B, H, L, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        training = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad
                                                or v.requires_grad)
        # mask : (B,H,Mb,Nb) int8 when training (the backward kernels need it),
        # otherwise the router returns the raw scores and only `lut` is used
        mask, lut, topk, _ = self.router(q, k, self.topk, soft=False, need_mask=training)

        qh, kh, vh = q.to(self.dtype), k.to(self.dtype), v.to(self.dtype)
        kphi = F.softmax(kh, dim=-1).contiguous()                     # (B,H,L,D)
        alpha = self._get_alpha(H, mask.shape[-2], q.device)          # (H, Mb)
        if not training:
            if self.fused:
                # Algorithm 2 read literally: one kernel, one pass over the selected
                # blocks, both branches updated inside that loop.  Measured slower
                # here -- two (BLOCK_M, D) fp32 accumulators cost more occupancy than
                # the HBM traffic they save (see records/).
                block_m = self.block_m or max(32, self.bq // 2)
                quantized = (quantize_qkv(qh, block_m), quantize_qkv(kh, self.bk),
                             quantize_qkv(vh, self.bk)) if self.use_int8 else None
                # kernel_fused takes a head-shared (Mb,) alpha; average over heads
                return sla2_forward_fused(
                    qh, kh, vh, kphi, lut, topk, self.bq, self.bk,
                    alpha.float().mean(0).contiguous(), block_m=block_m,
                    use_int8=self.use_int8, quantized=quantized).to(out_dtype)
            # default inference path: sparse branch, then the linear branch built from
            # precomputed per-key-block states, with the alpha combination fused into
            # the epilogue
            want_phi = self.backend == 'cutedsl'
            if self.use_int8:
                o_s = sparse_attention_int8(qh, kh, vh, lut, topk, self.bq, self.bk,
                                            phi_out=want_phi)
                if want_phi:
                    o_s, qphi = o_s
            else:
                qphi = torch.empty_like(qh) if want_phi else None
                o_s = _sparse_attention_qat.apply(qh, kh, vh, mask, lut, topk,
                                                  self.bq, self.bk, False, None, qphi)
            if want_phi:
                from .cute_linear import complement_linear_states_cute
                o = complement_linear_states_cute(
                    qphi, kphi, vh, lut, topk, self.bq, self.bk,
                    o_s, alpha.float().contiguous())
            else:
                o = complement_linear_states(qh, kphi, vh, lut, topk, self.bq, self.bk,
                                             o_s, alpha.float().contiguous())
        else:
            o_s = _sparse_attention_qat.apply(qh, kh, vh, mask, lut, topk,
                                              self.bq, self.bk, self.use_int8)
            qphi = F.softmax(qh, dim=-1).contiguous()                 # (B,H,L,D)
            o_l = _complement_linear_attention.apply(qphi, kphi, vh, mask, lut, topk,
                                                     self.bq, self.bk)
            a = alpha.repeat_interleave(self.bq, dim=-1)[..., :L]     # (H, L)
            a = a.view(1, H, L, 1).to(o_s.dtype)
            o = a * o_s + (1.0 - a) * o_l
        if return_sparsity:
            return o.to(out_dtype), 1.0 - topk / mask.shape[-1]
        return o.to(out_dtype)


# ---------------------------------------------------------------------------
# Dense reference implementations (fp32, no kernels) -- used for correctness
# checks and for stage-1 router training, where gradients must flow through the
# SoftTop-k mask (the block-sparse kernels take a hard mask and cannot).
# ---------------------------------------------------------------------------

def expand_mask(mc, bq, bk, L):
    """(B,H,Mb,Nb) block mask -> (B,H,L,L) token mask."""
    m = mc.repeat_interleave(bq, dim=-2).repeat_interleave(bk, dim=-1)
    return m[..., :L, :L]


def sla2_dense(q, k, v, mc, alpha, bq, bk):
    """SLA2 Eq. 12-14 in dense fp32.

    q, k, v : (B, H, L, D)
    mc      : (B, H, Mb, Nb) block mask, hard {0,1} or soft [0,1]
    alpha   : (Mb,) in (0,1)
    """
    B, H, L, D = q.shape
    m = expand_mask(mc.float(), bq, bk, L)                 # (B,H,L,L)
    s = (q.float() @ k.float().transpose(-1, -2)) * D ** -0.5
    p = torch.softmax(s.masked_fill(m <= 0, float("-inf")), dim=-1) if mc.dtype == torch.int8 \
        else torch.softmax(s + torch.log(m + 1e-9), dim=-1)
    o_s = p @ v.float()

    qphi, kphi = softmax_feature_map(q), softmax_feature_map(k)
    w = (qphi @ kphi.transpose(-1, -2)) * (1.0 - m)        # (B,H,L,L)
    o_l = (w @ v.float()) / (w.sum(-1, keepdim=True) + 1e-5)

    a = alpha.repeat_interleave(bq)[:L].view(1, 1, L, 1)
    return a * o_s + (1.0 - a) * o_l


def full_attention(q, k, v):
    """Reference full attention. q,k,v: (B,H,L,D) -> (B,H,L,D) fp32."""
    D = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) * D ** -0.5
    return torch.softmax(s, dim=-1) @ v.float()
