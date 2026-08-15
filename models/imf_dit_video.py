"""PyTorch video imfDiT: port of refs/imfDiT.py (JAX/Flax) extended from
images to video latents.

Input latents: (B, C, T, H, W) with C = 16 (Wan2.1 VAE latent channels).
3D patchify with patch size (1, 2, 2) -> tokens = T * (H/2) * (W/2).

Every op is torch.func.jvp-compatible (forward-mode AD): no in-place ops on
autograd tensors, no .item() in forward.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################################
#                          Pluggable Attention Kernels                           #
#################################################################################


def eager_math_attention(q, k, v):
    """Attention written out as einsums, softmax accumulated in float32.

    Shape symbols
        b  : batch size
        l  : query sequence length
        s  : key/value sequence length (equal to l for self-attention)
        H  : number of heads
        dk : query/key head dim
        dv : value head dim (may differ from dk, as it does under MLA)

    Every op here (matmul, softmax, mul) has a forward-mode AD formula on all
    backends, which the fused and MPS-specific SDPA kernels do not. Softmax
    scale is 1/sqrt(dk), matching F.scaled_dot_product_attention's default.

    Args:
        q: (b, l, H, dk)
        k: (b, s, H, dk)
        v: (b, s, H, dv)

    Returns:
        (b, l, H, dv)
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("blhd,bshd->bhls", q.float(), k.float()) * scale
    #   scores: (b, H, l, s) attention logits
    probs = torch.softmax(scores, dim=-1)  # (b, H, l, s)
    out = torch.einsum("bhls,bshd->blhd", probs, v.float())  # (b, l, H, dv)
    return out.to(v.dtype)


def sdpa_math_attention(q, k, v):
    """Default attention: torch SDPA restricted to the math backend.

    The math backend is a plain softmax(q k^T / sqrt(d)) v decomposition and
    therefore supports forward-mode AD (torch.func.jvp). Fused flash /
    mem-efficient kernels do not; a CuTeDSL flash-attention JVP op can be
    plugged in later via the `attn_impl` constructor argument.

    On MPS (Apple Metal), SDPA dispatches to
    aten::_scaled_dot_product_attention_math_for_mps, which has no forward-AD
    formula and raises under torch.func.jvp, so the einsum path is used there.

    Args:
        q, k, v: (batch, seq_len, num_heads, head_dim); v may carry a different
            head dim than q/k (MLA uses v_head_dim != qk_head_dim).

    Returns:
        (batch, seq_len, num_heads, v_head_dim)
    """
    if q.device.type == "mps":
        return eager_math_attention(q, k, v)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> (B, H, S, D)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)


def sdpa_flash_attention(q, k, v):
    """SDPA with fused flash / mem-efficient backends (no math fallback).

    O(S) memory, so it fits long sequences, but it has NO forward-mode AD
    formula: use only with jvp_impl="fast" (du/dt comes from the hand-rolled
    Triton pass, never from torch.func.jvp over this function).

    Args:
        q, k, v: (batch, seq_len, num_heads, head_dim); v may carry a
            different head dim than q/k (MLA uses v_head_dim != qk_head_dim).

    Returns:
        (batch, seq_len, num_heads, v_head_dim)
    """
    dt = q.dtype
    # flash kernels need fp16/bf16; the CuTeDSL flash_jvp path also runs
    # attention in bf16 internally, so this matches production numerics.
    q, k, v = (x.transpose(1, 2).to(torch.bfloat16) for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel([
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    ]):
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, Dv) bf16
    return out.transpose(1, 2).to(dt)  # (B, S, H, Dv)


#################################################################################
#                              Basic Torch Modules                               #
#################################################################################


def scaled_variance_linear(in_features, out_features, bias, init_constant):
    """Linear layer with weight ~ Normal(0, init_constant / sqrt(in_features))."""
    linear = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(linear.weight, std=init_constant / math.sqrt(in_features))
    if bias:
        nn.init.zeros_(linear.bias)
    return linear


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (float32 accumulation)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f32 = x.float()
        mean_square = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(mean_square + self.eps)
        return normed.to(x.dtype) * self.weight


class SituAndMul(nn.Module):
    """Kimi-K3 gated activation (modeling_kimi_linear.SituAndMul).

    situ(g) = beta * tanh(g / beta) * sigmoid(g): identical to SiLU for
    |g| << beta and saturating at +-beta * sigmoid(g) for large |g|, so the
    gate branch is bounded (outlier / massive-activation control). When
    linear_beta is set the up branch is also soft-clamped by
    linear_beta * tanh(up / linear_beta).
    """

    def __init__(self, beta=1.0, linear_beta=None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x, up=None):
        # x: (..., 2F) rows [gate | up], or gate (..., F) when `up` given.
        # returns (..., F)
        if up is None:
            d = x.shape[-1] // 2
            gate, up = x[..., :d], x[..., d:]            # (..., F) each
        else:
            gate = x                                     # (..., F)
        dtype = gate.dtype
        gate = gate.to(torch.float32)
        up = up.to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(dtype)


class SwiGLUMlp(nn.Module):
    """Gated MLP with the Kimi-K3 SituAndMul activation (bounded SiLU
    gate; plain SiLU is the beta -> inf limit)."""

    def __init__(self, in_features, hidden_features, weight_init_constant=1.0,
                 situ_beta=1.0, situ_linear_beta=None):
        super().__init__()
        linear = partial(
            scaled_variance_linear, bias=False, init_constant=weight_init_constant
        )
        self.w1 = linear(in_features, hidden_features)
        self.w3 = linear(in_features, hidden_features)
        self.w2 = linear(hidden_features, in_features)
        self.act = SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)

    def forward(self, x):
        # x: (b, l, d) -> gate/up (b, l, F) each -> (b, l, d)
        # (no cat: at 29k tokens the (b, l, 2F) fp32 concat is a 0.65 GB
        # transient; SituAndMul takes the pair directly)
        return self.w2(self.act(self.w1(x), self.w3(x)))


class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    def __init__(self, hidden_size, frequency_embedding_size=256, init_constant=1.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        linear = partial(scaled_variance_linear, bias=True, init_constant=init_constant)
        self.mlp = nn.Sequential(
            linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings. t: (B,)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """Embeds class labels (index num_classes = null/unconditional label)."""

    def __init__(self, num_classes, hidden_size, init_constant=1.0):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(
            self.embedding_table.weight, std=init_constant / math.sqrt(hidden_size)
        )

    def forward(self, labels):
        return self.embedding_table(labels)


class VideoPatchEmbedder(nn.Module):
    """Video latents to patch embedding via Conv3d with patch (1, 2, 2)."""

    def __init__(self, in_channels, hidden_size, patch_size=(1, 2, 2), bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.proj.weight.view(hidden_size, -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


#################################################################################
#                   Modern Transformer Components with Vec Gates                 #
#################################################################################


def apply_rotary_pos_emb(x, rope_cos, rope_sin):
    """Rotate adjacent-pair channels of x by precomputed angles.

    Matches the JAX complex-view convention: adjacent channel pairs
    (x[..., 2i], x[..., 2i+1]) form (real, imag).

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        rope_cos, rope_sin: (seq_len, head_dim // 2)
    """
    x_f32 = x.float()
    x_real = x_f32[..., 0::2]
    x_imag = x_f32[..., 1::2]
    cos = rope_cos[None, :, None, :]
    sin = rope_sin[None, :, None, :]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    return out.to(x.dtype)


def attn_res_apply(prefix, snaps, gain, proj_w, eps=1e-6):
    """Kimi-K3-style attention residual (modeling_kimi_linear._apply_attn_res).

    Per token, a 1-query softmax attention over the token's own residual-
    stream history: J = len(snaps) + 1 candidate vectors (earlier block
    snapshots + the current prefix sum) are RMS-normalized, scored against a
    single learned direction (norm gain folded with the 1-dim projection),
    and mixed by softmax.

    Args:
        prefix: (b, l, d) current residual prefix sum.
        snaps: list of (b, l, d) block snapshots (may be empty).
        gain: (d,) RMSNorm gain of the residual-attention norm.
        proj_w: (1, d) or (d,) weight of the Linear(d, 1) score projection.
        eps: RMSNorm epsilon.

    Returns:
        (b, l, d) re-mixed residual stream, in prefix.dtype.
    """
    v = torch.stack(list(snaps) + [prefix], dim=2)      # (b, l, J, d)
    w = gain.float() * proj_w.reshape(-1).float()       # (d,) fused score dir
    if prefix.is_cuda and prefix.dtype == torch.float32:
        # fused Triton path (forward, backward and forward-mode JVP)
        from models.triton_attn_res_jvp import attn_res_op

        b, l, J, d = v.shape
        return attn_res_op(v.reshape(b * l, J, d), w, eps).view(b, l, d)
    vf = v.float()
    var = vf.pow(2).mean(-1, keepdim=True)              # (b, l, J, 1)
    k = vf * torch.rsqrt(var + eps)                     # (b, l, J, d) RMS-normed
    scores = torch.einsum("bljd,d->blj", k, w)          # (b, l, J)
    probs = scores.softmax(-1)                          # (b, l, J)
    out = torch.einsum("blj,bljd->bld", probs, vf)      # (b, l, d)
    return out.to(prefix.dtype)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with decoupled ("partial") RoPE.

    Shape symbols used throughout this class:
        b  : batch size
        l  : sequence length (num_prefix_tokens + num_patch_tokens)
        d  : hidden_size, model width
        H  : num_heads (query heads; MLA gives every query head its own
             up-projected key/value, so there is no grouped-query sharing)
        dq : q_lora_rank, query down-projection (latent) dim
        dc : kv_lora_rank, shared key/value down-projection (latent) dim
        dn : qk_nope_head_dim, position-free part of each q/k head vector
        dr : qk_rope_head_dim, rotary part of each q/k head vector; the key
             half is ONE shared head broadcast over all H query heads
        dv : v_head_dim, value head dim

    Why MLA in a DiT. A diffusion transformer runs one bidirectional forward
    pass per denoising step and keeps no KV cache, so MLA's usual selling point
    (cache compression for autoregressive decoding) does not apply. What it
    does buy here is a low-rank bottleneck on the q/k/v projections: with the
    defaults below the attention block holds ~0.68x the parameters of the
    equivalent full-rank multi-head attention, and the shared dc-dim latent
    constrains keys and values to a common subspace.

    Decoupled / partial RoPE. Only the trailing dr channels of each q/k head
    vector carry rotary phase; the leading dn channels are position-free and
    come straight from the compressed latent. The key side of the rotary band
    (k_rope) is computed once per token from x and shared across all H heads,
    which is what makes the band "decoupled" from the low-rank path.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        weight_init_constant=1.0,
        norm_eps=1e-6,
        attn_impl=sdpa_math_attention,
        use_output_gate=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # attn_impl is either a plain function (q, k, v) -> o shared by all
        # blocks, or a module factory (e.g. partial(SLA2AttentionImpl, ...))
        # that builds a stateful per-block attention (router + alpha params);
        # factories are instantiated here so the params register as submodules.
        from functools import partial as _partial
        is_factory = isinstance(attn_impl, type) or (
            isinstance(attn_impl, _partial) and isinstance(attn_impl.func, type)
        )
        if is_factory:
            assert self.qk_head_dim == v_head_dim, (
                "SLA2 attention needs qk_head_dim == v_head_dim, got "
                f"{self.qk_head_dim} vs {v_head_dim}"
            )
            attn_impl = attn_impl(self.qk_head_dim)
        self.attn_impl = attn_impl

        linear = partial(
            scaled_variance_linear, bias=False, init_constant=weight_init_constant
        )
        # query: d -> dq -> H * (dn + dr)
        self.q_a_proj = linear(hidden_size, q_lora_rank)
        self.q_b_proj = linear(q_lora_rank, num_heads * self.qk_head_dim)
        # key/value: d -> (dc latent | dr shared rotary key), then dc -> H * (dn + dv)
        self.kv_a_proj = linear(hidden_size, kv_lora_rank + qk_rope_head_dim)
        self.kv_b_proj = linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
        )
        self.out_proj = linear(num_heads * v_head_dim, hidden_size)
        # Kimi-K3 MLA output gate (mla_use_output_gate): the attention
        # output is modulated per channel by sigmoid(g_proj(x)) before the
        # output projection.
        self.use_output_gate = use_output_gate
        if use_output_gate:
            self.g_proj = linear(hidden_size, num_heads * v_head_dim)

        # Latent-space norms (DeepSeek-V2/V3 placement).
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=norm_eps)
        # Per-head QK norm on the position-free band only. The rotary band is
        # left un-normed so that k_rope stays a single shared head; it is
        # bounded upstream by q_a_layernorm (query side) and by the small
        # kv_a_proj init scale (key side).
        self.q_norm = RMSNorm(qk_nope_head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(qk_nope_head_dim, eps=norm_eps)

    def forward(self, x, rope_cos, rope_sin):
        """
        Args:
            x: (b, l, d) input token features.
            rope_cos, rope_sin: (l, dr // 2) precomputed rotary angles, float32.

        Returns:
            (b, l, d) attention output.
        """
        b, l, _ = x.shape
        H = self.num_heads
        dn, dr, dv = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim

        # ---- query path: (b,l,d) -> latent (b,l,dq) -> heads (b,l,H,dn+dr) ----
        q_latent = self.q_a_layernorm(self.q_a_proj(x))  # (b, l, dq)
        q = self.q_b_proj(q_latent).reshape(b, l, H, dn + dr)  # (b, l, H, dn+dr)
        q_nope, q_rope = q.split([dn, dr], dim=-1)  # (b,l,H,dn), (b,l,H,dr)

        # ---- key/value path: (b,l,d) -> (b,l,dc) latent + (b,l,dr) shared key ----
        kv_a = self.kv_a_proj(x)  # (b, l, dc+dr)
        kv_latent, k_rope = kv_a.split([self.kv_lora_rank, dr], dim=-1)
        #   kv_latent: (b, l, dc)      k_rope: (b, l, dr)  <- shared over heads
        kv_latent = self.kv_a_layernorm(kv_latent)  # (b, l, dc)
        kv = self.kv_b_proj(kv_latent).reshape(b, l, H, dn + dv)  # (b, l, H, dn+dv)
        k_nope, v = kv.split([dn, dv], dim=-1)  # (b,l,H,dn), (b,l,H,dv)

        q_nope = self.q_norm(q_nope)  # (b, l, H, dn)
        k_nope = self.k_norm(k_nope)  # (b, l, H, dn)

        # ---- partial RoPE: rotate only the trailing dr channels ----
        q_rope = apply_rotary_pos_emb(q_rope, rope_cos, rope_sin)  # (b, l, H, dr)
        k_rope = apply_rotary_pos_emb(
            k_rope.unsqueeze(2), rope_cos, rope_sin
        )  # (b, l, 1, dr)
        k_rope = k_rope.expand(b, l, H, dr)  # (b, l, H, dr)

        q = torch.cat([q_nope, q_rope], dim=-1)  # (b, l, H, dn+dr)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (b, l, H, dn+dr)

        # attn_impl leaves softmax_scale at its default 1/sqrt(q.shape[-1]),
        # which for MLA is 1/sqrt(dn + dr) -- the full query head dim.
        attn = self.attn_impl(q, k, v)  # (b, l, H, dv)
        attn = attn.reshape(b, l, H * dv)                 # (b, l, H*dv)
        if self.use_output_gate:
            attn = attn * torch.sigmoid(self.g_proj(x))   # (b, l, H*dv)
        return self.out_proj(attn)  # (b, l, d)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        mlp_ratio=8 / 3,
        weight_init_constant=1.0,
        norm_eps=1e-6,
        attn_impl=sdpa_math_attention,
        use_attn_res=False,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        mla_use_output_gate=True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MLAAttention(
            hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            weight_init_constant=weight_init_constant,
            norm_eps=norm_eps,
            attn_impl=attn_impl,
            use_output_gate=mla_use_output_gate,
        )
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUMlp(
            hidden_size, mlp_hidden_dim, weight_init_constant=weight_init_constant,
            situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
        )

        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

        # Kimi-K3 attention-residual heads (used only when the parent model
        # runs with attn_res_block_size > 0): an RMSNorm + Linear(d, 1) pair
        # per sublayer scores the residual-stream snapshots.
        if use_attn_res:
            self.attn_res_norm = RMSNorm(hidden_size, eps=norm_eps)
            self.mlp_res_norm = RMSNorm(hidden_size, eps=norm_eps)
            self.attn_res_proj = scaled_variance_linear(
                hidden_size, 1, bias=False, init_constant=weight_init_constant
            )
            self.mlp_res_proj = scaled_variance_linear(
                hidden_size, 1, bias=False, init_constant=weight_init_constant
            )

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero-init weights."""

    def __init__(self, hidden_size, patch_size, out_channels, norm_eps=1e-6):
        super().__init__()
        patch_t, patch_h, patch_w = patch_size
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.linear = nn.Linear(
            hidden_size, patch_t * patch_h * patch_w * out_channels, bias=True
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                           Rotary Position Helpers                              #
#################################################################################


def precompute_axial_rope_3d(
    head_dim, grid_size, num_prefix_tokens, axis_dims=None, theta=10000.0
):
    """3D axial RoPE angles for video patch tokens plus prefix tokens.

    head_dim is split into (t, h, w) parts (default 16/24/24 for head_dim 64);
    each part gets standard 1D RoPE on its axis position.

    Prefix conditioning tokens use the identity rotation (cos=1, sin=0), i.e.
    they are position-free: they are distinguished by their learnable token
    content, not by position, so no rotary phase is applied to them.

    Returns:
        rope_cos, rope_sin: (num_prefix_tokens + T*Gh*Gw, head_dim // 2) float32.
    """
    if axis_dims is None:
        dim_t = 2 * ((head_dim // 4) // 2)  # even split, t gets ~1/4
        dim_h = 2 * (((head_dim - dim_t) // 2) // 2)
        dim_w = head_dim - dim_t - dim_h
        axis_dims = (dim_t, dim_h, dim_w)
    assert sum(axis_dims) == head_dim
    assert all(d % 2 == 0 for d in axis_dims)

    grid_t, grid_h, grid_w = grid_size
    pos_t, pos_h, pos_w = torch.meshgrid(
        torch.arange(grid_t, dtype=torch.float32),
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )  # each (T, Gh, Gw), matching t-major token order of VideoPatchEmbedder

    angle_parts = []
    for positions, axis_dim in zip(
        (pos_t.flatten(), pos_h.flatten(), pos_w.flatten()), axis_dims
    ):
        freqs = 1.0 / (
            theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        )
        angle_parts.append(torch.outer(positions, freqs))
    angles = torch.cat(angle_parts, dim=-1)  # (N, head_dim // 2)

    # Identity rotation (angle 0) on prefix conditioning tokens.
    prefix_angles = torch.zeros(num_prefix_tokens, head_dim // 2)
    angles = torch.cat([prefix_angles, angles], dim=0)

    return torch.cos(angles), torch.sin(angles)


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning              #
#################################################################################


class IMFDiTVideo(nn.Module):
    """improved MeanFlow DiT for video latents.

    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    def __init__(
        self,
        input_size=16,
        num_frames=3,
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=8 / 3,
        num_classes=1000,
        aux_head_depth=8,
        # --- MLA (multi-head latent attention) geometry; 0 = derive from head_dim ---
        q_lora_rank=0,
        kv_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
        num_class_tokens=8,
        num_time_tokens=4,
        num_cfg_tokens=4,
        num_interval_tokens=2,
        freq_embedding_size=256,
        token_init_constant=1.0,
        embedding_init_constant=1.0,
        weight_init_constant=0.32,
        rmsnorm_eps=1e-6,
        rope_theta=10000.0,
        attn_impl=sdpa_math_attention,
        eval_mode=False,
        attn_res_block_size=0,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        mla_use_output_gate=True,
        grad_checkpoint=False,
    ):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        # Kimi-K3 attention residual: snapshot the residual stream every
        # attn_res_block_size blocks; 0 disables the mechanism entirely.
        self.attn_res_block_size = attn_res_block_size
        # recompute sublayer activations in backward (long-sequence training)
        self.grad_checkpoint = grad_checkpoint

        # MLA defaults, all derived from head_dim so that
        #   qk_nope_head_dim + qk_rope_head_dim == v_head_dim == head_dim,
        # which keeps the layout drop-in compatible with the fixed-head_dim-64
        # flash JVP op while still giving the low-rank q/kv bottleneck.
        self.v_head_dim = v_head_dim or self.head_dim
        self.qk_rope_head_dim = qk_rope_head_dim or max(2, self.head_dim // 4)
        self.qk_nope_head_dim = (
            qk_nope_head_dim or self.head_dim - self.qk_rope_head_dim
        )
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Latent ranks must scale with hidden_size, not head_dim: a rank larger
        # than the dim it compresses makes the "bottleneck" cost more than the
        # full-rank projection it replaces. d/2 and d/4 keep the attention block
        # at ~0.69x the parameters of equivalent full-rank multi-head attention.
        self.q_lora_rank = q_lora_rank or max(self.head_dim, hidden_size // 2)
        self.kv_lora_rank = kv_lora_rank or max(self.head_dim, hidden_size // 4)

        # apply_rotary_pos_emb pairs adjacent channels, and precompute_axial_rope_3d
        # splits the rotary band across the (t, h, w) axes; both need even dims.
        assert self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even"
        assert self.qk_nope_head_dim % 2 == 0, (
            "qk_nope_head_dim must be even so the rotary band starts on an even "
            "channel offset"
        )
        if getattr(attn_impl, "__name__", "") == "flash_jvp_attention":
            assert self.qk_head_dim == 64 and self.v_head_dim == 64, (
                "flash_jvp_attention is compiled for head_dim 64: need "
                "qk_nope_head_dim + qk_rope_head_dim == v_head_dim == 64, got "
                f"{self.qk_nope_head_dim} + {self.qk_rope_head_dim} and "
                f"{self.v_head_dim}"
            )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.eval_mode = eval_mode

        patch_t, patch_h, patch_w = patch_size
        # input_size: int (square) or (H, W) tuple of latent spatial dims
        in_h, in_w = (input_size, input_size) if isinstance(input_size, int)             else input_size
        self.grid_size = (
            num_frames // patch_t,
            in_h // patch_h,
            in_w // patch_w,
        )
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.x_embedder = VideoPatchEmbedder(
            in_channels, hidden_size, patch_size=patch_size, bias=True
        )

        # TimestepEmbedder takes the sinusoidal basis width; LabelEmbedder does not.
        time_embed_kwargs = dict(
            hidden_size=hidden_size,
            frequency_embedding_size=freq_embedding_size,
            init_constant=embedding_init_constant,
        )
        self.h_embedder = TimestepEmbedder(**time_embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**time_embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**time_embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**time_embed_kwargs)
        self.y_embedder = LabelEmbedder(
            num_classes,
            hidden_size=hidden_size,
            init_constant=embedding_init_constant,
        )

        token_std = token_init_constant / math.sqrt(hidden_size)
        self.time_tokens = nn.Parameter(
            torch.randn(num_time_tokens, hidden_size) * token_std
        )
        self.class_tokens = nn.Parameter(
            torch.randn(num_class_tokens, hidden_size) * token_std
        )
        self.omega_tokens = nn.Parameter(
            torch.randn(num_cfg_tokens, hidden_size) * token_std
        )
        self.t_min_tokens = nn.Parameter(
            torch.randn(num_interval_tokens, hidden_size) * token_std
        )
        self.t_max_tokens = nn.Parameter(
            torch.randn(num_interval_tokens, hidden_size) * token_std
        )

        self.prefix_tokens = (
            num_class_tokens
            + num_cfg_tokens
            + 2 * num_interval_tokens
            + num_time_tokens
        )

        # Partial RoPE: the table covers only the rotary band (dr channels),
        # not the whole head vector.
        rope_cos, rope_sin = precompute_axial_rope_3d(
            self.qk_rope_head_dim,
            self.grid_size,
            self.prefix_tokens,
            theta=rope_theta,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        head_depth = aux_head_depth
        shared_depth = depth - head_depth

        # A module-factory attn_impl (SLA2) needs the full sequence length
        # L = prefix_tokens + num_patches and the head count to size its
        # per-(head, query-block) alpha eagerly; bind them here where L is
        # first known, before the blocks instantiate the factory per block.
        from functools import partial as _partial
        self.tile_major = False
        if isinstance(attn_impl, _partial) and isinstance(attn_impl.func, type):
            attn_impl = _partial(
                attn_impl,
                seq_len=self.prefix_tokens + num_patches,
                num_heads=num_heads,
            )
            # Tile-major residency for cube attention: keep the WHOLE
            # sequence in 3D-tile token order for the entire forward, so
            # the per-block permute gathers (6 per block per pass) become
            # ONE permute after the sequence embed and ONE inverse before
            # the output heads. Every block op except attention and RoPE
            # is per-token and therefore order-agnostic; RoPE tables are
            # permuted once below.
            if getattr(attn_impl.func, "TILE_MAJOR", False):
                from models.mla_video_sparse_jvp import tile_permutation
                tile = attn_impl.keywords.get("tile", (4, 4, 4))
                tm_perm, tm_inv, _ = tile_permutation(
                    self.grid_size, tile, self.prefix_tokens,
                    torch.device("cpu"))
                # tm_perm/tm_inv: (L,) int64 model-order <-> tile-order
                self.register_buffer("tm_perm", tm_perm, persistent=False)
                self.register_buffer("tm_inv", tm_inv, persistent=False)
                self.tile_major = True
                attn_impl = _partial(attn_impl, pre_permuted=True)
                # rope_cos/rope_sin: (L, dr//2) angle tables aligned with
                # token positions -> permute rows into tile order once
                self.rope_cos = self.rope_cos[tm_perm]
                self.rope_sin = self.rope_sin[tm_perm]

        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            mlp_ratio=mlp_ratio,
            weight_init_constant=weight_init_constant,
            norm_eps=rmsnorm_eps,
            attn_impl=attn_impl,
            use_attn_res=attn_res_block_size > 0,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            mla_use_output_gate=mla_use_output_gate,
        )
        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth)]
        )
        # We don't need the v heads during evaluation
        self.v_heads = nn.ModuleList(
            [
                TransformerBlock(**block_kwargs)
                for _ in range(head_depth if not eval_mode else 0)
            ]
        )

        self.u_final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, norm_eps=rmsnorm_eps
        )
        self.v_final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, norm_eps=rmsnorm_eps
        )

        if attn_res_block_size > 0:
            # Output-side attention residual (one per head branch), applied
            # to the final prefix sum before the head's FinalLayer -- the
            # analogue of KimiLinearModel.output_attn_res_{norm,proj}.
            self.rmsnorm_eps = rmsnorm_eps
            self.u_out_res_norm = RMSNorm(hidden_size, eps=rmsnorm_eps)
            self.v_out_res_norm = RMSNorm(hidden_size, eps=rmsnorm_eps)
            self.u_out_res_proj = scaled_variance_linear(
                hidden_size, 1, bias=False, init_constant=weight_init_constant
            )
            self.v_out_res_proj = scaled_variance_linear(
                hidden_size, 1, bias=False, init_constant=weight_init_constant
            )

    def unpatchify(self, x):
        """(B, N, patch_t*patch_h*patch_w*C) -> (B, C, T, H, W)."""
        batch = x.shape[0]
        channels = self.out_channels
        patch_t, patch_h, patch_w = self.patch_size
        grid_t, grid_h, grid_w = self.grid_size
        assert grid_t * grid_h * grid_w == x.shape[1]

        x = x.reshape(
            batch, grid_t, grid_h, grid_w, patch_t, patch_h, patch_w, channels
        )
        x = torch.einsum("nthwpqrc->nctphqwr", x)
        return x.reshape(
            batch, channels, grid_t * patch_t, grid_h * patch_h, grid_w * patch_w
        )

    def _build_sequence(self, x, h, w, t_min, t_max, y):
        """
        Build the input token sequence for the transformer.
        1. Embed the input video latent patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input video latents (B, C, T, H, W)
            h: timestep difference t - r, shape (B,)
            w: CFG scale, shape (B,)
            t_min, t_max: CFG interval, shape (B,)
            y: Class labels, shape (B,)
        """
        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / w)
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + h_embed.unsqueeze(1)
        omega_tokens = self.omega_tokens + omega_embed.unsqueeze(1)
        t_min_tokens = self.t_min_tokens + t_min_embed.unsqueeze(1)
        t_max_tokens = self.t_max_tokens + t_max_embed.unsqueeze(1)
        class_tokens = self.class_tokens + y_embed.unsqueeze(1)

        return torch.cat(
            [
                class_tokens,
                omega_tokens,
                t_min_tokens,
                t_max_tokens,
                time_tokens,
                x_embed,
            ],
            dim=1,
        )

    def _run_attn_res_blocks(self, prefix, snaps, blocks, start_idx):
        """Run blocks under the Kimi-K3 attention-residual schedule.

        Mirrors KimiDecoderLayer._forward_attn_residual (see the record).
        With grad_checkpoint, each block iteration runs as ONE functional
        checkpoint segment (inputs: prefix + snapshot tensors), so neither
        the sublayer activations nor the attention-residual candidate
        stacks are retained for backward -- essential at 29k-token
        sequences, where one stacked (n, J, d) candidate copy is ~0.7 GB.

        Args:
            prefix: (b, l, d) running residual prefix sum entering blocks.
            snaps: list of (b, l, d) earlier snapshots (not mutated).
            blocks: iterable of TransformerBlock.
            start_idx: global index of blocks[0] (snapshot schedule is
                indexed globally across shared trunk and head branch).

        Returns:
            (prefix, snaps): (b, l, d) final prefix sum and the extended
            snapshot list (a new list; the input list is left untouched).
        """
        bs = self.attn_res_block_size
        eps = self.rmsnorm_eps
        snaps = list(snaps)
        for j, block in enumerate(blocks):
            fresh = (start_idx + j) % bs == 0

            def step(prefix, *snaps_t, blk=block, fr=fresh):
                # prefix (b,l,d); snaps_t: tuple of (b,l,d) snapshots
                snl = list(snaps_t)
                if snl:
                    h1 = attn_res_apply(
                        prefix, snl, blk.attn_res_norm.weight,
                        blk.attn_res_proj.weight, eps,
                    )                                 # (b, l, d) remixed
                else:
                    h1 = prefix
                a = blk.attn(
                    blk.norm1(h1), self.rope_cos, self.rope_sin
                ) * blk.attn_scale                    # (b, l, d) gated attn
                if fr:
                    snl2 = snl + [prefix]             # snapshot pre-block sum
                    p2 = a                            # prefix restart
                else:
                    snl2 = snl
                    p2 = prefix + a
                h2 = attn_res_apply(
                    p2, snl2, blk.mlp_res_norm.weight,
                    blk.mlp_res_proj.weight, eps,
                )                                     # (b, l, d) remixed
                m = blk.mlp(blk.norm2(h2)) * blk.mlp_scale
                return p2 + m                         # (b, l, d)

            args = (prefix, *snaps)
            if self.grad_checkpoint and torch.is_grad_enabled():
                new_prefix = torch.utils.checkpoint.checkpoint(
                    step, *args, use_reentrant=False
                )
            else:
                new_prefix = step(*args)
            if fresh:
                snaps = snaps + [prefix]
            prefix = new_prefix
        return prefix, snaps

    def _out_res_apply(self, prefix, snaps, gain_mod, proj_mod):
        """Output-side attention residual, checkpointed under grad_checkpoint.

        Args:
            prefix: (b, l, d) final prefix sum;  snaps: list of (b, l, d).
            gain_mod: RMSNorm module;  proj_mod: Linear(d, 1) module.

        Returns:
            (b, l, d) remixed stream.
        """
        def apply(prefix, *snaps_t):
            return attn_res_apply(
                prefix, list(snaps_t), gain_mod.weight, proj_mod.weight,
                self.rmsnorm_eps,
            )

        if self.grad_checkpoint and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                apply, prefix, *snaps, use_reentrant=False
            )
        return apply(prefix, *snaps)

    def forward(self, x, t, h, w, t_min, t_max, y):
        """
        Forward pass of the video imfDiT model.

        Args:
            x: Input video latents (B, C, T, H, W)
            t, h: time step and time difference t - r, shape (B,)
            w: CFG scale, shape (B,)
            t_min, t_max: CFG interval, shape (B,)
            y: Class labels, shape (B,)

        Returns:
            u: Average velocity field (B, C, T, H, W)
            v: Instantaneous velocity field (B, C, T, H, W)
        """
        # We don't explicitly condition on time t, only on h = t - r
        # following https://arxiv.org/abs/2502.13129
        seq = self._build_sequence(x, h, w, t_min, t_max, y)
        if self.tile_major:
            # one permute for the whole forward: (B, L, d) model order ->
            # 3D-tile order; every per-token op is order-agnostic and the
            # RoPE tables were permuted at build time.
            seq = seq[:, self.tm_perm]

        if self.attn_res_block_size > 0:
            shared_depth = len(self.shared_blocks)
            prefix, snaps = self._run_attn_res_blocks(
                seq, [], self.shared_blocks, 0
            )
            u_prefix, u_snaps = self._run_attn_res_blocks(
                prefix, snaps, self.u_heads, shared_depth
            )
            u_seq = self._out_res_apply(
                u_prefix, u_snaps, self.u_out_res_norm, self.u_out_res_proj
            )
            if len(self.v_heads) > 0:
                v_prefix, v_snaps = self._run_attn_res_blocks(
                    prefix, snaps, self.v_heads, shared_depth
                )
                v_seq = self._out_res_apply(
                    v_prefix, v_snaps, self.v_out_res_norm, self.v_out_res_proj
                )
            else:
                v_seq = u_seq
        else:
            def run_block(blk, z):
                if self.grad_checkpoint and torch.is_grad_enabled():
                    return torch.utils.checkpoint.checkpoint(
                        lambda zz: blk(zz, self.rope_cos, self.rope_sin),
                        z, use_reentrant=False,
                    )
                return blk(z, self.rope_cos, self.rope_sin)

            for block in self.shared_blocks:
                seq = run_block(block, seq)

            u_seq = v_seq = seq
            for block in self.u_heads:
                u_seq = run_block(block, u_seq)

            for block in self.v_heads:
                v_seq = run_block(block, v_seq)

        if self.tile_major:
            # back to model token order before the position-dependent
            # prefix slice + unpatchify
            u_seq = u_seq[:, self.tm_inv]
            v_seq = v_seq[:, self.tm_inv]
        u_tokens = u_seq[:, self.prefix_tokens :]
        v_tokens = v_seq[:, self.prefix_tokens :]

        u = self.unpatchify(self.u_final_layer(u_tokens))
        v = self.unpatchify(self.v_final_layer(v_tokens))

        return u, v


#################################################################################
#                             iMF Video DiT Configs                              #
#################################################################################


def imf_dit_video_S(num_classes, **kwargs):
    return IMFDiTVideo(
        hidden_size=384,
        depth=8,
        num_heads=6,
        aux_head_depth=4,
        num_classes=num_classes,
        **kwargs,
    )


def imf_dit_video_B(num_classes, **kwargs):
    return IMFDiTVideo(
        hidden_size=768,
        depth=12,
        num_heads=12,
        aux_head_depth=8,
        num_classes=num_classes,
        **kwargs,
    )
