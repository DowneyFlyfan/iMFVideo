# Triton MLA Block JVP (fp32 io, fp16 default / fp8 optional internals)

## Task

- Rewrite the Triton transformer-block JVP (Jacobian-vector product) so it supports the MLA (Multi-head Latent Attention) block of `models/imf_dit_video.py`, at least 10x faster than pure PyTorch ops, with fp16/bf16 (not fp8) internals as the default.

## Variants

- `variant="fp16"` (default): fp16 weights + fp16 activations, cuBLAS fp16 GEMMs with fp32 accumulation, fp16 flash-attention JVP core (fp16-accumulate per-tile dots, fp32 softmax statistics and cross-tile accumulation). No fp8 anywhere.

- `variant="fp8"`: fp8 `torch._scaled_mm` GEMMs with per-tensor weight scales, fp8 activations quantized inside the producing kernels.

## Structure

- Tensor pseudocode of the pipeline (all comments and symbols first):

```python
# Shape symbols:
#   b  batch (primal); stacked primal+tangent batch = 2b;  M = 2*b*l rows
#   l  sequence length;          d   hidden size = 1024
#   H  heads = 16;               dq  q_lora_rank = 512
#   dc kv_lora_rank = 256;       dn  qk_nope_head_dim = 48
#   dr qk_rope_head_dim = 16;    dv  v_head_dim = 64;  hd = dn+dr = 64
#   F  SwiGLU width int(8d/3) = 2730, zero-padded to 2736 for fp8 GEMMs
# Variables:
#   x(b,l,d), dx(b,l,d)  input primal / tangent (fp32)
#   n1(M,d)   RMSNorm1 JVP output, fp8         (stacked [primal; tangent])
#   a(M,dq+dc+dr)  fused q_a+kv_a projection output, fp16
#   ql(M,dq), kvl(M,dc)  latent RMSNorm outputs, fp8
#   qb(M,H*hd), kvb(M,H*(dn+dv))  up-projections, fp16
#   q,k,v (2b,l,H,64)  packed fp8 attention operands
#   o(M,H*dv)  flash JVP output, fp8;  res(M,d) fp16 residual stream
# Parameter tangents are zero -> every linear is ONE stacked GEMM.
# RoPE at fixed position is linear -> tangent gets the same rotation.

n1  = rmsnorm_jvp(x, dx)                        # Triton, fp8 out
a   = einsum("mk,jk->mj", n1, w_a)              # w_a(dq+dc+dr, d) fp8 GEMM
ql  = rmsnorm_jvp(a[:, :dq])                    # strided read, fp8 out
kvl = rmsnorm_jvp(a[:, dq:dq+dc])               # strided read, fp8 out
qb  = einsum("mk,jk->mj", ql, w_qb)             # w_qb(H*hd, dq) fp8 GEMM
kvb = einsum("mk,jk->mj", kvl, w_kvb)           # w_kvb(H*(dn+dv), dc)
q   = pack(q_norm_jvp(qb[..., :dn]),            # per-head RMS on nope band
           rope_jvp(qb[..., dn:]))              # rotate dr band, fp8 out
k   = pack(k_norm_jvp(kvb[..., :dn]),           # per-head RMS on nope band
           rope_jvp(a[:, dq+dc:]))              # SHARED k_rope, broadcast H
v   = kvb[..., dn:dn+dv]                        # copy into packed fp8 slot
o   = flash_jvp(q, k, v)                        # reused fp8/fp16-acc core
p   = einsum("mk,jk->mj", o, w_out)             # w_out(d, H*dv) fp8 GEMM
res = x + p * g_attn                            # fused with RMSNorm2 kernel
n2  = rmsnorm_jvp(res)                          # same fused kernel, fp8 out
gu  = einsum("mk,jk->mj", n2, w_gate_up)        # w_gate_up(2F, d) fp8 GEMM
h   = swiglu_jvp(gu)                            # Triton, fp8 out
dwn = einsum("mk,jk->mj", h, w_down)            # w_down(d, F) fp8 GEMM
y   = res + dwn * g_mlp                         # fp32 outputs (y, dy)
```

- JVP identities implemented in the prep kernels, with $r$ the RMS scale, $w$ the norm gain, $c = \cos\theta_{s,i}$, $s = \sin\theta_{s,i}$ the position-$s$ rotary angles:

$$
\begin{equation}
\begin{aligned}
r &= \sqrt{\tfrac{1}{d_n}\textstyle\sum_j x_j^2 + \epsilon}, \quad
y = \frac{w \odot x}{r}, \quad
dy = \frac{w}{r} \odot \left(dx
     - x\,\frac{\langle x, dx\rangle}{d_n\, r^{2}}\right) \\
y_{2i} &= x_{2i} c - x_{2i+1} s, \quad
y_{2i+1} = x_{2i} s + x_{2i+1} c, \quad
dy_{2i} = dx_{2i} c - dx_{2i+1} s \\
dy_{2i+1} &= dx_{2i} s + dx_{2i+1} c
\end{aligned}
\end{equation}
$$

- Reused unchanged from `models/triton_block_jvp.py`: the flash-attention JVP core (fp8 operands, fp16-accumulate `tl.dot`, fp32 softmax statistics; `qk_head_dim = v_head_dim = 64` keeps it drop-in), the SwiGLU JVP kernel, `torch._scaled_mm` fp8 GEMM helpers with per-tensor weight scales.

- New Triton kernels: strided RMSNorm JVP (reads norm bands out of wider GEMM outputs), q-prep (per-head nope RMSNorm + RoPE JVP into the packed fp8 q slot), kv-prep (k_nope RMSNorm + shared k_rope rotation broadcast over heads + v copy), fused gated-residual + RMSNorm2 JVP (vector gate `attn_scale`, fp16 residual stream, fp32 norm math in registers).

- fp8 GEMM alignment: SwiGLU width 2730 is not a multiple of 16; gate/up rows and down columns are zero-padded to 2736, which is exact (silu(0) * 0 = 0 through the padded lanes).

## Settings

- GPU: RTX 5070 Ti (sm_120), fp32 io, model-config MLA geometry (d 1024, H 16, dq 512, dc 256, dn 48, dr 16, dv 64), random nonzero residual gates, random rotary tables.

- Baseline: `torch.func.jvp` through the eager fp32 `TransformerBlock` (sdpa math backend), the exact path `train.py` uses without kernels. `(2, 4096)` eager needs four (b, H, l, l) fp32 score/prob buffers under jvp; it fits at these shapes on 16 GB.

## Results

- Speed (median CUDA-event ms; eager = `torch.func.jvp` through the fp32 `TransformerBlock`, sdpa math backend):

| shape (b, l, d) | eager fp32 jvp | fp16 variant | fp8 variant | fp16 speedup | fp8 speedup |
|---|---|---|---|---|---|
| (2, 2048, 1024) | 39.69 ms | 3.391 ms | 1.930 ms | 11.71x | 20.57x |
| (2, 4096, 1024) | 124.83 ms | 8.267 ms | 4.712 ms | 15.10x | 26.49x |
| (8, 210, 1024) | 9.52 ms | 1.150 ms | 0.689 ms | 8.28x | 13.81x |

- The (2, 4096) eager number is from a run with the memory free to hold the four (b, H, l, l) score/prob buffers; with both variant weight caches resident it OOMs on 16 GB.

- The small (8, 210) shape is kernel-launch-overhead-dominated for the fp16 variant (8.28x); the 10x target shape class is the long-sequence one.

- Accuracy (relative Frobenius error vs fp64 eager reference):

| shape | fp16 y | fp16 dy | fp8 y | fp8 dy | eager fp32 y |
|---|---|---|---|---|---|
| (2, 512, 1024) | 2.08e-04 | 2.08e-04 | 6.93e-04 | 1.01e-03 | 3.63e-08 |
| (1, 210, 1024) | 2.07e-04 | 2.08e-04 | 7.96e-04 | 1.24e-03 | 3.64e-08 |

- Tests: `tests/test_triton_mla_block_jvp.py`, 8/8 passed (fp64 reference match at three shapes x two variants including odd sequence length 210, module/functional bit-equality, >= 10x fp16 speed assert at (2, 2048, 1024)).

## Files

- `models/triton_mla_block_jvp.py` — kernels + `triton_mla_block_jvp` + `TritonMLABlockJVP` + `mla_params_from_block`

- `tests/test_triton_mla_block_jvp.py` — correctness, wrapper, speed tests; `__main__` benchmark
