# Kimi-K3 Attention Residual: Model Integration + Fused Triton JVP Kernel

## Mechanism (ported from Kimi-K3 modeling_kimi_linear.py)

- Source: `KimiDecoderLayer._forward_attn_residual` (lines 1061-1135), `_apply_attn_res` (1164-1177), model-level output apply (1321-1338) in `/home/downeyflyfan/Research_Projects/AI/LM/Kimi-K3/modeling_kimi_linear.py`.

- Per token, the residual stream keeps a running prefix sum plus snapshots taken every `attn_res_block_size` blocks. Before each attention and each MLP sublayer, a 1-query softmax attention over [snapshots..., prefix] re-mixes the stream; at snapshot blocks the prefix restarts from the sublayer output (history lives in the snapshots). A final output-side apply mixes the stream before the head.

- Tensor pseudocode of one apply (all symbols and variables first):

```python
# Shape symbols:
#   n  token rows (batch * seq);  J  candidates = snapshots + 1;  d  hidden
# Variables:
#   v(n,J,d)   candidate stack [snapshots..., prefix sum]
#   dv(n,J,d)  input tangents (JVP; parameter tangents are zero)
#   g(d)       RMSNorm gain;  pw(d)  Linear(d,1) score projection weight
#   w(d) = g * pw   fused score direction
#   r(n,J)     per-candidate RMS;  s(n,J) scores;  p(n,J) softmax probs
#   y(n,d)     mixed stream;  dy(n,d) its tangent

r  = sqrt(mean(v**2, -1) + eps)                    # (n, J)
s  = einsum("njd,d->nj", v, w) / r                 # (n, J)
p  = softmax(s, -1)                                # (n, J)
y  = einsum("nj,njd->nd", p, v)                    # (n, d)
ds = einsum("njd,d->nj", dv, w) / r \
     - s * mean(v * dv, -1) / r**2                 # (n, J)
dp = p * (ds - einsum("nj,nj->n", p, ds)[:, None]) # (n, J)
dy = einsum("nj,njd->nd", dp, v) \
     + einsum("nj,njd->nd", p, dv)                 # (n, d)
```

## Model integration (`models/imf_dit_video.py`)

- `attn_res_apply(prefix, snaps, gain, proj_w, eps)` functional helper (torch, forward-AD compatible).

- `TransformerBlock(use_attn_res=True)` adds `attn_res_norm`, `mlp_res_norm` (RMSNorm(d)) and `attn_res_proj`, `mlp_res_proj` (Linear(d, 1, bias=False), scaled-variance init).

- `IMFDiTVideo(attn_res_block_size=k)`: `_run_attn_res_blocks` runs the Kimi schedule with a global block index across the shared trunk and each head branch (u/v heads continue from the shared prefix and a copied snapshot list); per-head output applies `u/v_out_res_norm`, `u/v_out_res_proj` before the FinalLayers. `attn_res_block_size=0` restores the previous architecture exactly.

- `config.py`: `ModelConfig.attn_res_block_size = 4` (0 disables); threaded through `train.py` `build_model()`.

- Parameter count at config defaults: 264.8M (cap 300M). Snapshot count at depth 19, block size 4: 5 snapshots -> J <= 6.

- Verified: full-model forward and `torch.func.jvp` finite with `flash_jvp_attention` at config defaults.

## Triton kernel (`models/triton_attn_res_jvp.py`)

- `triton_attn_res_jvp(v, dv, w, eps)` -> (y, dy) and `triton_attn_res_fwd(v, w, eps)` -> y; one program per token row, two register-resident passes over the J candidates (statistics + scores, then the probability-weighted mix; second pass re-reads V/DV from L2). Softmax over a JP = next_power_of_2(J) slot vector with -inf masking; J <= 32.

- Eager torch under `torch.func.jvp` runs ~10 kernels and materializes the (n, J, d) fp32 stack for primal and tangent separately; the fused kernel is one launch.

## Results

- Correctness: rel Frobenius error vs fp64 eager jvp < 1e-5 at (n, J) in {(512, 6), (333, 2), (256, 1), (128, 20)}; bit-identical primal between jvp and fwd-only entry points; matches `attn_res_apply` on the list API < 1e-6.

- Speed (median CUDA-event ms, RTX 5070 Ti, fp32, d = 1024):

| (n, J) | eager fp32 jvp | Triton jvp | speedup | Triton fwd-only |
|---|---|---|---|---|
| (8192, 6) | 6.981 ms | 0.763 ms | 9.15x | 0.323 ms |
| (8192, 3) | 3.800 ms | 0.378 ms | 10.06x | 0.174 ms |
| (16384, 6) | 13.927 ms | 1.489 ms | 9.36x | 0.637 ms |
| (1680, 6) | 1.403 ms | 0.132 ms | 10.63x | 0.037 ms |

- Tests: `tests/test_attn_res.py` 6/6 passed (fp64 reference at four (n, J) shapes including J = 1 and J = 20, model-helper equivalence, speed assert >= 3x). Existing suites unaffected: `tests/test_triton_mla_block_jvp.py` 9/9.

## Files

- `models/imf_dit_video.py` — `attn_res_apply`, block/`IMFDiTVideo` integration

- `models/triton_attn_res_jvp.py` — fused kernel

- `tests/test_attn_res.py` — correctness + speed; `__main__` benchmark

- `config.py` — `attn_res_block_size` knob; `train.py` — wiring

## Training-path integration (autograd + forward AD)

- `attn_res_op` (`models/triton_attn_res_jvp.py`): `torch.autograd.Function` with the fused Triton forward, a fused Triton backward kernel (input gradients; the score-direction gradient reduces the per-row gs/r workspace with one matmul: `gw = einsum("x,xd->d", gsr, v)`), and a functorch `jvp` staticmethod calling the fused primal+tangent kernel (supports a nonzero score-direction tangent). `imf_dit_video.attn_res_apply` dispatches to it for CUDA fp32; eager math remains the fallback and the reference.

- Gradients and jvp verified against fp64 eager autograd (rel Frobenius < 1e-5), including gradients into snapshots, prefix, norm gain and projection (`tests/test_attn_res.py`, 9/9 passed).

- 400-step Wan-Syn run with the kernel: loss trajectory IDENTICAL to the eager attention-residual run at every logged step (same seeds, numerically equivalent op).

- Throughput/memory, matched probe harness (same session, idle GPU, 50 timed steps, micro-batch 4 x grad_accum 2 unless stated):

| mode | samples/s | ms/step | peak memory |
|---|---|---|---|
| eager attn-res, accum | 6.56 | 1220 | 13.14 GiB |
| Triton attn-res, accum | 7.28 | 1099 | 8.02 GiB |
| Triton attn-res, batch 8 (no accum) | 8.63 | 927 | 12.05 GiB |

- Batch 8 without accumulation OOMs on 16 GB with the eager op (it stores the RMS-normed stack, scores and probabilities in the autograd graph per apply) and fits with the Triton op (saves only the candidate stack and the fused direction): +32% throughput over the eager configuration that fits.

- Absolute samples/s are not comparable across sessions on this desktop: concurrent CPU load from unrelated jobs (load average 13-16 during these probes) shifts the partly CPU-bound training loop; the earlier eager training run logged 10.6 samples/s at lower background load. Within-session A/B: Triton 7.3-7.4 vs eager 6.4-6.6 samples/s over 100 real `train.py` steps, GPU clocks pinned at 2790-2850 MHz (no thermal throttling).
