# Wan-Syn Smoke Training: Attention Residual vs Baseline (1x RTX 5070 Ti)

## Settings

- Identical to `records/wan_syn_smoke_train.md` (same 6048 Wan-Syn latent crops, same seed 0, 400 steps, warmup 40, lr 1e-4 cosine, moonlight optimizer, EMA off, `flash_jvp_attention`), plus `attn_res_block_size = 4` (Kimi-K3 attention residual; 264.8M params vs 264.7M baseline).

- Memory: batch 8 with attention residual OOMs on 16 GB (snapshot history + per-apply (b, l, J, d) autograd intermediates); run used micro-batch 4 x grad_accum 2 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so each optimizer step averages the same 8 samples as the baseline.

## Results

- loss_u, mean over logged points per window:

| steps | baseline | attn-res |
|---|---|---|
| 10-100 | 1.3930 | 1.4596 |
| 110-200 | 0.9019 | 0.8575 |
| 210-300 | 0.8713 | 0.7582 |
| 310-400 | 0.8601 | 0.6937 |
| 210-400 | 0.8657 | 0.7260 |

- loss_v mean over steps 210-400: baseline 0.7554, attn-res 0.6832.

- Milestone values: step 100 loss_u 0.8995 (baseline) vs 1.3781 (attn-res); step 200 0.7987 vs 0.5188; step 400 0.7279 vs 0.7780.

- Throughput: baseline 15.8 samples/s, attn-res 10.6 samples/s; wall time for 400 steps: 203 s vs 302 s (1.49x longer; includes the grad_accum=2 split, which the baseline did not need).

- Curves: `records/wan_syn_smoke_attnres_vs_base.png`; raw series `records/wan_syn_smoke_attnres_history.json` vs `records/wan_syn_smoke_history.json`.
