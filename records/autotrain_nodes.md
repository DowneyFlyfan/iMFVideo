# Auto-Tuning Node Sweep + Continuous Training (Wan-Syn crops, 1x RTX 5070 Ti)

## Protocol

- Node = 400 optimizer steps, 8 samples/step (micro-batch 4 x grad_accum 2), same data and seed 0 unless stated, fresh init per tuning node. Metric: window means of raw `loss_u`; stability: spike count (loss_u > 1.5x run median) and max grad norm.

- Architecture throughout: 288.9M MLA DiT with Kimi-K3 attention residual (block size 4), SituAndMul (beta 4, linear_beta 25), MLA output gate, Triton attn-res kernel path, `flash_jvp_attention`.

## Tuning nodes (fresh init, one knob per node)

| node | change | loss_u windows (10-100 / 110-200 / 210-300 / 310-400) | spikes | gn_max |
|---|---|---|---|---|
| 01 | baseline lr 1e-4, wd 1e-5, warmup 40 | 1.431 / 0.897 / 0.862 / 0.844 | 4 | 6.2 |
| 02 | wd 0.1 (Moonlight) | 1.435 / 0.939 / 0.819 / 0.736 | 5 | 5.8 |
| 03 | + lr 2e-4 | 1.269 / 0.856 / 0.690 / 0.598 | 8 | 13.3 |
| 04 | + lr 3e-4 | 1.214 / 0.756 / 0.602 / 0.528 | 8 | 9.7 |
| 05 | + lr 5e-4 | 1.126 / 0.699 / 0.557 / 0.487 | 8 | 6.1 |
| 06 | + lr 1e-3 | 1.047 / 0.709 / 0.565 / 0.464 | 10 | 7.0 |
| 07 | lr 5e-4, warmup 100 | 1.264 / 0.740 / 0.573 / 0.496 | 9 | 6.7 |
| 08 | lr 5e-4, clip 0.5 | 1.121 / 0.688 / 0.559 / 0.487 | 8 | 7.1 |
| 09 | lr 5e-4, attn-res OFF | 1.089 / 0.872 / 0.608 / 0.556 | 7 | 2.8 |
| 10 | node-05 config, seed 1 | 1.050 / 0.678 / 0.564 / 0.574 | 8 | - |
| 11 | lr 7e-4 | 1.069 / 0.694 / 0.565 / 0.471 | 8 | - |
| 12 | lr 5e-4, attn-res block size 2 | (210-400 mean 0.5195) | 9 | - |

- Seed replica (node 10 vs 05): final-window spread 0.487 vs 0.574 -> noise floor ~0.09 on single windows; decisions above rest on multi-window trends (lr ladder cumulative -0.36) or larger deltas.

- Adopted into `config.py` (commit f50b35f): lr 5e-4, weight_decay 0.1. Null results (kept old values): warmup 100, clip 0.5, attn-res block size 2; lr 7e-4 / 1e-3 within noise of 5e-4. Attention-residual ablation at tuned lr: 0.556 vs 0.487 final window -> mechanism retained.

## Continuous chain (resume every 400 steps, tuned config)

| node | steps | first / last window loss_u | spikes |
|---|---|---|---|
| C01 | 1-400 | 1.126 / 0.487 | 8 |
| C02 | 401-800 | 0.535 / 0.429 | 4 |
| C03 | 801-1200 | 0.462 / 0.406 | 2 |
| C04 | 1201-1600 | 0.421 / 0.383 | 2 |
| C05 | 1601-2000 | 0.393 / 0.357 | 3 |
| C06 | 2001-2400 | 0.373 / 0.335 | 3 |
| C07 | 2401-2800 | 0.362 / 0.319 | 4 |
| C08 | 2801-3200 | 0.347 / 0.307 | 4 |

- Curve: `records/autotrain_continuous_loss.png`; raw series `records/autotrain_continuous_history.json`. Chain continues; checkpoints in `.cache/autotrain_ckpt/` (latest two kept).
