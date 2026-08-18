# Wan2.2-Syn 32k Full Training, Run 1 (4x A100, option c)

## Data

- `FastVideo/Wan2.2-Syn-121x704x1280_32k`: 33,336 latents, shape (C=48, T=31, H=44, W=80), Wan2.2-5B high-compression VAE; streaming-converted with frozen 32-chunk bootstrap normalization stats; `.cache/wan22_full/` (~700 GB, persistent CephFS volume).

- Tokens per sample: 31 x 22 x 40 patches + 18 conditioning = 27,298. Cube tile (1, 2, 8) = 16 (grid (31, 22, 40); 31 prime, only pow-2 tile), multi-block prefix tails N_TAIL = 2, int8 PV falls back to fp16 at E = 16 (int8 MMA needs inner dim >= 32; QK stays int8).

## Settings

- 290.1M MLA DiT (hidden 1024, depth 19, heads 16, in_channels 48), attn `sla2_cube_qat` topk 0.03, tile-major residency, bf16 autocast, TF32, grad checkpointing.

- Rules-derived parameters: lr 8e-4 (2e-3 x (K/4k)^(-1/4), K = 160k samples), warmup 250 (2.5%), total 10,000 steps at global batch 16 (bs 4/GPU), WSD decay_fraction 0.2 (1-sqrt), P_std 0.8, token/embedding init constants 32 = sqrt(d), per-layer-class lr width factors (Muon hidden 0.5, Muon res score heads 0.25, AdamW output projections 0.25, Theta(1) classes 1.0).

- Ops: `num_workers` 4/rank (94/rank exhausted the 16 GiB pod `/dev/shm` and crashed the first launch attempt; 510 leaked workers held unlinked shm segments), Moonlight with per-shape NS coefficient cache, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, launched via kubectl exec (pod sshd defunct after reschedule).

## Results

- Throughput: 0.5 samples/s = 32 s/step, constant over the whole run.

- loss_u_ema trajectory (step: value): 50: 1.585, 250: 1.310, 500: 0.939, 1000: ~0.66, 1450: 0.588, 2000: 0.550, 2950: 0.521, 3400: 0.502, 3900: 0.501. Oscillation band ~0.50-0.58 after step 1800; grad_norm 0.6-1.14 with clip 1.0 engaging on spikes at steps 850, 2000, 2350, 3550, 3700.

- step 3900: loss_u 0.4636, loss_u_ema 0.5009, grad_norm 0.684. step 3950: all quantities nan (loss, loss_u, loss_u_ema, loss_v, grad_norm). Single 50-step window from healthy to full NaN, no preceding grad_norm ramp, lr constant 8e-4 (stable phase).

- No checkpoint existed (ckpt_every 5000, first save would have been step 5000). Run killed at step 3950; log archived as `train_full_nan3950.log` on the pod.
