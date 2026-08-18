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

## Root cause (stress-battery forensics, 2026-08-18)

- Reproduction: with unit-scale random inputs the op is clean, but scaling q, k (and for some channels v or the tangents) by only ~20x makes fwd/bwd/JVP emit inf/nan while every observable one logging window earlier is healthy -- matching the step 3900 -> 3950 cliff with no grad_norm ramp. Battery: scratch `stress_nan.py`, cases per tensor and jointly at x10/20/30/50/70/100, bf16 autocast on and off.

- Four independent numeric channels, all triggered by attention logit / feature growth during training (no qk-norm in the model):

- Channel 1 (the likely killer): backward kernels recompute p = exp2(fp16 scores - lse) but lse comes from the INT8-quantized forward. The quantization mismatch grows with logit scale, the exponent goes spuriously positive, and the fp16 cast of p (and of ds = p(dp - delta)) becomes inf -> whole rows of dk/dv/dq -> all-reduce spreads it to every rank -> optimizer step poisons all weights -> next forward prints loss=nan. Fixed by clamping the exponent at +4 (true p <= 1, so inert when healthy) plus saturating the fp16 casts in `_sparse_bwd_dkdv_lut`, `_attn_bwd_dq`, `_attn_bwd_dkdv`.

- Channel 2: linear branch, den = phi(q) z + eps_l bottoms out at eps_l = 1e-5 when the phi channel a query selects has no global mass (phi saturates one-hot at large features); g = dOl/den ~ 1e5-1e7 then overflowed the fp16 dqphi/dkphi/dv buffers. Fixed with fp32 grad buffers, saturated g/dh casts, and a clamp before the final fp16 grad cast (clip_grad_norm bounds the magnitude afterwards).

- Channel 3: fused JVP linear-branch denominator used plain +eps_l while the complement state zc = ztot - routed can cancel slightly negative in fp32, letting den cross zero -> nan. Fixed with the vendored signed-epsilon form (|den| >= eps_l, sign kept), which `_lin_fwd2` already used.

- Channel 4: fused JVP score-tangent dots and the P V accumulation used out_dtype=float16 (fp16 accumulate) and overflow at 65504 once primals and tangents are jointly ~20x -- the same class as the 66a4e5b fp16 output-cast NaN. Fixed with fp32 accumulate (same HMMA tensor-core rate on A100) and a saturated fp16 cast of the tangent operand.

- After the fixes the entire battery is finite in fwd, bwd and JVP up to x100 (only 1000-sigma inputs that overflow the fp16 input tensors themselves remain non-finite), and `test_e16` passes with fused-vs-module rel 4e-4. Fix commit 933f227; run 2 hot-swapped onto the fixed kernels at its step-1000 checkpoint via config.run.resume.
