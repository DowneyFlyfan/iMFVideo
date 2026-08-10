# Full Uncropped Wan-Syn Training (1x RTX 5070 Ti)

## Data

- 56 full Wan-Syn latents, shape (C=16, T=20, H=56, W=104) each -- NO crops; per-channel normalized (stats over the set), pseudo-label = md5(caption) mod 1000; `.cache/wan_syn_full/`.

- Tokens per sample: 20 x 28 x 52 patches + 18 conditioning = 29,138 (139x the 210-token crop runs).

## Settings

- 288.9M MLA DiT (rectangular input_size (56, 104) support added), attention residual (block size 4), `flash_jvp_attention`, fast detached Triton du/dt engine (`jvp_impl="fast"`), gradient checkpointing per block iteration (functional segments: neither sublayer activations nor attention-residual candidate stacks retained; one stacked candidate copy alone is ~0.7 GB at this length).

- Optimizer: moonlight, lr 5e-4, wd 0.1, warmup 40, WSD schedule with decay_fraction 0.5 (flat to step 200, 1-sqrt tail to 5e-5); micro-batch 1 x grad_accum 2 (2 samples/step); 400 steps; seed 0; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

- Memory: peak 11.57 GiB. Without per-iteration checkpointing the first step OOMs on 16 GB.

## Results

- loss_u window means (10-100 / 110-200 / 210-300 / 310-400): 1.1708 / 0.8333 / 0.6134 / 0.7798; spikes (>1.5x median) 7.

- Step milestones: step 10 = 1.9200, step 100 = 0.7846, step 250 = 0.4673, step 400 = 0.6979. Per-step values are 2-sample estimates and swing 0.3-1.2 late in the run.

- Wall time: ~3.6 h for 400 steps (~33 s/step nominal; segments slowed to ~50 s/step while unrelated jobs held CPU load average > 30 on the 20-core box -- GPU was exclusive throughout).

- Curve: `records/wan_syn_full_train_loss.png`; raw series `records/wan_syn_full_train_history.json`.

- Not directly comparable to the crop-run records: different token count, per-step sample count (2 vs 8), and data distribution (full 480p latents vs 16x16 crops).
