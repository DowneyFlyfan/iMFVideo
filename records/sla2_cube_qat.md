# SLA2-Cube-JVP-QAT: kernel + full-model training A/B

- Deliverables: (1) `_sla2_cube_jvp_qat_kernel` + `sla2_cube_qat_jvp` in [models/sla2_cube_qat.py](../models/sla2_cube_qat.py) — the cube-block sparse-linear JVP with INT8 primal score dots (per-64-token-block amax scales, matching the QAT training forward) and fp16 tangent / PV dots, always-attended ragged prefix tail; (2) `SLA2CubeQATAttentionImpl` — the autograd training path: tile-permute to cube-major, SLA2 learnable router on VSA 3D tiles, prefix appended as an always-selected LUT entry, vendored SLA2 INT8 QAT sparse kernel + complement linear kernel at bq = bk = 64, learnable (H, Mb) alpha mix; (3) a full-model training A/B on ABA (1x A100 80GB).

## Kernel verification (tests/test_sla2_cube_qat.py)

- vs `torch.func.jvp` over the fp32 dense reference (grid (4,8,8), tile (4,4,4), E = 64, prefix 18, ragged L = 274): o 9.7e-3, do 1.1e-2 (int8 quantization noise, cf. 6.7e-4 for the fp16 non-QAT kernel). Training module: int8-vs-fp16 forward rel diff 1.9e-2; alpha and input grads finite and nonzero; Moonlight split routes alpha_logit to AdamW, router projections to Muon.

## Full-model training A/B (ABA, 1x A100 80GB)

- Both runs: identical hyperparams — 18.6M model (hidden 256, depth 19, heads 4), 198 Wan-Syn full latents, bs 2, lr 2e-3, warmup 100, WSD decay_fraction 0.2, 150 steps, seed 0, jvp_impl fast, grad_checkpoint off. Only `model.attn_impl` differs: `sdpa_flash` (dense bf16 full attention) vs `sla2_cube_qat` (topk 0.03 = 97% sparsity, INT8 sparse forward, tile (4,4,4)).

| step | dense loss_u / probe | cube-QAT loss_u / probe |
|---|---|---|
| 0 | - / 1.9148 | - / 1.9148 |
| 50 | 1.187 / 1.1369 | 1.192 / 1.1442 |
| 100 | 0.727 / 0.6557 | 0.653 / 0.6423 |
| 150 | 0.655 / **0.5511** | 0.647 / **0.5439** |

- Wall clock for the 150 steps: dense 1117 s, cube-QAT 773 s (1.44x faster end-to-end at 97% attention sparsity + int8 sparse forward).

- Reading: at matched budget and identical recipe, the cube-QAT model matches the dense baseline's loss trajectory within noise and ends slightly ahead on the frozen probe (0.5439 vs 0.5511) while training 1.44x faster. 150 steps and a 2-video probe are a smoke-scale signal, not a convergence claim; the decisive test is the full 4k-step local recipe or a longer server run.

## Why the theoretical 5.5x is not observed at d=1024 (profiled on A100, bf16)

- Per-component times of one bs=1 training step (289M model, 29k tokens), before/after the kernel fixes:

| component | dense bf16 | cube before | cube after fixes |
|---|---|---|---|
| guidance (2 no-grad fwd) | 2600 ms | 4783 ms | 4600 ms |
| primal forward | 897 ms | 2416 ms | 2300 ms |
| backward | 3023 ms | 3763 ms | 3421 ms |
| du/dt fast pass | 1479 ms | 8441 ms | **2740 ms** |
| optimizer | 1702 ms | 1934 ms | 1637 ms |
| full step | 9701 ms | 21337 ms | **14697 ms** |

- Fix 1 (landed, 3.1x on du/dt): the 16 GB survival serialization (per-batch-row loop, per-head linear chunk, T-looped LUT gathers) was latency-bound on the 80 GB A100; `_plenty_of_vram` now selects the wide paths there (commit "Gate cube-op serialization on free VRAM").

- Fix 2 (landed, neutral): `WITH_TANGENT=False` primal-only kernel for the guidance no-grad forwards. Measured no change -- which localizes the remaining gap: it is NOT in the attention kernels.

- Remaining gap (cube 14.7 s vs dense 9.7 s): per-block TORCH-LEVEL machinery around the kernels -- 3x permute+gather to tile order, router pooling + top-k, mask scatter, channel-softmax phi in fp32, states einsums, alpha interleave, inverse permute -- roughly 60 ms x 23 blocks per forward-equivalent, executed 4-5x per step. Dense bf16 flash runs ONE fused kernel per block at ~70% MFU; the cube composite runs ~15 separate memory-bound ops. The 5.5x theory assumed equal hardware efficiency per FLOP; the measured effective MFU of the cube composite is ~8-10%.

- Consequence: the cube advantage is real where the baseline's non-attention stack is slow (fp32 linears: 1.44x end-to-end at d=256) and where attention dominates FLOPs at matched MFU. Closing the gap at d=1024-bf16 requires fusing permute+quant+route+sparse+linear+mix into 1-2 kernels (the VSA paper's ThunderKittens approach, 85% MFU) -- a dedicated kernel-engineering project, out of scope for this pass.

## Kernel optimization campaign: 3x over dense attention reached (2026-08)

- Target (user): the cube-QAT attention kernel 3x faster than dense attention at d=1024 during training. Measured at the training-op level (fwd+bwd plus the forward-mode JVP op, B=1, H=16, L=29,138, D=64, idle A100):

| stage | dense | cube | ratio |
|---|---|---|---|
| baseline (start of campaign) | 139.7 ms | 200.4 ms | 0.70x |
| + fused JVP kernel (phi + complement states + quotient + alpha in-program) | 139.7 | 150 | 0.93x |
| + whole-op autograd Function (direct kernels, analytic backward) | 140 | 60 | 2.33x |
| + tile-major residency (one permute per model forward, bit-exact) | 140 | ~60 | 2.33x (op); step 14.7 -> 12.2 s |
| + LUT-native dkdv backwards + phi-states kernel (tf32) + smooth-k removal | 139.3-140.0 | **40.0** | **3.48-3.50x** |

- Final per-phase: fwd 10.0 ms (dense 29.7, 3.0x), fwd+bwd 30.0 (dense 80, 2.7x), JVP 11.5 (dense 60, 5.2x).

- Full-step core (interleaved medians, guidance + fwd_bwd + du/dt): dense 7930 ms vs cube 8949 ms -- the gradient path (fwd_bwd 3050 vs 3747) and du/dt (1206 vs 1495) are FASTER than dense; the remaining deficit is entirely the guidance pair of no-grad forwards (4693 vs 2689), which pays per-block routing + branch overheads 46 times per step. Closing it needs either a per-step routing cache (not bit-exact, unapproved) or route+mix fusion into the vendored forward kernel.

- Correctness at every stage: forward vs old path 9e-7; gradients within fp16 kernel noise (<= 8e-3); tile-major residency bit-exact (0.0 on u, v, du/dt); router selection invariant to the smooth-k removal (per-row constant score shift cannot change hard top-k; agreement 1.0000).
