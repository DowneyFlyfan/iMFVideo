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
