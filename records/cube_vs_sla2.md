# Cube-block (VSA 3D tiles) vs pure SLA2 (1D blocks): approximation test

- Question: at equal compute budget, does the cube-block sparse-linear attention (models/mla_video_sparse_jvp.py) approximate full attention better than pure SLA2 1D blocks (models/sla2_mla_jvp.py)?

## Setting

- Post-hoc attention-approximation test on one dense model; no sparse model is trained. Server: 2x A100 80GB (ssh ABA), torch 2.7.1+cu126, triton 3.3.1.

- Dataset: Wan-Syn WanVAE latents (16, 20, 56, 104) per video; 200 on the server (198 train / 2 harness-eval); comparison captures from 4 videos.

- Model: tuned iMF DiT (hidden 256, depth 19 = 15 shared + 4 u + 4 v blocks, 4 heads, head_dim 64, MLA + partial RoPE, 18.6M params), dense bf16 flash attention (`sdpa_flash`), trained 150 steps at bs 2 (~1.5 epochs, lr 2e-3, warmup 100, WSD; final train loss_u 0.65, probe 0.55). Random-init capture also measured as a control.

- Captured tensors: post-RoPE q/k/v at shared blocks {0, 9}: (1, 4, 29138, 64) = (batch, heads, 18 prefix + 29120 patch tokens, head dim). Reference: full softmax attention, same shape.

- Operators on identical q/k/v at matched routed budget (same selected-token count, 64-token key blocks): SLA2-1D (bq 128 / bk 64 raster blocks, pooled router) vs cube (4x4x4 tiles = 455 blocks, SLA2 router on tiles, always-attended prefix tail). Both mix a sparse branch with the complement linear branch by scalar $a$.

- Metrics: softmax-mass recall of the routed sets (higher better) and relative Frobenius output error vs full attention at $a = 1$ (pure sparse) and $a = 0.9$ (lower better).

## Results (trained weights, mean over 4 videos)

| block | budget | recall SLA2 / cube | err a=1 SLA2 / cube | err a=0.9 SLA2 / cube |
|---|---|---|---|---|
| 0 | 3% | 0.071 / **0.114** | **0.719** / 0.734 | 0.641 / **0.639** |
| 0 | 10% | 0.199 / **0.292** | **0.514** / 0.595 | **0.451** / 0.508 |
| 0 | 25% | 0.412 / **0.537** | **0.375** / 0.418 | **0.322** / 0.342 |
| 9 | 3% | 0.070 / **0.112** | 0.677 / **0.673** | 0.611 / **0.592** |
| 9 | 10% | 0.212 / **0.302** | **0.498** / 0.566 | **0.443** / 0.491 |
| 9 | 25% | 0.438 / **0.558** | **0.373** / 0.400 | **0.324** / 0.340 |

## Results (random init, control; mean over 4 videos, blocks pooled)

| budget | recall SLA2 / cube | err a=1 SLA2 / cube | err a=0.9 SLA2 / cube |
|---|---|---|---|
| 3% | 0.041 / 0.055 | 1.10 / 1.27 | 0.98 / 1.12 |
| 10% | 0.132 / 0.168 | 0.89 / 1.06 | 0.78 / 0.92 |
| 25% | 0.310 / 0.371 | 0.63 / 0.81 | 0.53 / 0.67 |

## Reading

- Routing: cube captures 30-60% more attention mass than 1D blocks at every budget on trained weights (e.g. 0.114 vs 0.071 at 3%), and the recall gap grew from random init to 150 trained steps.

- Output error: at the production-relevant 3% budget (the SLA2 default sparsity regime) cube is equal to slightly better (block 9: 0.592 vs 0.611 at a=0.9); at 10% and 25% budgets the 1D raster blocks give lower output error despite lower recall — the raster diagonal band appears to capture the error-dominant peaked part of each row's distribution.

- The linear complement lowers error for both operators at every setting (a=0.9 < a=1 columns).

- Caveats: 150-step attention structure only; 4 videos, 2 blocks, one seed; approximation metrics, not end-task training loss. The 10-25% inversion (more mass, more error) is unexplained and worth a per-row error breakdown before drawing design conclusions at those budgets.
