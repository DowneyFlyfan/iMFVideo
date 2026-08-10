# Sizing a 2-Hour Run: Model Size vs Latent Shape (1x RTX 5070 Ti, 16 GB)

## Question

- Pick a model size and a Wan-Syn subset size for a training run that finishes in 2 h (7200 s) on one RTX 5070 Ti, given that the tuning sweep (`records/loss_curve_tuning.md`) found optimizer-step count to be the dominant lever at fixed sample budget.

## Method

- Timed one full training step (loss forward, backward, grad clip, Moonlight step) at micro-batch 1, `grad_accum` 1, gradient checkpointing on, `jvp_impl="fast"`, master dtype float32. Two warmup steps discarded, then 3 to 12 timed steps depending on cost. Peak memory from `torch.cuda.max_memory_allocated`.

- Tokens per sample follow the patchifier, `patch_size = (1, 2, 2)`:

$$
\begin{equation}
\begin{aligned}
N_{\textbf{tok}} = T \cdot \frac{H}{2} \cdot \frac{W}{2} + 18,
\end{aligned}
\end{equation}
$$

- where $T$, $H$, $W$ are the latent frame count and latent spatial dims, and the 18 extra tokens are the conditioning banks (8 class + 4 time + 4 cfg + 2 interval).

## Hard constraint found: num_heads must be a power of two

- `head_dim = hidden_size / num_heads` must be 64 for the flash-JVP path, but that is not sufficient. `hidden_size` 384 (6 heads) and 768 (12 heads) both FAIL, in both du/dt engines:

| hidden | heads | head_dim | `jvp_impl="fast"` | `jvp_impl="functorch"` |
|---|---|---|---|---|
| 384 | 6 | 64 | CompilationError | NotImplementedError |
| 768 | 12 | 64 | CompilationError | NotImplementedError |
| 512 | 8 | 64 | OK | NotImplementedError |
| 256 | 4 | 64 | OK | NotImplementedError |

- The plain forward pass works at 6 and 12 heads; only the Triton du/dt kernel rejects them, at `BN = next_power_of_2(DN)`. The functorch column fails everywhere because `flash_jvp_attention` has no forward-mode rule, which is why the fast engine exists.

- Usable widths are therefore 256, 512 and 1024. Intermediate model sizes must come from DEPTH, not width.

## Measured step cost

- Seconds per step at micro-batch 1, and the steps that buys in 7200 s:

| params | config | 29,138 tok | 11,666 tok | 7,298 tok | 2,930 tok | 1,838 tok |
|---|---|---|---|---|---|---|
| 8.6M | 256 x 8 | 1.425 | 0.434 | 0.327 | 0.266 | 0.267 |
| 12.4M | 256 x 12 | - | - | 0.410 | 0.307 | - |
| 33.0M | 512 x 8 | - | - | 0.496 | 0.305 | - |
| 48.3M | 512 x 12 | 4.648 | 1.207 | 0.719 | 0.377 | 0.330 |
| 63.7M | 512 x 16 | - | - | 0.937 | 0.453 | - |
| 88.3M | 512 x 24 | - | - | 1.309 | 0.581 | - |
| 166.1M | 1024 x 10 | - | - | 1.509 | 0.622 | - |
| 288.9M | 1024 x 19 | (OOM here) | 4.861 | 2.738 | 1.077 | 0.781 |

- Steps available in 2 h (= 7200 / s-per-step), which is the quantity to maximize:

| params | 7,298 tok | 2,930 tok |
|---|---|---|
| 12.4M | 17,542 | 23,433 |
| 33.0M | 14,527 | 23,637 |
| 48.3M | 10,017 | 19,082 |
| 63.7M | 7,681 | 15,896 |
| 88.3M | 5,499 | 12,403 |
| 166.1M | 4,770 | 11,569 |
| 288.9M | 2,630 | 6,685 |

- The 288.9M entry at 29,138 tokens OOMed in this benchmark only because five models were built in one process and the allocator was fragmented; the real 4 h runs in `records/loss_curve_tuning.md` did fit at 10 to 14 GiB. It is far too slow for a 2 h budget either way at 4.648 s/step for even the 48.3M model.

## Two results that set the answer

- **Below about 33M, shrinking the model buys nothing.** 512 x 8 (33.0M) costs 0.3046 s/step against 0.3073 s/step for 256 x 12 (12.4M) at 2,930 tokens: 2.7x the parameters for free. Re-timed with 12 steps to confirm. At these sizes the GPU is launch- and memory-bound rather than FLOP-bound, and the wider matrices use the tensor cores better. The same shows at 1,838 vs 2,930 tokens for the 8.6M model (0.267 vs 0.266 s/step): shrinking the input below roughly 3,000 tokens is also free, meaning it is pure loss of information at no speed gain.

- **Batch size is NOT free**, so the "more steps" lever still applies. At 48.3M and 2,930 tokens:

| micro-batch | s/step | vs batch 1 | steps in 2 h | samples in 2 h | peak |
|---|---|---|---|---|---|
| 1 | 0.374 | 1.00x | 19,259 | 19,259 | 0.83 GiB |
| 2 | 0.531 | 1.42x | 13,567 | 27,135 | 1.23 GiB |
| 4 | 0.933 | 2.50x | 7,717 | 30,867 | 2.05 GiB |
| 8 | 1.819 | 4.87x | 3,958 | 31,665 | 3.79 GiB |
| 16 | 3.633 | 9.72x | 1,982 | 31,712 | 7.32 GiB |

- Only the first doubling is sublinear (1.42x cost for 2x samples); past batch 2 the cost is linear and samples per hour saturates near 31,700 while steps collapse 10x. Since the sweep showed steps beating samples at fixed budget, micro-batch 1 stays the default.

## Data available and what it costs

- Source: HuggingFace dataset `FastVideo/Wan-Syn_77x448x832_600k`, already VAE-encoded, so no encoding pass is needed. Latents are `(C=16, T=20, H=56, W=104)` float32 per video.

- Locally present: `Part_25` only, 7 chunks, **56 videos**. That is the entire local dataset; the 6,048 crops in `.cache/wan_syn_latents` are cut from those same 56 videos, so they add no new content.

- The repo holds hundreds of parts. A chunk is about 126 MB and holds 8 videos, so roughly **15.8 MB per video** to download. Free disk is 836 GB, so download time is the only real limit.

- Stored as half-resolution `(16, 20, 28, 52)` float32, a video is 1.86 MB on disk (0.93 MB in float16), so even a few thousand videos is a few GB locally.

## Recommendation from throughput alone

- **Latent shape: keep all 20 latent frames, halve the spatial dims to (28, 52)**, i.e. 7,298 tokens. Halving space costs 4x the tokens; cutting frames instead destroys the temporal structure that is the point of a video model. Only drop to `T=8` (2,930 tokens) if steps are worth more than temporal span. This part was not contradicted by the bake-off and is what the final runs used.

- **Dataset: 1,000 to 2,500 videos**, i.e. 125 to 313 chunks or 16 to 40 GB of download, so that a run seeing 10,000 to 20,000 samples makes a handful of passes rather than dozens. 2,000 videos were used below, giving 6.8 to 10.4 passes.

- Both settings are far inside the 300M parameter cap and use under 1.7 GiB, so the 16 GB card is not the constraint at this size; wall clock is.

- **The model-size half of this prediction was WRONG and is superseded by the next section.** It said `hidden_size` 512, `depth` 12, 48.3M parameters, reasoning that 33M is the speed floor and 48.3M buys 46% more capacity for 24% fewer steps. The bake-off found 48.3M to be the second worst of five arms. Throughput arithmetic did not predict the depth effect, which is the whole reason the arms were actually trained.

## Measured bake-off: depth is the axis that matters, not parameter count

- The throughput reasoning above was then TESTED. 2,000 Wan-Syn videos were downloaded and converted to half resolution, and five model shapes were trained at EQUAL WALL CLOCK (about 40 min each, micro-batch 1, 7,298 tokens, tuned knobs from `records/loss_curve_tuning.md`, 32 videos held out for the frozen probe). Each arm gets the step count its own speed allows, which is the real trade.

| arm | params | shape | s/step | steps in 40 min | final probe |
|---|---|---|---|---|---|
| E | 8.6M | 256 x 8 | 0.341 | 7,033 | 0.2400 |
| A | 33.0M | 512 x 8 | 0.526 | 4,637 | 0.2624 |
| D | 12.4M | 256 x 12 | 0.438 | 5,609 | 0.3384 |
| B | 48.3M | 512 x 12 | 0.773 | 3,198 | 0.4869 |
| C | 63.7M | 512 x 16 | 1.037 | 2,454 | 0.5626 |

- Sorting by DEPTH rather than by parameter count explains the table: both depth-8 arms (0.2400, 0.2624) beat both depth-12 arms (0.3384, 0.4869), which beat depth-16 (0.5626). Width is the weaker axis, and at fixed depth the narrower arm is only marginally ahead (0.2400 vs 0.2624 at depth 8, a gap near the noise band; 0.3384 vs 0.4869 at depth 12).

- Parameter count alone does NOT order the results: 33.0M beats 12.4M despite running 17% fewer steps. So this is not simply "more steps wins". Depth costs step time linearly and, at a budget of only a few epochs with a short schedule, does not pay that cost back.

- This overturns the depth-12 recommendation made earlier in this document from throughput reasoning alone. The revised pick is **depth 8**: `512 x 8` (33.0M) or `256 x 8` (8.6M).

- Between those two, prefer 512 x 8 (33.0M) when the budget may later grow, since it holds 4x the capacity for a 26% step-time penalty; prefer 256 x 8 (8.6M) for the tightest budget. Its max grad norm is much wilder though (108.9 vs 56.0), so the smaller arm is the less stable of the two.

- Curves: `records/model_size_bakeoff.png`; raw series `records/model_size_bakeoff_history.json`.

## The two depth-8 candidates at the real 2 h budget

- The 40 min bake-off left 8.6M and 33.0M within noise of each other, so both were re-run at the full budget on the same 2,000 videos. Each got about 7,000 s of training, so the step counts differ by the ratio of their speeds.

| arm | params | shape | steps | epochs over 1,968 train videos | wall | `probe_last3` | final probe |
|---|---|---|---|---|---|---|---|
| F | 33.0M | 512 x 8 | 13,292 | 6.8 | 115.2 min | 0.1427 | 0.1417 |
| G | 8.6M | 256 x 8 | 20,564 | 10.4 | 117.1 min | 0.1464 | 0.1462 |

- Final probe against wall clock, which is the axis the budget fixes:

| minutes | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 115 |
|---|---|---|---|---|---|---|---|---|
| 33.0M | 0.5786 | 0.5133 | 0.2487 | 0.1923 | 0.1713 | 0.1552 | 0.1472 | 0.1449 |
| 8.6M | 0.5672 | 0.3943 | 0.2349 | 0.1864 | 0.1847 | 0.1628 | 0.1473 | 0.1473 |

- They finish in a dead heat, 0.1417 vs 0.1462, a gap of 0.0045 that is far inside the noise band. The smaller model leads for the first hour and the larger one closes the gap and edges ahead after about 100 min, which is the expected shape: the extra capacity needs enough steps to be worth its slower step.

- Both roughly halve the 40 min result (0.2624 and 0.2400), so the budget itself was still the binding constraint at 40 min; neither model had saturated.

- **Pick 512 x 8 (33.0M).** It matches the 8.6M at 2 h, holds 4x the capacity if the budget ever grows, and its lead is widening rather than shrinking at the end of the run. Choose 256 x 8 (8.6M) only if the budget might shrink below about 1.5 h, where it is genuinely ahead.

- Curves: `records/model_size_2h.png`; raw series `records/model_size_2h_history.json`.

## Final recipe

- Data: 2,000 Wan-Syn videos, `python prepare_wan_syn.py download --videos 2000` then `convert --factor 2 --mode pool`, giving `(16, 20, 28, 52)` latents at 7,298 tokens in `.cache/wan_syn_half` (3.5 GB on disk, 29.5 GiB of parquet downloaded in 24 min at about 73 GiB/h).

- Model: `hidden_size` 512, `num_heads` 8, `depth` 8, `aux_head_depth` 3, 33.0M parameters.

- Optimizer and loss: the committed `config.py` defaults, i.e. the values adopted in `records/loss_curve_tuning.md`, with `warmup_steps` 100 and `total_steps` about 13,300 so the WSD tail lands at the end of the run.

- Batch: `batch_size_per_gpu` 1, `grad_accum` 1.

- Expect about 13,300 steps, 6.8 passes over the training split, 1.9 h, held-out probe near 0.142.

## Depth-19 shapes, priced against the full dataset

- Timed the same way (micro-batch 1, `grad_accum` 1, gradient checkpointing on, `jvp_impl="fast"`, float32 master, 2 warmup steps discarded then 8 timed at 7,298 tokens / 3 at 29,138):

| shape | params | tokens | s/step | peak | steps in 2 h |
|---|---|---|---|---|---|
| 256 x 19 | 18.6M | 7,298 | 0.569 | 0.79 GiB | 12,653 |
| 256 x 19 | 18.6M | 29,138 | 3.358 | 2.69 GiB | 2,144 |
| 1024 x 19 | 288.9M | 7,298 | 2.736 | 4.79 GiB | 2,631 |
| 1024 x 19 | 288.9M | 29,138 | 16.796 | 12.02 GiB | 429 |

- One pass over the full 600k-video Wan-Syn set at micro-batch 1 is 600,000 optimizer steps, so wall clock for one epoch on this single RTX 5070 Ti is:

| shape | tokens | 1 epoch |
|---|---|---|
| 256 x 19 | 7,298 | 94.8 h (4.0 days) |
| 256 x 19 | 29,138 | 559.7 h (23.3 days) |
| 1024 x 19 | 7,298 | 456.0 h (19.0 days) |
| 1024 x 19 | 29,138 | 2,799.3 h (116.6 days) |

- No convergence measurement exists at this scale. The longest run recorded here is 13,292 steps (6.8 passes over 1,968 videos) and its held-out probe was still descending at the end, so the step count required for a plateau on 600k videos is not bounded by any measurement in this repo.

## Caveats

- Every bake-off arm is a single seed. The 8.6M vs 33.0M gap (0.0224) is close to the seed-level noise measured in the tuning sweep, so those two should be treated as tied; the depth-8 vs depth-12 vs depth-16 ordering is far outside it.

- Ranking was established at a 40 min budget. Larger models improve for longer, so the ordering can shift at 2 h; the two depth-8 arms are therefore being re-run at the full budget.

- Probe values here are NOT comparable to those in `records/loss_curve_tuning.md`. Different data (2,000 half-resolution videos vs 56 full or 6,048 crops), so only within-table comparisons are valid.

- The half-resolution latents are average-pooled, which puts them off the Wan VAE decoder's manifold. Fine for studying training dynamics under a wall-clock budget; samples from such a model will not decode to clean video. Use `--mode crop` in `prepare_wan_syn.py` if decodable output matters more than field of view.
