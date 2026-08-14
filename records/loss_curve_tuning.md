# Loss-Curve Parameter Tuning (Wan-Syn crops, 1x RTX 5070 Ti)

## Why a frozen probe was needed first

- Several knobs under test change the training objective itself, not just the optimization path: $P_{\textbf{mean}}$, $P_{\textbf{std}}$, $p_{\textbf{data}}$ (the flow-matching proportion), $\beta_{\textbf{cfg}}$, $s_{\max}$ and $p_{\textbf{norm}}$ all move the law of the nuisance draws $(t, r, \omega, t_{\min}, t_{\max})$. The raw training `loss_u` of two such nodes is therefore not comparable: shrinking the guidance scale, or drawing more flow-matching samples, lowers the logged number without improving the model.

- Every node is instead scored by ONE fixed objective. A held-out batch of 32 latents (never trained on) is evaluated under the canonical distribution

$$
\begin{equation}
\begin{aligned}
P_{\textbf{mean}} = -0.4,
\quad
P_{\textbf{std}} = 1.0,
\quad
p_{\textbf{data}} = 0.5,
\\
\beta_{\textbf{cfg}} = 1.0,
\quad
s_{\max} = 7.0,
\quad
p_{\textbf{drop}} = 0.1,
\end{aligned}
\end{equation}
$$

- with a fixed RNG seed per eval batch, so the draws $(t, r, \omega, e, \textbf{label-drop})$ are bit-identical across every node in every round. The metric is `probe`, the same quantity `train.py` logs as `loss_u`, namely $\textbf{mean}((V - v_g)^2)$ over the eval samples. `probe_last3` is the mean of the final three probes.

- The RNG state is saved and restored around each probe, so probing does not perturb the training stream. The probe is verified objective-invariant: two nodes whose training objectives differ (`P_mean` -0.4 vs 0.8, `s_max` 7 vs 3) report a bit-identical step-0 probe of 2.0251.

- That this was necessary is shown directly by node A5. Raising $p_{\textbf{data}}$ to 0.75 gives the LOWEST training loss of round 1 (`train_late` 0.5500 vs the baseline 0.5852) while its probe is clearly WORSE (0.5954 vs 0.5834): the extra flow-matching samples make the logged objective easier, not the model better.

## Protocol

- Node = fresh init, 400 optimizer steps, 8 samples/step (micro-batch 4 x grad_accum 2), Wan-Syn crops `(C=16, T=3, H=16, W=16)`, 6016 training crops with 32 held out for the probe, seed 0 unless stated, warmup 40, `total_steps` set to the node's step count so the WSD tail lands inside the run.

- Architecture throughout: 288.9M MLA DiT, attention residual (block size 4), SituAndMul, MLA output gate, `flash_jvp_attention`, fast Triton du/dt engine (`jvp_impl="fast"`).

- Baseline = `config.py` as committed at 380b792: lr 5e-4, wd 0.1, moonlight, WSD `decay_fraction` 0.15, `muon_momentum` 0.95, `weight_init_constant` 0.32, $p_{\textbf{data}}$ 0.5.

- Columns: `probe_last3` / `probe_fin` = canonical held-out loss (comparable across ALL nodes); `train_late` = raw training `loss_u` over the final half (comparable only among nodes sharing an objective); `cv` = its coefficient of variation; `jump` = mean absolute step-to-step change; `spk` = steps above 1.5x the run median; `gn_max` = max grad norm.

## Measurement noise

- Three separate replicate pairs, because a single pair badly understates it:

| pair | difference in `probe_last3` |
|---|---|
| base config, seed 0 vs 1 (A0 / A0s1) | 0.0002 |
| combo config, seed 0 vs 1 (C0 / C7) | 0.0138 |
| same seed, RNG stream only (D9 / D10) | 0.0033 |

- The 0.0002 agreement of the first pair is a coincidence of averaging three probes: the individual probe points of that same pair differ by up to 0.04. Init variation (different seed) dominates and gives a spread near 0.014; run-to-run variation at fixed init is near 0.003.

- Decisions below therefore rest on differences above ~0.015, or on monotone trends across three or more nodes. Single-knob gains smaller than that are reported as null.

## Round 1: objective and distribution knobs

| node | change | `probe_last3` | `train_late` | cv | gn_max |
|---|---|---|---|---|---|
| A4 | $p_{\textbf{data}}$ 0.25 | 0.5799 | 0.6146 | 0.338 | 6.18 |
| A6 | $\beta_{\textbf{cfg}}$ 0.5 | 0.5831 | 0.5856 | 0.240 | 8.01 |
| A0 | baseline | 0.5834 | 0.5852 | 0.239 | 8.11 |
| A0s1 | baseline, seed 1 | 0.5836 | 0.5788 | 0.259 | 9.28 |
| A8 | $p_{\textbf{norm}}$ 0.5 | 0.5837 | 0.5868 | 0.264 | 421.50 |
| A7 | $\beta_{\textbf{cfg}}$ 2.0 | 0.5838 | 0.5848 | 0.236 | 8.38 |
| A9 | $p_{\textbf{drop}}$ 0.2 | 0.5858 | 0.5854 | 0.234 | 8.53 |
| A5 | $p_{\textbf{data}}$ 0.75 | 0.5954 | 0.5500 | 0.233 | 11.17 |
| A1 | $P_{\textbf{mean}}$ -1.0 | 0.5997 | 0.6049 | 0.211 | 9.39 |
| A3 | $P_{\textbf{std}}$ 1.6 | 0.6425 | 0.9077 | 0.699 | 11.16 |
| A2 | $P_{\textbf{mean}}$ 0.4 | 0.6794 | 0.6554 | 0.395 | 9.44 |

- The committed time distribution is already at a local optimum: both $P_{\textbf{mean}}$ moves are worse, and widening $P_{\textbf{std}}$ to 1.6 is much worse on both level and stability.

- $\beta_{\textbf{cfg}}$ and $p_{\textbf{drop}}$ are null within noise.

- $p_{\textbf{norm}}$ 0.5 reaches a max grad norm of 421.5 (baseline 8.1) and is only survivable because of the global clip; rejected.

## Round 2: optimizer, initialization, loss balance

| node | change | `probe_last3` | `train_late` | cv | gn_max |
|---|---|---|---|---|---|
| B9 | `decay_fraction` 0.4 | 0.5642 | 0.5646 | 0.231 | 7.38 |
| B5 | `muon_momentum` 0.9 | 0.5739 | 0.5727 | 0.235 | 7.61 |
| B12 | `weight_init_constant` 0.5 | 0.5750 | 0.5765 | 0.236 | 10.07 |
| B15 | `stratified_interval` on | 0.5814 | 0.5683 | 0.249 | 13.99 |
| B10 | `grad_clip` 0.5 | 0.5815 | 0.5830 | 0.226 | 8.31 |
| B7 | `warmup_steps` 20 | 0.5829 | 0.5822 | 0.236 | 8.18 |
| B14 | `loss_v_weight` 2.0 | 0.5830 | 0.5797 | 0.218 | 17.12 |
| B6 | `betas` (0.9, 0.98) | 0.5841 | 0.5795 | 0.225 | 11.84 |
| B2 | lr 8e-4 | 0.5844 | 0.5987 | 0.270 | 7.29 |
| B13 | `loss_v_weight` 0.5 | 0.5869 | 0.5895 | 0.226 | 8.26 |
| B3 | `muon_lr_scale_constant` 0.1 | 0.5902 | 0.5916 | 0.243 | 12.07 |
| B1 | lr 3e-4 | 0.5916 | 0.5881 | 0.237 | 12.74 |
| B4 | `muon_lr_scale_constant` 0.4 | 0.5916 | 0.6058 | 0.284 | 8.90 |
| B16 | `loss_v_weight` 0.25 | 0.5990 | 0.5954 | 0.231 | 10.49 |
| B8 | `warmup_steps` 100 | 0.5991 | 0.5916 | 0.219 | 20.24 |
| B11 | `weight_init_constant` 0.2 | 0.6013 | 0.5978 | 0.250 | 8.91 |

- Three knobs beat the baseline by more than the run-to-run spread: a longer WSD decay tail (0.15 to 0.4), a lower Muon momentum (0.95 to 0.9), and a larger init scale (0.32 to 0.5). Only the first clears the seed-level spread on its own.

- `loss_v_weight` (added in this work; the u/v balance was previously hardcoded at 1:1) is null in both directions. `muon_lr_scale_constant` is worse either way, confirming the committed 0.2.

## Round 3: combinations

| node | change | `probe_last3` | `train_late` | cv | jump |
|---|---|---|---|---|---|
| C8 | combo, 4 samples/step x 800 steps | 0.5264 | 0.5485 | 0.372 | 0.192 |
| C6 | combo + lr 8e-4 | 0.5472 | 0.5949 | 0.368 | 0.193 |
| C0 | combo | 0.5517 | 0.5874 | 0.336 | 0.181 |
| C10 | combo + `grad_clip` 0.5 | 0.5522 | 0.5861 | 0.319 | 0.178 |
| C1 | combo without $p_{\textbf{data}}$ 0.25 | 0.5573 | 0.5615 | 0.250 | 0.157 |
| C5 | combo + `stratified_interval` | 0.5617 | 0.5642 | 0.247 | 0.148 |
| C7 | combo, seed 1 | 0.5655 | 0.5809 | 0.349 | 0.154 |
| C2 | `decay_fraction` 0.6 | 0.5665 | 0.5554 | 0.228 | 0.143 |
| C3 | `muon_momentum` 0.85 | 0.5760 | 0.5728 | 0.256 | 0.157 |
| C4 | `weight_init_constant` 0.7 | 0.5905 | 0.5850 | 0.231 | 0.159 |
| C9 | combo, 16 samples/step x 200 steps | 0.6493 | 0.6147 | 0.153 | 0.102 |

- combo = `decay_fraction` 0.4 + `muon_momentum` 0.9 + `weight_init_constant` 0.5 + $p_{\textbf{data}}$ 0.25.

- The four gains are sub-additive but real: 0.5834 to 0.5517, against individual gains summing to 0.0406.

- Both refinements overshoot: `decay_fraction` 0.6 and `weight_init_constant` 0.7 are worse than 0.4 / 0.5, and `muon_momentum` 0.85 is worse than 0.9, so each knob is bracketed.

- `stratified_interval` (added in this work: jittered-quantile draws for $t_{\min}$, $t_{\max}$, verified to cut their batch-mean std 4x at unchanged marginals) trades level for stability inside the combo: cv 0.247 vs 0.336 and jump 0.148 vs 0.181, but probe 0.5617 vs 0.5517.

## The dominant effect: optimizer steps, not batch size

- All four points below train on exactly 3200 samples with the combo config; only the split between samples-per-step and step count changes.

| samples/step x steps | `probe_last3` | `probe_fin` | cv | wall (min) |
|---|---|---|---|---|
| 16 x 200 | 0.6493 | 0.6024 | 0.153 | 4.4 |
| 8 x 400 | 0.5517 | 0.5357 | 0.336 | 6.5 |
| 4 x 800 | 0.5264 | 0.5156 | 0.372 | 11.2 |
| 2 x 1600 | 0.4934 | 0.4913 | 0.402 | 13.4 |

- Monotone across four points and far outside the 0.015 noise band: 0.6493 to 0.4934, a 24% reduction at identical sample cost. For this model, data and optimizer, the binding constraint is the NUMBER of Muon updates, not gradient noise. Wall time rises with step count (optimizer overhead), but quality per minute still favours the small-batch end.

- The per-step training curve moves the opposite way (cv 0.153 to 0.402, spikes 36 to 217): fewer samples per logged point is a noisier ESTIMATE of the same objective. The probe curve of the 2 x 1600 node is smooth and monotone. Curve stability and convergence quality point in opposite directions here, so the two must not be read off the same series.

- Micro-batch shape alone is irrelevant: 8 samples/step as 1 x accum 8 (D0, 0.5532) matches 4 x accum 2 (C0, 0.5517). Only samples/step and step count matter.

## Learning rate is flat over a wide band, and 5e-4 is kept

| samples/step x steps | lr 3e-4 | lr 5e-4 | lr 6e-4 | lr 8e-4 | lr 1.2e-3 | lr 2e-3 |
|---|---|---|---|---|---|---|
| 8 x 400 | - | 0.5517 | 0.5464 | 0.5472 | 0.5418 | 0.5502 |
| 2 x 1600 | - | 0.4934 | - | 0.5108 | 0.4970 | 0.5760 |
| 8 x 400, baseline knobs | 0.5916 | 0.5834 | - | 0.5844 | - | - |

- At 8 samples/step the four values from 5e-4 to 1.2e-3 span 0.5418 to 0.5517, a range of 0.010 that sits INSIDE the 0.015 noise band; the apparent downward trend to 1.2e-3 does not survive adding 6e-4 and 2e-3, which break the monotonicity. At 2 samples/step the same is true of 5e-4 to 1.2e-3 (0.4934 to 0.5108).

- What is outside noise is the failure at 2e-3, in both regimes. The lr is on a wide plateau from roughly 5e-4 to 1.2e-3 and falls off above it, so the committed 5e-4 is kept: no replicated evidence supports moving it, and it sits in the middle of the plateau rather than at its edge.

## Null result: stratifying across the accumulation group

- `_strat_uniform` spreads its jittered-quantile draws over the micro-batch. At micro-batch 1 there is a single stratum spanning $(0, 1)$, which is exactly a uniform draw, so the whole stratification is INERT at the batch size that full uncropped latents force on 16 GB.

- `strat_group` was added to spread strata over `batch_size * grad_accum` samples instead, with one queue per draw site so $t$, $\omega$ and the flow-matching split stay independent. Verified at micro-batch 1 with accum 8: step-mean std of $\omega$ falls 6.8x (0.687 to 0.102) and of $t$ 2.2x (0.063 to 0.029), with the marginal quantiles unchanged to 0.001.

- It does not help. Two paired A/B tests, same seed:

| pair | without group | with group |
|---|---|---|
| micro-batch 1 x accum 8 | 0.5532 | 0.5578 |
| micro-batch 1 x accum 2 | 0.6920 | 0.6933 |

- Both differences are inside the noise band and both point the wrong way. The variance this removes is not the variance that limits the gradient; the data sample and the noise vector $e$ dominate. `strat_group_auto` therefore defaults to False.

## Replicated 2x2: the two effects are independent and both real

- Every cell trains on 3200 samples. `combo` = the four adopted knobs; `base` = `config.py` at 380b792.

| knobs | 8 samples/step x 400 | 2 samples/step x 1600 |
|---|---|---|
| base | 0.5834 (s0), 0.5836 (s1) | 0.5150 (s0), 0.5277 (s2) |
| combo | 0.5517 (s0), 0.5655 (s1) | 0.4934 (s0), 0.5048 (s2) |

- The knob gain replicates across two seeds at the small-batch point: 0.0216 at seed 0 and 0.0229 at seed 2, both above the 0.015 noise band and nearly identical.

- The batch-ladder gain is larger than the knob gain, and the two compose: 0.5834 to 0.4934 overall, a 15.4% reduction at unchanged sample cost. The baseline knobs at 2 samples/step (0.5150) already beat the tuned knobs at 8 samples/step (0.5517), so the step budget matters more than any single hyperparameter tested.

- The ladder holds at the last rung too. At a matched 800-sample budget, 1 sample/step x 800 steps scores 0.6440 against 0.6920 for 2 samples/step x 400 steps, though its max grad norm rises to 57.2.

## Adopted in config.py

- `OptimConfig.decay_fraction` 0.15 to 0.4, `OptimConfig.muon_momentum` 0.95 to 0.9, `ModelConfig.weight_init_constant` 0.32 to 0.5, `LossConfig.data_proportion` 0.5 to 0.25. Evidence is from crops and replicated over two seeds there; the full-latent A/B above could not confirm it, so these are adopted as best-available rather than as a validated gain at 29k tokens.

- Unchanged after being tested and found null or worse: lr 5e-4, `weight_decay` 0.1, `betas`, `muon_lr_scale_constant` 0.2, `warmup_steps`, `grad_clip` 1.0, $P_{\textbf{mean}}$, $P_{\textbf{std}}$, $\beta_{\textbf{cfg}}$, $s_{\max}$, `class_dropout_prob`, `norm_p`.

- Added but defaulted OFF, both null or level-negative: `LossConfig.loss_v_weight` 1.0, `LossConfig.stratified_interval` False, `LossConfig.strat_group_auto` False.

- Raw series for all 57 nodes: `records/loss_curve_tuning_history.json`. All nodes: `records/loss_curve_tuning_rounds.png`. The 2x2 plus ladder: `records/loss_curve_tuning_headline.png`, drawn against SAMPLES rather than steps, since nodes with different samples-per-step are not comparable at equal step count.

## Transfer to full uncropped latents is NOT demonstrated

- The sweep above was run on crops for throughput. The local target is full uncropped Wan-Syn latents `(C=16, T=20, H=56, W=104)` = 29,138 tokens. Two 400-step runs, 2 samples/step (micro-batch 1 x grad_accum 2, the most 16 GB allows), 50 training videos with 6 held out for the probe, seed 0, 3.9 h each.

| node | knobs | `probe_last3` | `probe_fin` | `train_late` | cv | jump | gn_max |
|---|---|---|---|---|---|---|---|
| G1 | combo | 0.6055 | 0.5907 | 0.6417 | 0.368 | 0.208 | 13.82 |
| G0 | baseline | 0.6135 | 0.5775 | 0.6388 | 0.359 | 0.205 | 10.08 |

- The two summary metrics DISAGREE: `probe_last3` favours the tuned config by 0.0080, `probe_fin` favours the baseline by 0.0132. Per-checkpoint differences (tuned minus base):

| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|
| diff | +0.0101 | -0.0609 | -0.0305 | -0.0232 | +0.0114 | -0.0262 | -0.0109 | +0.0132 |

- The tuned config is ahead at 5 of 8 checkpoints and by 0.0146 on the mean over all eight, but the checkpoint-to-checkpoint spread of that difference is 0.0741, five times the mean. One seed cannot separate the two here.

- The full-latent probe is also far weaker than the crop probe: 6 held-out videos instead of 32, against 50 training videos that each run sees about 16 times. So this is a null result from an underpowered test, not evidence that the knobs are wrong. The four adopted values stay, because they are replicated on crops and are at worst neutral here, but the honest summary is that the 15.4% crop improvement has NOT been reproduced at full resolution.

- Confirming transfer properly needs several seeds per arm at full resolution (about 8 h per seed-pair on this GPU), or a larger held-out set.

## What DOES transfer: the step-count lever

- A third full-latent run, G2, keeps the tuned knobs and spends the same 800 samples as 1 sample/step x 800 steps instead of 2 samples/step x 400 steps. Same data, same seed, and the same wall clock to within 4%.

| node | samples/step x steps | `probe_last3` | `probe_fin` | cv | jump | spk | gn_max | wall (min) |
|---|---|---|---|---|---|---|---|---|
| G2 | 1 x 800, combo | 0.5795 | 0.5694 | 0.402 | 0.243 | 138 | 28.99 | 241 |
| G1 | 2 x 400, combo | 0.6055 | 0.5907 | 0.368 | 0.208 | 68 | 13.82 | 232 |
| G0 | 2 x 400, baseline | 0.6135 | 0.5775 | 0.359 | 0.205 | 67 | 10.08 | 232 |

- Probe against samples consumed, which is what is held equal:

| samples | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|
| G2, 1/step | 1.0166 | 0.8321 | 0.7069 | 0.7063 | 0.6538 | 0.6141 | 0.5875 | 0.5694 |
| G1, 2/step | 0.9076 | 0.8213 | 0.6976 | 0.6846 | 0.6567 | 0.6225 | 0.6034 | 0.5907 |
| G0, 2/step base | 0.8976 | 0.8822 | 0.7281 | 0.7078 | 0.6453 | 0.6487 | 0.6143 | 0.5775 |

- G2 starts worst, crosses over around 500 to 600 samples and finishes lowest on both summary metrics. Against G1, which differs ONLY in the samples-per-step split, it is 0.0213 better on the final probe and 0.0260 better on `probe_last3`, and unlike the knob A/B the two metrics agree in direction.

- So the finding that dominated the crop sweep survives at 29,138 tokens: given a fixed sample budget, spend it on more optimizer steps rather than on wider steps. The committed defaults `batch_size_per_gpu = 1` and `grad_accum = 1` already sit at this end; the earlier full-latent record (`records/wan_syn_full_train.md`) overrode `grad_accum` to 2 and was therefore on the wrong side of it.

- Same stability trade as on crops, and larger here: cv 0.402 vs 0.368, spikes 138 vs 68, max grad norm 29.0 vs 13.8. The better run again has the uglier per-step curve.

- Curves: `records/loss_curve_tuning_full.png`; raw series `records/loss_curve_tuning_full_history.json`.

## Reading the curve

- Two different series are called "the loss curve" in this project and they disagree about the small-batch end of the ladder. The per-step training `loss_u` is an average over `batch_size * grad_accum` samples with freshly drawn $(t, r, \omega)$; halving that count makes the logged series visibly noisier (cv 0.153 at 16 samples/step to 0.402 at 2) while the model it is estimating is strictly better.

- So the noisier-looking run is the better run. `train.py` already logs `loss_u_ema` next to the instantaneous value; that, or a fixed-batch probe like the one used here, is the series to judge convergence by. Judging by the raw per-step series alone selects for large batches, which this sweep shows to be the worst use of a fixed sample budget.

---

# Rounds R1 / R2 / R3 — 4k-step full-latent sweep (2026-08)

Restarted tuning against the CLAUDE.md target of 4,000 training steps on the
2k original (uncropped) Wan-Syn latents.

## Fixed across all nodes

| item | value |
|---|---|
| latents | `.cache/wan_syn_2k_full`, `(C=16, T=20, H=56, W=104)` = 29,138 tokens |
| model | `hidden_size=256`, `depth=19`, `aux_head_depth=4`, `num_heads=4` → 18.6M params |
| attention | `attn_impl="flash_jvp"` (CuTeDSL), `attn_res_block_size=4` |
| du/dt engine | `jvp_impl="fast"` (Triton forward-mode, detached) |
| optimizer | `moonlight` (Muon + AdamW split), `muon_coeff_mode="per_shape"` |
| schedule | `lr_schedule="wsd"`, `decay_shape="1-sqrt"`, `min_lr_ratio=0.1` |
| sampling | `stratified_time=True`, `stratified_interval=False` |
| split | train 1968 files / eval 32 files, `PROBE_SEED=1234` |
| ranking metric | frozen held-out probe `loss_u` (never raw train loss) |

`probe_fin` = probe at the last logged checkpoint. `cv` = coefficient of
variation of the train `loss_u` over the second half of the run.

## R1 — batch_size_per_gpu = 1, total_steps = 4000

| node | lr | warmup | probe_fin | cv | note |
|---|---|---|---|---|---|
| **R1_warm100** | 2e-3 | 100 | **0.2772** | 1.17 | best overall |
| R1_lr2e3 | 2e-3 | 400 | 0.2848 | 1.33 | |
| R1_lr1e3 | 1e-3 | 400 | 0.3462 | 1.21 | |
| R1_base | 5e-4 | 400 | 0.5355 | 0.99 | |
| R1_lr4e3 | 4e-3 | 400 | NaN | 0.55 | diverged |

`decay_fraction=0.2` throughout → 800 decay steps.

## R2 — batch_size_per_gpu = 5, total_steps = 800

Same 4,000-sample budget as R1, spent as 5 samples/step x 800 steps.

| node | lr | probe_fin | cv |
|---|---|---|---|
| R2_base | 2e-3 | 0.4000 | 0.50 |
| R2_lr1e3 | 1e-3 | 0.7621 | 0.50 |

`decay_fraction=0.2` → only 160 decay steps.

## R3 — batch_size_per_gpu = 5, one-knob sweep off R2_base

All eight nodes were killed by the harness early-stop rule (probe worse than
the running best at or past the halfway mark), so all report probe at step 400.

| node | knob | probe@400 | cv |
|---|---|---|---|
| R3_wd005 | `optim.weight_decay=0.05` | 0.5679 | 0.25 |
| R3_decay04 | `optim.decay_fraction=0.4` | 0.5760 | 0.25 |
| R3_decay01 | `optim.decay_fraction=0.1` | 0.5762 | 0.25 |
| R3_wic1 | `model.weight_init_constant=1.0` | 0.5794 | 0.29 |
| R3_pmean08 | `loss.P_mean=-0.8` | 0.5807 | 0.21 |
| R3_dp05 | `loss.data_proportion=0.5` | 0.5971 | 0.23 |
| R3_wic032 | `model.weight_init_constant=0.32` | 0.5977 | 0.24 |
| R3_warm80 | `optim.warmup_steps=80` | 0.5986 | 0.24 |

R2_base's own probe at step 400 was 0.5762 — inside the spread of every R3
node. No knob moved the trajectory before the decay tail opened.

## Conclusions

- R2_base reached 0.4000 entirely inside its decay tail: probe 0.5762 at step
  400, 0.5736 at step 500, then 0.4565 / 0.3966 / 0.4000 once the WSD decay
  began at step 640. The decay phase does most of the visible learning.

- That is why bs=5 loses to bs=1 at an equal sample budget. Both see 4,000
  samples, but `decay_fraction=0.2` buys bs=1 800 decay steps and bs=5 only
  160. The bs=5 arm is decay-starved, not knob-starved, which is why the whole
  R3 sweep was flat.

- Raising `decay_fraction` does not rescue it: R3_decay04 and R3_decay01 are
  indistinguishable from each other and from baseline at step 400, i.e. the
  stable phase is insensitive to how much of the run is reserved for decay.

- This reproduces the step-count lever from the earlier G1/G2 comparison at the
  same 29,138 tokens: at a fixed sample budget, buy optimizer steps, not wider
  steps.

- The stability/quality trade also reproduces and is now extreme. The best node
  has the worst-looking train curve (cv 1.17) and the flat R3 nodes have the
  cleanest ones (cv 0.21–0.29). The per-step train `loss_u` is an average over
  `batch_size` samples with freshly drawn nuisances, so its cv tracks batch
  size, not convergence. Judge by the frozen probe.

- Open: the "stable AND fast-converging" target is only half met. R1_warm100
  converges fastest but its raw train series is the noisiest of the sweep.
  Untested middle ground: bs=1 with more than 4,000 steps, or an intermediate
  batch size with `decay_fraction` raised to hold decay steps constant.

- Curves for all 15 nodes: `records/tuning_4k_loss_curves.png`.

## Round PT — P_mean / P_std / data_proportion sweep on the tuned schedule (2026-08)

- Settings: bs 1, 1000 steps (compressed WSD: warmup 100, decay_fraction 0.2 -> decay from step 800), lr 2e-3, full uncropped latents, frozen 32-video probe (PROBE_SEED 1234), harness `autotune.py`. Early stop: probe > 0.70 at >= 500 steps (base's pre-decay band; note the first attempt used base's post-decay 0.5554 as the threshold, which would kill every node before its decay phase — two P_mean nodes were re-run after that fix).

| node | override | probe final | at step | train loss_u cv (late half) | early stop |
|---|---|---|---|---|---|
| PT_ps08 | P_std 0.8 | **0.5506** | 1000 | 0.39 | |
| PT_base | none | 0.5554 | 1000 | 0.40 | |
| PT_dp05 | data_proportion 0.5 | 0.5611 | 1000 | 0.49 | |
| PT_pm02 | P_mean -0.2 | 0.7009 | 500 | 0.35 | yes (0.7009 > 0.70) |
| PT_pm06 | P_mean -0.6 | 0.7028 | 600 | 0.32 | yes (0.7028 > 0.70) |
| PT_ps12 | P_std 1.2 | 0.7381 | 600 | 0.43 | yes |
| PT_dp01 | data_proportion 0.1 | (killed at ~step 250, sweep stopped on request) | | | |

- PT_ps08 and PT_dp05 both led base during the stable phase (0.6559 / 0.6897 vs 0.6926 at step 600) but only PT_ps08 held the lead through decay. The two P_mean kills were marginal (0.001-0.003 over the threshold); both trailed base at every matched checkpoint before the stop.
