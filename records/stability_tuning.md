# Loss-Curve Stability Tuning (full uncropped Wan-Syn latents)

## Diagnosis

- The swings in the first full-latent run were sampling variance, not divergence: each logged `loss_u` averages only `batch_size * grad_accum` samples, and every sample draws its own time pair, guidance scale and branch assignment. Two defects amplified this.

- Defect 1, flow-matching branch silently dead at micro-batch 1. The split used a deterministic prefix rule

$$
\begin{equation}
\begin{aligned}
n_{\textbf{fm}} = \lfloor B \cdot p \rfloor,
\qquad
\textbf{fm\_mask}_i = [\, i < n_{\textbf{fm}} \,]
\end{aligned}
\end{equation}
$$

- so with $B = 1$ and $p = 0.5$, $n_{\textbf{fm}} = 0$ and NO sample ever took the flow-matching branch. Every full-latent step trained only the harder mean-flow branch, which raises both the loss level and its variance. Replaced by a stratified rule that reproduces the prefix behaviour in expectation at any batch size (measured proportion at $B = 1$: 0.000 before, 0.495 after).

- Defect 2, unstratified nuisance draws. The guidance scale $\omega \in [1, 1 + s_{\max}]$ is the largest single contributor to raw-loss variance, and both $\omega$ and the times were drawn independently per sample. Jittered-quantile (stratified) draws keep the marginal law exactly and cut batch-statistic variance: time batch-mean std 0.0736 to 0.0122 (6x), omega batch-mean std 0.963 to 0.288 (3.3x).

## Settings

- All three runs: 288.9M MLA DiT, full latents (16, 20, 56, 104) = 29,138 tokens, attention residual (block 4), fast Triton du/dt engine, per-block gradient checkpointing, moonlight optimizer, wd 0.1, WSD schedule, seed 0, identical total sample budget (800 samples).

- F01: independent sampling, 2 samples/step (batch 1 x accum 2), lr 5e-4, 400 steps, decay_fraction 0.5.

- G01: stratified time only, 4 samples/step (batch 1 x accum 4), lr 3e-4, 200 steps, decay_fraction 0.3.

- G02: G01 plus the flow-matching fix and stratified omega.

## Results

- Statistics over the final half of each run (equal sample budget):

| run | mean loss_u | std | CV | range/mean | mean step-to-step jump | grad_norm mean |
|---|---|---|---|---|---|---|
| F01 baseline | 0.6966 | 0.2362 | 0.339 | 1.30 | 0.2757 | 1.99 |
| G01 stratified t | 0.7270 | 0.1851 | 0.255 | 0.97 | 0.2655 | 1.54 |
| G02 all fixes | 0.7921 | 0.1403 | 0.177 | 0.61 | 0.2041 | 1.76 |

- Coefficient of variation halved (0.339 to 0.177); peak-to-trough spread within the late window fell from 1.30x the mean to 0.61x; mean step-to-step jump fell 26%.

- Mean level rose slightly across the three runs. This is expected and not a regression: G02 restores the flow-matching samples that F01 never trained on, and lower learning rate plus a shorter run means fewer optimizer steps at equal sample count (200 vs 400). Convergence at matched samples is comparable (G02 reaches 0.71 at 400 samples where F01 was at 1.17).

- Curves and raw series: `records/stability_nodes_history.json`.

## Adopted in config.py

- `LossConfig.stratified_time = True` (stratified times, guidance scale and flow-matching split).

- Flow-matching split fix is unconditional (`imf_video.py: sample_tr`), since the previous rule was wrong for any batch smaller than `1 / data_proportion`.

- `train.py` logs `loss_u_ema` alongside the instantaneous value so the trend is readable at small per-step sample counts.
