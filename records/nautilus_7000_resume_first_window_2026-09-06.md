# Step-7000 A100 resume: first verified window — 2026-09-06

## Configuration actually loaded

- Four NVIDIA A100 GPUs; model: 290.1 million parameters.
- Checkpoint: `checkpoints/step_0007000.pt`.
- Latents: 48 channels, 31 frames, 44 by 80.
- Optimizer schedule: 10,000 total steps; batch 4 per GPU; no gradient
  accumulation.
- Resume Q/K preconditioner: scale 0.3 on 23 modules, applied to online
  weights, Exponential Moving Average weights, and optimizer state on all four
  ranks.

## First post-resume metric

At step 7050 after roughly 29.5 minutes:

| metric | value |
| --- | ---: |
| loss | 0.7913 |
| loss_u | 0.4131 |
| loss_v | 0.3783 |
| gradient norm | 0.856 |
| learning rate | 8.00e-04 |
| QK-Clip | 0 / 8 |
| phi-Clip | 7 / 8.4 / 0.988 |
| throughput | 0.5 samples/s |

No non-finite loss or gradient was logged after the step-7000 resume marker.
This contrasts with the prior uncorrected resume, which produced non-finite
gradients at steps 7001 and 7002.  The observed throughput matches the prior
full-resolution A100 baseline; the long first logging window is expected, not
a hang.

## Second verified window

At step 7100, the run remained finite and at the same 0.5 samples/s:

| metric | value |
| --- | ---: |
| loss | 0.8940 |
| loss_u | 0.4703 |
| loss_u EMA | 0.4245 |
| loss_v | 0.4237 |
| gradient norm | 0.730 |
| QK-Clip | 0 / 8 |
| phi-Clip | 8 / 8.5 / 0.978 |

The two 50-step windows establish finite, non-skipped optimization over 100
steps after the formerly explosive checkpoint boundary.

## Third verified window

At step 7150, optimization remained finite:

| metric | value |
| --- | ---: |
| loss | 0.9041 |
| loss_u | 0.4610 |
| loss_u EMA | 0.4318 |
| loss_v | 0.4430 |
| gradient norm | 1.213 |
| QK-Clip | 0 / 8 |
| phi-Clip | 8 / 8.5 / 0.985 |
| throughput | 0.5 samples/s |

The resumed run has now completed 150 finite steps (7001 through 7150) at the
historical full-resolution throughput, with no recurrence of the old T2/QK
gradient explosion.
