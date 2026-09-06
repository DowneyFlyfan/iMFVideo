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
