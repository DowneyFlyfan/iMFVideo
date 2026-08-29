# EMA resume recovery — 2026-08-27

## Root cause

The step-7,000 checkpoint was resumed from step 6,000 with
`resume_linear_qk_scale=0.3`. The resume path scaled only the online model's
linear Query-Key producers, then loaded the unscaled Exponential Moving
Average (EMA). Inference selected that invalid EMA.

For `shared_blocks.0.attn.q_norm`, the online and EMA norms in the step-7,000
checkpoint are 2.187 and 6.666 respectively, a 0.328 ratio consistent with
the one-off preconditioner. Across a matched noise sample, online one-step
sampling has output/noise cosine similarity 0.439 while the old EMA is 0.973:
the old EMA leaves the input almost unchanged.

## Repair

- `rescale_linear_qk_producers` now also transforms every floating-point,
  parameter-shaped optimizer state tensor, including Moonlight momentum.
- `rescale_resumed_linear_qk` applies the preconditioner to the online model
  and optimizer, then initializes EMA from the corrected online model. An old
  EMA cannot be made valid merely by scaling its Query-Key rows.
- Checkpoints now persist `linear_qk_preconditioned`; a later resume will not
  apply the same scale again.
- `generate_7k_video.py --weights model` explicitly selects online weights for
  this legacy checkpoint. The default EMA mode remains available for repaired
  future checkpoints.
- `repair_checkpoint_ema.py` atomically migrates a completed legacy checkpoint
  in place: it resets EMA from online weights, records the preconditioning
  marker, and updates the saved resume scale to 1.0.

## Local inference result

`step_0007000_repaired_online_1step.mp4` was generated with online weights,
the checkpoint's native one MeanFlow step, seed 7000, and the Wan 2.2 TI2V
variational autoencoder. It is H.264, 1280 by 704 pixels, 121 frames, 16 FPS,
and 7.5625 seconds long.

The previous EMA output was pure noise. The repaired online output has
low-frequency video structure but remains blocky and not semantically
coherent. A four-step integration collapsed to black, so the one-step result
is the only valid sample from this checkpoint. Further training after the
resume repair is required for usable visual quality.

## Verification

```text
.venv/bin/python -m pytest -q tests/test_generate_7k_video.py \
    tests/test_resume_linear_qk_scale.py
6 passed
```

`python -m py_compile train.py generate_7k_video.py` completed successfully.
