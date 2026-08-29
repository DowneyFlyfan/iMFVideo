# Adaptive-loss saturation diagnosis and metric repair

## Observation

The four-A100 log reported `loss=2.0000` at steps 7050 through 8000.  This
looked like a late training loss increase, but the same line reported finite
raw per-element components at step 8000:

```text
loss_u=0.5894  loss_v=0.5572  grad_norm=1.227
```

The preconditioned step-6000 audit had larger raw components (`loss_u=1.0330`,
`loss_v=0.6711`), so the 8k record does not establish an exploding error.

## Root cause

For each per-sample summed squared error `l`, the configured objective uses

```text
l / detach((l + 0.01)^1)
```

With `norm_p=1`, every positive branch is strictly below one.  The optimized
objective is the sum of the `u` and `v` branches, each with unit weight, so it
is strictly below two and rounds to `2.0000` whenever the full-video summed
error is much larger than `0.01`.  This is a reporting saturation, not a
gradient explosion or a changed hyperparameter.

## Change

- `adaptive_loss_metrics` retains the exact detached adaptive objective used
  for backward propagation.
- It additionally returns weighted raw per-element mean squared error.
- Training logs and Weights & Biases now expose that quantity as `loss` and
  preserve the bounded optimization quantity as `objective`.

No optimizer, schedule, model, input shape, or training loss gradient was
changed.

## Verification

`tests/test_adaptive_loss_metrics.py` proves a raw error of `50.0` remains
visible while the adaptive objective is in `(1.999, 2.0)`.  The production-like
CUDA end-to-end test completed eight finite forward passes and one finite
backward pass:

```text
bwd grad-norm-sum 0.32 finite True
finite loss_u in 8/8 seeds
PASS
```
