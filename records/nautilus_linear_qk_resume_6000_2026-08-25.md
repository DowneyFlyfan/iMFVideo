# Nautilus Linear QK Resume From Step 6000 — 2026-08-25

## Server settings

- Host: `Nautilus-A100`; four NVIDIA A100 80 GiB graphics processing units.
- Checkpoint: `checkpoints/step_0006000.pt`, size 3.3 GiB.
- Model: 290.1 million parameters.
- Distributed world size: 4.
- Input configuration retained from the checkpoint: channels 48, latent frames 31,
  latent size `(44, 80)`, cube tile `(1, 2, 8)`.
- Resume step: 6000.

## Synced implementation

- `train.py`
- `models/sla2_cube_qat.py`
- `models/mla_jvp_fast.py`
- Optimizer coefficient mode: Jordan.
- Linear Query-Key denominator floor: `0.1`.

## Result

```text
resumed from checkpoints/step_0006000.pt at step 6000
step 6001: non-finite grad_norm, step skipped; loss_u=nan loss_v=nan
```

## GPU occupancy after stop

```text
GPU 0: 1487 MiB, 100%
GPU 1: 1487 MiB, 100%
GPU 2: 1487 MiB, 100%
GPU 3: 1487 MiB, 100%
```

- Watchdog interval: 15 seconds.
- Fallback task: one bfloat16 matrix multiplication process per graphics
  processing unit.

## Checkpoint and preconditioner audit

- `step_0006000.pt`: 290,118,896 floating-point model values; zero non-finite
  values.
- Optimizer state: 297,608,672 floating-point values; zero non-finite values.
- First non-finite attention output on `nan_batch_r0.pt`: `shared_blocks.3.attn`.
- Block 3 Q/K channel ranges: 11.5934 and 11.5478.
- Whole-model Q/K preconditioner scale `0.30` on the same batch:

```text
loss=2.0
loss_u=1.032982349395752
loss_v=0.6711363196372986
```

- Resume code applies `resume_linear_qk_scale` after checkpoint loading and
  restores the configured Moonlight Newton-Schulz coefficient mode after the
  optimizer state is loaded.
