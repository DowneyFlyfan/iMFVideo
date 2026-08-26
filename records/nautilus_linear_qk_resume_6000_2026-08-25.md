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
