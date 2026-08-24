# Linear QK-Clip Local Test — 2026-08-23

## Scope

- Attention implementation: `sla2_cube_qat`.
- Input: fixed Wan-Syn local latents, `(16, 20, 56, 104)`; no resize or crop.
- Model: 289.3 million parameters.
- Device: NVIDIA GeForce RTX 5070 Ti, 16 GiB.

## Implemented changes

- The feature-softmax range signal is `max(channel) - min(channel)` per Q/K head.
- The clip factor uses the complement key-token count and the configured denominator floor `0.1`.
- Q/K RMSNorm (root mean square normalization) gains and RoPE (rotary position embedding) projection rows receive the same factor.
- The Cube JVP (Jacobian-vector product) keeps the tangent attention buffer in float32 until its output projection.
- Moonlight uses Jordan Newton–Schulz coefficients. The local `per_shape` coefficient for `(1024, 1024)` reported mean squared error `9.565e-01` and wrote non-finite `shared_blocks.{0,3,6}.attn.out_proj.weight` at update 6.

## Local run settings

- Diagnostic run only: 20 steps, seed 0, batch size 1, gradient accumulation 1.
- Learning-rate values recorded at steps 1 through 20: `0` through `3.80e-4` in increments of `2e-5`.
- Production configuration restored after the run: 4,000 steps, logging every 50 steps, checkpoint directory `checkpoints`.

## Local run results

| Step | loss_u | loss_v | gradient norm | QK clip | linear QK clip (modules/range/factor) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.9801 | 1.9801 | 2.692 | 0 / 10 | 23 / 14.7 / 0.569 |
| 6 | 1.8145 | 1.8144 | 2.394 | 0 / 8 | 4 / 8.7 / 0.969 |
| 10 | 1.6072 | 1.6071 | 2.120 | 0 / 8 | 2 / 8.6 / 0.971 |
| 15 | 1.6134 | 1.6100 | 2.490 | 0 / 8 | 1 / 8.5 / 0.988 |
| 20 | 1.4290 | 1.4263 | 1.948 | 0 / 8 | 0 / 8.4 / 1.000 |

## Test commands and results

```text
.venv/bin/python tests/test_linear_qk_clip.py
PASS gamma=0.594123 range_sum=18.000000 rho=10.694215

.venv/bin/python tests/test_sla2_cube_qat.py
qat jvp kernel: o 9.74e-03 do 1.08e-02
module int8=True: out rms 0.8523 alpha-grad absmax 1.76e-02
module int8=False: out rms 0.8545 alpha-grad absmax 1.77e-02
int8 vs fp16 forward rel diff 1.85e-02
PASS

.venv/bin/python tests/test_cube_tangent_buffer.py
PASS
```
