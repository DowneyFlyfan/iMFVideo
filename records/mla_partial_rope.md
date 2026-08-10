# MLA with Decoupled RoPE in iMFDiTVideo

- Date: 2026-07-29

## Change

- Replaced `RoPEAttention` (full-rank multi-head attention, rotary position
embedding applied over the whole head vector) with `MLAAttention` in
`models/imf_dit_video.py`.

### Symbols

| Symbol | Meaning | Default at `hidden_size=1024`, `num_heads=16` |
|---|---|---|
| $b$ | batch size | — |
| $l$ | sequence length, `prefix_tokens + T*(Hp/2)*(Wp/2)` | 20 + 192 = 212 |
| $d$ | `hidden_size` | 1024 |
| $H$ | `num_heads` | 16 |
| $d_q$ | `q_lora_rank`, query latent dim | 512 = `d // 2` |
| $d_c$ | `kv_lora_rank`, shared key/value latent dim | 256 = `d // 4` |
| $d_n$ | `qk_nope_head_dim`, position-free q/k channels | 48 |
| $d_r$ | `qk_rope_head_dim`, rotary q/k channels | 16 |
| $d_v$ | `v_head_dim` | 64 |

- $d_n + d_r = 64 = d_v$, so the layout stays compatible with the
fixed-head-dim-64 `flash_jvp_attention` CuTeDSL op.

### Forward Path

```
x(b,l,d)
  -> q_a_proj  -> q_a_layernorm -> q_b_proj -> q(b,l,H,dn+dr)
                        split -> q_nope(b,l,H,dn), q_rope(b,l,H,dr)
  -> kv_a_proj -> split -> kv_latent(b,l,dc), k_rope(b,l,dr)
                        k_rope is ONE shared head
     kv_latent -> kv_a_layernorm -> kv_b_proj -> kv(b,l,H,dn+dv)
                        split -> k_nope(b,l,H,dn), v(b,l,H,dv)
  q_nope, k_nope -> per-head RMSNorm(dn)
  q_rope         -> rope(dr)                       (b,l,H,dr)
  k_rope         -> rope(dr) -> expand over H      (b,l,H,dr)
  q = cat[q_nope, q_rope]; k = cat[k_nope, k_rope]  (b,l,H,dn+dr)
  attn(q,k,v) -> (b,l,H,dv) -> out_proj -> (b,l,d)
```

- The softmax scale is taken from the full query head dim:

$$
\begin{equation}
\begin{aligned}
\textbf{scale} &= \frac{1}{\sqrt{d_n + d_r}} \\
\end{aligned}
\end{equation}
$$

- Norm placement follows DeepSeek-V2/V3: latent-space norms
(`q_a_layernorm`, `kv_a_layernorm`). Per-head RMSNorm is applied to the
$d_n$ band only; the $d_r$ band is left un-normed so that `k_rope` stays a
single shared head.

- A diffusion transformer keeps no key-value cache, so MLA here is a low-rank
bottleneck on the query/key/value projections rather than a cache-compression
trick.

### Rotary Table

- `precompute_axial_rope_3d` is now called with $d_r$ instead of `head_dim`,
so `rope_cos` and `rope_sin` have shape $(l, d_r/2)$.

- The three-axis split of $d_r = 16$ is
$(\textbf{dim}_t, \textbf{dim}_h, \textbf{dim}_w) = (4, 6, 6)$.

## Other Changes

- `models/imf_dit_video.py`: added `eager_math_attention`, an einsum plus
softmax attention. `sdpa_math_attention` dispatches to it on MPS, where
`aten::_scaled_dot_product_attention_math_for_mps` has no forward-mode
autodiff formula and raises under `torch.func.jvp`.

- `imf_video.py`: added `sample_one_step` and `sample`, ports of the `imf.py`
sampler to $(B, C, T, H, W)$; the `fm_mask` argument of
`sample_cfg_interval` is now required.

- `train.py`: single-process and multi-GPU device selection covering CUDA, MPS
and CPU; `DistributedDataParallel` replaced by one explicit flat `all_reduce`
of gradients; `fused` AdamW and `pin_memory` gated on CUDA; a `flash_jvp`
import failure falls back to `sdpa_math_attention` with a printed warning.

- `config.py`: added the five MLA knobs, where 0 means derive from
`hidden_size` or `head_dim`; comments added for `data_proportion`,
`grad_clip` and `ema_decay`; `num_workers` guards `os.cpu_count()` returning
`None`.

- `tests/test_model_pt.py`: no longer requires CUDA; added attention-kernel
equivalence, MLA geometry and sampler checks.

## Bugs Found And Fixed

### Latent Ranks Scaled Off The Wrong Dimension

- The first `q_lora_rank` / `kv_lora_rank` default was derived from `head_dim`
($8 \cdot 64 = 512$, $4 \cdot 64 = 256$). At `hidden_size=256` that gives
$d_q = 512$, a latent wider than the model itself, and the attention block came
out at 1.956x the parameters of full-rank MHA instead of below 1.0x.

- Fixed to derive from `hidden_size`: $d_q = d/2$, $d_c = d/4$, floored at
`head_dim`.

### MPS Has No Forward-Mode Autodiff For SDPA

- `torch.func.jvp` through
`aten::_scaled_dot_product_attention_math_for_mps` raises
`NotImplementedError`, so the whole iMF training step failed on this machine.

- Fixed by adding `eager_math_attention` and dispatching to it when
`q.device.type == "mps"`. Verified against the SDPA math backend at
`max_abs_diff = 4.94e-07`.

### Checkpoint Resume Was Completely Broken

- `torch.save` wrote `"config": vars(config)`, which embeds `ModelConfig` and
the other dataclass instances. `torch.load` defaults to `weights_only=True`
since PyTorch 2.6 and refuses to unpickle arbitrary classes, so every resume
died with:

```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL config.ModelConfig was not an allowed global by default.
```

- Fixed with `config_as_dict()` using `dataclasses.asdict`, so the checkpoint
holds only dicts, numbers, strings and tuples. `weights_only=True` is kept
explicit on load rather than being weakened to `False`.

### Guidance Scale Range Was Mis-documented

- `cfg_s_max = 7.0` reads as "maximum guidance scale 7", but both branches of
`sample_cfg_scale` map $u \in [0, 1]$ onto $\omega \in [1, 1 + s_{\max}]$, so
the true maximum is 8.0. Confirmed at both endpoints for
$\beta \in \{0, 0.5, 1, 2, 3\}$: $\omega(u{=}0) = 1.000000$ and
$\omega(u{=}1) = 8.000000$ in every case. Documented in `config.py`; no code
change, the behaviour matches `imf.py`.

## Settings

- Environment:
`uv venv --system-site-packages --python /opt/miniconda3/bin/python3 .venv`,
torch 2.11.0, macOS Darwin 25.5.0, MPS available.

| Run | Config |
|---|---|
| `tests/test_model_pt.py` | `imf_dit_video_S`: `hidden_size=384`, `depth=8`, `num_heads=6`, `aux_head_depth=4`, `num_classes=10`, latents `(2,16,3,16,16)` |
| `train.py` smoke | `hidden_size=256`, `depth=4`, `num_heads=4`, `aux_head_depth=2`, `num_classes=16`, `batch_size_per_gpu=2`, `grad_accum=2`, `total_steps=6`, `warmup_steps=2`, `attn_impl="flash_jvp"` |
| Full-config param count | `hidden_size=1024`, `depth=19`, `num_heads=16`, `aux_head_depth=4`, `num_classes=1000` |

## Results

### Parameter Counts

| Model | MLA attn params/layer | Full-rank MHA params/layer | Ratio |
|---|---|---|---|
| `hidden_size=256`, H=4 | 180,512 | 262,144 | 0.689x |
| `hidden_size=384`, H=6 | — | — | 0.683x |
| `hidden_size=1024`, H=16 | 2,835,296 | 4,194,304 | 0.676x |

- Full config total: 264.7M parameters, cap 300M.

### Correctness Checks

```
[1] eager vs sdpa  max_abs_diff=4.402e-07
[1] dv != dk       eager (2, 7, 4, 48) sdpa (2, 7, 4, 48)
[2] same offset (1,2,3) logits=['-2.417367', '-2.417366', '-2.417367']
[2] spread=8.603e-07
[2] offset (1,2,4) logit=-1.037471  differs=True
[3] k_rope identical across all 4 heads: True
[4] full model 264.7M params (cap 300M) -> under_cap=True
[4] per-layer attn: MLA 2835296 vs MHA 4194304 -> 0.676x
```

### Model And Loss Test

- MPS:

```
attention kernels agree (max abs diff 4.94e-07), dv != dk OK
model params: 20.08M
MLA geometry OK: dq=192 dc=96 dn=48 dr=16 dv=64; attn params 0.683x MHA
forward shapes OK: (2, 16, 3, 16, 16)
torch.func.jvp path OK, |du_dt| mean: 0.0
perturbed jvp |du_dt| mean: 0.3100799024105072
loss: 1.999999
  loss_u: 2.145571
  loss_v: 2.144818
finite grads on 222 parameter tensors
sample(1 steps) OK: (2, 16, 3, 16, 16) std=1.0638
sample(4 steps) OK: (2, 16, 3, 16, 16) std=1.0605
ALL TESTS PASSED
```

- CPU: identical structure, with
`perturbed jvp |du_dt| mean: 0.31284675002098083`, `loss_u: 2.125149`,
`loss_v: 2.123339`, `sample(1) std=1.0651`, `sample(4) std=1.0596`.

### Training Loop End To End On MPS

```
WARNING: attn_impl='flash_jvp' unavailable (ModuleNotFoundError: No module named 'flash_attn_2_cuda'); falling back to sdpa_math_attention (much slower).
model: 4.8M params, world=1, device=mps, attn=sdpa_math_attention
MLA: heads=4 q_lora=128 kv_lora=64 qk_nope=48 qk_rope=16 v=64
step 1 loss=2.0000 loss_u=2.0050 loss_v=2.0050 grad_norm=0.168 lr=0.00e+00 4.1 samples/s
step 2 loss=2.0000 loss_u=2.0025 loss_v=2.0025 grad_norm=0.162 lr=5.00e-05 10.0 samples/s
step 3 loss=2.0000 loss_u=1.9623 loss_v=1.9623 grad_norm=0.158 lr=1.00e-04 16.2 samples/s
step 4 loss=2.0000 loss_u=2.0101 loss_v=2.0101 grad_norm=0.200 lr=8.68e-05 16.8 samples/s
step 5 loss=2.0000 loss_u=1.9915 loss_v=1.9915 grad_norm=0.148 lr=5.50e-05 16.8 samples/s
saved .../ckpt/step_0000005.pt
step 6 loss=2.0000 loss_u=1.9666 loss_v=1.9666 grad_norm=0.200 lr=2.32e-05 7.8 samples/s
```

### Edge Case Sweep

- 27 cases, all passing on both CPU and MPS:

| Group | Cases |
|---|---|
| `cfg_beta` power-distribution branch | $\beta \in \{0, 0.5, 2, 3\}$, scale range and full loss backward |
| `eval_mode=True` (`v_heads` empty) | forward and sampler |
| degenerate mixing ratios | `data_proportion` $\times$ `class_dropout_prob` over $\{0, 0.5, 1\}^2$ |
| batch sizes | 1, 3, 5 |
| low-precision master dtype | `bfloat16`, `float16` |
| sampler schedules | custom `t_steps`, `num_steps=8` with a narrowed interval |
| checkpoint | `state_dict` round trip, `rope_*` absent (`persistent=False`) |
| forward-mode autodiff | `du_dt` nonzero after parameter perturbation |

- `train.py` paths covered separately: `grad_accum=3`, `num_workers=2` worker
processes, checkpoint save then resume, `flash_jvp` fallback, `fused` AdamW
auto-disable off CUDA.

### Gradient Distribution At Initialization

- 4 of 126 parameter tensors carry nonzero gradient at step 0:
`u_final_layer.linear.weight`, `u_final_layer.linear.bias`,
`v_final_layer.linear.weight`, `v_final_layer.linear.bias`.

- After one optimizer step, `shared_blocks[0].attn.q_a_proj.weight` changes on
the next step.

### Reported Total Loss

- With `norm_p` $=1.0$, `norm_eps` $=0.01$, and per-sample sums over
$C \cdot T \cdot H \cdot W = 12288$ elements, `adp_wt_fn` returns

$$
\begin{equation}
\begin{aligned}
\frac{\mathcal{L}}{(\mathcal{L} + \epsilon)^{p}}
\Big|_{p=1} &= \frac{\mathcal{L}}{\mathcal{L} + 0.01} \\
&\approx 1.0000 \quad \textbf{for} \quad \mathcal{L} \sim 2.4 \times 10^{4} \\
\end{aligned}
\end{equation}
$$

- The printed `loss` is therefore $2.0000$ independent of `loss_u` and
`loss_v`.
