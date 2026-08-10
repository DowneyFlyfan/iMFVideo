# Moonlight Optimizer: Muon On Matrices, AdamW On Vectors And Scalars

- Date: 2026-07-29

## Change

- Added `moonlight.py` implementing Moonshot AI's Moonlight variant of Muon
(MomentUm Orthogonalized by Newton-Schulz), and made it the default optimizer
via `config.optim.optimizer = "moonlight"`. `"adamw"` keeps the previous
behaviour.

- Muon takes the 2-D hidden weight matrices. Every 1-D parameter and every
embedding-like or output-like matrix goes to AdamW, in one optimizer with two
parameter groups tagged `use_muon`.

## Update Rule

- Momentum, optional Nesterov lookahead, Newton-Schulz orthogonalization, then
a decoupled weight decay and an RMS-matched step:

$$
\begin{equation}
\begin{aligned}
M_t &= \mu M_{t-1} + G_t \\
D_t &= G_t + \mu M_t \quad \textbf{(Nesterov)}, \quad D_t = M_t
\quad \textbf{(otherwise)} \\
\Phi_t &= \textbf{NewtonSchulz}_5(D_t) \\
s_{nm} &= 0.2 \sqrt{\max(n, m)} \\
W_t &= (1 - \eta_t \lambda) W_{t-1} - \eta_t s_{nm} \Phi_t \\
\end{aligned}
\end{equation}
$$

- Newton-Schulz iterates the quintic with the fixed coefficients
$(a, b, c) = (3.4445, -4.7750, 2.0315)$:

$$
\begin{equation}
\begin{aligned}
X_0 &= \frac{D_t}{\|D_t\|_F + \epsilon} \\
X_{k+1} &= a X_k + b X_k X_k^{\top} X_k
+ c (X_k X_k^{\top})^2 X_k \\
\end{aligned}
\end{equation}
$$

## Coefficients Do Depend On Shape

- Two statements that are both true and easy to conflate.

> A FIXED $(a, b, c)$ is shape-blind. The iteration acts as a scalar polynomial
applied elementwise to the singular values,

$$
\begin{equation}
\begin{aligned}
\sigma &\mapsto p(\sigma) = a \sigma + b \sigma^{3} + c \sigma^{5} \\
\end{aligned}
\end{equation}
$$

> and $p$ has no access to $n$ or $m$. Verified: five different shapes given the
same input spectrum produce output spectra agreeing to $1.73\times 10^{-6}$,
all landing in $[0.6822, 1.1344]$ under Jordan's triple.

> The OPTIMAL $(a, b, c)$ is not shape-blind. Newton-Schulz starts from
$X_0 = M/\|M\|_F$, and the distribution of $\sigma_i/\|M\|_F$ depends on the
shape: for an $(n, m)$ Gaussian the squared singular values follow the
Marchenko-Pastur law in the aspect ratio, and dividing by
$\|M\|_F \sim \sqrt{nm}$ makes the individual values scale as
$1/\sqrt{\min(n, m)}$. Larger and squarer matrices therefore start with smaller,
more spread-out singular values and need a more aggressive polynomial.

- Following Su Jianlin (https://kexue.fm/archives/10592), after @leloykun, the
quintic is reparameterized by its fixed points $\{0, \pm x_1, \pm x_2\}$:

$$
\begin{equation}
\begin{aligned}
g(x) &= x + \kappa x (x^2 - x_1^2)(x^2 - x_2^2) \\
a &= 1 + \kappa x_1^2 x_2^2, \quad
b = -\kappa (x_1^2 + x_2^2), \quad c = \kappa \\
\end{aligned}
\end{equation}
$$

- $(\kappa, x_1, x_2)$ are then fitted by gradient descent to minimize
$\mathbb{E}[(g^{T}(\sigma) - 1)^2]$ over that shape's empirical spectrum.
Implemented in `moonlight.py:solve_ns_coefficients`, cached per
$(\min(n,m), \max(n,m), T)$ since $\sigma(M) = \sigma(M^{\top})$.

- Jordan's fixed triple $(3.4445, -4.7750, 2.0315)$ is approximately the
square-matrix, $T = 5$ optimum, which is why it does poorly on rectangular
weights.

### Solver Validated Against The Published Table

| $n$ | $m$ | $T$ | ours $(a, b, c)$ | Su $(a, b, c)$ | our mse | Su mse |
|---|---|---|---|---|---|---|
| 1024 | 1024 | 3 | +4.329, -9.666, +7.018 | +4.328, -9.666, +7.020 | 0.10250 | 0.10257 |
| 1024 | 1024 | 5 | +3.303, -4.137, +1.721 | +3.297, -4.136, +1.724 | 0.02739 | 0.02733 |
| 2048 | 1024 | 3 | +4.089, -9.260, +6.941 | +4.095, -9.327, +7.028 | 0.01631 | 0.01628 |
| 2048 | 1024 | 5 | +2.644, -3.128, +1.477 | +2.644, -3.128, +1.476 | 0.00038 | 0.00038 |

### Measured Gain Over The Fixed Triple

- Residual $\mathbb{E}[(\sigma_T - 1)^2]$ at $T = 5$:

| Shape | Jordan mse | fitted mse | improvement |
|---|---|---|---|
| (256, 256) | 0.04369 | 0.01577 | 2.8x |
| (1024, 256) | 0.02334 | 0.00000 | 5,979x |
| (2048, 256) | 0.02076 | 0.00000 | 4,313x |
| (1024, 128) | 0.04617 | 0.00000 | 905,358x |

- The gain is small on square matrices, which is what Jordan tuned for, and
enormous on rectangular ones. This model's Muon-side weights are almost all
rectangular.

- Also note the separate per-shape update scale $0.2\sqrt{\max(n, m)}$ in
`Moonlight.muon_lr_scale`, which is an independent mechanism from the
coefficients.

## Why The Update Scale Is 0.2 sqrt(max(n, m))

- Newton-Schulz drives $M$ toward $\Phi = U V^{\top}$ with $r = \min(n, m)$
orthonormal columns per side, so

$$
\begin{equation}
\begin{aligned}
\|\Phi\|_F^2 &= \operatorname{tr}(\Phi^{\top}\Phi)
= \operatorname{tr}(V U^{\top} U V^{\top})
= \operatorname{tr}(V^{\top} V) = r \\
\textbf{RMS}(\Phi) &= \frac{\|\Phi\|_F}{\sqrt{nm}}
= \sqrt{\frac{r}{nm}} = \frac{1}{\sqrt{\max(n, m)}} \\
\end{aligned}
\end{equation}
$$

- That shrinks as a layer widens, so one learning rate would under-train wide
layers. Multiplying by $0.2\sqrt{\max(n,m)}$ pins the update RMS at 0.2 for
every shape. AdamW's update entries are roughly $\pm \eta$ because
$m/\sqrt{v}$ is sign-like, i.e. RMS $\approx \eta$, which is what makes an
AdamW-tuned learning rate transfer.

## Parameter Routing

- Rule in `moonlight.py:split_params`: a parameter goes to AdamW if
`ndim < 2`, or if its name contains `embedding_table`, `_tokens`, `_embedder`,
`x_embedder` or `final_layer`. Everything else goes to Muon.

- At the full config (`hidden_size=1024`, `depth=19`, `num_heads=16`,
`num_classes=1000`): Muon 184 tensors / 258.1M params (97.5%), AdamW 214
tensors / 6.6M params (2.5%).

- The 2-D tensors deliberately excluded from Muon:

```
time_tokens                          (4, 1024)
class_tokens                         (8, 1024)
omega_tokens                         (4, 1024)
t_min_tokens                         (2, 1024)
t_max_tokens                         (2, 1024)
x_embedder.proj.weight               (1024, 16, 1, 2, 2)
h_embedder.mlp.0.weight              (1024, 256)
h_embedder.mlp.2.weight              (1024, 1024)
omega_embedder.mlp.0.weight          (1024, 256)
omega_embedder.mlp.2.weight          (1024, 1024)
cfg_t_start_embedder.mlp.0.weight    (1024, 256)
cfg_t_start_embedder.mlp.2.weight    (1024, 1024)
cfg_t_end_embedder.mlp.0.weight      (1024, 256)
cfg_t_end_embedder.mlp.2.weight      (1024, 1024)
y_embedder.embedding_table.weight    (1001, 1024)
u_final_layer.linear.weight          (64, 1024)
v_final_layer.linear.weight          (64, 1024)
```

- Muon therefore receives exactly the MLA projections (`q_a_proj`, `q_b_proj`,
`kv_a_proj`, `kv_b_proj`, `out_proj`) and the SwiGLU MLP projections (`w1`,
`w2`, `w3`) in all 19 blocks.

## Settings

- Environment: `.venv` from `/opt/miniconda3/bin/python3`, torch 2.11.0,
macOS Darwin 25.5.0, MPS available.

| Knob | Default | Meaning |
|---|---|---|
| `optimizer` | `"moonlight"` | `"moonlight"` or `"adamw"` |
| `muon_momentum` | 0.95 | $\mu$ |
| `muon_nesterov` | True | step along $G + \mu M$ |
| `muon_ns_steps` | 5 | Newton-Schulz iterations |
| `muon_lr_scale_constant` | 0.2 | target update RMS |
| `muon_coeff_mode` | `"per_shape"` | `"per_shape"` or `"jordan"` |
| `weight_decay` | 1e-5 | see the warning below |

### Startup Cost Of The Per-Shape Solve

- At the full config the 184 Muon tensors collapse to 7 distinct shapes, solved
once in 23.0 s on CPU:

```
Moonlight: fitting Newton-Schulz coefficients for 7 distinct shapes (steps=5)
  NS coeffs (  272, 1024) T=5: a=+2.5459 b=-2.4806 c=+0.9366 mse=5.862e-06
  NS coeffs (  512, 1024) T=5: a=+2.8971 b=-3.0681 c=+1.1623 mse=4.489e-04
  NS coeffs ( 1024, 1024) T=5: a=+3.2952 b=-4.1289 c=+1.7211 mse=2.735e-02
  NS coeffs ( 1024, 2730) T=5: a=+2.5415 b=-2.8626 c=+1.3227 mse=2.045e-04
  NS coeffs (  256, 1792) T=5: a=+2.8810 b=-3.1185 c=+1.2396 mse=7.710e-06
```

- The square (1024, 1024) `out_proj` has residual $2.7\times10^{-2}$, three to
four orders of magnitude worse than any rectangular shape. This reproduces Su's
remark that non-square matrices converge more easily than square ones, and its
fitted triple matches his 1024/1024/T=5 row.

- Newton-Schulz compute dtype defaults to bfloat16 on CUDA and float32
elsewhere, because MPS and CPU bfloat16 matmul coverage is patchier.

## Results

### Newton-Schulz Orthogonality, Well-Conditioned Input

- Input spectrum $\sigma \in [0.3, 1.0]$ linearly spaced, 5 steps:

```
( 64, 64) sigma in [0.682, 1.134], ||X||_F^2=50.9 (r=64)
(128, 64) sigma in [0.682, 1.134], ||X||_F^2=50.9 (r=64)
( 64,128) sigma in [0.682, 1.134], ||X||_F^2=50.9 (r=64)
(256, 48) sigma in [0.682, 1.133], ||X||_F^2=36.3 (r=48)
( 48,256) sigma in [0.682, 1.133], ||X||_F^2=36.3 (r=48)
```

- $\|X\|_F^2 / r = 0.795$ rather than 1, because the singular values sit in
$[0.68, 1.14]$ instead of exactly at 1. This is the whole reason the measured
update RMS lands near 0.18 instead of exactly 0.20:
$\sqrt{0.795} \times 0.2 = 0.178$.

### Newton-Schulz On Near-Singular Input

```
square (256,256) Gaussian: 5 steps -> sigma_min=0.0834 median=0.863; 10 steps -> sigma_min=0.6818
tall   (1024,256) Gaussian: 5 steps -> sigma in [0.682, 1.134] (well conditioned)
```

- A square iid Gaussian is near-singular, with $\sigma_{\min} \approx 10^{-4}$
after Frobenius normalization. Since $p(\sigma) \approx a\sigma$ for small
$\sigma$, five steps amplify the small tail by only about $a^5$, so
$\sigma_{\min}$ stays at 0.08. The bulk still reaches ~0.86, which is what
makes the update a usable direction. Tall and wide Gaussians are well
conditioned and do reach the full envelope at 5 steps.

- Relevance to this model: `out_proj` is square (1024, 1024), as is
`h_embedder.mlp.2` (which is on the AdamW side anyway). Raising
`muon_ns_steps` to 10 would tighten those, at roughly double the Newton-Schulz
cost.

### RMS Scale Shape-Invariance

```
(  64,  64) scale=  1.600 update RMS=0.1740
(1024, 256) scale=  6.400 update RMS=0.1909
( 256,1024) scale=  6.400 update RMS=0.1913
(2048,  64) scale=  9.051 update RMS=0.1464
( 128,4096) scale= 12.800 update RMS=0.1626
unscaled RMS ratio (64,64)/(128,4096) = 8.58, predicted sqrt(4096/64) = 8.00
```

- Scaled RMS stays in [0.146, 0.191] across a 64x span of aspect ratios,
against a 0.2 target. Without the scale the RMS would vary by the measured
8.58x, matching the predicted $\sqrt{4096/64} = 8.00$.

### AdamW Branch Against The Reference

```
max |ours - torch.optim.AdamW| after 20 steps: 5.96e-08
```

### Training Loop

- `optimizer="moonlight"`, `weight_decay=0.1`, `hidden_size=128`, `depth=3`,
`grad_accum=2`, `num_workers=2`, MPS:

```
Moonlight split: Muon 32 tensors / 0.8M params, AdamW 62 tensors / 0.2M params
step 2 loss=2.0000 loss_u=2.0151 loss_v=2.0151 grad_norm=0.116 lr=1.00e-04
step 4 loss=2.0000 loss_u=1.9888 loss_v=1.9888 grad_norm=0.141 lr=3.25e-05
resumed from .../step_0000004.pt at step 4
step 6 loss=2.0000 loss_u=2.0138 loss_v=2.0138 grad_norm=0.116 lr=4.50e-05
step 8 loss=2.0000 loss_u=1.9885 loss_v=1.9885 grad_norm=0.141 lr=1.45e-05
```

- Paths covered: fresh run, checkpoint save, resume with Muon momentum buffers
restored, the low-weight-decay warning, and the `"adamw"` fallback.

- `tests/test_moonlight.py` passes on CPU and MPS: shape-independence of the
coefficients, orthogonality envelope, near-singular limitation, RMS
shape-invariance, parameter split coverage and rule, AdamW reference match,
learning-rate wiring across both groups, `state_dict` round trip for 222
parameters, and a 6-step end-to-end run with both branches confirmed moving.

## Open Item

- `config.optim.weight_decay` is 1e-5. Moonlight's first stated finding is that
weight decay is what makes Muon scale, and the paper's runs use $\lambda = 0.1$.
At 1e-5 the Muon branch is effectively undecayed and the published recipe does
not apply. `train.py:build_optimizer` prints a warning when
`weight_decay < 1e-3`; the value was left unchanged rather than altered
silently.
