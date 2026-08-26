# Linear-Attention T2 Stability Record

Date: 2026-08-26

## Failure analysis

The sparse QK score path and the complement linear path have different
instability mechanisms.  The existing QK Clip limits sparse attention logits
after each optimizer step.  It cannot prevent the linear complement from
encountering a singular denominator on that step's first forward pass.

The linear JVP quotient has a T2 contribution:

$$
\begin{equation}
\begin{aligned}
\text{T2} =
-\frac{o_{\ell}\left(dq_{\phi}^{\mathsf T}z_c+
q_{\phi}^{\mathsf T}dz_c\right)}
{q_{\phi}^{\mathsf T}z_c}.
\end{aligned}
\end{equation}
$$

Ordinary channel softmax allows an arbitrarily negative key channel, making
the complement state `z_c` arbitrarily close to zero.  This makes T2 and the
same denominator-dependent reverse-mode gradient unbounded.  The observed
pattern of finite sparse statistics followed by skipped/non-finite gradients
is therefore consistent with the linear branch, not evidence that sparse QK
logits alone are the cause.

## Change

The linear feature map is now

$$
\begin{equation}
\begin{aligned}
\phi_{\varepsilon}(x) &=
(1-\varepsilon)\operatorname{softmax}(x)+\frac{\varepsilon}{D},\\
d\phi_{\varepsilon}(x) &=
(1-\varepsilon)d\operatorname{softmax}(x).
\end{aligned}
\end{equation}
$$

For `N_c` complement tokens it guarantees

$$
\begin{equation}
\begin{aligned}
q_{\phi}^{\mathsf T} z_c \geq N_c\frac{\varepsilon}{D}.
\end{aligned}
\end{equation}
$$

`config.model.linear_den_floor` sets the target floor and the cube-attention
module derives `epsilon = D * floor / N_c`.  The same map is used by the
training forward/backward, no-gradient path, and fused JVP state builder.
Input geometry and model size are unchanged.  Existing QK Clip remains as the
sparse branch's separate guard.

## Verification

- `tests/test_linear_t2_stability.py`
  - analytic JVP agrees with `torch.func.jvp` in float64;
  - adversarial one-hot keys: denominator `4.135000e+02` is above bound
    `1.033750e-01`, and T2 is finite;
  - CUDA custom-autograd plus fused-JVP smoke test passed with
    `epsilon=5e-3`, finite input gradient, maximum magnitude `1.22e-4`.
- `tests/test_sla2_cube_qat.py`
  - legacy `epsilon=0` compatibility passed: QAT JVP relative primal error
    `9.74e-03`, tangent error `1.08e-02`, and finite module gradients.

## Deployment decision

The step-7000 checkpoint is not a valid resume point: gradient skips began
before it was saved.  Resume from the known finite `step_0006000.pt` with the
new bounded map and preserve all A100 input/model settings.
