# Linear-Attention T2 Stability Design

## Diagnosis target

The candidate failure is the complement linear-attention quotient tangent.
For feature maps `phi_q` and `phi_k`, the relevant computation is

$$
\begin{equation}
\begin{aligned}
o_{\ell} &= \frac{q_{\phi}^{\mathsf T} H_c}{q_{\phi}^{\mathsf T} z_c},\\
d o_{\ell} &=
\frac{d q_{\phi}^{\mathsf T} H_c + q_{\phi}^{\mathsf T} dH_c
- o_{\ell}(d q_{\phi}^{\mathsf T}z_c + q_{\phi}^{\mathsf T}dz_c)}
{q_{\phi}^{\mathsf T} z_c}.
\end{aligned}
\end{equation}
$$

The second quotient term is T2.  With an ordinary channel softmax, an
arbitrarily negative key channel makes a component of `z_c` arbitrarily
small.  Therefore a finite checkpoint can create an arbitrarily small
denominator during its very next forward pass.  An after-step QK Clip cannot
protect that first pass.

The production diagnostic records the linear denominator and T2 maximum next
to the existing sparse MaxLogit statistic.  The diagnosis is confirmed when a
non-finite or growing JVP has a collapsing linear denominator/T2 while the
sparse score statistic remains finite.

## Stabilizing invariant

Replace the linear-branch feature map only with an epsilon-smoothed channel
softmax:

$$
\begin{equation}
\begin{aligned}
\phi_{\varepsilon}(x) &=
(1-\varepsilon)\operatorname{softmax}(x)+\frac{\varepsilon}{D},\\
d\phi_{\varepsilon}(x) &=
(1-\varepsilon)d\operatorname{softmax}(x),\\
q_{\phi}^{\mathsf T}z_c &\geq
N_c\frac{\varepsilon}{D}.
\end{aligned}
\end{equation}
$$

Here `D` is the head dimension and `N_c` is the number of complement keys.
The last inequality follows because each query feature sums to one and every
smoothed key feature is at least `epsilon / D`.  It is a forward invariant,
not a heuristic threshold or a post-hoc clip.

`linear_den_floor` in `config.py` specifies the desired lower bound.  At
construction, `epsilon = D * linear_den_floor / N_c`, clamped below one.  For
the A100 run (`D=64`, `N_c≈26464`, floor `0.1`) this is about `2.42e-4`, so
the kernel remains effectively the original softmax in the healthy regime.

The existing QK Clip remains enabled for the sparse INT8 quantization-aware
training branch; it solves the distinct sparse MaxLogit problem described by
Su Jianlin's QK-Clip article.  The old post-step linear-range clip is retained
only as a conservative guard during migration, not as the correctness basis.

## Scope and acceptance criteria

- Do not change input shape, sequence length, model size, routing, or data.
- Apply exactly the same feature map and derivative in training forward,
  training backward, guidance forward, and the fused JVP path.
- Add tests for the analytic derivative and for an adversarial one-hot key
  distribution where ordinary softmax would make the denominator collapse.
- Run a local finite-gradient smoke test before synchronizing to Nautilus.
- Resume from the known finite step-6000 checkpoint, not the already-skipping
  step-7000 checkpoint.
