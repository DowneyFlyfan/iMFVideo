# Target

- Move improved Mean Flow method to video models

# TO BE Improved and Ascertained

- Parameter Tune

- Wan2.1 ?

- Align Parameters/optimizers for model comparisons

- Examine the efficiency of iMF and see if we can further improve this arch

- in self-attention, $\sqrt{d}$ is always not needed. Since you can put it in initialization

# MuP Rules for Special Layers (verified)

- Deduced from `kexue.fm/archives/10795, 11605, 11647` (spectral condition / Muon steepest descent / special-case layers), anchored at reference width $d_0 = 256$; full derivations and settings in [records/mup_special_layers.md](records/mup_special_layers.md).

| Layer | MuP class | Init std rule | Current code | lr width factor (deduced / current) |
|---|---|---|---|---|
| attn/mlp/out res score heads `Linear(d,1)` | LM-Head | $\frac{0.5}{\sqrt{d_0}}\cdot\frac{d_0}{d}$ | $0.5/\sqrt{d}$ (right at $d_0$, wrong exponent beyond) | $d_0/d$ / $1$ |
| token banks + label embedding | Embedding | $\Theta(1)$: constant $=\sqrt{d}$ | $1/\sqrt{d}$ ($\sqrt{d}$ too small) | $1$ / $1$ (Adam, correct) |
| RMSNorm gains, res gains | Hadamard | $1$ (correct) | $1$ | $1$ / $1$ (Adam $\approx$ SignSGD, correct) |
| attn_scale / mlp_scale | Hadamard (ReZero) | $0$ by design (correct) | $0$ | $1$ / $1$ (correct) |
| x_embedder Conv3d | Input | $\Theta(1/\sqrt{64})$ width-free | Xavier $\sqrt{2/(64+d)}$ (wrong exponent) | $1$ / $1$ (correct) |
| u/v final layers | Output | $0$ (correct) | $0$ | $d_0/d$ / $1$ |
| g_proj + hidden matrices | Linear | aspect rule (correct) | $0.5/\sqrt{d_{in}}$ | Moonlight shape scale (correct) |

- Key deduction: the residual score heads are LM-Head-type layers, since logits come from RMS-normed snapshots, so both init std and lr must carry a $1/d$ width factor, while Muon's $\mathrm{msign}$ on a $1 \times d$ matrix already reduces to exactly the prescribed Normalized SGD:

$$
\begin{equation}
\begin{aligned}
|\ell_j| &\le d\,\Vert w\Vert_{RMS},\quad
|\Delta\ell_j| \le d\,\Vert\Delta w\Vert_{RMS}
\ \Rightarrow\ \sigma_w = \Theta(1/d),\ \eta_{res} \propto 1/d \\
\Vert E_i\Vert_{RMS} &= \Theta(1) \Rightarrow \sigma_E = \Theta(1),\quad
\mathrm{msign}(G_{1\times d}) = G/\Vert G\Vert_2
\ \textbf{(Normalized SGD)}
\end{aligned}
\end{equation}
$$

- Verified by a width coordinate check ($d \in \{64, 128, 256\}$, 10 optimizer steps each, real 29k-token latents, Moonlight, fixed lr $2\times 10^{-3}$): every predicted exponent measured within $\pm 0.03$:

| quantity | prediction | d=64 | d=128 | d=256 | measured exponent |
|---|---|---|---|---|---|
| token-bank row RMS, current init | $\propto d^{-1/2}$ | 0.126 | 0.089 | 0.062 | $-0.51$ |
| token-bank row RMS, constant $=\sqrt{d}$ | $\Theta(1)$ | 1.00 | 1.00 | 0.99 | $-0.01$ |
| res-head logit std at init | $\Theta(1)$ | 0.25 | 0.23 | 0.24 | $-0.03$ |
| res-head $\Vert\Delta w\Vert_{RMS}$ per step | width-free | 1.6e-4 | 1.4e-4 | 1.6e-4 | $+0.01$ |
| res-head logit drift bound per step | $\propto d$ | 0.0099 | 0.0184 | 0.0403 | $+1.01$ |
| token-bank Adam $\Vert\Delta E\Vert_{RMS}$ per step | width-free | 1.1e-4 | 1.2e-4 | 1.4e-4 | $+0.17$ |

# Latest Loss Curve

![tuning_4k_loss_curves](records/tuning_4k_loss_curves.png)

# Done

- JVP MLA Kernel

- Moonlight (scaled + weight decay version of muon)
