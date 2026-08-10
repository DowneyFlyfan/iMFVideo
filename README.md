# Target

- Move improved Mean Flow method to video models

# TO BE Improved and Ascertained

- Parameter Tune

- Wan2.1 ?

- Align Parameters/optimizers for model comparisons

- Examine the efficiency of iMF and see if we can further improve this arch

- in self-attention, $\sqrt{d}$ is always not needed. Since you can put it in initialization

# Scaling Rules

- Deduced from `kexue.fm/archives/10770, 10795, 11605, 11647` (MuP primer / spectral condition / Muon steepest descent / special-case layers), anchored at reference width $d_0 = 256$. Verified by a width coordinate check ($d \in \{64, 128, 256\}$, 10 optimizer steps, real 29k-token latents, Moonlight, fixed lr $2\times 10^{-3}$); full derivations, settings and per-width numbers in [records/mup_special_layers.md](records/mup_special_layers.md).

| Layer | MuP class | Branch | Init std: deduced rule | Init std: current code | lr width factor: deduced / current | Verified exponent (predicted) |
|---|---|---|---|---|---|---|
| attn/mlp/out res score heads `Linear(d,1)` | LM-Head | Muon ($\mathrm{msign}_{1\times d}$ = Normalized SGD, correct direction) | $\frac{0.5}{\sqrt{d_0}}\cdot\frac{d_0}{d}$ | $0.5/\sqrt{d}$ (right at $d_0$, wrong exponent beyond) | $d_0/d$ / $1$ | logit std at init $-0.03$ ($0$); logit drift bound $+1.01$ ($+1$) |
| g_proj + hidden MLA/MLP matrices | Linear | Muon | aspect rule (correct) | $0.5/\sqrt{d_{in}}$ | $\sqrt{d_0/d}$ under Moonlight's Adjust-LR / $1$ | $\Vert\Delta W\Vert_2$ per step $+0.48, +0.44$ ($+1/2$) |
| token banks + label embedding | Embedding | AdamW | $\Theta(1)$: constant $=\sqrt{d}$ | $1/\sqrt{d}$ ($\sqrt{d}$ too small) | $1$ / $1$ (correct) | row RMS $-0.51$ ($-1/2$); fixed init $-0.01$ ($0$); $\Vert\Delta E\Vert_{RMS}$ $+0.17$ ($0$) |
| RMSNorm gains, res gains | Hadamard | AdamW $\approx$ SignSGD | $1$ (correct) | $1$ | $1$ / $1$ (correct) | analytic (diag $\mathrm{msign}$ = SignSGD) |
| attn_scale / mlp_scale | Hadamard (ReZero) | AdamW | $0$ by design (correct) | $0$ | $1$ / $1$ (correct) | analytic |
| x_embedder Conv3d | Input | AdamW | $\Theta(1/\sqrt{64})$ width-free | Xavier $\sqrt{2/(64+d)}$ (wrong exponent) | $1$ / $1$ (correct) | analytic (fan-in fixed) |
| u/v final layers | Output | AdamW | $0$ (correct) | $0$ | $d_0/d$ / $1$ | analytic (zero init) |

- Two key formulas behind the table — the res score heads are LM-Head-type layers (logits from RMS-normed snapshots), and Moonlight's Adjust-LR scale $0.2\sqrt{\max(n,m)} \propto \sqrt{d}$ makes hidden msign updates exact isometries (pure MuP-Muon $\Delta W = -\eta\sqrt{d_{out}/d_{in}}\,\mathrm{msign}$ would keep lr width-free instead):

$$
\begin{equation}
\begin{aligned}
|\ell_j| &\le d\,\Vert w\Vert_{RMS},\quad
|\Delta\ell_j| \le d\,\Vert\Delta w\Vert_{RMS}
\ \Rightarrow\ \sigma_w = \Theta(1/d),\ \eta_{res} \propto 1/d \\
\Vert x\,\Delta W \Vert_{RMS} &= \Vert \Delta W \Vert_2\,\Vert x\Vert_{RMS}
= 0.2\,\eta\sqrt{d}
\ \Rightarrow\ \eta_{muon}(d) = \eta\,\sqrt{d_0/d}
\end{aligned}
\end{equation}
$$

# Latest Loss Curve

![tuning_4k_loss_curves](records/tuning_4k_loss_curves.png)

# Done

- JVP MLA Kernel

- Moonlight (scaled + weight decay version of muon)
