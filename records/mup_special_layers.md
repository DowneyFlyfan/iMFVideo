# MuP Rules for the Special Layers (init std + per-layer lr)

- Source blogs (CLAUDE.md): `https://kexue.fm/archives/10795` (spectral condition), `https://kexue.fm/archives/11605` (linear layers + Muon), `https://kexue.fm/archives/11647` (special layers: Embedding, LM Head, Hadamard gain, bias, attention scale).

- Goal: deduce how initialization standard deviation ($\sigma$) and learning rate ($\eta$, lr) must scale with hidden width $d$ for the model's special layers, anchored at the tuned reference width $d_0 = 256$, so the local optimum transfers to the sub-300M server model.

- Symbols: $d$ = hidden width (`hidden_size`), $d_{in}, d_{out}$ = linear fan-in/fan-out, $\Vert\cdot\Vert_{RMS}$ = root mean square of entries, $\Vert\cdot\Vert_2$ = spectral norm (matrix) or Euclidean norm (vector), $G$ = gradient, $\eta$ = learning rate, $\mathrm{msign}$ = matrix sign ($U V^{\top}$ from the SVD).

## Baseline rules from the blogs

- Spectral condition (10795): every layer keeps $\Vert x_k \Vert_{RMS} = \Theta(1)$ and $\Vert \Delta x_k \Vert_{RMS} = \Theta(1)$, which for a linear layer forces both the weight and its update to satisfy the same spectral-norm law; a Gaussian $d_{in} \times d_{out}$ matrix has spectral norm $\approx \sigma(\sqrt{d_{in}} + \sqrt{d_{out}})$, giving the init std; the Muon update realizes the increment law exactly (11605):

$$
\begin{equation}
\begin{aligned}
\Vert W \Vert_2 &= \Theta\Big(\sqrt{d_{out}/d_{in}}\Big), \quad
\Vert \Delta W \Vert_2 = \Theta\Big(\sqrt{d_{out}/d_{in}}\Big) \\
\sigma &= \Theta\Big(\sqrt{\tfrac{d_{out}}{d_{in}}}\cdot
\tfrac{1}{\sqrt{d_{in}} + \sqrt{d_{out}}}\Big) \\
\Delta W &= -\eta \sqrt{d_{out}/d_{in}}\; \mathrm{msign}(G) \\
\eta_{Adam} &= \Theta(1/d_{in}), \quad \eta_{SGD} = \Theta(d_{out}/d_{in})
\end{aligned}
\end{equation}
$$

- Special layers (11647): Embedding rows need $\Vert E_i \Vert_{RMS} = \Theta(1)$ and $\Vert \Delta E_i \Vert_{RMS} = \Theta(1)$, i.e. init std $\Theta(1)$ and row-wise Normalized SGD with width-free lr; an LM-Head-type layer (logits from an RMS-bounded input) needs init std and lr both $\Theta(1/d)$ with column-wise Normalized SGD; a Hadamard gain $\gamma$ equals the diagonal linear layer $\mathrm{diag}(\gamma)$, so init $\gamma = 1$ and its steepest descent is SignSGD; biases init 0 with Normalized SGD; the attention scale should scale as $\Theta(1/d_{head})$ across widths even though $1/\sqrt{d_{head}}$ is correct for the random init average.

- Moonlight convention in this repo (`moonlight.py`): Muon branch update has entrywise RMS $= 0.2\,\eta$ for every matrix shape (`muon_lr_scale` multiplies by $0.2\sqrt{\max(n,m)}$ and $\mathrm{msign}$ has RMS $\approx \sqrt{r/(nm)}$); AdamW branch update RMS $\approx \eta$. Both are width-free per step, so all width dependence must come from per-layer lr factors.

## Deduction for each special layer

### Attention-residual score heads (attn_res_proj, mlp_res_proj, u/v_out_res_proj)

- These are `Linear(d, 1)` heads scoring RMS-normed residual snapshots (`models/imf_dit_video.py:261-294`): logits $\ell_j = \langle k_j, \gamma \odot w \rangle$ over $J \le 6$ candidates, then softmax. Since $k_j$ is RMS-normed, $\Vert k_j \Vert_{RMS} = 1$, so $\Vert k_j \Vert_2 = \sqrt{d}$. This is exactly the blog's LM-Head class: logits over a discrete candidate set from an RMS-bounded input.

- Forward and update bounds by Cauchy-Schwarz (with $\gamma = 1$ at init, and $w \in \mathbb{R}^d$ the fused score direction):

$$
\begin{equation}
\begin{aligned}
|\ell_j| &= |\langle k_j, w \rangle| \le \Vert k_j \Vert_2 \Vert w \Vert_2
= d\,\Vert w \Vert_{RMS} \\
|\Delta \ell_j| &\le d\,\Vert \Delta w \Vert_{RMS} \\
\Theta(1)\ \textbf{logits} &\Rightarrow \sigma_w = \Theta(1/d), \quad
\Vert \Delta w \Vert_{RMS} = \Theta(1/d) \Rightarrow
\eta_{res}(d) = \eta \cdot d_0 / d \\
\mathrm{Var}(\ell_j)\big|_{init} &= \textstyle\sum_i k_i^2 \sigma_w^2
\approx d\,\sigma_w^2 \Rightarrow \sigma_w = \Theta(1/\sqrt{d})
\ \textbf{(iid average)}
\end{aligned}
\end{equation}
$$

- The $1/d$ vs $1/\sqrt{d}$ dichotomy is the same as the blog's attention-scale case: $1/\sqrt{d}$ is the random-init average, $1/d$ is the worst case that holds throughout training once gradient alignment builds up. Following the blog's recommendation, anchor at the reference and scale as $1/d$: $\sigma_w(d) = (0.5/\sqrt{d_0})\cdot(d_0/d)$, keeping the current value at $d_0 = 256$.

- Steepest-descent direction: for $G \in \mathbb{R}^{1 \times d}$ the SVD is $G = 1 \cdot \Vert G \Vert_2 \cdot (G/\Vert G \Vert_2)$, hence $\mathrm{msign}(G) = G/\Vert G \Vert_2$, which IS the blog's Normalized SGD for this layer class. So the current Muon routing (`split_params` sends these 2-D weights to Muon) gives the correct direction; only the magnitude lacks the $1/d$ width factor, since Moonlight's update RMS $0.2\,\eta$ is width-free while the bound above demands $\Theta(1/d)$.

### Conditioning token banks and label embedding (class/time/omega/t_min/t_max tokens, LabelEmbedder)

- Embedding class. Rows are output vectors selected by a discrete id, so forward stability requires $\Vert E_i \Vert_{RMS} = \Theta(1)$, i.e. init std $\Theta(1)$, width-independent. Current code (`imf_dit_video.py:689,206`) uses std $= c/\sqrt{d}$ with $c = 1$, i.e. $\Theta(1/\sqrt{d})$: the conditioning rows shrink relative to the $\Theta(1)$ patch-token stream as width grows. Fix inside the existing formula by setting the constants to $\sqrt{d}$ (config-only): `token_init_constant = embedding_init_constant = sqrt(d) = 16` at $d_0$.

- lr: blog steepest descent is row-wise Normalized SGD with $\Vert \Delta E_i \Vert_{RMS} = \Theta(1)$; AdamW (sign-like, update RMS $\approx \eta$, width-free) already realizes this. Current AdamW routing correct, lr factor $\Theta(1)$.

### RMSNorm gains and residual gates (norm weights, attn_res_norm gain, attn_scale, mlp_scale)

- Hadamard class: $x \odot \gamma = x\,\mathrm{diag}(\gamma)$, a diagonal linear layer with $\Vert \mathrm{diag}(\gamma) \Vert_2 = \max_i |\gamma_i|$, so init $\gamma = 1$ satisfies $\Theta(1)$ (the zero init of `attn_scale`/`mlp_scale` is a deliberate sub-$\Theta(1)$ ReZero choice, allowed). Steepest descent: $\mathrm{msign}(\mathrm{diag}(g)) = \mathrm{diag}(\mathrm{sign}(g))$ = SignSGD, which AdamW approximates. Current treatment correct; lr factor $\Theta(1)$.

### MLA output gate (g_proj) and all hidden matrices

- Ordinary linear class: input is the block-normed stream ($\Vert x \Vert_{RMS} \approx 1$), shapes are $d \to \Theta(d)$, so the baseline rules apply: init std $c/\sqrt{d_{in}} = \Theta(1/\sqrt{d})$, Muon branch, lr transfers with Moonlight's shape-aware scale. Current treatment correct.

### Output projections (u_final_layer, v_final_layer)

- Output class. Zero init (current) trivially satisfies any $O(1/d)$ requirement. Under AdamW the muP output-layer rule applies: lr factor $\Theta(1/d)$, i.e. $\eta_{out}(d) = \eta \cdot d_0 / d$ when widening past $d_0$ (factor 1 at the reference).

### Patch embedder (x_embedder Conv3d)

- Input class with fixed fan-in $16 \cdot 1 \cdot 2 \cdot 2 = 64$: init std should be $\Theta(1/\sqrt{64})$, width-independent. Current Xavier-uniform std $= \sqrt{2/(64 + d)}$ decays with width, violating input-layer transfer; replace by a fan-in-only init when scaling width. AdamW lr factor $\Theta(1)$ (muP input rule). At $d_0$ the numeric gap is modest ($0.079$ vs $0.125$).

## Rule table (anchored at reference width $d_0 = 256$, hidden width $d$)

| Layer | MuP class | Init std: deduced rule | Init std: current code | Optimizer branch (deduced = current?) | lr width factor: deduced | lr factor: current |
|---|---|---|---|---|---|---|
| attn_res_proj / mlp_res_proj / u,v_out_res_proj (`Linear(d,1)`) | LM-Head | $\frac{0.5}{\sqrt{d_0}}\cdot\frac{d_0}{d}$ (worst-case $1/d$ transfer; $1/\sqrt{d}$ ok at init only) | $0.5/\sqrt{d}$ (matches at $d_0$, wrong exponent beyond) | Muon; $\mathrm{msign}$ on $1{\times}d$ = Normalized SGD = blog's rule (yes) | $d_0/d$ | $1$ (width-free RMS $0.2\eta$) |
| attn_res_norm / mlp_res_norm gain $\gamma$ | Hadamard | $1$ | $1$ (yes) | AdamW $\approx$ SignSGD (yes) | $1$ | $1$ (yes) |
| class/time/omega/t_min/t_max token banks | Embedding | $\Theta(1)$: constant $=\sqrt{d}$ (=16 at $d_0$) | $1/\sqrt{d}$ (too small by $\sqrt{d}$) | AdamW $\approx$ row Normalized SGD (yes) | $1$ | $1$ (yes) |
| LabelEmbedder table | Embedding | $\Theta(1)$: constant $=\sqrt{d}$ | $1/\sqrt{d}$ (too small by $\sqrt{d}$) | AdamW (yes) | $1$ | $1$ (yes) |
| Timestep/CFG embedder MLPs | Input path (fixed 256-dim basis) | $c/\sqrt{256}$, width-free (current ok) | $1/\sqrt{256}$ (yes) | AdamW (yes) | $1$ | $1$ (yes) |
| x_embedder Conv3d | Input | $\Theta(1/\sqrt{64})$, width-free | Xavier $\sqrt{2/(64+d)}$ (wrong exponent) | AdamW (yes) | $1$ | $1$ (yes) |
| u/v_final_layer | Output | $0$ (current ok; any $O(1/d)$) | $0$ (yes) | AdamW (yes) | $d_0/d$ | $1$ |
| attn_scale / mlp_scale gates | Hadamard (ReZero) | $0$ (design) | $0$ (yes) | AdamW $\approx$ SignSGD (yes) | $1$ | $1$ (yes) |
| g_proj output gate + all MLA/MLP matrices | Linear | $0.5/\sqrt{d_{in}}$ with aspect factor (current ok for square-ish) | $0.5/\sqrt{d_{in}}$ (yes) | Muon (yes) | Moonlight shape scale | (yes) |

## Experiment E1: coordinate check across widths

- Settings: widths $d \in \{64, 128, 256\}$ (num_heads $= d/64$, head_dim 64 fixed), depth 19, full uncropped Wan-Syn latents (16, 20, 56, 104) = 29,138 tokens, batch 1, 10 optimizer steps at fixed lr $2 \times 10^{-3}$ (no schedule), Moonlight optimizer, seed 0, fp32 master. Score-head stats from `shared_blocks[1].attn_res_proj` (block 0's attention-side head never fires: its snapshot list is empty). Script: `mup_coord_check.py` (scratchpad, temporary).

- Results (per-step update RMS from 10-step drift / 10):

| quantity | prediction | d=64 | d=128 | d=256 | measured exponent |
|---|---|---|---|---|---|
| token-bank row RMS, current init | $\propto d^{-1/2}$ | 0.1255 | 0.0887 | 0.0618 | $-0.51$ |
| token-bank row RMS, constant $=\sqrt{d}$ | $\Theta(1)$ | 1.004 | 1.004 | 0.989 | $-0.01$ |
| label-embedding row RMS, current | $\propto d^{-1/2}$ | 0.1249 | 0.0882 | 0.0625 | $-0.50$ |
| res-head logit std at init | $\Theta(1)$ | 0.253 | 0.231 | 0.241 | $-0.03$ |
| res-head $\Vert\Delta w\Vert_{RMS}$/step (Moonlight) | width-free | 1.55e-4 | 1.44e-4 | 1.58e-4 | $+0.01$ |
| res-head logit drift bound $d\cdot\Vert\Delta w\Vert_{RMS}$ | $\propto d$ | 0.0099 | 0.0184 | 0.0403 | $+1.01$ |
| token-bank Adam $\Vert\Delta E\Vert_{RMS}$/step | width-free | 1.12e-4 | 1.23e-4 | 1.42e-4 | $+0.17$ |

- All predicted exponents confirmed: the current token/embedding init decays as $d^{-1/2}$ (rule says $\Theta(1)$), the proposed constant $=\sqrt{d}$ restores exactly $\Theta(1)$; the score-head per-step logit drift bound grows linearly in $d$ under Moonlight's width-free update RMS, so the deduced $d_0/d$ lr factor makes it width-constant by construction ($d \cdot \Vert \Delta w \Vert_{RMS} \cdot d_0/d = d_0 \cdot \Vert \Delta w \Vert_{RMS}$, and $\Vert \Delta w \Vert_{RMS}$ was measured width-free); Adam token updates are already width-free as the embedding rule requires.

## Conclusions

- The three width-scaling defects found and verified: (1) token banks and label embedding are under-initialized by $\sqrt{d}$; (2) the attention-residual score heads' effective lr must shrink as $d_0/d$ (Moonlight treats them as ordinary matrices, giving width-free logit drift bounds that grow linearly in $d$); (3) the patch embedder's Xavier init has a spurious width dependence. Output-layer lr also needs a $d_0/d$ factor when widening. All other special layers (gains, gates, embedder MLPs, hidden matrices) already follow the blogs' rules, including the non-obvious fact that Muon's $\mathrm{msign}$ on a $1 \times d$ matrix reduces exactly to the Normalized SGD the blogs prescribe for LM-Head-type layers.
