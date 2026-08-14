# MLA_Video_Sparse_JVP (VSA x SLA2 fused Triton kernel)

- Goal: combine VSA (Video Sparse Attention, `../fastvideo`) with SLA2 (`../SLA/SLA2`) into one attention op for the MLA pipeline, with forward-mode JVP. Final spec: VSA's 3D cubes as blocks, SLA2's learnable router as the router, NO coarse branch — only the sparse and linear branches. Triton kernel `_mla_video_sparse_jvp_kernel` in [models/mla_video_sparse_jvp.py](../models/mla_video_sparse_jvp.py).

## Composition

- Tokens are re-ordered TILE-MAJOR (`tile_permutation`): 3D video tiles of $E$ tokens first, then the $P$ conditioning prefix tokens as one ragged, always-attended tail block. Per head:

$$
\begin{equation}
\begin{aligned}
O &= a \odot O_s + (1 - a) \odot O_l \\
O_s &= \mathrm{softmax}\big(Q K^{\top} d^{-1/2}\ \textbf{on top-k tiles
+ prefix tail}\big) V \\
O_l &= \frac{(\phi(Q)\phi(K)^{\top} \odot \bar{M})\,V}
{(\phi(Q)\phi(K)^{\top} \odot \bar{M})\,\mathbf{1} + \epsilon_l}
\quad \textbf{(complement over non-selected tiles)} \\
LUT &= \mathrm{topk}\big(\mathrm{pool}(Q)W_q\,(\mathrm{pool}(K)W_k)^{\top}
d^{-1/2}\big)\ \textbf{(SLA2 router, smooth-k)}
\end{aligned}
\end{equation}
$$

- From VSA: 3D-cube tiling (video-native blocks; tile (4, 4, 4) divides the production grid (20, 28, 52) exactly: $Np = 455$ tiles $\times\ 64 = 29{,}120$ patch tokens, zero padding; prefix 18 = ragged tail). From SLA2: learnable router (`route_tiles`: per-tile mean pooling, smooth-k mean-key subtraction, identity-initialized $D \times D$ proj_q / proj_k, hard top-k) and the sparse + complement-linear structure with per-(head, block) mixing ratio $a$.

- JVP: hard LUT piecewise constant (zero tangent a.e., reused from the primal); $a$ constant; sparse branch = routed-tile flash-JVP recurrence in the Triton kernel (dual accumulators, interleaved $[dq|q]$ dot of width $2d$, $T{+}1$ key-block iterations with the masked ragged prefix tail); linear branch = per-tile states $H_b = \sum k_\phi v^{\top}$, $z_b = \sum k_\phi$, complement = global minus LUT-gathered, tangents by bilinearity (einsum); $\phi$ = channel softmax with $d\phi = \phi \odot (dx - \langle\phi, dx\rangle)$.

## Verification (tests/test_mla_video_sparse_jvp.py)

- Settings: grid (4, 8, 8), tile (2, 4, 4) so $E = 32$, $Np = 8$, prefix 18, $L = 274$ (ragged), B = 2, H = 2, D = 64, topk 0.25, per-(head, block) alpha, NON-identity router projections (identity + 0.05 noise). Reference: `torch.func.jvp` over the dense implementation `mvs_dense_ref`. Relative Frobenius error:

| path | o | do |
|---|---|---|
| fp32 (IEEE dots) | 2.74e-07 | 3.21e-07 |
| fp16 (fp16 MMA) | 6.66e-04 | 8.42e-04 |

- Tile permutation round trip exact.

- Sparse-branch cost at the production 29k-token shape is bounded by the SLA2 measurement (same kernel recurrence: x3.35 over dense flash JVP at 3% routed fraction) plus the router pooling (negligible) and the same linear-states einsums; a direct benchmark is queued behind the tuning sweep holding the GPU.

## Not yet done

- Model integration (an `attn_impl` module owning proj_q / proj_k / alpha, plus the fast-path branch in mla_jvp_fast) — same wiring pattern as SLA2AttentionImpl.
- Production-shape benchmark and a training A/B.
