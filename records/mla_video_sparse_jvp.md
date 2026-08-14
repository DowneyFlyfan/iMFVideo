# MLA_Video_Sparse_JVP (VSA x SLA2 fused Triton kernel)

- Goal: combine VSA (Video Sparse Attention, `../fastvideo`, "Faster Video Diffusion with Trainable Sparse Attention") with SLA2 (`../SLA/SLA2`) into one attention op for the MLA pipeline, with forward-mode JVP: Triton kernel `_mla_video_sparse_jvp_kernel` in [models/mla_video_sparse_jvp.py](../models/mla_video_sparse_jvp.py).

## Composition

- Tokens are re-ordered TILE-MAJOR (3D video tiles of $E$ tokens; the $P$ conditioning prefix tokens ride at the end as one ragged, always-attended tail block). Per head:

$$
\begin{equation}
\begin{aligned}
O &= g \odot O_c + a \odot O_s + (1 - a) \odot O_l \\
O_c &= \mathrm{softmax}(Q_c K_c^{\top} d^{-1/2})\,V_c
\quad \textbf{(VSA compression branch, pooled tiles, broadcast)} \\
O_s &= \mathrm{softmax}\big(Q K^{\top} d^{-1/2}\ \textbf{on top-k tiles
+ prefix tail}\big) V \\
O_l &= \frac{(\phi(Q)\phi(K)^{\top} \odot \bar{M})\,V}
{(\phi(Q)\phi(K)^{\top} \odot \bar{M})\,\mathbf{1} + \epsilon_l}
\quad \textbf{(SLA2 complement over non-selected tiles)} \\
LUT &= \mathrm{topk}_{video\ tiles}\big(Q_c K_c^{\top} d^{-1/2}\big)
\quad \textbf{(routing = the coarse scores, VSA-style trainable)}
\end{aligned}
\end{equation}
$$

- Differences vs the plain SLA2 fusion (records/sla2_mla_jvp.md): 1D bq/bk blocks become 3D video tiles; SLA2's separate pooled router is replaced by VSA's coarse branch, which both routes and contributes gated output $g \odot O_c$ (routing trainable through $g$); the conditioning prefix is always visible to every query in the sparse branch and excluded from the linear complement.

- JVP structure: hard LUT is piecewise constant (zero tangent a.e., reused from the primal); $(g, dg)$ enter by the product rule; coarse branch = small dense softmax JVP in torch on $(Np{+}1)^2$ pooled scores; sparse branch = routed-tile flash-JVP recurrence in the Triton kernel (dual accumulators, interleaved $[dq|q]$ dot of width $2d$, $T{+}1$ key-block iterations with the ragged prefix tail masked); linear branch = per-tile states $H_b = \sum k_\phi v^{\top}$, $z_b = \sum k_\phi$, complement by global-minus-gathered, tangents by bilinearity (einsum).

- Geometry fit: full latents (16, 20, 56, 104), patch (1, 2, 2) give grid (20, 28, 52); tile (4, 4, 4) divides it exactly: $Np = 5 \cdot 7 \cdot 13 = 455$ tiles $\times\ 64 = 29{,}120$ patch tokens, zero padding; prefix 18 = ragged tail block.

## Verification (tests/test_mla_video_sparse_jvp.py)

- Settings: grid (4, 8, 8), tile (2, 4, 4) so $E = 32$, $Np = 8$, prefix 18, $L = 274$ (ragged), B = 2, H = 2, D = 64, topk 0.25, per-(head, block) alpha, random per-token gate WITH nonzero gate tangent. Reference: `torch.func.jvp` over the dense implementation `mvs_dense_ref`. Relative Frobenius error:

| path | o | do |
|---|---|---|
| fp32 (IEEE dots) | 2.28e-07 | 3.01e-07 |
| fp16 (fp16 MMA) | 5.91e-04 | 7.89e-04 |

- Tile permutation round trip (`tile_permutation`) exact.

- Benchmark at the production 29k-token shape deferred: the GPU is held by the P_mean/P_std/data_proportion tuning sweep; run `_sparse_jvp` cost is bounded by the SLA2 measurement (same recurrence, x3.35 over dense flash JVP at 3% routed fraction) plus the coarse branch ($456^2$ scores, negligible) and the same linear-states einsums.

## Not yet done

- Model integration (an `attn_impl` counterpart of SLA2AttentionImpl with the gate projection producing $g$, plus the fast-path branch in mla_jvp_fast) — the op-level contract matches the SLA2 one, so the wiring is the same pattern.
- Production-shape benchmark and a training A/B.
