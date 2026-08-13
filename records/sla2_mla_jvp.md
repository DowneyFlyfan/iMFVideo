# SLA2 x MLA JVP Fusion (sparse-linear attention with forward-mode tangents)

- Goal: fuse SLA2 (Sparse-Linear Attention with learnable routing, `../SLA/SLA2`) with the MLA (multi-head latent attention) iMF pipeline, i.e. a JVP (Jacobian-vector product) multi-head latent sparse linear attention: SLA2 replaces the dense flash attention core in both the training forward (autograd) and the hand-rolled fast du/dt pass.

## Design

- SLA2 attention per head, with the mixing ratio $a$ and the block mask $M$ from a learnable router (hard top-k over pooled block scores):

$$
\begin{equation}
\begin{aligned}
O &= a \odot O_s + (1 - a) \odot O_l \\
O_s &= \mathrm{softmax}\big(Q K^{\top} d^{-1/2}\ \textbf{on routed blocks}\big) V \\
O_l &= \frac{(\phi(Q)\,\phi(K)^{\top} \odot (1 - M))\,V}
{(\phi(Q)\,\phi(K)^{\top} \odot (1 - M))\,\mathbf{1} + \epsilon_l},\quad
\phi = \mathrm{softmax}_{channel} \\
\end{aligned}
\end{equation}
$$

- JVP split (tangents w.r.t. network input only; the hard mask $M$, LUT and $a$ are piecewise constant, zero tangent a.e., reused from the primal):

$$
\begin{equation}
\begin{aligned}
dO_s &: \ \textbf{flash-JVP recurrence over routed blocks only} \\
dP &= P \odot (dS - \mathrm{rowsum}(P \odot dS)),\quad
dO_s = dP\,V + P\,dV \\
O_l &= \frac{q_\phi^{\top} H_c}{q_\phi^{\top} z_c + \epsilon_l},\quad
H_c = H_{all} - \textstyle\sum_{n \in LUT} H_b[n],\quad
H_b[n] = \textstyle\sum_{j \in n} k_{\phi j} v_j^{\top} \\
dO_l &= \frac{dq_\phi^{\top} H_c + q_\phi^{\top} dH_c
- O_l\,(dq_\phi^{\top} z_c + q_\phi^{\top} dz_c)}{q_\phi^{\top} z_c + \epsilon_l},
\quad d\phi = \phi \odot (dx - \langle \phi, dx \rangle)
\end{aligned}
\end{equation}
$$

- The sparse-branch tangent is bilinear in the same accumulators the flash-JVP kernel already streams, so the kernel is the existing `_flash_jvp_kernel_impl` (models/triton_block_jvp.py) with the key-tile loop replaced by a walk over the LUT entries (BLOCK_N = bk). The linear branch is bilinear in (phi-features, per-key-block states), so its tangent is plain batched einsum on (D, Dv) states, no custom kernel.

## Files

- [models/sla2_mla_jvp.py](../models/sla2_mla_jvp.py): `route_blocks` (pooled top-k router, identity-init projections), `_sla2_sparse_jvp_kernel` (routed-block flash JVP, fp16 MMA / IEEE-fp32 test path), `_linear_jvp` (complement states + tangents), `sla2_mla_jvp` (full op), `sla2_dense_ref` (dense reference for tests).
- [models/sla2_vendor/](../models/sla2_vendor/): vendored SLA2 triton backend (core, router, sparse-QAT + linear + fused kernels, SLA backward kernels) so server training needs no SLA checkout. CuTeDSL files not vendored; `backend="triton"`.
- [models/sla2_attention.py](../models/sla2_attention.py): `SLA2AttentionImpl`, a stateful per-block attn_impl for the training forward (autograd through the vendored kernels); creates the (H, Mb) `alpha_logit` eagerly so the Moonlight split sees it.
- [models/mla_jvp_fast.py](../models/mla_jvp_fast.py): `build_fast_jvp_state` snapshots alpha/router per block; `_attn_sublayer_jvp` branches to `sla2_mla_jvp` when the block carries SLA2.
- [models/imf_dit_video.py](../models/imf_dit_video.py): MLA accepts a module-factory attn_impl; IMFDiTVideo binds seq_len/num_heads into the factory.
- [config.py](../config.py): `attn_impl="sla2_jvp"` + `sla2_topk/sla2_bq/sla2_bk/sla2_alpha_init` (To Be Tuned).
- [moonlight.py](../moonlight.py): `alpha_logit` routed to AdamW (Hadamard-class mixing gates); router projections stay on Muon.
- Tests: [tests/test_sla2_jvp.py](../tests/test_sla2_jvp.py) (op vs `torch.func.jvp` over the dense reference), [tests/test_sla2_e2e.py](../tests/test_sla2_e2e.py) (d=256 model: loss fwd/bwd + fast du/dt over 8 seeds).

## Results

- Correctness (B=2, H=2, D=64, bq=128, bk=64, topk 0.25; relative Frobenius error vs `torch.func.jvp` of the dense reference):

| L | fp32 o / do | fp16 o / do |
|---|---|---|
| 512 | 3.4e-07 / 3.8e-07 | 7.6e-04 / 9.2e-04 |
| 640 | 3.3e-07 / 3.9e-07 | 7.6e-04 / 9.4e-04 |
| 583 (ragged tail) | 2.8e-07 / 3.7e-07 | 7.2e-04 / 9.1e-04 |

- Speed at production shape (B=1, H=4, L=29138, D=64, fp16; GPU shared with a training sweep, ratios reliable):

| topk | T blocks | sla2_jvp + route | dense flash JVP | speedup |
|---|---|---|---|---|
| 0.03 | 13 | 5.21 + 0.37 ms | 18.69 ms | x3.35 |
| 0.10 | 45 | 9.21 + 0.38 ms | 18.69 ms | x1.95 |
| 0.25 | 114 | 17.83 + 0.38 ms | 18.69 ms | x1.03 |

- At topk 0.03 the linear-branch einsums dominate (~4.5 of 5.2 ms); a Triton states kernel is the next optimization if SLA2 becomes the default.

- End-to-end (hidden 256, heads 4, depth 4, SLA2 topk 0.4): training forward/backward finite, Moonlight split routes 2 x alpha_logit per block to AdamW and router projections to Muon, `IMFVideoLoss(jvp_impl="fast")` loss_u finite over 8/8 seeds.

## Known limitations

- Routing consistency: the fast du/dt pass recomputes routing with a torch mean-pool (tail block averaged over real tokens only), while the training forward uses the vendored triton `mean_pool_smooth`; top-k sets can differ near ties. The mask is inside stop-gradient, so this perturbs only the detached du/dt.
- Pre-existing (NOT SLA2): at small geometry (hidden 128, kv_lora_rank 32) the fast-path kv-latent RMSNorm JVP tangent overflows fp16 and NaNs du/dt, reproduced with plain sdpa attention. Production geometry (hidden 256, rank 64) measured clean. Flagged as a separate task.
- `alpha` and router weights are snapshotted by `build_fast_jvp_state` (already rebuilt after every optimizer step, so no extra staleness).
