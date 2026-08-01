"""All configurable parameters for iMF video training.

Every knob lives here; train_8gpu.py reads only this file.
Launch: torchrun --nproc-per-node 8 train_8gpu.py
"""

from dataclasses import dataclass, field
import os


@dataclass
class ModelConfig:
    # imf_dit_video backbone. Hard cap enforced in train_8gpu.py: <= 300M params.
    hidden_size: int = 1024
    depth: int = 19  # total blocks = depth (shared = depth - aux_head_depth)
    aux_head_depth: int = 4  # u-head and v-head each get this many blocks
    num_heads: int = 16  # head_dim = hidden_size / num_heads = 64 (required)

    # --- Multi-head Latent Attention (MLA) geometry; 0 = derive from head_dim ---
    # A DiT keeps no KV (key-value) cache, so MLA here is a low-rank bottleneck on
    # the q/k/v projections (~0.68x the attention params of full-rank MHA), not a
    # cache-compression trick. Defaults satisfy
    #   qk_nope_head_dim + qk_rope_head_dim == v_head_dim == 64,
    # which the flash_jvp CuTeDSL op requires.
    q_lora_rank: int = 0  # dq: query latent dim, default hidden_size // 2 = 512
    kv_lora_rank: int = 0  # dc: key/value latent dim, default hidden_size // 4 = 256
    qk_nope_head_dim: int = 0  # dn: position-free q/k channels, default 48
    qk_rope_head_dim: int = 0  # dr: rotary q/k channels (shared key head), default 16
    v_head_dim: int = 0  # dv: value head dim, default head_dim = 64

    patch_size: tuple = (1, 2, 2)
    in_channels: int = 16  # Wan2.1 VAE latent channels
    num_classes: int = 1000
    attn_impl: str = "flash_jvp"  # "flash_jvp" (CuTeDSL kernels) | "sdpa" (math, slow)
    # Kimi-K3 mla_use_output_gate: attn output modulated by sigmoid(g_proj(x)).
    mla_use_output_gate: bool = True

    # --- Feed-forward and normalization ---
    mlp_ratio: float = 8 / 3  # SwiGLU hidden = mlp_ratio * hidden_size
    # Kimi-K3 SituAndMul activation: gate act = beta*tanh(g/beta)*sigmoid(g)
    # (bounded SiLU; SiLU is the beta -> inf limit). situ_linear_beta also
    # soft-clamps the up branch when set (None disables).
    situ_beta: float = 4.0  # Kimi-K3 production value (activation_situ_beta)
    situ_linear_beta: float | None = 25.0  # Kimi-K3 activation_situ_linear_beta
    rmsnorm_eps: float = 1e-5  # RMSNorm epsilon (Kimi-K3 rms_norm_eps)

    # --- In-context conditioning token banks (prepended to the patch tokens) ---
    num_class_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2
    # Sinusoidal basis width inside each TimestepEmbedder.
    freq_embedding_size: int = 256

    # --- Initialization scales: weight ~ Normal(0, constant / sqrt(fan_in)) ---
    token_init_constant: float = 1.0  # learned conditioning token banks
    embedding_init_constant: float = 1.0  # timestep / label embedders
    weight_init_constant: float = 0.32  # transformer block projections

    # --- 3D axial RoPE ---
    rope_theta: float = 10000.0  # base period; angles use theta^(-2i/axis_dim)

    # Kimi-K3 attention residual: every attn_res_block_size blocks the
    # residual-stream prefix sum is snapshotted, and before each sublayer a
    # 1-query softmax over the snapshots re-mixes the stream. 0 disables.
    attn_res_block_size: int = 4

    # Recompute sublayer activations in backward: required for full-latent
    # (29k-token) sequences on 16 GB; costs ~30% extra forward compute.
    grad_checkpoint: bool = True

    # Drops the v-heads; only valid for sampling, never for training.
    eval_mode: bool = False

    max_params: int = 300_000_000


@dataclass
class DataConfig:
    # Latent source: directory of .pt/.npy latent tensors (C,T,H,W) or "synthetic".
    # Wan-Syn parquet on the Nautilus PVC: preprocess to tensors first (see
    # fastvideo preprocessing) or point latent_dir at the converted output.
    latent_dir: str = ".cache/wan_syn_full"
    latent_frames: int = 20  # latent T (video frames = 4*(T-1)+1)
    # latent spatial dims: int (square) or (H, W); Wan-Syn 480p = (56, 104)
    latent_size: tuple = (56, 104)
    batch_size_per_gpu: int = 1
    num_workers: int = max(0, (os.cpu_count() or 1) - 2)
    pin_memory: bool = True


@dataclass
class LossConfig:
    P_mean: float = -0.4  # NOTE
    P_std: float = 1.0  # NOTE
    # fraction of each batch forced to r = t (flow-matching samples, fm_mask=True);
    # the remaining 1 - data_proportion keep r < t (mean-flow samples).
    # fm samples also bypass CFG interval gating: [t_min, t_max] = [0, 1], so the
    # sampled cfg scale w applies at every t instead of only inside the interval.
    data_proportion: float = 0.5
    # CFG (classifier-free guidance) scale omega is drawn per sample as
    #   omega(u) = (1 + u * ((1 + s_max)^(1-beta) - 1))^(1/(1-beta)),  u ~ U(0,1)
    # (beta = 1 uses the limiting form omega = (1 + s_max)^u). Both branches map
    # u in [0,1] onto omega in [1, 1 + s_max], so the largest guidance scale is
    # 1 + cfg_s_max = 8.0, NOT cfg_s_max. beta tilts the density: beta < 1 favours
    # small omega, beta > 1 favours large omega.
    cfg_beta: float = 1.0
    cfg_s_max: float = 7.0
    class_dropout_prob: float = 0.1
    norm_p: float = 1.0
    norm_eps: float = 0.01
    # du/dt engine: "fast" = detached hand-rolled Triton forward-mode pass
    # (requires attn_res_block_size > 0 and CUDA); "functorch" = torch.func.jvp
    jvp_impl: str = "fast"


@dataclass
class OptimConfig:
    # "moonlight": Muon (orthogonalized momentum) on the 2-D hidden weight
    #   matrices, AdamW on everything else -- all 1-D params (norm scales,
    #   biases, residual gates), the embedding/token tables, the patch-embed
    #   conv and the two output projections. See moonlight.py:split_params.
    # "adamw": plain AdamW on every parameter.
    optimizer: str = "moonlight"

    lr: float = 5e-4  # tuned: 400-step node sweep 1e-4..1e-3, 5e-4 best
    # Moonlight recipe: weight decay 0.1 on the Muon branch. Confirmed on
    # 400-step nodes: final-window loss_u 0.7364 (wd 0.1) vs 0.8437 (1e-5).
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)  # TODO: Maybe decrease beta2 in deep stages of training

    # --- Muon branch (ignored when optimizer="adamw") ---
    muon_momentum: float = 0.95  # mu in M_t = mu * M_{t-1} + G_t
    muon_nesterov: bool = True  # step along G + mu * M instead of M
    muon_ns_steps: int = 5  # Newton-Schulz iterations per step
    # Update scale is muon_lr_scale_constant * sqrt(max(n, m)), which pins the
    # orthogonalized update's RMS at the constant for every matrix shape, making
    # an AdamW-tuned lr transferable. 0.2 is Moonlight's value.
    muon_lr_scale_constant: float = 0.2
    # Newton-Schulz coefficients (a, b, c).
    #   "per_shape": fit them to each distinct weight shape at construction time,
    #     following Su Jianlin (https://kexue.fm/archives/10592). The quintic acts
    #     as the scalar map sigma -> a*sigma + b*sigma^3 + c*sigma^5 and so never
    #     sees n or m, but the singular value distribution it must map to 1 does
    #     depend on shape (Marchenko-Pastur in the aspect ratio, with the
    #     Frobenius-normalized values scaling as 1/sqrt(min(n,m))). Fitting per
    #     shape cuts the residual by 1-2 orders of magnitude on non-square
    #     matrices. Costs a few seconds of SVD + Adam once at startup.
    #   "jordan": the fixed (3.4445, -4.7750, 2.0315) that stock Muon and
    #     Moonlight ship, which is roughly the square-matrix, steps=5 optimum.
    muon_coeff_mode: str = "per_shape"
    # Budget for the OFFLINE per-shape fit (ignored when muon_coeff_mode="jordan").
    # Do not confuse muon_coeff_iters with muon_ns_steps above: ns_steps=5 is the
    # Newton-Schulz iteration applied to every matrix on every training step,
    # whereas coeff_iters is the outer Adam loop that solves for (kappa, x1, x2)
    # once per distinct shape at startup. The two are nested -- the fit's loss
    # unrolls ns_steps applications of g() -- and coeff_iters costs nothing per
    # training step.
    muon_coeff_samples: int = 32  # random matrices SVD'd to estimate the spectrum
    # Converged: at (1024, 2730) T=5, 10000 iters reproduces 3000 bit-for-bit
    # (delta kappa = 1.8e-06, same mse 2.037e-04), while 1000 is 4x worse
    # (8.4e-04). Su's reference used 100k momentum-SGD steps at lr=0.01; Adam
    # needs far fewer.
    muon_coeff_iters: int = 3000  # Adam iterations for the (kappa, x1, x2) fit
    muon_coeff_lr: float = 2e-2  # Adam learning rate for that fit
    muon_coeff_seed: int = 0  # RNG seed, so the coefficients are reproducible
    # global L2 clip over all params concatenated (not per-tensor):
    # g <- g * min(1, grad_clip / ||g||_2)
    grad_clip: float = 1.0

    # --- learning-rate schedule ---
    # "wsd" (Warmup-Stable-Decay, trapezoidal; Moonlight/MiniCPM/DeepSeek
    #   style): linear warmup -> FLAT at lr -> decay tail over the final
    #   decay_fraction of total_steps.
    # "cosine": linear warmup -> cosine annealing (the previous behavior).
    lr_schedule: str = "wsd"
    warmup_steps: int = 1000
    total_steps: int = 100_000
    # fraction of total_steps spent in the final decay tail (wsd only)
    decay_fraction: float = 0.15
    # decay tail shape (wsd only): "1-sqrt" (Hagele et al. 2024 best) or
    # "cosine"
    decay_shape: str = "1-sqrt"
    min_lr_ratio: float = 0.1  # decay floor = lr * min_lr_ratio
    grad_accum: int = 1
    fused_adamw: bool = True


@dataclass
class RunConfig:
    seed: int = 0
    out_dir: str = "checkpoints"
    log_every: int = 50
    ckpt_every: int = 5000
    resume: str = ""  # path to checkpoint to resume from
    # shadow weights for sampling: theta_ema <- d * theta_ema + (1 - d) * theta,
    # averaging horizon 1 / (1 - d) = 1e4 steps. no bias correction / no ramp,
    # so the EMA stays init-biased for roughly the first horizon. 0 disables EMA.
    ema_decay: float = 0.9999
    wandb_project: str = ""  # empty disables wandb
    master_dtype: str = "float32"  # model params dtype; attention runs bf16 inside


@dataclass
class SampleConfig:
    """Inference-time knobs for IMFVideoLoss.sample (integrates t = 1 -> 0)."""

    # 1 gives one-step generation, which is the point of MeanFlow; more steps
    # trade compute for accuracy.
    num_steps: int = 1
    omega: float = 2.0  # CFG scale held fixed across steps, in [1, 1 + cfg_s_max]
    t_min: float = 0.0  # guidance interval lower bound
    t_max: float = 1.0  # guidance interval upper bound


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


config = Config()
