"""All configurable parameters for iMF video training.

Every knob lives here; train_8gpu.py reads only this file.
Launch: torchrun --nproc-per-node 8 train_8gpu.py
"""

from dataclasses import dataclass, field
import os


@dataclass
class ModelConfig:
    # imf_dit_video backbone. Hard cap enforced in train_8gpu.py: <= 300M params.
    # NOTE: Fix these parameters
    hidden_size: int = 1024
    depth: int = 19  # total blocks = depth (shared = depth - aux_head_depth)
    aux_head_depth: int = 4  # u-head and v-head each get this many blocks
    num_heads: int = 16  # head_dim = hidden_size / num_heads = 64 (required)

    # NOTE: Fix these parameters
    q_lora_rank: int = 0  # dq: query latent dim, default hidden_size // 2 = 512
    kv_lora_rank: int = 0  # dc: key/value latent dim, default hidden_size // 4 = 256
    qk_nope_head_dim: int = 0  # dn: position-free q/k channels, default 48
    qk_rope_head_dim: int = 0  # dr: rotary q/k channels (shared key head), default 16
    v_head_dim: int = 0  # dv: value head dim, default head_dim = 64

    # NOTE: Fix these parameters
    patch_size: tuple = (1, 2, 2)
    in_channels: int = 16  # Wan2.1 VAE latent channels
    num_classes: int = 1000
    attn_impl: str = (
        "sla2_cube_qat"  # "flash_jvp" | "sla2_jvp" | "sla2_cube_qat" | "sdpa_flash" | "sdpa"
    )
    mla_use_output_gate: bool = True

    # --- SLA2 sparse-linear attention (attn_impl="sla2_jvp") ---
    # NOTE: To Be Tuned
    sla2_topk: float = 0.03  # routed fraction of key blocks (sparse branch)
    sla2_bq: int = 128  # query block length
    sla2_bk: int = 64  # key block length
    sla2_alpha_init: float = 0.9  # initial sparse/linear mixing ratio in (0,1)
    # 3D tile (t, h, w) for cube attention; prod = tile tokens E. Must
    # divide the patch grid. (4,4,4)=64 for Wan2.1 448x832; (1,2,8)=16
    # for Wan2.2 704x1280 (grid (31,22,40): 31 is prime).
    sla2_tile: tuple = (4, 4, 4)

    # --- Feed-forward and normalization ---
    # NOTE: Fix these parameters
    mlp_ratio: float = 8 / 3  # SwiGLU hidden = mlp_ratio * hidden_size

    # FIX: KimiMLP Params
    situ_beta: float = 4.0  # Kimi-K3 production value (activation_situ_beta)
    situ_linear_beta: float | None = 25.0  # Kimi-K3 activation_situ_linear_beta
    rmsnorm_eps: float = 1e-5

    # --- In-context conditioning token banks (prepended to the patch tokens) ---
    # NOTE: Fix these parameters
    num_class_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2
    # Sinusoidal basis width inside each TimestepEmbedder.
    freq_embedding_size: int = 256

    # --- Initialization scales: weight ~ Normal(0, constant / sqrt(fan_in)) ---
    token_init_constant: float = 1.0  # learned conditioning token banks
    embedding_init_constant: float = 1.0  # timestep / label embedders
    # transformer block projections. Tuned on the 400-step crop node sweep
    # (records/loss_curve_tuning.md): 0.5 beats both 0.32 and 0.2/0.7.
    # NOTE: To Be Tuned
    weight_init_constant: float = 0.5

    # --- 3D axial RoPE ---
    # NOTE: Fix these parameters
    rope_theta: float = 10000.0  # base period; angles use theta^(-2i/axis_dim)
    attn_res_block_size: int = 4
    grad_checkpoint: bool = True
    eval_mode: bool = False

    max_params: int = 300_000_000


@dataclass
class DataConfig:
    # NOTE: To Be Tuned
    # At a fixed sample budget, optimizer steps beat wider steps: bs=1 x 4000
    # steps reaches probe 0.2772 where bs=5 x 800 steps (same 4000 samples)
    # stalls at 0.4000, because decay_fraction buys the former 800 WSD decay
    # steps and the latter only 160. See records/loss_curve_tuning.md (R1/R2/R3).
    batch_size_per_gpu: int = 1

    # NOTE: Fix these parameters for local machines
    latent_dir: str = ".cache/wan_syn_2k_full"
    latent_frames: int = 20  # latent T (video frames = 4*(T-1)+1)
    latent_size: tuple = (56, 104)
    num_workers: int = max(0, (os.cpu_count() or 1) - 2)
    pin_memory: bool = True


@dataclass
class LossConfig:
    # NOTE: To Be Tuned
    P_mean: float = -0.4
    P_std: float = 1.0
    data_proportion: float = 0.25
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
    loss_v_weight: float = 1.0
    jvp_impl: str = "fast"
    # bf16 autocast for every network forward + backward (fp32 master and
    # loss math unchanged). Big win on server GPUs whose fp32 rate is far
    # below tensor-core rate; requires jvp_impl="fast".
    autocast_bf16: bool = False
    # Stratified (jittered-quantile) sampling of t and r: identical marginal
    # logit-normal law, but each batch covers the whole time range, which is
    # the dominant variance source at small per-step sample counts.
    stratified_time: bool = True
    # Also stratify the guidance-interval bounds t_min / t_max. Separate flag:
    # the tuning harness pins this to False inside its canonical scoring
    # distribution, so folding it into stratified_time would redefine every
    # recorded probe score.
    stratified_interval: bool = False
    # Spread the strata over batch_size_per_gpu * grad_accum samples instead of
    # over one micro-batch. Set automatically at that value when
    # strat_group_auto is on; within-micro-batch stratification is inert at
    # micro-batch 1, which is what full uncropped latents force on 16 GB.
    strat_group_auto: bool = False


@dataclass
class OptimConfig:
    optimizer: str = "moonlight"
    # NOTE: Fix these parameters
    lr: float = 2e-3
    weight_decay: float = 0.1

    # NOTE: Fix these parameters for local machines
    betas: tuple = (0.9, 0.95)  # TODO: Maybe decrease beta2 in deep stages of training

    # --- Muon branch (ignored when optimizer="adamw") ---
    # NOTE: To Be Tuned
    muon_momentum: float = 0.9

    # NOTE: Fix these parameters for local machines
    muon_nesterov: bool = True  # step along G + mu * M instead of M
    muon_ns_steps: int = 5  # Newton-Schulz iterations per step
    muon_lr_scale_constant: float = 0.2
    muon_coeff_mode: str = "jordan"
    muon_coeff_samples: int = 32  # random matrices SVD'd to estimate the spectrum
    muon_coeff_iters: int = 3000  # Adam iterations for the (kappa, x1, x2) fit
    muon_coeff_lr: float = 2e-2  # Adam learning rate for that fit
    muon_coeff_seed: int = 0  # RNG seed, so the coefficients are reproducible
    grad_clip: float = 1.0
    # QK-Clip (kexue.fm/archives/11126, Kimi K2 MuonClip): after each
    # optimizer step, any attention head whose MaxLogit exceeded this
    # threshold (nats) has its un-normed rope query rows rescaled by
    # tau / S_max. Cures the MaxLogit explosion that produced backward NaN
    # at step ~3500-3950 in Wan2.2 runs 1 and 3. Kimi uses 100 for fp32/bf16
    # kernels; the INT8-QAT backward here starts emitting NaN near logit
    # ~60-66 (run 4 skips began at step 3912 with S_max 66 and tau 100
    # never engaging), so the threshold sits at 40 -- inside the op's
    # proven-stable region (<= 50) and per Su even tau 30 is lossless.
    qk_clip_tau: float = 40.0
    # Linear-QK-Clip: lower bound for the complement linear-attention
    # denominator.  After every optimizer step the complete q/k feature
    # producers are rescaled so the channel-softmax denominator is at least
    # this value. This bounds the T2 quotient-JVP term directly.
    phi_clip_den_floor: float = 1e-1
    # Resume preconditioner. A finite checkpoint can still contain Q/K
    # ranges large enough to make the very first resumed linear-attention
    # pass singular before post-step QK-Clip has a chance to run.
    resume_linear_qk_scale: float = 1.0

    lr_schedule: str = "wsd"

    # NOTE: To Be Tuned
    # Tuned on the 4k-step full-latent sweep (records/loss_curve_tuning.md).
    # Best node R1_warm100: lr 2e-3, warmup 100, 4000 steps -> probe 0.2772,
    # ahead of the same run at warmup 400 (0.2848). lr 4e-3 diverges to NaN.
    # decay_fraction 0.2 gives 800 decay steps, where most of the gain lands;
    # 0.1 and 0.4 were indistinguishable in the stable phase.
    warmup_steps: int = 100
    total_steps: int = 4000
    decay_fraction: float = 0.2
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
    ema_decay: float = 0.9999
    wandb_project: str = ""  # empty disables wandb
    master_dtype: str = "float32"  # model params dtype; attention runs bf16 inside


@dataclass
class SampleConfig:
    """Inference-time knobs for IMFVideoLoss.sample (integrates t = 1 -> 0)."""

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
