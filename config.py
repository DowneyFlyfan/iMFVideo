"""All configurable parameters for iMF video training.

Every knob lives here; train_8gpu.py reads only this file.
Launch: torchrun --nproc-per-node 8 train_8gpu.py
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # imf_dit_video backbone. Hard cap enforced in train_8gpu.py: <= 300M params.
    hidden_size: int = 1024
    depth: int = 19              # total blocks = depth (shared = depth - aux_head_depth)
    aux_head_depth: int = 4      # u-head and v-head each get this many blocks
    num_heads: int = 16          # head_dim = hidden_size / num_heads = 64 (required)
    patch_size: tuple = (1, 2, 2)
    in_channels: int = 16        # Wan2.1 VAE latent channels
    num_classes: int = 1000
    attn_impl: str = "flash_jvp"  # "flash_jvp" (CuTeDSL kernels) | "sdpa" (math, slow)

    max_params: int = 300_000_000


@dataclass
class DataConfig:
    # Latent source: directory of .pt/.npy latent tensors (C,T,H,W) or "synthetic".
    # Wan-Syn parquet on the Nautilus PVC: preprocess to tensors first (see
    # fastvideo preprocessing) or point latent_dir at the converted output.
    latent_dir: str = "synthetic"
    latent_frames: int = 3       # latent T (video frames = 4*(T-1)+1)
    latent_size: int = 16        # latent H = W
    batch_size_per_gpu: int = 8
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LossConfig:
    # improved MeanFlow (imf_video.IMFVideoLoss) hyperparameters.
    P_mean: float = -0.4
    P_std: float = 1.0
    data_proportion: float = 0.5
    cfg_beta: float = 1.0
    cfg_s_max: float = 7.0
    class_dropout_prob: float = 0.1
    norm_p: float = 1.0
    norm_eps: float = 0.01


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    total_steps: int = 100_000
    min_lr_ratio: float = 0.1    # cosine floor = lr * min_lr_ratio
    grad_accum: int = 1
    fused_adamw: bool = True


@dataclass
class RunConfig:
    seed: int = 0
    out_dir: str = "checkpoints"
    log_every: int = 50
    ckpt_every: int = 5000
    resume: str = ""             # path to checkpoint to resume from
    ema_decay: float = 0.9999    # 0 disables EMA
    wandb_project: str = ""      # empty disables wandb
    master_dtype: str = "float32"  # model params dtype; attention runs bf16 inside


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)


config = Config()
