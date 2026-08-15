"""iMF (improved MeanFlow) video training: single-device or multi-GPU.
All parameters come from config.py. A 300M model needs no sharding: full
Adam (adaptive moment estimation) states are ~4.8 GB per GPU, so plain data
parallelism with a single gradient all-reduce per step suffices.

Gradient synchronisation is done explicitly rather than through
DistributedDataParallel. The iMF loss calls the network several times per step
(the classifier-free-guidance passes under no_grad, plus one torch.func.jvp
forward-mode pass), and DDP's reducer only arms its autograd hooks inside
DDP.forward(). Routing those sub-forwards through the DDP wrapper is not
possible for the functorch jvp path, so instead every rank backpropagates
locally and one all_reduce averages the gradients before the optimizer step.
This gives up compute/communication overlap in exchange for being correct.

Launch (8 GPUs):     torchrun --nproc-per-node 8 train.py
Launch (1 process):  python train.py
"""

import dataclasses
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from models.imf_dit_video import IMFDiTVideo, sdpa_math_attention

from config import config


def config_as_dict():
    """Flatten the config dataclass tree into plain dicts.

    Checkpoints must not embed dataclass instances: torch.load defaults to
    weights_only=True since PyTorch 2.6 and refuses to unpickle arbitrary
    classes, which would make every resume fail. Plain dicts, strings, numbers
    and tuples are all on the weights_only allowlist.
    """
    return dataclasses.asdict(config)


class LatentDataset(Dataset):
    """Video latents (C, T, H, W) + integer class label.

    latent_dir="synthetic": random unit-variance latents (pipeline debugging).
    Otherwise: directory of .pt files, each {"latent": tensor, "label": int}.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.latent_dir == "synthetic":
            self.files = None
            self.length = 100_000
        else:
            self.files = sorted(
                os.path.join(cfg.latent_dir, f)
                for f in os.listdir(cfg.latent_dir)
                if f.endswith(".pt")
            )
            self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        c = self.cfg
        if self.files is None:
            gen = torch.Generator().manual_seed(idx)
            latent = torch.randn(
                config.model.in_channels,
                c.latent_frames,
                c.latent_size,
                c.latent_size,
                generator=gen,
            )
            label = idx % config.model.num_classes
            return latent, label
        item = torch.load(self.files[idx], map_location="cpu")
        return item["latent"].float(), int(item.get("label", 0))


def resolve_device():
    """Pick the best available device and the matching collective backend.

    Returns:
        (device, backend): torch.device and the process-group backend name.
    """
    if torch.cuda.is_available():
        # TF32 tensor-core GEMMs for all fp32 matmuls: on A100 this is the
        # difference between 19.5 (CUDA cores) and 156 TFLOPS. Attention
        # already runs bf16/fp16 internally, so TF32 (10-bit mantissa) on
        # the Linear layers matches the repo's precision philosophy.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda"), "nccl"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "gloo"
    return torch.device("cpu"), "gloo"


def build_model():
    """Instantiate the MLA-based iMF video DiT and check the parameter cap.

    Returns:
        (net, num_params, attn_name): the module, its total parameter count, and
        the name of the attention implementation actually bound.
    """

    m = config.model
    attn = sdpa_math_attention
    if m.attn_impl == "sdpa_flash":
        # Fused sdpa backends: O(S) memory for long sequences on GPUs
        # without the CuTeDSL kernels. jvp_impl must be "fast".
        from models.imf_dit_video import sdpa_flash_attention

        attn = sdpa_flash_attention
    elif m.attn_impl == "sla2_cube_qat":
        # Cube-block (VSA 3D tiles) sparse-linear attention with INT8 QAT
        # training forward; one stateful instance per block. seq_len and
        # num_heads are bound inside IMFDiTVideo; the patch grid is known
        # here from the data config.
        from functools import partial

        from models.sla2_cube_qat import SLA2CubeQATAttentionImpl

        pt, ph, pw = m.patch_size
        lh, lw = config.data.latent_size
        attn = partial(
            SLA2CubeQATAttentionImpl,
            grid=(config.data.latent_frames // pt, lh // ph, lw // pw),
            tile=(4, 4, 4),
            topk=m.sla2_topk,
            alpha_init=m.sla2_alpha_init,
        )
    elif m.attn_impl == "sla2_jvp":
        # SLA2 sparse-linear attention: a module factory, one stateful
        # instance (router + alpha params) per transformer block. seq_len
        # and num_heads are bound inside IMFDiTVideo where L is known.
        from functools import partial

        from models.sla2_attention import SLA2AttentionImpl

        attn = partial(
            SLA2AttentionImpl,
            topk=m.sla2_topk,
            bq=m.sla2_bq,
            bk=m.sla2_bk,
            alpha_init=m.sla2_alpha_init,
        )
    elif m.attn_impl == "flash_jvp":
        # The CuTeDSL / flash-attn kernels are CUDA-only, so importing them on a
        # Mac raises. Fall back loudly rather than crashing, so the same config
        # runs for a laptop smoke test and an H200 job.
        try:
            from models.attention_op import flash_jvp_attention

            attn = flash_jvp_attention
        except Exception as exc:
            print(
                f"WARNING: attn_impl='flash_jvp' unavailable ({type(exc).__name__}: "
                f"{exc}); falling back to sdpa_math_attention (much slower).",
                flush=True,
            )

    net = IMFDiTVideo(
        input_size=config.data.latent_size,
        num_frames=config.data.latent_frames,
        patch_size=m.patch_size,
        in_channels=m.in_channels,
        hidden_size=m.hidden_size,
        depth=m.depth,
        num_heads=m.num_heads,
        num_classes=m.num_classes,
        aux_head_depth=m.aux_head_depth,
        q_lora_rank=m.q_lora_rank,
        kv_lora_rank=m.kv_lora_rank,
        qk_nope_head_dim=m.qk_nope_head_dim,
        qk_rope_head_dim=m.qk_rope_head_dim,
        v_head_dim=m.v_head_dim,
        mlp_ratio=m.mlp_ratio,
        num_class_tokens=m.num_class_tokens,
        num_time_tokens=m.num_time_tokens,
        num_cfg_tokens=m.num_cfg_tokens,
        num_interval_tokens=m.num_interval_tokens,
        freq_embedding_size=m.freq_embedding_size,
        token_init_constant=m.token_init_constant,
        embedding_init_constant=m.embedding_init_constant,
        weight_init_constant=m.weight_init_constant,
        rmsnorm_eps=m.rmsnorm_eps,
        rope_theta=m.rope_theta,
        eval_mode=m.eval_mode,
        attn_impl=attn,
        attn_res_block_size=m.attn_res_block_size,
        situ_beta=m.situ_beta,
        situ_linear_beta=m.situ_linear_beta,
        mla_use_output_gate=m.mla_use_output_gate,
        grad_checkpoint=m.grad_checkpoint,
    )
    num_params = sum(p.numel() for p in net.parameters())
    assert (
        num_params <= m.max_params
    ), f"model has {num_params/1e6:.1f}M params, exceeds cap {m.max_params/1e6:.0f}M"
    # Report the impl actually bound, not the one requested: they differ whenever
    # the flash fallback above fired.
    return net, num_params, getattr(attn, "__name__", str(attn))


def build_optimizer(net, o, device, verbose=False):
    """Build the optimizer named by `o.optimizer`.

    Args:
        net: the model to optimize.
        o: an OptimConfig.
        device: torch.device the parameters live on (gates the fused AdamW path).
        verbose: print the Muon/AdamW parameter split.

    Returns:
        a torch.optim.Optimizer whose param_groups all expose "lr", so lr_at()
        can drive them uniformly.
    """
    if o.optimizer == "moonlight":
        from moonlight import build_moonlight

        if verbose and o.weight_decay < 1e-3:
            print(
                f"WARNING: optimizer='moonlight' with weight_decay="
                f"{o.weight_decay:g}. Moonlight attributes Muon's scalability to "
                f"weight decay and uses 0.1; the Muon branch is effectively "
                f"undecayed at this value.",
                flush=True,
            )
        return build_moonlight(
            net,
            lr=o.lr,
            weight_decay=o.weight_decay,
            momentum=o.muon_momentum,
            nesterov=o.muon_nesterov,
            ns_steps=o.muon_ns_steps,
            lr_scale_constant=o.muon_lr_scale_constant,
            adamw_betas=o.betas,
            coeff_mode=o.muon_coeff_mode,
            coeff_samples=o.muon_coeff_samples,
            coeff_iters=o.muon_coeff_iters,
            coeff_lr=o.muon_coeff_lr,
            coeff_seed=o.muon_coeff_seed,
            verbose=verbose,
        )

    if o.optimizer == "adamw":
        return torch.optim.AdamW(
            net.parameters(),
            lr=o.lr,
            weight_decay=o.weight_decay,
            betas=o.betas,
            # The fused AdamW kernel exists only for CUDA tensors.
            fused=o.fused_adamw and device.type == "cuda",
        )

    raise ValueError(
        f"unknown optimizer {o.optimizer!r}; expected 'moonlight' or 'adamw'"
    )


def lr_at(step):
    """Learning rate at `step` under config.optim.lr_schedule.

    "wsd": linear warmup -> flat at lr -> decay tail ("1-sqrt" or cosine)
    over the final decay_fraction of total_steps, down to lr*min_lr_ratio.
    "cosine": linear warmup -> cosine annealing to the same floor.
    """
    o = config.optim
    if step < o.warmup_steps:
        return o.lr * step / max(1, o.warmup_steps)
    floor = o.lr * o.min_lr_ratio
    if o.lr_schedule == "wsd":
        decay_start = int(o.total_steps * (1.0 - o.decay_fraction))
        if step < decay_start:
            return o.lr                      # flat (stable) phase
        x = (step - decay_start) / max(1, o.total_steps - decay_start)
        x = min(x, 1.0)
        if o.decay_shape == "1-sqrt":
            return floor + (o.lr - floor) * (1.0 - math.sqrt(x))
        return floor + 0.5 * (o.lr - floor) * (1 + math.cos(math.pi * x))
    t = (step - o.warmup_steps) / max(1, o.total_steps - o.warmup_steps)
    return floor + 0.5 * (o.lr - floor) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def ema_update(ema, model, decay):
    """theta_ema <- decay * theta_ema + (1 - decay) * theta, parameters only.

    Buffers are not touched; the only buffers in this model are the constant
    rope_cos / rope_sin tables, which never change.
    """
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


@torch.no_grad()
def all_reduce_grads(params, world):
    """Average gradients across ranks with a single flat all-reduce.

    Args:
        params: iterable of parameters whose .grad should be averaged in place.
        world: number of ranks participating.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])  # (total_numel,)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


def main():
    # torchrun sets WORLD_SIZE / RANK; without them run as a single process so
    # the same script works for a laptop smoke test and an 8-GPU job.
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    device, backend = resolve_device()

    if distributed:
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
    else:
        rank, world = 0, 1

    torch.manual_seed(config.run.seed + rank)

    from imf_video import IMFVideoLoss

    net, num_params, attn_name = build_model()
    net = net.to(device, dtype=getattr(torch, config.run.master_dtype))
    if rank == 0:
        a = net.shared_blocks[0].attn
        print(
            f"model: {num_params/1e6:.1f}M params, world={world}, "
            f"device={device}, attn={attn_name}",
            flush=True,
        )
        print(
            f"MLA: heads={a.num_heads} q_lora={a.q_lora_rank} "
            f"kv_lora={a.kv_lora_rank} qk_nope={a.qk_nope_head_dim} "
            f"qk_rope={a.qk_rope_head_dim} v={a.v_head_dim}",
            flush=True,
        )

    ema = None
    if config.run.ema_decay > 0:
        import copy

        ema = copy.deepcopy(net).eval().requires_grad_(False)

    lcfg = config.loss
    loss_mod = IMFVideoLoss(
        net,
        config.model.num_classes,
        P_mean=lcfg.P_mean,
        P_std=lcfg.P_std,
        data_proportion=lcfg.data_proportion,
        cfg_beta=lcfg.cfg_beta,
        class_dropout_prob=lcfg.class_dropout_prob,
        jvp_impl=lcfg.jvp_impl,
        stratified_time=lcfg.stratified_time,
        stratified_interval=lcfg.stratified_interval,
        strat_group=(
            config.data.batch_size_per_gpu * config.optim.grad_accum
            if lcfg.strat_group_auto
            else 0
        ),
        s_max=lcfg.cfg_s_max,
        norm_p=lcfg.norm_p,
        norm_eps=lcfg.norm_eps,
        loss_v_weight=lcfg.loss_v_weight,
        autocast_bf16=lcfg.autocast_bf16,
    )

    o = config.optim
    assert o.grad_accum >= 1, "grad_accum must be at least 1"
    optimizer = build_optimizer(net, o, device, verbose=rank == 0)

    dataset = LatentDataset(config.data)
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size_per_gpu,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.data.num_workers,
        # pin_memory only helps for CUDA host-to-device copies.
        pin_memory=config.data.pin_memory and device.type == "cuda",
        drop_last=True,
        persistent_workers=config.data.num_workers > 0,
    )

    start_step = 0
    if config.run.resume:
        # weights_only=True is the PyTorch >= 2.6 default and is kept: every
        # value we save is a tensor, number, string, tuple or dict.
        ckpt = torch.load(config.run.resume, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        if rank == 0:
            print(f"resumed from {config.run.resume} at step {start_step}", flush=True)

    use_wandb = bool(config.run.wandb_project) and rank == 0
    if use_wandb:
        import wandb

        wandb.init(project=config.run.wandb_project, config=config_as_dict())

    os.makedirs(config.run.out_dir, exist_ok=True)
    # running mean of loss_u over ~20 steps: the per-step value is an average
    # over batch_size*grad_accum samples with randomly drawn (t, r, omega),
    # so at small per-step sample counts it is a noisy estimate of the trend.
    loss_u_ema = None
    ema_beta = 0.8  # ~5 logged points; log_every steps apart
    data_iter = iter(loader)
    epoch = 0
    t_log = time.time()
    step = start_step
    while step < o.total_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(o.grad_accum):
            try:
                latents, labels = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                latents, labels = next(data_iter)
            # latents: (B, C, T, H, W) video latents;  labels: (B,) class ids
            latents = latents.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            loss, dict_losses = loss_mod(latents, labels)
            # Divide so the accumulated gradient is the mean over micro-batches.
            (loss / o.grad_accum).backward()
            loss_sum += loss.item() / o.grad_accum

        if world > 1:
            all_reduce_grads(list(net.parameters()), world)

        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), o.grad_clip)
        optimizer.step()
        if ema is not None:
            ema_update(ema, net, config.run.ema_decay)
        step += 1

        if rank == 0 and step % config.run.log_every == 0:
            dt = time.time() - t_log
            imgs_s = (
                config.run.log_every
                * config.data.batch_size_per_gpu
                * world
                * o.grad_accum
                / dt
            )
            lu = dict_losses["loss_u"].item()
            loss_u_ema = lu if loss_u_ema is None else (
                ema_beta * loss_u_ema + (1 - ema_beta) * lu
            )
            print(
                f"step {step} loss={loss_sum:.4f} "
                f"loss_u={lu:.4f} loss_u_ema={loss_u_ema:.4f} "
                f"loss_v={dict_losses['loss_v'].item():.4f} "
                f"grad_norm={grad_norm.item():.3f} lr={lr:.2e} "
                f"{imgs_s:.1f} samples/s",
                flush=True,
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "loss": loss_sum,
                        "grad_norm": grad_norm.item(),
                        "lr": lr,
                        "samples_per_s": imgs_s,
                    },
                    step=step,
                )
            t_log = time.time()

        if rank == 0 and step % config.run.ckpt_every == 0:
            path = os.path.join(config.run.out_dir, f"step_{step:07d}.pt")
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "step": step,
                    "config": config_as_dict(),
                },
                path,
            )
            print(f"saved {path}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
