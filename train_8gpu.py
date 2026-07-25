"""Distributed iMF video training (DDP) for 8xH200 (or any nproc).

All parameters come from config.py. A 300M model needs no sharding: full
Adam states ~4.8 GB per GPU, so plain DDP with overlapped all-reduce is the
fastest configuration (see records/ for the parallelism analysis).

Launch:  torchrun --nproc-per-node 8 train_8gpu.py
Single-GPU smoke: torchrun --nproc-per-node 1 train_8gpu.py
"""

import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from config import config


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
                config.model.in_channels, c.latent_frames, c.latent_size, c.latent_size,
                generator=gen,
            )
            label = idx % config.model.num_classes
            return latent, label
        item = torch.load(self.files[idx], map_location="cpu")
        return item["latent"].float(), int(item.get("label", 0))


def build_model():
    from models.imf_dit_video import IMFDiTVideo
    from models.attention_op import flash_jvp_attention, sdpa_math_attention

    m = config.model
    attn = flash_jvp_attention if m.attn_impl == "flash_jvp" else sdpa_math_attention
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
        attn_impl=attn,
    )
    num_params = sum(p.numel() for p in net.parameters())
    assert num_params <= m.max_params, (
        f"model has {num_params/1e6:.1f}M params, exceeds cap {m.max_params/1e6:.0f}M"
    )
    return net, num_params


def lr_at(step):
    o = config.optim
    if step < o.warmup_steps:
        return o.lr * step / max(1, o.warmup_steps)
    t = (step - o.warmup_steps) / max(1, o.total_steps - o.warmup_steps)
    floor = o.lr * o.min_lr_ratio
    return floor + 0.5 * (o.lr - floor) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(config.run.seed + rank)

    from imf_video import IMFVideoLoss

    net, num_params = build_model()
    net = net.to(device, dtype=getattr(torch, config.run.master_dtype))
    if rank == 0:
        print(f"model: {num_params/1e6:.1f}M params, world={world}, "
              f"attn={config.model.attn_impl}", flush=True)

    ema = None
    if config.run.ema_decay > 0:
        import copy
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    ddp = DDP(net, device_ids=[local_rank], gradient_as_bucket_view=True)
    lcfg = config.loss
    loss_mod = IMFVideoLoss(
        ddp.module, config.model.num_classes,
        P_mean=lcfg.P_mean, P_std=lcfg.P_std,
        data_proportion=lcfg.data_proportion, cfg_beta=lcfg.cfg_beta,
        class_dropout_prob=lcfg.class_dropout_prob, s_max=lcfg.cfg_s_max,
        norm_p=lcfg.norm_p, norm_eps=lcfg.norm_eps,
    )

    o = config.optim
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=o.lr, weight_decay=o.weight_decay,
        betas=o.betas, fused=o.fused_adamw,
    )

    dataset = LatentDataset(config.data)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=config.data.batch_size_per_gpu, sampler=sampler,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
        drop_last=True, persistent_workers=config.data.num_workers > 0,
    )

    start_step = 0
    if config.run.resume:
        ckpt = torch.load(config.run.resume, map_location="cpu")
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
        wandb.init(project=config.run.wandb_project, config=vars(config))

    os.makedirs(config.run.out_dir, exist_ok=True)
    data_iter = iter(loader)
    epoch = 0
    t_log = time.time()
    step = start_step
    # DDP syncs grads on loss.backward(); the iMF jvp/guidance passes run
    # inside loss_mod on ddp.module directly (no extra DDP hooks fired there).
    while step < o.total_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(o.grad_accum):
            try:
                latents, labels = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iter = iter(loader)
                latents, labels = next(data_iter)
            latents = latents.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            maybe_sync = (
                ddp.no_sync() if micro < o.grad_accum - 1
                else torch.enable_grad()
            )
            with maybe_sync:
                loss, dict_losses = loss_mod(latents, labels)
                (loss / o.grad_accum).backward()
            loss_sum += loss.item() / o.grad_accum

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
            imgs_s = config.run.log_every * config.data.batch_size_per_gpu * world * o.grad_accum / dt
            print(
                f"step {step} loss={loss_sum:.4f} "
                f"loss_u={dict_losses['loss_u'].item():.4f} "
                f"loss_v={dict_losses['loss_v'].item():.4f} "
                f"grad_norm={grad_norm.item():.3f} lr={lr:.2e} "
                f"{imgs_s:.1f} samples/s", flush=True,
            )
            if use_wandb:
                import wandb
                wandb.log({"loss": loss_sum, "grad_norm": grad_norm.item(),
                           "lr": lr, "samples_per_s": imgs_s}, step=step)
            t_log = time.time()

        if rank == 0 and step % config.run.ckpt_every == 0:
            path = os.path.join(config.run.out_dir, f"step_{step:07d}.pt")
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "step": step,
                "config": vars(config),
            }, path)
            print(f"saved {path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
