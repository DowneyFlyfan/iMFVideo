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
import gc
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
            tile=m.sla2_tile,
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
def qk_clip(net, tau, world):
    """QK-Clip (kexue.fm/archives/11126, Kimi K2 MuonClip): after the
    optimizer step, rescale attention weights of any head whose MaxLogit
    exceeded tau, by exactly the overshoot ratio.

    In this MLA the q/k nope bands are per-head RMS-normed (scale-invariant
    to their projections), so the only unbounded bilinear pair is
    q_rope (per-head rows of q_b_proj) x k_rope (shared, from kv_a_proj).
    Per Su's (qr, kr) rule the shared kr is never touched and qr takes the
    full factor: W_qr^h *= tau / S_max^h.

    Args:
        net: IMFDiTVideo; blocks hold MLAAttention whose attn_impl exposes
            logit_max_log2: (H,) fp32 per-head max lse of the last training
            forward, log2 domain with the 1/sqrt(dn+dr) scale folded in.
        tau: MaxLogit threshold in nats (Kimi K2 uses 100).
        world: ranks; per-rank batches differ, so S_max is all-reduced with
            MAX to keep the clip factors (and thus weights) identical.

    Returns:
        (num_clipped_heads, global_max_logit_nats) for logging.
    """
    from models.imf_dit_video import MLAAttention

    ln2 = math.log(2.0)
    n_clipped, s_global = 0, 0.0
    for m in net.modules():
        if not isinstance(m, MLAAttention):
            continue
        impl = m.attn_impl
        stats = getattr(impl, "logit_max_log2", None)
        if stats is None:
            continue
        if world > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        s_nat = stats * ln2                       # (H,) MaxLogit in nats
        s_global = max(s_global, s_nat.max().item())
        gamma = (tau / s_nat).clamp(max=1.0)      # (H,) clip factors <= 1
        hot = gamma < 1.0                         # (H,) bool
        if not bool(hot.any()):
            continue
        n_clipped += int(hot.sum())
        dn, dr = m.qk_nope_head_dim, m.qk_rope_head_dim
        # q_b_proj.weight: (H*(dn+dr), dq); head h rope rows are
        # [h*(dn+dr)+dn, (h+1)*(dn+dr)).
        w = m.q_b_proj.weight                     # (H*(dn+dr), dq)
        wv = w.view(m.num_heads, dn + dr, -1)     # (H, dn+dr, dq)
        wv[:, dn:, :] *= gamma.to(w.dtype).view(-1, 1, 1)
    return n_clipped, s_global


@torch.no_grad()
def phi_clip(net, denominator_floor, world):
    """Bound the linear-attention tangent's quotient denominator.

    The feature maps are channel softmaxes.  For a complement with N keys
    and D channels, full q/k channel ranges Rq/Rk imply
    denominator >= N * exp(-(Rq + Rk)) / D.  This routine uses that bound to
    choose a QK-Clip factor after the optimizer step.  It scales both the
    RMSNorm gains and rope projections: scaling only the rope projections
    leaves the RMS-normalized NoPE bands unchanged and cannot bound T2.
    """
    from models.imf_dit_video import MLAAttention

    n_clipped, r_global, gamma_min = 0, 0.0, 1.0
    for m in net.modules():
        if not isinstance(m, MLAAttention):
            continue
        impl = m.attn_impl
        rq = getattr(impl, "phi_range_q", None)
        rk = getattr(impl, "phi_range_k", None)
        if rq is None or rk is None:
            continue
        if world > 1:
            dist.all_reduce(rq, op=dist.ReduceOp.MAX)
            dist.all_reduce(rk, op=dist.ReduceOp.MAX)
        complement_tokens = getattr(impl, "linear_complement_tokens", None)
        if complement_tokens is None or complement_tokens <= 0:
            continue
        range_sum = (rq + rk).max().item()
        r_global = max(r_global, range_sum)
        rho = math.log(
            complement_tokens / (m.qk_head_dim * denominator_floor)
        )
        if rho <= 0:
            raise ValueError(
                "phi_clip_den_floor exceeds the linear-attention "
                "denominator bound"
            )
        gamma = min(1.0, rho / range_sum) if range_sum > 0 else 1.0
        gamma_min = min(gamma_min, gamma)
        if gamma == 1.0:
            continue
        n_clipped += 1
        dn, dr = m.qk_nope_head_dim, m.qk_rope_head_dim
        g = torch.as_tensor(gamma, device=m.q_norm.weight.device,
                            dtype=m.q_norm.weight.dtype)
        # RMSNorm makes q_b/kv_b NoPE projection scaling a no-op.  Their
        # gains and the unnormalized rope rows together are the complete
        # raw q/k feature producers, so this is an exact whole-vector scale.
        m.q_norm.weight *= g
        m.k_norm.weight *= g
        m.q_b_proj.weight.view(m.num_heads, dn + dr, -1)[:, dn:, :] *= g
        m.kv_a_proj.weight[-dr:, :] *= g
    return n_clipped, r_global, gamma_min


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
    # Per-LAYER-CLASS lr width factors (records/mup_special_layers.md,
    # verified by width coordinate checks), applied by splitting the two
    # Moonlight groups on parameter names:
    #   Muon hidden matrices / router projs  : sqrt(d0/d)
    #   Muon res score heads (LM-Head class) : d0/d
    #   AdamW output projections (u/v final) : d0/d
    #   AdamW Theta(1) classes (embeddings, tokens, gains, gates, alpha): 1
    ratio = config.model.hidden_size / 256.0        # d / d0
    name_of = {id(pp): nn for nn, pp in net.named_parameters()}
    new_groups = []
    for g in optimizer.param_groups:
        buckets = {}
        for pp in g["params"]:
            nm = name_of.get(id(pp), "")
            if g.get("use_muon") and "_res_proj" in nm:
                mult = 1.0 / ratio                  # res score heads
            elif g.get("use_muon"):
                mult = ratio ** -0.5                # hidden matrices
            elif "final_layer" in nm and ".norm." not in nm:
                mult = 1.0 / ratio                  # output projections
            else:
                mult = 1.0                          # Theta(1) classes
            buckets.setdefault(mult, []).append(pp)
        for mult, ps in buckets.items():
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = ps
            ng["lr_width_mult"] = mult
            new_groups.append(ng)
    optimizer.param_groups = new_groups
    if rank == 0:
        for ng in new_groups:
            print(f"lr group: muon={ng.get('use_muon', False)} "
                  f"mult={ng['lr_width_mult']:.3f} "
                  f"params={sum(pp.numel() for pp in ng['params'])/1e6:.2f}M",
                  flush=True)

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
        # Rank-serialized load: 4 ranks decompressing the 3.5 GB checkpoint
        # simultaneously peak past the pod's 64 GiB cgroup limit and the
        # whole job gets a silent SIGKILL (observed on Nautilus). One rank
        # loads at a time; the others wait at the barrier.
        for r in range(world):
            if rank == r:
                # weights_only=True is the PyTorch >= 2.6 default and is
                # kept: every value we save is a tensor, number, string,
                # tuple or dict.
                ckpt = torch.load(
                    config.run.resume, map_location="cpu", weights_only=True)
                net.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                if ema is not None and "ema" in ckpt:
                    ema.load_state_dict(ckpt["ema"])
                start_step = ckpt["step"]
                del ckpt
                gc.collect()
            if world > 1:
                dist.barrier()
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
            if not torch.isfinite(loss):
                # Forward-NaN forensics: dump the offending batch once per
                # rank so the failure reproduces offline (the NaN is batch-
                # dependent -- healthy steps interleave with nan ones).
                # latents: (B, C, T, H, W) fp32; labels: (B,) int64.
                _p = os.path.join(
                    config.run.out_dir, f"nan_batch_r{rank}.pt")
                if not os.path.exists(_p):
                    torch.save(
                        {"latents": latents.cpu(), "labels": labels.cpu(),
                         "step": step}, _p)
                    print(f"rank {rank}: dumped non-finite-loss batch to {_p}",
                          flush=True)
            # Divide so the accumulated gradient is the mean over micro-batches.
            (loss / o.grad_accum).backward()
            loss_sum += loss.item() / o.grad_accum

        if world > 1:
            all_reduce_grads(list(net.parameters()), world)

        lr = lr_at(step)
        for g in optimizer.param_groups:
            # per-group width factor (records/mup_special_layers.md): the
            # Muon branch under Moonlight's Adjust-LR needs lr * sqrt(d0/d)
            # when widening past the tuned d0=256; AdamW groups keep 1.
            g["lr"] = lr * g.get("lr_width_mult", 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), o.grad_clip)
        # grad_norm: () scalar, total pre-clip L2 norm over all parameter grads.
        # Non-finite => this batch produced inf/nan somewhere in backward; the
        # grads are identical on every rank (all-reduce above), so every rank
        # skips the same step and stays in sync. Skipping keeps the weights and
        # EMA finite instead of poisoning the run (Wan2.2 run 1 died this way
        # at step 3950 with no checkpoint, records/wan22_full_train.md).
        if torch.isfinite(grad_norm):
            optimizer.step()
            # QK-Clip: bound per-head MaxLogit right after the update so the
            # bilinear rope logits cannot enter the fp16/quantization NaN
            # regime that stalled runs 1 and 3 at step ~3500-3950.
            n_hot, s_max = qk_clip(net, o.qk_clip_tau, world)
            n_phi, r_max, phi_gamma = phi_clip(
                net, o.phi_clip_den_floor, world
            )
            if ema is not None:
                ema_update(ema, net, config.run.ema_decay)
        else:
            n_hot, s_max = 0, float("nan")
            n_phi, r_max, phi_gamma = 0, float("nan"), float("nan")
            if rank == 0:
                # name the culprits: first parameters (by module order) whose
                # grad went non-finite, with counts -- pinpoints which layer
                # and which sub-path (q/k, v, phi, mlp) sources the NaN.
                bad = []
                for nm, p in net.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad.append((nm, int((~torch.isfinite(p.grad)).sum())))
                    if len(bad) >= 6:
                        break
                lv = dict_losses["loss_v"].item()
                print(
                    f"step {step + 1}: non-finite grad_norm, step skipped; "
                    f"loss_u={dict_losses['loss_u'].item():.4g} "
                    f"loss_v={lv:.4g} bad={bad}"
                )
            optimizer.zero_grad(set_to_none=True)
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
            # lu: python float, last micro-batch loss_u. A non-finite value
            # would poison the displayed EMA forever; keep the EMA finite.
            if loss_u_ema is None:
                loss_u_ema = lu
            elif math.isfinite(lu):
                loss_u_ema = ema_beta * loss_u_ema + (1 - ema_beta) * lu
            print(
                f"step {step} loss={loss_sum:.4f} "
                f"loss_u={lu:.4f} loss_u_ema={loss_u_ema:.4f} "
                f"loss_v={dict_losses['loss_v'].item():.4f} "
                f"grad_norm={grad_norm.item():.3f} lr={lr:.2e} "
                f"qkclip={n_hot}/{s_max:.0f} "
                f"phiclip={n_phi}/{r_max:.1f}/{phi_gamma:.3f} "
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
