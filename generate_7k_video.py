"""Generate one Wan 2.2 video locally from the MFVideo step-7000 checkpoint."""

import argparse
import copy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def apply_checkpoint_config(target, checkpoint_config):
    """Populate an in-memory Config from the checkpoint's saved settings."""
    for section, values in checkpoint_config.items():
        destination = getattr(target, section, None)
        if destination is None or not isinstance(values, dict):
            continue
        for name, value in values.items():
            if hasattr(destination, name):
                setattr(destination, name, value)


def cast_for_vae(latents, dtype):
    """Match generated latent precision to the VAE convolution parameters."""
    return latents.to(dtype=dtype)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(".cache/checkpoints/step_0007000.pt"),
    )
    parser.add_argument(
        "--stats", type=Path, default=Path(".cache/wan22_full/stats.npz"),
    )
    parser.add_argument(
        "--vae-model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )
    parser.add_argument("--vae-cache", type=Path, default=Path(".cache/wan22_vae"))
    parser.add_argument("--output", type=Path, default=Path("step_0007000_sample.mp4"))
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU is required for video inference")
    if not args.checkpoint.is_file() or not args.stats.is_file():
        raise FileNotFoundError("checkpoint and Wan 2.2 normalization statistics are required")

    from config import config as base_config

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if checkpoint.get("step") != 7000:
        raise ValueError(f"expected step 7000, found step {checkpoint.get('step')}")
    runtime_config = copy.deepcopy(base_config)
    apply_checkpoint_config(runtime_config, checkpoint["config"])

    import train
    from imf_video import IMFVideoLoss

    train.config = runtime_config
    net, parameter_count, _ = train.build_model()
    weights = checkpoint.get("ema") or checkpoint["model"]
    net.load_state_dict(weights, strict=True)
    del checkpoint
    net.eval().to(device="cuda", dtype=torch.bfloat16)
    print(f"loaded step 7000 ({parameter_count / 1e6:.1f}M parameters)", flush=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    shape = (
        1,
        runtime_config.model.in_channels,
        runtime_config.data.latent_frames,
        *runtime_config.data.latent_size,
    )
    noise = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    labels = torch.tensor([args.label], device="cuda", dtype=torch.long)
    loss = IMFVideoLoss(
        net,
        num_classes=runtime_config.model.num_classes,
        jvp_impl=runtime_config.loss.jvp_impl,
        autocast_bf16=runtime_config.loss.autocast_bf16,
    )
    with torch.inference_mode():
        normalized_latents = loss.sample(
            noise, labels, num_steps=args.steps, omega=runtime_config.sample.omega,
            t_min=runtime_config.sample.t_min, t_max=runtime_config.sample.t_max,
        )

    statistics = np.load(args.stats)
    mean = torch.from_numpy(statistics["mean"]).to("cuda", torch.bfloat16)
    std = torch.from_numpy(statistics["std"]).to("cuda", torch.bfloat16)
    vae_latents = cast_for_vae(
        normalized_latents * std.unsqueeze(0) + mean.unsqueeze(0), torch.bfloat16
    )
    del normalized_latents, loss, net
    torch.cuda.empty_cache()

    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        args.vae_model,
        subfolder="vae",
        cache_dir=args.vae_cache,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    vae.enable_tiling()
    vae.enable_slicing()
    with torch.inference_mode():
        frames = vae.decode(vae_latents).sample
    frames = ((frames[0].float().cpu().permute(1, 2, 3, 0) + 1) / 2)
    frames = frames.clamp(0, 1).mul(255).byte().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"wrote {args.output} with {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
