"""Regression test for checkpoint-driven local video inference."""

import copy

import torch

from config import config
from generate_7k_video import apply_checkpoint_config, cast_for_vae


def test_apply_checkpoint_config_uses_server_latent_geometry():
    trial = copy.deepcopy(config)
    checkpoint_config = {
        "model": {"in_channels": 48, "sla2_tile": (1, 2, 8)},
        "data": {"latent_frames": 31, "latent_size": (44, 80)},
        "loss": {"autocast_bf16": True, "jvp_impl": "fast"},
    }

    apply_checkpoint_config(trial, checkpoint_config)

    assert trial.model.in_channels == 48
    assert trial.model.sla2_tile == (1, 2, 8)
    assert trial.data.latent_frames == 31
    assert trial.data.latent_size == (44, 80)
    assert trial.loss.autocast_bf16 is True
    assert trial.loss.jvp_impl == "fast"


def test_cast_for_vae_matches_decoder_dtype():
    latent = torch.randn(1, 48, 1, 1, 1, dtype=torch.float32)

    cast = cast_for_vae(latent, torch.bfloat16)

    assert cast.dtype is torch.bfloat16
