"""Atomically replace a legacy checkpoint's invalid EMA with online weights."""

import argparse
import copy
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if temporary_path.exists():
        raise FileExistsError(f"refusing to overwrite incomplete migration {temporary_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint has no model state dictionary")

    checkpoint.pop("ema", None)
    checkpoint["ema"] = {
        name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in model.items()
    }
    checkpoint["linear_qk_preconditioned"] = True
    checkpoint.setdefault("config", {}).setdefault("optim", {})[
        "resume_linear_qk_scale"
    ] = 1.0
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    print(f"repaired EMA in {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
