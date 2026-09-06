#!/usr/bin/env python3
"""Materialize the configuration saved in a checkpoint into a config.py template."""

from __future__ import annotations

import argparse
import copy
import os
import pprint
import re
from pathlib import Path

import torch


SECTION_CLASSES = {
    "model": "ModelConfig",
    "data": "DataConfig",
    "loss": "LossConfig",
    "optim": "OptimConfig",
    "run": "RunConfig",
    "sample": "SampleConfig",
}
CONFIG_ASSIGNMENT = re.compile(r"^config\s*=\s*Config\(\)\s*$", re.MULTILINE)


def render_config(saved_config: dict[str, dict[str, object]]) -> str:
    """Render the saved nested configuration as a Config constructor."""
    if set(saved_config) != set(SECTION_CLASSES):
        raise ValueError(
            "checkpoint config sections must be exactly "
            f"{sorted(SECTION_CLASSES)}, got {sorted(saved_config)}"
        )

    sections: list[str] = []
    for section, class_name in SECTION_CLASSES.items():
        values = saved_config[section]
        if not isinstance(values, dict):
            raise ValueError(f"checkpoint config.{section} must be a dictionary")
        rendered_values = pprint.pformat(values, width=100, sort_dicts=True)
        sections.append(f"    {section}={class_name}(**{rendered_values}),")
    return "config = Config(\n" + "\n".join(sections) + "\n)"


def restore(checkpoint_path: Path, config_path: Path, resume: str | None) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError(f"{checkpoint_path} has no dictionary config")
    saved_config = copy.deepcopy(saved_config)
    if resume is not None:
        run_config = saved_config.get("run")
        if not isinstance(run_config, dict):
            raise ValueError("checkpoint config.run must be a dictionary")
        run_config["resume"] = resume

    source = config_path.read_text()
    restored, count = CONFIG_ASSIGNMENT.subn(render_config(saved_config), source, count=1)
    if count != 1:
        raise ValueError(
            f"{config_path} must contain exactly one standalone 'config = Config()' template"
        )

    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary_path.write_text(restored)
    os.chmod(temporary_path, config_path.stat().st_mode)
    temporary_path.replace(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume")
    args = parser.parse_args()
    restore(args.checkpoint, args.config, args.resume)


if __name__ == "__main__":
    main()
