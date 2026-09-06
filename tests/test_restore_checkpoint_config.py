import runpy
import subprocess
import sys
from pathlib import Path

import torch


def test_restores_checkpoint_config_without_changing_checkpoint_input(tmp_path):
    template = tmp_path / "config.py"
    template.write_text(
        """class Section:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

ModelConfig = DataConfig = LossConfig = OptimConfig = RunConfig = SampleConfig = Section

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

config = Config()
"""
    )
    checkpoint = tmp_path / "step_0008000.pt"
    torch.save(
        {
            "step": 8000,
            "config": {
                "model": {"in_channels": 48, "sla2_tile": (1, 2, 8)},
                "data": {"latent_frames": 31, "latent_size": (44, 80)},
                "loss": {"autocast_bf16": True},
                "optim": {"total_steps": 10000},
                "run": {"resume": "checkpoints/step_0008000.pt"},
                "sample": {"num_steps": 1},
            },
        },
        checkpoint,
    )

    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "restore_checkpoint_config.py",
            str(checkpoint),
            str(template),
            "--resume",
            "checkpoints/step_0007000.pt",
        ],
        check=True,
        cwd=repo,
    )

    restored = runpy.run_path(template)["config"]
    assert restored.model.in_channels == 48
    assert restored.model.sla2_tile == (1, 2, 8)
    assert restored.data.latent_frames == 31
    assert restored.data.latent_size == (44, 80)
    assert restored.optim.total_steps == 10000
    assert restored.run.resume == "checkpoints/step_0007000.pt"
