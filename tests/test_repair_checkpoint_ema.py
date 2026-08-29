import subprocess
import sys
from pathlib import Path

import torch


def test_repair_checkpoint_replaces_legacy_ema_atomically(tmp_path):
    source = tmp_path / "step_0008000.pt"
    model = {"weight": torch.tensor([1.0, 2.0])}
    torch.save(
        {
            "model": model,
            "ema": {"weight": torch.tensor([9.0, 9.0])},
            "config": {"optim": {"resume_linear_qk_scale": 0.3}},
            "step": 8000,
        },
        source,
    )

    subprocess.run(
        [sys.executable, "repair_checkpoint_ema.py", str(source)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    repaired = torch.load(source, map_location="cpu", weights_only=True)
    assert repaired["linear_qk_preconditioned"] is True
    assert repaired["config"]["optim"]["resume_linear_qk_scale"] == 1.0
    assert torch.equal(repaired["ema"]["weight"], repaired["model"]["weight"])
    assert not source.with_suffix(".pt.tmp").exists()
