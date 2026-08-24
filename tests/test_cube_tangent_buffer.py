"""Regression: cube linear-attention tangent must not re-enter fp16.

Run: .venv/bin/python tests/test_cube_tangent_buffer.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.mla_jvp_fast import cube_attention_jvp_buffers


primal, tangent = cube_attention_jvp_buffers(
    batch_size=1, sequence_length=3, attention_width=4, device="cuda"
)
assert primal.dtype == torch.float16
assert tangent.dtype == torch.float32

# T2 can exceed float16's largest finite number before its output projection.
# It must remain finite until the fp32 projection, rather than becoming inf on
# the temporary attention-buffer assignment.
tangent.fill_(100_000.0)
assert torch.isfinite(tangent).all()
assert tangent.abs().max().item() == 100_000.0
print("PASS")
