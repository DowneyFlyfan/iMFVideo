#!/usr/bin/env bash
# Keep Nautilus GPUs occupied if a named training parent exits unexpectedly.
set -euo pipefail

train_pid="${1:?usage: gpu_heartbeat_watchdog.sh TRAIN_PID}"
while kill -0 "$train_pid" 2>/dev/null; do
    sleep 30
done

mapfile -t gpu_uuids < <(
    nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '\r'
)
(( ${#gpu_uuids[@]} > 0 )) || {
    echo '[heartbeat] no NVIDIA GPU UUIDs available for fallback' >&2
    exit 1
}

for gpu_uuid in "${gpu_uuids[@]}"; do
    MFVIDEO_A100_FILLER_SIZE=24576 CUDA_VISIBLE_DEVICES="$gpu_uuid" \
        nohup .venv/bin/python -u -c '
import os
import torch

marker = "a100_goal_filler"
size = int(os.environ["MFVIDEO_A100_FILLER_SIZE"])
left = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
right = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(left)
torch.cuda.synchronize()
while True:
    torch.mm(left, right, out=out)
' >> gpu-heartbeat-fallback.log 2>&1 &
done
wait
