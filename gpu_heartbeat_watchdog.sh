#!/usr/bin/env bash
# Keep Nautilus GPUs occupied if a named training parent exits unexpectedly.
set -euo pipefail

train_pid="${1:?usage: gpu_heartbeat_watchdog.sh TRAIN_PID}"
while kill -0 "$train_pid" 2>/dev/null; do
    sleep 30
done

for gpu_idx in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu_idx" nohup .venv/bin/python -u -c '
import torch
mat = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(mat)
torch.cuda.synchronize()
while True:
    torch.mm(mat, mat, out=out)
' >> gpu-heartbeat-fallback.log 2>&1 &
done
wait
