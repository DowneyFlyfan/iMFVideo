#!/usr/bin/env bash
# Start the MFVideo four-A100 job after a Nautilus pod boot.
# This file is mounted from the nautilus-init ConfigMap at /init.
set -euo pipefail

project_dir=/root/downeyflyfan/MFVideo
log_file=train_linear_t2_resume.log

if [[ ! -d "$project_dir" ]]; then
    echo "[auto-resume] project is absent; not starting training" >&2
    exit 0
fi

cd "$project_dir"

if pgrep -f '[t]orchrun.*train.py' >/dev/null; then
    echo "[auto-resume] training already exists; not starting a duplicate" >&2
    exit 0
fi

if [[ ! -x .venv/bin/torchrun ]]; then
    echo "[auto-resume] missing .venv/bin/torchrun; not starting training" >&2
    exit 1
fi

# The requested stable restart point is step 7000.  Prefer a later checkpoint
# only when it really exists, so a completed checkpoint is never overwritten.
checkpoint=checkpoints/step_0007000.pt
latest_checkpoint=$(find checkpoints -maxdepth 1 -type f -name 'step_*.pt' \
    -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "$latest_checkpoint" ]]; then
    checkpoint="checkpoints/$latest_checkpoint"
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "[auto-resume] no usable checkpoint found under checkpoints/" >&2
    exit 1
fi

export MFVIDEO_RESUME="$checkpoint"
.venv/bin/python - <<'PY'
import os
import re
from pathlib import Path

path = Path("config.py")
text = path.read_text()
resume = os.environ["MFVIDEO_RESUME"]
updated, count = re.subn(
    r'^(\s*resume:\s*str\s*=\s*)"[^"]*"(\s*(?:#.*)?)$',
    lambda match: f'{match.group(1)}"{resume}"{match.group(2)}',
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    raise RuntimeError("config.py must contain exactly one RunConfig.resume field")
path.write_text(updated)
PY

echo "[auto-resume] resuming from $checkpoint" >&2
nohup .venv/bin/torchrun --nproc-per-node 4 train.py \
    >> "$log_file" 2>&1 < /dev/null &
train_pid=$!
echo "[auto-resume] training pid=$train_pid" >&2
nohup ./gpu_heartbeat_watchdog.sh "$train_pid" \
    >> gpu-heartbeat-watchdog.log 2>&1 < /dev/null &
