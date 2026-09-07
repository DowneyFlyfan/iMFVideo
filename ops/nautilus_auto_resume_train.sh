#!/usr/bin/env bash
# Start the MFVideo four-A100 job after a Nautilus pod boot.
# This file is mounted from the nautilus-init ConfigMap at /init.
set -euo pipefail

project_dir=${MFVIDEO_PROJECT_DIR:-/root/downeyflyfan/MFVideo}
log_file=train_linear_t2_resume.log

if [[ ! -d "$project_dir" ]]; then
    echo "[auto-resume] project is absent; not starting training" >&2
    exit 0
fi

cd "$project_dir"

# Both the mounted Pod bootstrap and the allocation monitor can invoke this
# script during a fresh start.  Only one may materialize checkpoint config and
# launch torchrun; the other must leave the shared PVC untouched.
mkdir -p .cache
exec 9>.cache/nautilus-auto-resume.lock
if ! flock -n 9; then
    echo '[auto-resume] another recovery invocation owns the resume lock' >&2
    exit 0
fi

if pgrep -f '[t]orchrun.*train.py' >/dev/null; then
    echo "[auto-resume] training already exists; not starting a duplicate" >&2
    exit 0
fi

if [[ ! -x .venv/bin/torchrun ]]; then
    echo "[auto-resume] missing .venv/bin/torchrun; not starting training" >&2
    exit 1
fi

# The requested restart point is step 8000.  Older step-8000 files may need
# an EMA repair before their saved configuration is materialized below.
checkpoint=checkpoints/step_0008000.pt
if [[ ! -f "$checkpoint" ]]; then
    echo "[auto-resume] requested checkpoint $checkpoint is absent" >&2
    exit 1
fi

# The legacy 8k checkpoint scaled only online Q/K producers, leaving its EMA
# in a different parameterization.  Repair it before generating config.py so
# the restored resume scale and the checkpoint state remain paired.
if .venv/bin/python - "$checkpoint" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True,
                        mmap=True)
sys.exit(not (
    checkpoint.get("step") == 8000
    and not checkpoint.get("linear_qk_preconditioned", False)
))
PY
then
    repair_script=repair_checkpoint_ema.py
    if [[ ! -f "$repair_script" ]]; then
        echo "[auto-resume] legacy 8k checkpoint needs EMA repair, but ${repair_script} is absent" >&2
        exit 1
    fi
    echo "[auto-resume] repairing legacy EMA in $checkpoint" >&2
    .venv/bin/python "$repair_script" "$checkpoint"
fi

# The shared PVC can retain a stale or damaged config.py from a prior pod.
# Rebuild the server config from the checkpoint before launching so the model
# architecture, input geometry, and optimizer schedule remain checkpoint-safe.
config_restore_script=restore_checkpoint_config.py
if .venv/bin/python - "$checkpoint" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True,
                        mmap=True)
sys.exit(not isinstance(checkpoint.get("config"), dict))
PY
then
    if [[ ! -f "$config_restore_script" ]]; then
        echo "[auto-resume] missing ${config_restore_script}" >&2
        exit 1
    fi
    .venv/bin/python "$config_restore_script" "$checkpoint" config.py \
        --resume "$checkpoint"
else
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
fi

echo "[auto-resume] resuming from $checkpoint" >&2
nohup .venv/bin/torchrun --nproc-per-node 4 train.py \
    >> "$log_file" 2>&1 < /dev/null &
train_pid=$!
echo "[auto-resume] training pid=$train_pid" >&2
nohup ./gpu_heartbeat_watchdog.sh "$train_pid" \
    >> gpu-heartbeat-watchdog.log 2>&1 < /dev/null &

# The heartbeat keeps the allocation occupied after a failure; the supervisor
# additionally resumes training from the newest incomplete checkpoint.
if [[ -x ./ops/nautilus_train_supervisor.sh ]] \
    && ! pgrep -f '[n]autilus_train_supervisor.sh' >/dev/null; then
    nohup ./ops/nautilus_train_supervisor.sh \
        >> train-supervisor.log 2>&1 < /dev/null &
fi
