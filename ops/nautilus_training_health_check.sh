#!/usr/bin/env bash
# Fail closed on repeated post-resume non-finite gradients.  The watchdog then
# keeps the allocated GPUs busy while the failure remains available for debug.
set -euo pipefail

project_dir=${MFVIDEO_PROJECT_DIR:-/root/downeyflyfan/MFVideo}
log_file=${MFVIDEO_TRAIN_LOG:-train_linear_t2_resume.log}
threshold=${MFVIDEO_NONFINITE_THRESHOLD:-2}

cd "$project_dir"
[[ -f "$log_file" ]] || exit 0

resume_line=$(awk '/resumed from checkpoints\/step_0007000\.pt at step 7000/ {line=NR} END {print line}' "$log_file")
[[ -n "$resume_line" ]] || exit 0
nonfinite_count=$(tail -n "+$resume_line" "$log_file" \
    | grep -c 'non-finite grad_norm' || true)
(( nonfinite_count >= threshold )) || exit 0

mkdir -p .cache
marker=.cache/nautilus-training-instability.marker
if [[ ! -e "$marker" ]]; then
    printf 'post_resume_nonfinite=%s threshold=%s detected_at=%s\n' \
        "$nonfinite_count" "$threshold" "$(date -u +%FT%TZ)" > "$marker"
fi
echo "[training-health] stopping unstable run after $nonfinite_count non-finite gradients" >&2

# Stop only the supervisor and four-rank MFVideo torchrun parent.  Do not
# touch generic Python workloads (for example vLLM) that might share a node.
supervisor_pids=$(ps -eo pid=,comm=,args= | awk \
    '$2 == "bash" && /bash \.\/ops\/nautilus_train_supervisor\.sh$/ {print $1}')
[[ -z "$supervisor_pids" ]] || kill -TERM $supervisor_pids || true
train_pids=$(ps -eo pid=,comm=,args= | awk \
    '$2 ~ /^python/ && /torchrun --nproc-per-node 4 train\.py$/ {print $1}')
[[ -z "$train_pids" ]] || kill -TERM $train_pids || true
