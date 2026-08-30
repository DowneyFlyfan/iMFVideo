#!/usr/bin/env bash
# Synchronize the verified local training code to a newly allocated Nautilus
# Pod, then restart it from the latest complete checkpoint under supervision.
set -euo pipefail

pod_name=${1:?usage: nautilus_sync_and_resume.sh POD_NAME}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
namespace=${MFVIDEO_NAMESPACE:-ecepxie}
project_dir=${MFVIDEO_REMOTE_PROJECT_DIR:-/root/downeyflyfan/MFVideo}
ready_attempts=${MFVIDEO_READY_ATTEMPTS:-60}
ready_sleep=${MFVIDEO_READY_SLEEP_SECONDS:-5}

log() {
    printf '[nautilus-sync-resume] %s %s\n' "$(date -u +%FT%TZ)" "$*"
}

for ((attempt = 1; attempt <= ready_attempts; attempt++)); do
    if kubectl -n "$namespace" exec "$pod_name" -- sh -lc \
        "test -x '$project_dir/.venv/bin/python'" >/dev/null 2>&1; then
        break
    fi
    if (( attempt == ready_attempts )); then
        log "ERROR: pod $pod_name did not expose the MFVideo environment"
        exit 1
    fi
    sleep "$ready_sleep"
done

log "synchronizing verified local source to $pod_name"
for source_file in train.py imf_video.py moonlight.py repair_checkpoint_ema.py \
                   gpu_heartbeat_watchdog.sh; do
    kubectl -n "$namespace" cp "$repo_dir/$source_file" \
        "$namespace/$pod_name:$project_dir/$source_file"
done
for source_dir in models ops; do
    kubectl -n "$namespace" cp "$repo_dir/$source_dir" \
        "$namespace/$pod_name:$project_dir"
done

# The bootstrap can have already started an older PVC copy.  Stop only the
# exact recovery/training/fallback processes, never a generic Python process.
kubectl -n "$namespace" exec -i "$pod_name" -- bash -s <<'REMOTE'
set -euo pipefail
cd /root/downeyflyfan/MFVideo
supervisor_pids=$(ps -eo pid=,args= | awk '/[b]ash \.\/ops\/nautilus_train_supervisor\.sh$/ {print $1}')
[[ -z "$supervisor_pids" ]] || kill -TERM $supervisor_pids || true
train_pids=$(ps -eo pid=,args= | awk '/[t]orchrun --nproc-per-node 4 train\.py$/ {print $1}')
[[ -z "$train_pids" ]] || kill -TERM $train_pids || true
for _ in $(seq 1 36); do
    pgrep -f '[t]orchrun --nproc-per-node 4 train.py' >/dev/null || break
    sleep 5
done
if pgrep -f '[t]orchrun --nproc-per-node 4 train.py' >/dev/null; then
    echo '[nautilus-sync-resume] training did not stop after 180 seconds' >&2
    exit 1
fi
fallback_pids=$(ps -eo pid=,args= | awk \
    '/torch\.mm\(mat, mat, out=out\)/ && /4096/ {print $1}')
[[ -z "$fallback_pids" ]] || kill -TERM $fallback_pids || true
chmod 755 gpu_heartbeat_watchdog.sh ops/nautilus_auto_resume_train.sh \
    ops/nautilus_train_supervisor.sh
./ops/nautilus_auto_resume_train.sh
REMOTE

log "recovery launch requested for $pod_name"
