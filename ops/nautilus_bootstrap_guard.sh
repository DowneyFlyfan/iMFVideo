#!/usr/bin/env bash
# Do not replace a recovery that Pod bootstrap has already begun.  The slow
# sync-and-restart path is only a fallback for a bootstrap that is absent.
set -euo pipefail

pod_name=${1:?usage: nautilus_bootstrap_guard.sh POD_NAME}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
namespace=${MFVIDEO_NAMESPACE:-ecepxie}
grace_seconds=${MFVIDEO_BOOTSTRAP_GRACE_SECONDS:-20}
poll_seconds=${MFVIDEO_BOOTSTRAP_GUARD_POLL_SECONDS:-1}
sync_and_resume=${MFVIDEO_SYNC_AND_RESUME:-"$repo_dir/ops/nautilus_sync_and_resume.sh"}

[[ $grace_seconds =~ ^[0-9]+$ ]] || {
    echo 'MFVIDEO_BOOTSTRAP_GRACE_SECONDS must be a non-negative integer' >&2
    exit 2
}

deadline=$((SECONDS + grace_seconds))
while true; do
    if kubectl -n "$namespace" exec "$pod_name" -- bash -lc \
        "pgrep -f '[t]orchrun --nproc-per-node 4 train\\.py|[n]autilus_auto_resume_train\\.sh' >/dev/null"
    then
        echo "[bootstrap-guard] preserving bootstrap recovery on $pod_name" >&2
        exit 0
    fi
    (( SECONDS >= deadline )) && break
    sleep "$poll_seconds"
done

echo "[bootstrap-guard] no bootstrap recovery found; using fallback on $pod_name" >&2
exec "$sync_and_resume" "$pod_name"
