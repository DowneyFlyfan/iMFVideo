#!/usr/bin/env bash
# Resume MFVideo after an unexpected torchrun exit, without restarting a run
# that has already reached config.optim.total_steps.
set -euo pipefail

project_dir=${MFVIDEO_PROJECT_DIR:-/root/downeyflyfan/MFVideo}
poll_seconds=${MFVIDEO_SUPERVISOR_POLL_SECONDS:-60}

checkpoint_step() {
    local checkpoint_name=${1##*/}
    checkpoint_name=${checkpoint_name#step_}
    checkpoint_name=${checkpoint_name%.pt}
    [[ $checkpoint_name =~ ^[0-9]+$ ]] || return 1
    printf '%d\n' "$((10#$checkpoint_name))"
}

should_resume() {
    local completed_step=$1 total_steps=$2
    (( completed_step < total_steps ))
}

latest_checkpoint() {
    find "$project_dir/checkpoints" -maxdepth 1 -type f \
        -name 'step_???????.pt' -printf '%f\n' 2>/dev/null \
        | sort -V | tail -n 1
}

total_steps() {
    (
        cd "$project_dir"
        .venv/bin/python - <<'PY'
from config import config
print(config.optim.total_steps)
PY
    )
}

training_running() {
    pgrep -f '[t]orchrun.*train.py' >/dev/null
}

stop_heartbeat_fallback() {
    local fallback_pids
    fallback_pids=$(ps -eo pid=,args= | awk \
        '/torch\.mm\(mat, mat, out=out\)/ && /4096/ {print $1}')
    [[ -z $fallback_pids ]] || kill $fallback_pids 2>/dev/null || true
}

main() {
    cd "$project_dir"
    while true; do
        if training_running; then
            sleep "$poll_seconds"
            continue
        fi

        local checkpoint_name checkpoint_path completed_step configured_total
        checkpoint_name=$(latest_checkpoint)
        [[ -n $checkpoint_name ]] || {
            echo '[supervisor] no checkpoint exists; refusing to start' >&2
            exit 1
        }
        checkpoint_path="checkpoints/$checkpoint_name"
        completed_step=$(checkpoint_step "$checkpoint_name")
        configured_total=$(total_steps)
        if ! should_resume "$completed_step" "$configured_total"; then
            echo "[supervisor] training completed at step $completed_step" >&2
            exit 0
        fi

        echo "[supervisor] torchrun absent; resuming $checkpoint_path" >&2
        stop_heartbeat_fallback
        sleep 5
        training_running || ./ops/nautilus_auto_resume_train.sh
        sleep "$poll_seconds"
    done
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
