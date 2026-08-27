#!/usr/bin/env bash
# Persistently monitor the Nautilus A100 pod without blocking Codex.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
namespace=ecepxie
pod_name=gpu-dev2-55886bbcc8-kt7qm
state_file=${MFVIDEO_MONITOR_STATE:-"$repo_dir/.cache/nautilus_a100_monitor.state"}
poll_seconds=${MFVIDEO_MONITOR_POLL_SECONDS:-60}

if [[ ${1:-} == --once ]]; then
    once=true
elif [[ $# -eq 0 ]]; then
    once=false
else
    echo "usage: $0 [--once]" >&2
    exit 2
fi

mkdir -p "$(dirname "$state_file")"

check_once() {
    local phase previous title body
    phase=$(kubectl get pod "$pod_name" -n "$namespace" \
        -o jsonpath='{.status.phase}' 2>/dev/null || printf 'Unavailable')
    [[ -n "$phase" ]] || phase=Unavailable
    previous=$(cat "$state_file" 2>/dev/null || true)

    if [[ "$phase" != "$previous" ]]; then
        case "$phase" in
            Running)
                title='Nautilus-A100 allocated'
                body='The four-A100 pod is running; the bootstrap is resuming MFVideo.'
                ;;
            Failed|Unknown|Unavailable)
                title='Nautilus-A100 requires attention'
                body="The allocation monitor observed pod state: $phase."
                ;;
            *)
                title=''
                body=''
                ;;
        esac
        if [[ -n "$title" ]]; then
            notify-send -u normal "$title" "$body" || true
        fi
        printf '%s\n' "$phase" > "$state_file"
    fi
}

while true; do
    check_once
    "$once" && exit 0
    sleep "$poll_seconds"
done
