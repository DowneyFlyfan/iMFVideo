#!/usr/bin/env bash
# Persistently monitor the Nautilus A100 pod without blocking Codex.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
namespace=ecepxie
pod_selector=${MFVIDEO_MONITOR_SELECTOR:-app=gpu-dev2}
pod_name_override=${MFVIDEO_MONITOR_POD_NAME:-}
on_running=${MFVIDEO_MONITOR_ON_RUNNING:-}
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
    local pod_name phase previous current title body
    if [[ -n "$pod_name_override" ]]; then
        pod_name=$pod_name_override
    else
        pod_name=$(kubectl get pods -n "$namespace" -l "$pod_selector" \
            --field-selector='status.phase!=Succeeded,status.phase!=Failed' \
            --sort-by=.metadata.creationTimestamp \
            -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true)
    fi
    if [[ -z "$pod_name" ]]; then
        phase=Unavailable
    else
        phase=$(kubectl get pod "$pod_name" -n "$namespace" \
            -o jsonpath='{.status.phase}' 2>/dev/null || printf 'Unavailable')
    fi
    [[ -n "$phase" ]] || phase=Unavailable
    current="$pod_name $phase"
    previous=$(cat "$state_file" 2>/dev/null || true)

    if [[ "$current" != "$previous" ]]; then
        case "$phase" in
            Running)
                title='Nautilus-A100 allocated'
                body="The four-A100 pod ${pod_name} is running; the bootstrap is resuming MFVideo."
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
        if [[ "$phase" == Running && -n "$on_running" ]]; then
            "$on_running" "$pod_name" || true
        fi
        printf '%s\n' "$current" > "$state_file"
    fi
}

while true; do
    check_once
    "$once" && exit 0
    sleep "$poll_seconds"
done
