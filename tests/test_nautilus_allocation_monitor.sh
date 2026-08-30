#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
monitor="$repo_dir/ops/nautilus_allocation_monitor.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat > "$work_dir/bin/kubectl" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MFVIDEO_KUBECTL_LOG"
if [[ " $* " == *" get pods "* ]]; then
    printf 'gpu-dev2-55886bbcc8-replacement\n'
else
    printf 'Running\n'
fi
SH
cat > "$work_dir/bin/notify-send" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MFVIDEO_NOTIFY_LOG"
SH
cat > "$work_dir/bin/on-running" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MFVIDEO_RECOVERY_LOG"
SH
chmod +x "$work_dir/bin/kubectl" "$work_dir/bin/notify-send" \
    "$work_dir/bin/on-running"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_MONITOR_STATE="$work_dir/state"
export MFVIDEO_NOTIFY_LOG="$work_dir/notifications"
export MFVIDEO_KUBECTL_LOG="$work_dir/kubectl.log"
export MFVIDEO_MONITOR_ON_RUNNING="$work_dir/bin/on-running"
export MFVIDEO_RECOVERY_LOG="$work_dir/recovery.log"

bash "$monitor" --once

grep -Fq 'Nautilus-A100 allocated' "$MFVIDEO_NOTIFY_LOG"
grep -Fxq 'gpu-dev2-55886bbcc8-replacement Running' "$MFVIDEO_MONITOR_STATE"
grep -Fq 'get pods -n ecepxie -l app=gpu-dev2' "$MFVIDEO_KUBECTL_LOG"
grep -Fxq 'gpu-dev2-55886bbcc8-replacement' "$MFVIDEO_RECOVERY_LOG"
printf 'PASS: allocation transition sends one notification and records state\n'
