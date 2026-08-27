#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
monitor="$repo_dir/ops/nautilus_allocation_monitor.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat > "$work_dir/bin/kubectl" <<'SH'
#!/usr/bin/env bash
printf 'Running\n'
SH
cat > "$work_dir/bin/notify-send" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MFVIDEO_NOTIFY_LOG"
SH
chmod +x "$work_dir/bin/kubectl" "$work_dir/bin/notify-send"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_MONITOR_STATE="$work_dir/state"
export MFVIDEO_NOTIFY_LOG="$work_dir/notifications"

bash "$monitor" --once

grep -Fq 'Nautilus-A100 allocated' "$MFVIDEO_NOTIFY_LOG"
grep -Fxq 'Running' "$MFVIDEO_MONITOR_STATE"
printf 'PASS: allocation transition sends one notification and records state\n'
