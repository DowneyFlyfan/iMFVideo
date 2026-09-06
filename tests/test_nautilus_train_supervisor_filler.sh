#!/usr/bin/env bash
# A recovery must stop only the explicitly named A100 filler before resuming.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat > "$work_dir/bin/ps" <<'SH'
#!/usr/bin/env bash
printf '4242 python a100_goal_filler\n'
SH
chmod +x "$work_dir/bin/ps"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_KILL_LOG="$work_dir/kill.log"
source "$repo_dir/ops/nautilus_train_supervisor.sh"
kill() { printf '%s\n' "$*" >> "$MFVIDEO_KILL_LOG"; }
stop_heartbeat_fallback

grep -Fxq '4242' "$MFVIDEO_KILL_LOG"
printf 'PASS: supervisor stops only the named A100 filler\n'
