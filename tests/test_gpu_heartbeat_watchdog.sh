#!/usr/bin/env bash
# The idle-A100 fallback must honor Kubernetes' UUID-based CUDA visibility.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

mkdir -p "$work_dir/bin" "$work_dir/.venv/bin"
cp "$repo_dir/gpu_heartbeat_watchdog.sh" "$work_dir/gpu_heartbeat_watchdog.sh"
chmod +x "$work_dir/gpu_heartbeat_watchdog.sh"

cat > "$work_dir/bin/kill" <<'SH'
#!/usr/bin/env bash
exit 1
SH
cat > "$work_dir/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
printf '%s\n' GPU-uuid-0 GPU-uuid-1 GPU-uuid-2 GPU-uuid-3
SH
cat > "$work_dir/bin/nohup" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$CUDA_VISIBLE_DEVICES" >> "$MFVIDEO_NOHUP_LOG"
printf '%s' "$*" >> "$MFVIDEO_NOHUP_ARGS_LOG"
SH
chmod +x "$work_dir/bin/kill" "$work_dir/bin/nvidia-smi" "$work_dir/bin/nohup"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_NOHUP_LOG="$work_dir/nohup.log"
export MFVIDEO_NOHUP_ARGS_LOG="$work_dir/nohup-args.log"
(cd "$work_dir" && ./gpu_heartbeat_watchdog.sh 12345)

test "$(wc -l < "$MFVIDEO_NOHUP_LOG")" -eq 4
for gpu_uuid in GPU-uuid-0 GPU-uuid-1 GPU-uuid-2 GPU-uuid-3; do
    grep -Fxq "$gpu_uuid" "$MFVIDEO_NOHUP_LOG"
done
grep -Fq 'a100_goal_filler' "$MFVIDEO_NOHUP_ARGS_LOG"
printf 'PASS: watchdog launches a named filler against every GPU UUID\n'
