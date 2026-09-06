#!/usr/bin/env bash
# Consecutive post-resume non-finite gradients must stop only MFVideo recovery.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
project_dir="$work_dir/project"
mkdir -p "$project_dir/.cache" "$work_dir/bin"
cat > "$project_dir/train_linear_t2_resume.log" <<'LOG'
resumed from checkpoints/step_0007000.pt at step 7000
step 7001: non-finite grad_norm, step skipped
step 7002: non-finite grad_norm, step skipped
LOG
cat > "$work_dir/bin/ps" <<'SH'
#!/usr/bin/env bash
printf '%s\n' \
  '111 bash bash ./ops/nautilus_train_supervisor.sh' \
  '222 python .venv/bin/torchrun --nproc-per-node 4 train.py' \
  '333 python vllm serve unrelated-model'
SH
cat > "$work_dir/bash_env" <<'SH'
kill() { printf '%s\n' "$*" >> "$MFVIDEO_KILL_LOG"; }
SH
chmod +x "$work_dir/bin/ps"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_PROJECT_DIR="$project_dir"
export MFVIDEO_KILL_LOG="$work_dir/kill.log"
BASH_ENV="$work_dir/bash_env" bash "$repo_dir/ops/nautilus_training_health_check.sh"

grep -Fxq -- '-TERM 111' "$MFVIDEO_KILL_LOG"
grep -Fxq -- '-TERM 222' "$MFVIDEO_KILL_LOG"
test "$(wc -l < "$MFVIDEO_KILL_LOG")" -eq 2
test -s "$project_dir/.cache/nautilus-training-instability.marker"
printf 'PASS: health check stops only confirmed MFVideo processes\n'
