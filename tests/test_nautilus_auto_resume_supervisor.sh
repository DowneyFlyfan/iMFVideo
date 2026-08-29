#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
project_dir="$work_dir/project"
mkdir -p "$project_dir/.venv/bin" "$project_dir/checkpoints" "$project_dir/ops" \
    "$work_dir/bin"
touch "$project_dir/checkpoints/step_0007000.pt"
printf '    resume: str = ""\n' > "$project_dir/config.py"
ln -s "$repo_dir/.venv/bin/python" "$project_dir/.venv/bin/python"
printf '#!/usr/bin/env bash\nexit 0\n' > "$project_dir/.venv/bin/torchrun"
printf '#!/usr/bin/env bash\nexit 0\n' > "$project_dir/gpu_heartbeat_watchdog.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "$project_dir/ops/nautilus_train_supervisor.sh"
chmod +x "$project_dir/.venv/bin/torchrun" "$project_dir/gpu_heartbeat_watchdog.sh" \
    "$project_dir/ops/nautilus_train_supervisor.sh"

cat > "$work_dir/bin/pgrep" <<'SH'
#!/usr/bin/env bash
exit 1
SH
cat > "$work_dir/bin/nohup" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MFVIDEO_NOHUP_LOG"
SH
chmod +x "$work_dir/bin/pgrep" "$work_dir/bin/nohup"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_PROJECT_DIR="$project_dir"
export MFVIDEO_NOHUP_LOG="$work_dir/nohup.log"
bash "$repo_dir/ops/nautilus_auto_resume_train.sh"
sleep 0.1

grep -Fq '.venv/bin/torchrun --nproc-per-node 4 train.py' "$MFVIDEO_NOHUP_LOG"
grep -Fq './gpu_heartbeat_watchdog.sh' "$MFVIDEO_NOHUP_LOG"
grep -Fq './ops/nautilus_train_supervisor.sh' "$MFVIDEO_NOHUP_LOG"
grep -Fq 'resume: str = "checkpoints/step_0007000.pt"' "$project_dir/config.py"
printf 'PASS: boot recovery launches training, heartbeat, and supervisor\n'
