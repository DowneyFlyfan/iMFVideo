#!/usr/bin/env bash
# A second boot-recovery invocation must exit before touching checkpoint state.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
project_dir="$work_dir/project"
mkdir -p "$project_dir/.cache" "$project_dir/.venv/bin" \
    "$project_dir/checkpoints" "$project_dir/ops"
cp "$repo_dir/ops/nautilus_auto_resume_train.sh" \
    "$project_dir/ops/nautilus_auto_resume_train.sh"
chmod +x "$project_dir/ops/nautilus_auto_resume_train.sh"
touch "$project_dir/checkpoints/step_0007000.pt"
printf '#!/usr/bin/env bash\nprintf invoked >> "$MFVIDEO_TORCHRUN_LOG"\n' \
    > "$project_dir/.venv/bin/torchrun"
chmod +x "$project_dir/.venv/bin/torchrun"

export MFVIDEO_PROJECT_DIR="$project_dir"
export MFVIDEO_TORCHRUN_LOG="$work_dir/torchrun.log"
flock "$project_dir/.cache/nautilus-auto-resume.lock" \
    bash "$project_dir/ops/nautilus_auto_resume_train.sh"

test ! -e "$MFVIDEO_TORCHRUN_LOG"
printf 'PASS: duplicate boot recovery exits while the resume lock is held\n'
