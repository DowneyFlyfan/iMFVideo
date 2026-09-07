#!/usr/bin/env bash
# A newer checkpoint may be used only when it records the corrected Q/K state.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
project_dir="$work_dir/project"
mkdir -p "$project_dir/.venv/bin" "$project_dir/checkpoints" "$work_dir/bin"
printf '    resume: str = ""\n' > "$project_dir/config.py"
ln -s "$repo_dir/.venv/bin/python" "$project_dir/.venv/bin/python"
printf '#!/usr/bin/env bash\nexit 0\n' > "$project_dir/.venv/bin/torchrun"
chmod +x "$project_dir/.venv/bin/torchrun"

PROJECT_DIR="$project_dir" "$project_dir/.venv/bin/python" - <<'PY'
import os
from pathlib import Path

import torch

directory = Path(os.environ["PROJECT_DIR"]) / "checkpoints"
torch.save({"step": 7000, "model": {}, "ema": {}, "optimizer": {}},
           directory / "step_0007000.pt")
torch.save({"step": 8000, "linear_qk_preconditioned": True,
            "model": {}, "ema": {}, "optimizer": {}},
           directory / "step_0008000.pt")
(directory / "step_0009000.pt").write_bytes(b"interrupted checkpoint write")
PY

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

grep -Fq 'resume: str = "checkpoints/step_0008000.pt"' "$project_dir/config.py"
printf 'PASS: boot recovery prefers the newest corrected checkpoint\n'
