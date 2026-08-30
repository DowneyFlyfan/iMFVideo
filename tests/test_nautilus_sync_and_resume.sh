#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat > "$work_dir/bin/kubectl" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MFVIDEO_KUBECTL_LOG"
if [[ " $* " == *" exec "* ]]; then
    cat > /dev/null
fi
SH
chmod +x "$work_dir/bin/kubectl"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_KUBECTL_LOG="$work_dir/kubectl.log"
bash "$repo_dir/ops/nautilus_sync_and_resume.sh" gpu-dev2-test

grep -Fq "cp $repo_dir/train.py ecepxie/gpu-dev2-test:/root/downeyflyfan/MFVideo/train.py" "$MFVIDEO_KUBECTL_LOG"
grep -Fq "cp $repo_dir/imf_video.py ecepxie/gpu-dev2-test:/root/downeyflyfan/MFVideo/imf_video.py" "$MFVIDEO_KUBECTL_LOG"
grep -Fq "cp $repo_dir/models ecepxie/gpu-dev2-test:/root/downeyflyfan/MFVideo" "$MFVIDEO_KUBECTL_LOG"
grep -Fq 'exec -i gpu-dev2-test -- bash -s' "$MFVIDEO_KUBECTL_LOG"
printf 'PASS: allocation recovery syncs local code and restarts the pod job\n'
