#!/usr/bin/env bash
# Preserve a just-started bootstrap recovery; use slow synchronization only
# when the remote Pod has no recovery process at all.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
guard="$repo_dir/ops/nautilus_bootstrap_guard.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat > "$work_dir/bin/kubectl" <<'SH'
#!/usr/bin/env bash
exit "${MFVIDEO_FAKE_REMOTE_EXIT:-1}"
SH
cat > "$work_dir/fallback.sh" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$1" >> "$MFVIDEO_FALLBACK_LOG"
SH
chmod +x "$work_dir/bin/kubectl" "$work_dir/fallback.sh"

export PATH="$work_dir/bin:$PATH"
export MFVIDEO_BOOTSTRAP_GRACE_SECONDS=0
export MFVIDEO_SYNC_AND_RESUME="$work_dir/fallback.sh"
export MFVIDEO_FALLBACK_LOG="$work_dir/fallback.log"

export MFVIDEO_FAKE_REMOTE_EXIT=1
bash "$guard" test-pod
grep -Fxq test-pod "$MFVIDEO_FALLBACK_LOG"

: > "$MFVIDEO_FALLBACK_LOG"
export MFVIDEO_FAKE_REMOTE_EXIT=0
bash "$guard" test-pod
test ! -s "$MFVIDEO_FALLBACK_LOG"
printf 'PASS: allocation guard preserves bootstrap recovery and falls back only when absent\n'
