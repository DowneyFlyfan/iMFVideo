#!/usr/bin/env bash
# A preemptible GPU pod must request training before optional boot provisioning.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
script="$repo_dir/ops/nautilus_init.sh"

test -x "$script"
resume_line=$(grep -nF '/init/nautilus_auto_resume_train.sh' "$script" | head -n 1 | cut -d: -f1)
apt_line=$(grep -nF 'apt-get update -qq' "$script" | head -n 1 | cut -d: -f1)
test -n "$resume_line"
test -n "$apt_line"
(( resume_line < apt_line ))
printf 'PASS: Pod bootstrap launches recovery before optional provisioning\n'
