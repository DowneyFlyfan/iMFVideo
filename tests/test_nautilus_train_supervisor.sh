#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/ops/nautilus_train_supervisor.sh"

[[ $(checkpoint_step checkpoints/step_0007000.pt) == 7000 ]]
[[ $(checkpoint_step checkpoints/step_0010000.pt) == 10000 ]]
should_resume 7000 10000
! should_resume 10000 10000
! should_resume 10001 10000

printf 'PASS: supervisor resumes only incomplete checkpoint runs\n'
