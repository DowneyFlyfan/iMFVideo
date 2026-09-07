# Nautilus immediate 8k resume — 2026-09-07

## Requirement

On the next four-A100 allocation, training must begin immediately from
`checkpoints/step_0008000.pt`, rather than waiting for a copy-and-restart
operation or falling back to step 7000.

## Root cause of the previous missed allocation

The prior allocation handler started `nautilus_sync_and_resume.sh` on the
first `Running` observation.  That path copied the repository, stopped exact
training processes, waited for them to exit, and only then launched recovery.
The short-lived Pod was preempted during this slow sequence.  Separately, the
checkpoint selector rejected the legacy 8k file before reaching its repair
branch, so a step-7000 fallback was selected.

## Enforced path

1. `nautilus_init.sh` launches `/init/nautilus_auto_resume_train.sh` before
   optional provisioning.
2. `nautilus_auto_resume_train.sh` requires `step_0008000.pt`.  If its Q/K
   correction marker is absent, it atomically replaces its Exponential Moving
   Average state with the online state and records the corrected marker before
   restoring `config.py`.
3. The allocation monitor invokes `nautilus_bootstrap_guard.sh`; an active
   bootstrap or `torchrun` is preserved.  The slow sync path is fallback-only.

## Evidence

The 8k regression test first failed on the old step-7000 behavior and then
passed after the change, logging both `repairing legacy EMA` and `resuming
from checkpoints/step_0008000.pt`.  The bootstrap order, guard, lock, and
safe-checkpoint tests also pass.  The live `nautilus-init` ConfigMap hashes
match the local bootstrap and auto-resume scripts, and the monitor service is
active with `Restart=always`.
