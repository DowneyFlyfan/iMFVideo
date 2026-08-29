# Nautilus training supervisor

The four-A100 bootstrap starts training only on Pod creation. Its original
watchdog preserves GPU utilization after a torchrun exit, but deliberately
does not restart training. `ops/nautilus_train_supervisor.sh` fills that gap.

The supervisor observes the torchrun parent. If it exits before the greatest
completed checkpoint reaches `config.optim.total_steps`, it stops the known
watchdog fallback matrix-multiplication processes and invokes the existing
auto-resume script. It exits instead when the latest checkpoint has reached
the configured total, so successful training is never restarted.

The shell regression test covers the filename-to-step parser and completion
gate. A second isolated bootstrap test proves that Pod initialization starts
torchrun, the GPU heartbeat, and this supervisor together. The updated
`nautilus_auto_resume_train.sh` was merged into the live `nautilus-init`
ConfigMap without restarting the current Pod; its remote SHA-256 matches the
local script.
