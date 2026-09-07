# Fast Nautilus allocation recovery — 2026-09-07

## Failure boundary

The Kubernetes event stream showed a replacement four-A100 Pod assigned and
started, followed by preemption about 85 seconds later.  At the same time the
allocation monitor logged the beginning of `nautilus_sync_and_resume.sh` and
then exit 137.  It did not log its recovery launch.  The old bootstrap also
ran package installation before calling the resume script.  Therefore the
short allocation could end before either path launched `torchrun`.

## Correction

`ops/nautilus_init.sh` launches the guarded resume script before optional
package, terminal, and SSH provisioning.  The allocation monitor now calls
`ops/nautilus_bootstrap_guard.sh` rather than directly calling the slow
copy-and-restart hook.  The guard checks the newly Running Pod for either the
bootstrap resume process or four-rank `torchrun` for 20 seconds.  If found,
it preserves the process; only an absent bootstrap uses the existing
`nautilus_sync_and_resume.sh` fallback.

This removes the prior event-path behavior that copied source and terminated
training during the first allocation minutes.  It does not alter input size,
optimizer settings, checkpoint selection, or Kubernetes priority.

## Verification

`tests/test_nautilus_bootstrap_guard.sh` proves both branches: a missing
bootstrap calls the fallback exactly once, while an active bootstrap prevents
it.  The allocation-monitor, synchronization, bootstrap-order,
safe-checkpoint, and health-check tests pass.  The live user service has
`MFVIDEO_MONITOR_ON_RUNNING` set to the guard, is active, and has
`Restart=always` with a five-second restart delay.
