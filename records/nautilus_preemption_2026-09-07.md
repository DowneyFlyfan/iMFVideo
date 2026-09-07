# Nautilus A100 preemption — 2026-09-07

## Observed sequence

Kubernetes events show that the running four-A100 Pod
`gpu-dev2-5f94756b9d-kd96k` was preempted on
`node-2-4.sdsc.optiputer.net` by a higher-priority Pod.  The Deployment
created replacement `gpu-dev2-5f94756b9d-c6zqc`, which was assigned to the
same node and then preempted again shortly after startup.  The current
replacement `gpu-dev2-5f94756b9d-5bz96` is Pending and requesting four A100
GPUs.

## Root cause

The Deployment explicitly uses Kubernetes priority class `opportunistic`
(priority `-2000000000`).  The event reason is `Preempted`; the monitor's
`exit 137` records the termination of its remote synchronization command
when the container was killed.  This is scheduler preemption, not an
out-of-memory condition or an MFVideo training failure.

## Recovery state

The local allocation monitor remains active.  It detects a newly Running Pod,
synchronizes the checked-out recovery scripts, and starts the supervised
resume path.  Because no corrected post-7k checkpoint has been saved yet,
the next run will correctly resume from `checkpoints/step_0007000.pt`.
