# Nautilus A100 allocation monitor

## Purpose

The four-A100 Kubernetes request can remain pending longer than an interactive
Codex turn. A user-level system service now observes the assigned pod without
stalling the training goal.

## Runtime settings

- Unit: `mfvideo-nautilus-monitor.service`
- Poll interval: 60 seconds
- Target: `ecepxie/gpu-dev2-55886bbcc8-kt7qm`
- State: `.cache/nautilus_a100_monitor.state`
- Notification: desktop notification on the transition to `Running`, or to a
  failed, unknown, or unavailable state.

The monitor does not modify Kubernetes scheduling priority, preempt workloads,
or change the training configuration. The pod bootstrap independently starts
the four-rank resume job from step 7000 or a later valid checkpoint.

## Verification

`tests/test_nautilus_allocation_monitor.sh` uses isolated fake Kubernetes and
desktop-notification commands. It proves a `Running` transition writes durable
state and emits the `Nautilus-A100 allocated` notification. The service was
verified active with a live main process and a 60-second sleep worker.
