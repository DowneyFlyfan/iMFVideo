# Boot-safe 8k EMA repair

## Context

The legacy run resumed from step 7000 after rescaling online linear-attention
query/key producers but left the Exponential Moving Average (EMA) weights in
the old parameterization.  It saved `checkpoints/step_0008000.pt` at 2026-08-29
01:27 UTC with `linear_qk_preconditioned` absent.  A subsequent node failure
made a replacement four-A100 Pod necessary before a manual controlled restart
could complete.

## Change

`ops/nautilus_auto_resume_train.sh` now inspects only the legacy step-8000
checkpoint before launching `torchrun`.  If and only if the checkpoint has
step 8000 and lacks `linear_qk_preconditioned`, it invokes
`repair_checkpoint_ema.py`.  That utility clones online weights into EMA,
writes the precondition marker and scale `1.0`, then atomically replaces the
checkpoint.  Marked checkpoints are not modified.

The same script was patched into the `ecepxie/nautilus-init` ConfigMap.  The
mounted bootstrap therefore repairs the persistent-volume checkpoint before it
can begin the replacement Pod's training process.

## Verification

The regression test creates a tiny unmarked step-8000 checkpoint with unequal
online and EMA tensors and asserts that auto-resume calls the repair utility
before launching the four-rank process.  Results:

```text
PASS: boot recovery launches training, heartbeat, and supervisor
1 passed: tests/test_repair_checkpoint_ema.py
PASS: supervisor resumes only incomplete checkpoint runs
```

The ConfigMap byte digest equals the local script digest:

```text
93c090385303db7c18515621c0d979c301f56f07672701dbcafab958d251c7dd
```

## Operational state

The old host `node-1-3.sdsc.optiputer.net` became NotReady and exposed no
healthy A100/CSI driver.  The Deployment created replacement Pod
`gpu-dev2-55886bbcc8-xdmqq`, requesting four A100 GPUs.  Its bootstrap will
perform the repair only after the scheduler assigns a healthy node.

The persistent local allocation monitor now resolves the newest non-terminal
`app=gpu-dev2` Pod rather than retaining the obsolete Pod name.  Its durable
state includes both Pod name and phase, so a replacement transition to
`Running` emits a new Nautilus-A100 allocation notification even if the old
Pod had previously been running.
