# Nautilus A100 automatic resume

## Purpose

The four-A100 request can remain queued longer than a foreground Codex terminal.
The pod bootstrap now invokes a guarded launcher after Secure Shell (SSH) starts.

## Behavior

- It does nothing if the MFVideo project or its virtual environment is absent.
- It refuses a duplicate `torchrun` job.
- It selects `checkpoints/step_0007000.pt`, unless a later numbered checkpoint
  exists, and updates only `RunConfig.resume`.
- It uses `.venv/bin/torchrun`, avoiding the system PyTorch/Protocol Buffers
  mismatch previously observed on Nautilus.
- It launches the four-rank T2-stabilized training run and attaches the existing
  heartbeat fallback to the exact training process identifier.

The 7k checkpoint remains the expected restart point; choosing a later checkpoint
is only a crash-recovery safeguard after a verified later save has appeared.
