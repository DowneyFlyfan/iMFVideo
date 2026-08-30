# Allocation-triggered Nautilus recovery monitor

## Objective

The replacement four-A100 Pod can wait in the Kubernetes scheduler for an
unbounded duration. Once assigned, it must resume from the latest complete
checkpoint with the verified local source, rather than continuing an old copy
left on the persistent volume claim.

## Mechanism

The existing user-level `mfvideo-nautilus-monitor.service` polls the newest
non-terminal `app=gpu-dev2` Pod every 60 seconds.  On its first `Running`
transition it calls `ops/nautilus_sync_and_resume.sh POD`.

The hook waits for the Python environment, copies only source/runtime files
from the local repository (`train.py`, `imf_video.py`, Moonlight optimizer,
models, operations scripts, checkpoint repair utility, and GPU heartbeat),
and deliberately does **not** copy `config.py`: server training uses the
server's full-dataset, 10k-step configuration.  It then stops only exact
`torchrun`, supervisor, and heartbeat-fallback processes, and invokes the
auto-resume script.  That script repairs the legacy 8k EMA checkpoint before
launching four-rank training; the in-Pod supervisor restarts future crashes
from the latest complete checkpoint.

## Safety

- No input size, optimizer, or hyperparameter is changed.
- The monitor runs the hook once per Pod/phase transition, preventing restart
  loops.
- It waits up to five minutes for the Pod environment and exits rather than
  acting on an incomplete container.
- The GPU heartbeat preserves allocation while a controlled restart occurs.

## Verification

`tests/test_nautilus_allocation_monitor.sh` proves that a first `Running`
transition invokes the recovery hook with the resolved replacement Pod name.
`tests/test_nautilus_sync_and_resume.sh` exercises the real recovery script
against a controlled Kubernetes command boundary and verifies synchronization
of `train.py`, `imf_video.py`, and `models`, followed by the exact recovery
command.  Both tests and shell syntax checks passed before enabling the service.
