# Nautilus resume-lock recovery — 2026-09-06

## Incident

A replacement four-A100 Pod became Running with two independent recovery
launchers: the allocation monitor used the current checked-out script pinned
to step 7000, while the mounted `nautilus-init` ConfigMap used an older script
that selected step 8000.  Both parsed multi-gigabyte checkpoints from the
shared Ceph Persistent Volume Claim at once.

## Root cause

The startup topology had no mutual exclusion between the ConfigMap bootstrap
and the monitor.  This was not a model or CUDA failure: the two confirmed
Python readers were blocked in filesystem page reads (`folio_`), one for 7k
and one for 8k.

## Correction

The obsolete 8k bootstrap process and only its child reader were terminated.
The current 7k reader then materialized its checkpoint-derived configuration
and launched four-rank training.  `ops/nautilus_auto_resume_train.sh` now holds
an exclusive `.cache/nautilus-auto-resume.lock` before it can inspect a
checkpoint or modify `config.py`.  A duplicate caller exits successfully
without touching the shared state.

## Verification

`tests/test_nautilus_auto_resume_lock.sh` holds the lock and proves that a
second invocation does not start `torchrun`.  The existing boot-resume and
allocation-sync tests pass with the lock enabled.
