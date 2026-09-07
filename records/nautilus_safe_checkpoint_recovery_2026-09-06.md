# Nautilus safe checkpoint recovery — 2026-09-06

## Problem

The four-A100 recovery script always selected `step_0007000.pt`.  This was
safe for the legacy checkpoint, but after a corrected checkpoint is saved it
would unnecessarily discard all later stable work after a Pod replacement.

## Selection rule

`ops/nautilus_auto_resume_train.sh` treats step 7000 as the baseline and
scans later `step_*.pt` files in descending order.  A later checkpoint is
selected only when all of the following hold:

- its stored `step` is greater than 7000;
- its `linear_qk_preconditioned` marker is true; and
- PyTorch can deserialize it as a dictionary.

Unreadable candidate files are skipped.  Thus a partial checkpoint left by a
node failure cannot prevent recovery, and the legacy unmarked step-8000 file
is never selected.

## Verification

The new `tests/test_nautilus_auto_resume_safe_checkpoint.sh` creates a stable
7k checkpoint, a marked 8k checkpoint, and a deliberately corrupt 9k file.
It proves recovery selects 8k.  The existing supervisor, lock, and training
health-monitor tests also pass.

## Scope

The current running job is not restarted.  The new selection is used only by
the next recovery invocation, after the local script has been synchronized to
the shared Persistent Volume Claim.
