# Nautilus training health monitor — 2026-09-06

## Gap

The allocation monitor originally acted only when a Pod changed state.  A
running job that repeatedly emitted non-finite gradients could remain alive,
skip every optimizer step, and hold four A100 GPUs without making progress.

## Behavior

While the selected Pod is Running, each monitor poll now executes the remote
health check.  It examines only the log segment after the latest confirmed
step-7000 resume.  Two or more non-finite-gradient events cause it to write an
instability marker and terminate only the MFVideo supervisor and its exact
four-rank `torchrun` parent.  The heartbeat watchdog then starts the named
UUID-aware random matrix-multiplication filler.

## Safety

The checker does not match generic Python processes, so it does not terminate
unrelated inference services such as vLLM.  The test fixture includes such an
unrelated process and proves it is not targeted.

## Current observation

The current four-A100 run has the intended 48-channel, 31-frame, 44-by-80
checkpoint configuration and Q/K scale 0.3.  No non-finite event appears after
the new step-7000 resume marker while all four GPUs report full utilization.
