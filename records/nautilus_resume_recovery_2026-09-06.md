# Nautilus A100 resume recovery — 2026-09-06

## Allocation

The replacement `gpu-dev2-5f94756b9d-7lss2` Pod received four NVIDIA A100
80 GiB GPUs.  Initial direct checks confirmed four CUDA-visible devices and
the same shared `yuw-home` persistent volume claim.

## Recovered defects

The persistent-volume `config.py` was zero bytes.  The allocation hook had
intentionally excluded `config.py` from its local-to-server synchronization,
so its resume regular expression could not find `RunConfig.resume` and every
automatic recovery exited before launch.

The 8k checkpoint contains the authoritative server configuration: 48 input
channels, 31 latent frames, latent size `(44, 80)`, four samples per GPU, and
10,000 total steps.  It is incompatible with the current local 4k tuning
configuration and must not be replaced by those defaults.

`restore_checkpoint_config.py` now atomically materializes the configuration
recorded in the selected checkpoint into a synchronized `config.py` template.
The recovery hook synchronizes the template and materializer before launching.
The launcher is pinned to the requested 7k checkpoint and passes that same
path as the materializer's effective `RunConfig.resume`; it no longer uses an
8k configuration with a 7k model/optimizer state.  Tests cover config
materialization, synchronization of `config.py`, and the automatic-resume
selection behavior.

## Runtime evidence

The first corrected launch built the 290.1M-parameter model and restored step
7000, with the intended 48-by-31-by-44-by-80 input geometry.  Its two
non-finite-gradient skips did not update weights.  Their direct cause was then
identified: the 7k checkpoint stores `resume_linear_qk_scale=0.3`, but the
old launcher had materialized the latest 8k configuration with scale 1.0 while
the actual runtime `RunConfig.resume` still pointed to 7k.  The affected node
then became `NodeNotReady`, making `kubectl exec` time out.  The unhealthy Pod
was replaced before further training.

While no training process is live, each allocated A100 runs one verified
24,576-by-24,576 BF16 random matrix-multiplication filler.  The synchronization
hook stops only the exact named `a100_goal_filler` Python processes before it
launches training.
