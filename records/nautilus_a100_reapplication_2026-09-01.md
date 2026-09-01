# Nautilus four-A100 reapplication — 2026-09-01

## Reason for reapplication

The prior `ecepxie/gpu-dev2` Deployment no longer existed, while the shared
`yuw-home` persistent volume claim and `nautilus-init` ConfigMap remained
available.  A replacement Deployment was required before the allocation
monitor could resume the MFVideo training process.

## Submitted specification

- Deployment and selector: `gpu-dev2` / `app=gpu-dev2`.
- Accelerator request and limit: four `nvidia.com/a100` GPUs.
- Preferred hardware: `NVIDIA-A100-SXM4-80GB`.
- Container: `nvcr.io/nvidia/pytorch:24.05-py3`, executing `/init/init.sh`.
- Persistent volume claim: `yuw-home` mounted at `/root`.
- Initialization ConfigMap: `nautilus-init` mounted at `/init`.
- Shared memory: 16 GiB memory-backed `emptyDir` mounted at `/dev/shm`.
- Resource requests / limits: 8 / 16 central processing units, 32 / 64 GiB
  memory, and 64 / 128 GiB ephemeral storage.

The namespace standard A100 quota had 6 of 8 GPUs in use, so a normal four-GPU
request would be rejected at admission.  The replacement therefore uses the
cluster's `opportunistic` priority class, which is excluded from that quota;
the allocation may be preempted by higher-priority work.

## Verification

`kubectl apply --dry-run=server` accepted the manifest, and the live Deployment
created ReplicaSet `gpu-dev2-5f94756b9d`.  Its Pod
`gpu-dev2-5f94756b9d-ftff4` was Pending immediately after submission with a
verified four-A100 request.  The existing monitor selects `app=gpu-dev2` and
will invoke the source synchronization and checkpoint-resume hook on the first
`Running` transition.
