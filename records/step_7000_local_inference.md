# Step-7000 local video inference

## Inputs

- Checkpoint: `checkpoints/step_0007000.pt` on the shared `yuw-home` PVC.
- Checkpoint step: 7,000.
- Model: 290.1 million parameters, 48 latent channels, 31 latent frames,
  and 44 by 80 latent spatial dimensions.
- Decoder: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` VAE. The T2V and I2V Wan 2.2
  VAE variants have 16 latent channels and are incompatible with this checkpoint.
- Normalization: the 48-channel mean and standard deviation from the server
  `wan22_full/stats.npz` were applied in reverse before VAE decoding.
- Device: local NVIDIA GeForce RTX 5070 Ti with 16 GiB memory.
- Sampling: seed 7000, class label 0, four MeanFlow steps, CFG scale 2.0.

## Result

`step_0007000_sample.mp4` was generated successfully.

- Codec: H.264.
- Resolution: 1280 by 704 pixels.
- Frames: 121.
- Frame rate: 16 frames per second.
- Duration: 7.5625 seconds.

## Validation and correction

The checkpoint transfer was resumed over a temporary HTTP server on the PVC
after Kubernetes direct copy timed out. Its final byte count matched the server
exactly and Python ZIP validation found no corrupt member across 2,000 entries.

The first decode failed because the sampler emitted float32 latents while the
VAE weights were bfloat16. `generate_7k_video.py` now casts denormalized
latents at the model-to-VAE boundary. The regression test passes, and the
successful full generation verifies the correction end to end.
