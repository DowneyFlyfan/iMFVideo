# Wan-Syn Local Smoke Training (1x RTX 5070 Ti)

## Data

- Source: HuggingFace `FastVideo/Wan-Syn_77x448x832_600k`, `Part_25/latents_chunk_0086..0092.parquet` (7 files, 846 MB total, in `.cache/wan_syn_parquet/`)

- 56 videos, each raw Wan2.1 VAE (Variational Autoencoder) latent of shape (C=16, T=20, H=56, W=104) float32

- Per-channel normalization over the whole set: mean range [-1.596, 1.937], std range [1.258, 4.060]; stats saved to `.cache/wan_syn_latents/stats.npz`

- Each video cut into non-overlapping crops of (16, 3, 16, 16): 6 temporal windows x 3 x 6 spatial grid = 108 crops/video, 6048 samples in `.cache/wan_syn_latents/`

- Pseudo-class label = md5(caption) mod 1000 (dataset is text-conditioned; label pathway exercised with deterministic hash classes)

## Settings

- Model: `IMFDiTVideo` from `config.py` defaults — hidden 1024, depth 19 (aux head depth 4), 16 heads, MLA (Multi-head Latent Attention: q_lora 512, kv_lora 256, qk_nope 48, qk_rope 16, v 64), patch (1,2,2), 264.7M params, attention `flash_jvp_attention` (CuTeDSL)

- Loss: `IMFVideoLoss` with config defaults (P_mean -0.4, P_std 1.0, data_proportion 0.5, cfg_beta 1.0, s_max 7.0, class_dropout 0.1, norm_p 1.0, norm_eps 0.01)

- Optimizer: moonlight (Muon + AdamW split), lr 1e-4, weight_decay 1e-5, muon per_shape coefficients, grad_clip 1.0

- Overrides for this run: batch 8, num_workers 0, total_steps 400, warmup 40, log_every 10, EMA (exponential moving average) off, no checkpoints

- Launch: `python run_smoke.py` (wrapper mutating `config` then calling `train.main()`), single process, fp32 master weights

## Results

- Throughput: 15.8 samples/s steady state (8.6 in first window, includes warmup/compile)

- `loss` (adaptive-weighted total) constant at 2.0000 by construction

- `loss_u`: step 10 = 1.8013, step 50 = 1.3405, step 100 = 0.8995, step 200 = 0.7987, step 400 = 0.7279

- `loss_v`: step 10 = 1.8012, step 400 = 0.6841

- grad_norm range over run: 0.744 - 1.467

- Full curves: `records/wan_syn_smoke_loss.png`, raw series `records/wan_syn_smoke_history.json`

## Reproduce

```bash
# download (846 MB)
python - <<'PY'
from huggingface_hub import hf_hub_download
for i in range(86, 93):
    hf_hub_download('FastVideo/Wan-Syn_77x448x832_600k',
                    f'Part_25/latents_chunk_{i:04d}.parquet',
                    repo_type='dataset', local_dir='.cache/wan_syn_parquet')
PY
# convert to .pt crops, then set config.data.latent_dir=".cache/wan_syn_latents" and run train.py
```
