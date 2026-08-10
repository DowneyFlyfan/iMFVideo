"""Fetch Wan-Syn latent chunks from HuggingFace and convert them to training .pt files.

The dataset ships Wan2.1 VAE latents already encoded, so there is no VAE pass:
each parquet row carries a (C=16, T=20, H=56, W=104) float32 latent plus a
caption. Full latents are 29,138 transformer tokens and cost ~4.6 s/step even
for a 48M model (records/compute_budget_sizing.md), so `convert` can shrink the
spatial dims to fit a wall-clock budget.

Usage:
    python prepare_wan_syn.py download --videos 1000 --out .cache/wan_syn_parquet_big
    python prepare_wan_syn.py convert  --parquet-dir .cache/wan_syn_parquet_big \
                                       --out .cache/wan_syn_half --factor 2
"""

import argparse
import concurrent.futures as cf
import hashlib
import io
import os
import time

import numpy as np
import torch

REPO_ID = "FastVideo/Wan-Syn_77x448x832_600k"
VIDEOS_PER_CHUNK = 8  # measured on Part_25: 56 videos / 7 chunks


def list_chunks():
    """All parquet paths in the dataset repo, grouped so parts interleave.

    Returns:
        list[str]: repo-relative paths like "Part_26/latents_chunk_0000.parquet",
        ordered part-by-part in round-robin so that taking a prefix of the list
        samples across many parts rather than exhausting one.
    """
    from huggingface_hub import HfApi

    files = [
        f
        for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    by_part = {}
    for f in files:
        by_part.setdefault(f.split("/")[0], []).append(f)
    for v in by_part.values():
        v.sort()
    parts = sorted(by_part, key=lambda p: int(p.split("_")[1]))
    out, i = [], 0
    while any(len(by_part[p]) > i for p in parts):
        for p in parts:
            if len(by_part[p]) > i:
                out.append(by_part[p][i])
        i += 1
    return out


def download(args):
    """Download enough chunks to cover args.videos, in parallel, resumable."""
    from huggingface_hub import hf_hub_download

    chunks = list_chunks()
    need = max(1, -(-args.videos // VIDEOS_PER_CHUNK))  # ceil division
    want = chunks[:need]
    print(
        f"{len(chunks)} chunks in repo; need {need} for ~{args.videos} videos "
        f"(~{need * 126 / 1024:.1f} GiB)",
        flush=True,
    )
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    done = [0]

    def grab(path):
        p = hf_hub_download(
            REPO_ID, path, repo_type="dataset", local_dir=args.out
        )
        done[0] += 1
        el = time.time() - t0
        gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(args.out)
            for f in fs
        ) / 2**30
        print(
            f"[{done[0]}/{len(want)}] {path}  {gb:.2f} GiB total  "
            f"{gb/max(el,1)*3600:.1f} GiB/h",
            flush=True,
        )
        return p

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(grab, want))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


def shrink(lat, factor, mode):
    """Reduce the spatial dims of a latent.

    Args:
        lat: (C, T, H, W) float32 latent straight from the parquet row.
        factor: integer spatial reduction, e.g. 2 maps (56, 104) -> (28, 52).
        mode: "pool" average-pools 2x2 latent cells, keeping the whole scene;
            "crop" takes the centre crop, keeping true un-averaged latents but
            only 1/factor^2 of the field of view.

    Returns:
        (C, T, H//factor, W//factor) float32.

    Note: "pool" produces latents the Wan VAE decoder never saw, so samples from
    a model trained on them will not decode cleanly. It is meant for studying
    training dynamics under a wall-clock budget, not for producing video.
    """
    if factor == 1:
        return lat
    c, t, h, w = lat.shape
    if mode == "crop":
        h2, w2 = h // factor, w // factor
        top, left = (h - h2) // 2, (w - w2) // 2
        return lat[:, :, top : top + h2, left : left + w2].contiguous()
    return torch.nn.functional.avg_pool3d(
        lat.unsqueeze(0), kernel_size=(1, factor, factor)
    ).squeeze(0)


def convert(args):
    """Parquet rows -> per-video .pt files, per-channel normalized.

    Writes {"latent": (C, T, H', W') float32, "label": int} per video plus a
    stats.npz holding the per-channel mean/std used, matching the convention of
    the existing .cache/wan_syn_full set.
    """
    import pyarrow.parquet as pq

    files = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(args.parquet_dir)
        for f in fs
        if f.endswith(".parquet")
    )
    if args.limit_chunks:
        files = files[: args.limit_chunks]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(files)} parquet files -> {args.out}", flush=True)

    # Pass 1: decode, shrink, write raw; accumulate per-channel sums so the
    # normalization matches the whole set rather than one chunk.
    n = 0
    csum = torch.zeros(16, dtype=torch.float64)   # (C,) sum of values
    csqs = torch.zeros(16, dtype=torch.float64)   # (C,) sum of squares
    count = 0                                     # elements per channel
    paths = []
    t0 = time.time()
    for fi, f in enumerate(files):
        tab = pq.read_table(
            f, columns=["caption", "vae_latent_bytes", "vae_latent_shape",
                        "vae_latent_dtype"]
        )
        for r in range(tab.num_rows):
            shape = [int(x) for x in tab["vae_latent_shape"][r].as_py()]
            dt = tab["vae_latent_dtype"][r].as_py()
            buf = tab["vae_latent_bytes"][r].as_py()
            # lat: (C=16, T=20, H=56, W=104) float32 as stored in the dataset
            lat = torch.from_numpy(
                np.frombuffer(buf, dtype=np.dtype(dt)).reshape(shape).copy()
            ).float()
            lat = shrink(lat, args.factor, args.mode)   # (C, T, H', W')
            cap = tab["caption"][r].as_py() or ""
            label = int(hashlib.md5(cap.encode()).hexdigest(), 16) % 1000
            p = os.path.join(args.out, f"video_{n:06d}.pt")
            torch.save({"latent": lat, "label": label}, p)
            paths.append(p)
            csum += lat.double().sum(dim=(1, 2, 3))
            csqs += (lat.double() ** 2).sum(dim=(1, 2, 3))
            count += lat.shape[1] * lat.shape[2] * lat.shape[3]
            n += 1
        if (fi + 1) % 5 == 0 or fi == len(files) - 1:
            print(
                f"[{fi+1}/{len(files)}] {n} videos, "
                f"{(time.time()-t0)/60:.1f} min",
                flush=True,
            )

    mean = (csum / count).float().reshape(16, 1, 1, 1)          # (C,1,1,1)
    var = (csqs / count).float().reshape(16, 1, 1, 1) - mean**2  # (C,1,1,1)
    std = var.clamp_min(1e-8).sqrt()
    np.savez(
        os.path.join(args.out, "stats.npz"),
        mean=mean.numpy(),
        std=std.numpy(),
    )
    print(
        f"stats: mean [{mean.min():.3f}, {mean.max():.3f}] "
        f"std [{std.min():.3f}, {std.max():.3f}]",
        flush=True,
    )

    # Pass 2: apply normalization in place.
    for i, p in enumerate(paths):
        d = torch.load(p, map_location="cpu")
        d["latent"] = (d["latent"] - mean) / std     # (C, T, H', W')
        torch.save(d, p)
        if (i + 1) % 200 == 0:
            print(f"normalized {i+1}/{len(paths)}", flush=True)
    ex = torch.load(paths[0], map_location="cpu")["latent"]
    tok = ex.shape[1] * (ex.shape[2] // 2) * (ex.shape[3] // 2) + 18
    print(
        f"done: {n} videos, latent {tuple(ex.shape)}, {tok} tokens/sample",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--videos", type=int, default=1000)
    d.add_argument("--out", default=".cache/wan_syn_parquet_big")
    d.add_argument("--workers", type=int, default=8)
    d.set_defaults(func=download)

    c = sub.add_parser("convert")
    c.add_argument("--parquet-dir", default=".cache/wan_syn_parquet_big")
    c.add_argument("--out", default=".cache/wan_syn_half")
    c.add_argument("--factor", type=int, default=2)
    c.add_argument("--mode", choices=["pool", "crop"], default="pool")
    c.add_argument("--limit-chunks", type=int, default=0)
    c.set_defaults(func=convert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
