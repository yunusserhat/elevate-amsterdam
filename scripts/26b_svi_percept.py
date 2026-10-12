# -*- coding: utf-8 -*-
"""Step 2.4b: perceived qualities with the ELEVATE team's own Amsterdam perception model.

Model: Spatial-Data-Science-and-GEO-AI-Lab/svi_percept on Hugging Face (GPL-3.0). It comes from the
Percept citizen-science survey in Amsterdam (Danish, Labib, Ricker and Helbich 2025, Computers,
Environment and Urban Systems 116: 22,637 ratings from 331 people) and scores five qualities on the
survey's 1-5 scale: walkability, bikeability, pleasantness, greenness and safety.

Method, as in the svi_percept package (model.py and feature_extractor.py, commit on GitHub master):
OpenCLIP ViT-H-14-378-quickgelu (pretrained dfn5b) image features, L2-normalised; for each quality,
the k = 40 most similar rated images by dot product; weights softmax(10 ** similarity); score = the
weighted mean of their ratings. It is re-implemented here so that whole batches run on the GPU without
the package's transformers pipeline. `--check-official N` scores N views with the package itself and
reports the largest difference.

Usage:
  python 26b_svi_percept.py LIST OUT_PREFIX [--shard K/N] [--chunk 4000] [--batch 64] [--limit N]
                             [--precision fp32|bf16] [--check-official N]
Writes OUT_PREFIX_shardK_partNNN.parquet every CHUNK images, so a rerun skips finished parts.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJ = Path(__file__).resolve().parents[1]
REPO = "Spatial-Data-Science-and-GEO-AI-Lab/svi_percept"
CLIP_NAME, CLIP_TAG, CLIP_HUB = "ViT-H-14-378-quickgelu", "dfn5b", "apple/DFN5B-CLIP-ViT-H-14-378"
CATEGORIES = ["walkability", "bikeability", "pleasantness", "greenness", "safety"]
K = 40


def clip_revision() -> str:
    """Commit of the CLIP weights that open_clip resolved in the local Hugging Face cache."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    ref = hub / f"models--{CLIP_HUB.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


class KNN:
    """The svi_percept scoring: ratings of the k most similar survey images, softmax(10**sim) weights."""

    def __init__(self, device):
        self.revision = HfApi().model_info(REPO).sha
        data = np.load(hf_hub_download(REPO, "data.npz", revision=self.revision))
        self.vecs = {c: torch.from_numpy(data[f"{c}_vecs"]).float().to(device) for c in CATEGORIES}
        self.scores = {c: torch.from_numpy(data[f"{c}_scores"]).float().to(device) for c in CATEGORIES}
        self.sizes = {c: int(self.scores[c].numel()) for c in CATEGORIES}

    @torch.inference_mode()
    def __call__(self, feats: torch.Tensor) -> dict[str, np.ndarray]:
        out = {}
        for c in CATEGORIES:
            sims = feats @ self.vecs[c].T
            top, idx = torch.topk(sims, k=K, dim=1)
            w = torch.softmax(torch.pow(10.0, top), dim=1)
            out[c] = (self.scores[c][idx] * w).sum(dim=1).cpu().numpy()
        return out


class Views(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert("RGB")), i


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_TAG, device=device)
    return model.eval(), preprocess


@torch.inference_mode()
def encode(model, x, precision):
    if precision == "bf16":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode_image(x)
    else:
        f = model.encode_image(x)
    f = f.float()
    return f / f.norm(dim=-1, keepdim=True)


def score(model, preprocess, knn, paths, device, precision, batch=64, workers=8) -> pd.DataFrame:
    res = {c: np.empty(len(paths), dtype=np.float32) for c in CATEGORIES}
    for x, idx in DataLoader(Views(paths, preprocess), batch_size=batch, num_workers=workers, pin_memory=True):
        s = knn(encode(model, x.to(device, non_blocking=True), precision))
        for c in CATEGORIES:
            res[c][idx.numpy()] = s[c]
    df = pd.DataFrame({"image_id": [Path(p).stem for p in paths], **{f"svp_{c}": res[c] for c in CATEGORIES}})
    df["svp_model"] = f"{REPO}@{knn.revision[:8]}"
    df["svp_clip"] = f"{CLIP_NAME}/{CLIP_TAG}@{clip_revision()[:8]}"
    df["svp_precision"] = precision
    return df


def check_official(paths, ours: pd.DataFrame):
    """Score the same views with the svi_percept package (fp32, on the CPU: its GPU path fails when it
    writes a CUDA tensor into a numpy array) and report the largest difference."""
    from svi_percept import SVIPerceptPipeline
    pipe = SVIPerceptPipeline(device="cpu")
    rows = []
    for p in paths:
        r = pipe(p)["results"]
        rows.append({"image_id": Path(p).stem, **{f"svp_{c}": float(r[c]) for c in CATEGORIES}})
    off = pd.DataFrame(rows).set_index("image_id")
    mine = ours.set_index("image_id").loc[off.index]
    diff = (mine[[f"svp_{c}" for c in CATEGORIES]] - off).abs()
    print("official vs re-implementation, max |diff| per quality:")
    print(diff.max().round(5).to_string())
    return float(diff.max().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_list")
    ap.add_argument("out_prefix")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N views only (timing and checks)")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--check-official", type=int, default=0, help="compare N views with the svi_percept package")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    lines = [l.strip() for l in (PROJ / args.image_list).read_text().splitlines() if l.strip()]
    paths = [str(PROJ / l) for i, l in enumerate(lines) if i % n == k]
    if args.limit:
        paths = paths[:args.limit]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model, preprocess = load_clip(device)
    knn = KNN(device)
    print(f"svi_percept {knn.revision[:8]}, survey images per quality: {knn.sizes}; CLIP {clip_revision()[:8]}", flush=True)
    out = PROJ / args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.check_official:
        sample = paths[:args.check_official]
        ours = score(model, preprocess, knn, sample, device, args.precision, batch=args.batch)
        ours.to_parquet(f"{out}_check_{args.precision}.parquet", index=False)
        print(f"max |diff| = {check_official(sample, ours):.5f}", flush=True)
        return
    for part, start in enumerate(range(0, len(paths), args.chunk)):
        path = Path(f"{out}_shard{k}_part{part:03d}.parquet")
        if path.exists():
            continue
        score(model, preprocess, knn, paths[start:start + args.chunk], device, args.precision, batch=args.batch).to_parquet(path, index=False)
        done = min(start + args.chunk, len(paths))
        print(f"shard {k}: {done}/{len(paths)} in {time.time() - t0:.0f}s", flush=True)
    print(f"SVP_SHARD_{k}_DONE", flush=True)


if __name__ == "__main__":
    main()
