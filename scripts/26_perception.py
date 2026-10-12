# -*- coding: utf-8 -*-
"""Step 2.4: perceived qualities with open Place Pulse 2.0 models (stand-in for ELEVATE's own).

Two model families, both used by ZenSVI (MIT; NUS Urban Analytics Lab), six Place Pulse
questions each. Column prefixes:
  vit_  Jiani11/human-perception-place-pulse (Apache-2.0; Ouyang; used in Global Streetscapes,
        Hou et al. 2024, ISPRS J. 215:216-238). ViT-B/16 at 384 px, binary head; score =
        softmax probability of class 1 x 10 (0-10), as in the authors' eval.py and ZenSVI's
        ClassifierPerceptionViT. Primary.
  r50_  seshing/placepulse (no licence file; ZenSVI's ClassifierPerception). ResNet-50 at
        224 px, five classes; score = sum(p_k * [1, 3, 5, 7, 9]) (1-9). Second opinion.
safety, lively, wealthy, beautiful: high = positive; boring, depressing: high = negative.

Loading never runs code from the checkpoints. seshing files are plain state dicts
(torch.load weights_only=True). Jiani11 files are pickled whole models whose pickle globals
were listed beforehand (torch, torchvision ViT classes, Model_01.Net); they are read with
weights_only=True and an explicit allow-list, and only their state dicts are kept. Both are
loaded with strict=True into architectures built here.

Usage:
  python 26_perception.py LIST OUT_PREFIX [--shard K/N] [--chunk 4000] [--batch 128] [--limit N]
Writes OUT_PREFIX_shardK_partNNN.parquet every CHUNK images, so a rerun skips finished parts.
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet50
from torchvision.models import vision_transformer as tv_vit

PROJ = Path(__file__).resolve().parents[1]
QUESTIONS = ["safety", "lively", "wealthy", "beautiful", "boring", "depressing"]
VIT = {"repo": "Jiani11/human-perception-place-pulse", "revision": "402164b5418525089fc6bc074478c7d3c6c64659",
       "files": {q: f"{q}.pth" for q in QUESTIONS}}
R50 = {"repo": "seshing/placepulse", "revision": "1d7f85c455303255d6d509ff0e5d27f54bdd4d4d",
       "files": {"safety": "safer.pth", "lively": "livelier.pth", "wealthy": "wealthier.pth",
                 "beautiful": "more beautiful.pth", "boring": "more boring.pth", "depressing": "more depressing.pth"}}
NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TF_VIT = T.Compose([T.Resize((384, 384)), T.ToTensor(), NORM])
TF_R50 = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])


class VitNet(nn.Module):
    """ViT-B/16 at 384 px with the authors' head (Model_01.Net), no pretrained download."""

    def __init__(self):
        super().__init__()
        self.model = tv_vit.vit_b_16(weights=None, image_size=384)
        self.model.heads.head = nn.Sequential(nn.Linear(768, 512), nn.ReLU(True), nn.Linear(512, 256), nn.ReLU(True),
                                              nn.Linear(256, 2))

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)[:, 1] * 10


class R50Net(nn.Module):
    """ZenSVI's PlacePulseClassificationModel: ResNet-50 features, four linear layers, 5 classes."""

    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50(weights=None)
        self.resnet50.fc = nn.Identity()
        self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(2048, 1024), nn.Linear(1024, 512), nn.Linear(512, 128), nn.Linear(128, 5)
        self.register_buffer("weights", torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]), persistent=False)

    def forward(self, x):
        x = F.relu(self.fc1(self.resnet50(x)))
        x = F.relu(self.fc3(F.relu(self.fc2(x))))
        return (torch.softmax(self.fc4(x), dim=1) * self.weights).sum(dim=1)


def _vit_state_dict(path: Path) -> dict:
    """Read a pickled Model_01.Net with the weights-only unpickler and keep its state dict."""
    mod = sys.modules.get("Model_01") or types.ModuleType("Model_01")
    if not hasattr(mod, "Net"):
        mod.Net = type("Net", (nn.Module,), {"__module__": "Model_01"})
        sys.modules["Model_01"] = mod
    allow = [mod.Net, set, functools.partial, tv_vit.VisionTransformer, tv_vit.Encoder, tv_vit.EncoderBlock,
             tv_vit.MLPBlock, nn.GELU, nn.MultiheadAttention, nn.ReLU, nn.Sequential, nn.Conv2d, nn.Dropout,
             nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear, nn.LayerNorm]
    with torch.serialization.safe_globals(allow):
        whole = torch.load(path, map_location="cpu", weights_only=True)
    return whole.state_dict()


def load_models(device) -> dict[str, nn.Module]:
    models = {}
    for prefix, spec, cls in (("vit", VIT, VitNet), ("r50", R50, R50Net)):
        local = Path(snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=["*.pth"]))
        for q, fname in spec["files"].items():
            sd = _vit_state_dict(local / fname) if prefix == "vit" else torch.load(local / fname, map_location="cpu", weights_only=True)
            m = cls()
            m.load_state_dict(sd, strict=True)
            models[f"{prefix}_{q}"] = m.eval().to(device)
    return models


class Views(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return TF_VIT(img), TF_R50(img), i


@torch.inference_mode()
def score(models, paths, device, batch=128, workers=8) -> pd.DataFrame:
    res = {name: np.empty(len(paths), dtype=np.float32) for name in models}
    for x384, x224, idx in DataLoader(Views(paths), batch_size=batch, num_workers=workers, pin_memory=True):
        x384, x224 = x384.to(device, non_blocking=True), x224.to(device, non_blocking=True)
        for name, m in models.items():
            res[name][idx.numpy()] = m(x384 if name.startswith("vit") else x224).float().cpu().numpy()
    df = pd.DataFrame({"image_id": [Path(p).stem for p in paths], **res})
    df["vit_model"] = f"{VIT['repo']}@{VIT['revision'][:8]}"
    df["r50_model"] = f"{R50['repo']}@{R50['revision'][:8]}"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_list")
    ap.add_argument("out_prefix")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="first N views only (timing test)")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    lines = [l.strip() for l in (PROJ / args.image_list).read_text().splitlines() if l.strip()]
    paths = [str(PROJ / l) for i, l in enumerate(lines) if i % n == k]
    if args.limit:
        paths = paths[:args.limit]
    device = torch.device("cuda:0")
    models = load_models(device)
    out = PROJ / args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for part, start in enumerate(range(0, len(paths), args.chunk)):
        path = Path(f"{out}_shard{k}_part{part:03d}.parquet")
        if path.exists():
            continue
        score(models, paths[start:start + args.chunk], device, batch=args.batch).to_parquet(path, index=False)
        done = min(start + args.chunk, len(paths))
        print(f"shard {k}: {done}/{len(paths)} in {time.time() - t0:.0f}s", flush=True)
    print(f"PERC_SHARD_{k}_DONE", flush=True)


if __name__ == "__main__":
    main()
