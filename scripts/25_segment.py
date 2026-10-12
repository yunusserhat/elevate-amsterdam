# -*- coding: utf-8 -*-
"""Step 2.3: Cityscapes class pixel fractions for the view images (SegFormer-B5).

Usage: python 25_segment.py LIST OUT_PREFIX [--shard K/N] [--chunk 2000]
Writes OUT_PREFIX_shardK_partNNN.parquet every CHUNK images, so a rerun skips finished parts.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from geoai_vlm import SemanticSegmenter

PROJ = Path(__file__).resolve().parents[1]
MODEL = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_list")
    ap.add_argument("out_prefix")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    lines = [l.strip() for l in (PROJ / args.image_list).read_text().splitlines() if l.strip()]
    paths = [str(PROJ / l) for i, l in enumerate(lines) if i % n == k]
    seg = SemanticSegmenter(model_name=MODEL)
    out = PROJ / args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for part, start in enumerate(range(0, len(paths), args.chunk)):
        path = Path(f"{out}_shard{k}_part{part:03d}.parquet")
        if path.exists():
            continue
        seg.measure(paths[start:start + args.chunk]).to_parquet(path, index=False)
        done = min(start + args.chunk, len(paths))
        print(f"shard {k}: {done}/{len(paths)} in {time.time() - t0:.0f}s", flush=True)
    print(f"SEG_SHARD_{k}_DONE", flush=True)


if __name__ == "__main__":
    main()
