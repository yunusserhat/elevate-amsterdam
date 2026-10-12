# -*- coding: utf-8 -*-
"""Step 2.5 check: can the Space embed search queries on a CPU?

The index vectors were made on a GPU in bfloat16. A Space without a GPU embeds the query on
the CPU in float32. For a set of queries this compares the two query vectors (cosine) and
the top-20 results (overlap), and times a CPU query.

Usage: python 27b_index_query_check.py
Writes outputs/wp3/index/query_check.csv.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geoai_vlm import ImageEmbedder
from geoai_vlm.service import SceneIndex

PROJ = Path(__file__).resolve().parents[1]
INDEX = PROJ / "outputs/wp3/index/embeddinggemma2_256"
QUERIES = ["red separated cycle track next to the road", "bench to sit under trees", "canal with boats and a bridge",
           "construction site with fences blocking the sidewalk", "busy shopping street with storefronts",
           "zebra crossing at an intersection", "narrow sidewalk with parked bicycles", "wide green park path",
           "speed bump on a residential street", "bus stop with shelter", "street without trees",
           "litter and rubbish bags on the pavement"]


def embedder(device: str, dtype: str) -> ImageEmbedder:
    return ImageEmbedder(backend="sentence-transformers", model_name="google/embeddinggemma-2", truncate_dim=256,
                         config_kwargs={"audio_config": None}, device=device, dtype=dtype)


def main():
    idx = SceneIndex.load(INDEX)
    gpu, cpu = embedder("cuda", "bfloat16"), embedder("cpu", "float32")
    qg = gpu.embed_queries(QUERIES)
    torch.set_num_threads(2)  # a free CPU Space has 2 vCPUs
    cpu.embed_queries(QUERIES[:1])  # warm-up
    t = time.time()
    qc = np.vstack([cpu.embed_queries([q]) for q in QUERIES])
    per_query = (time.time() - t) / len(QUERIES)
    rows = []
    for q, a, b in zip(QUERIES, qg, qc):
        ta, tb = idx.search_vector(a, 20), idx.search_vector(b, 20)
        cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        rows.append({"query": q, "cosine_gpu_cpu": cos, "top20_overlap": len(set(ta.image_id) & set(tb.image_id)) / 20,
                     "top1_same": ta.image_id.iloc[0] == tb.image_id.iloc[0], "top1_similarity": ta.similarity.iloc[0]})
    out = pd.DataFrame(rows)
    out.to_csv(INDEX.parent / "query_check.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"CPU (2 threads, float32) seconds per query: {per_query:.2f}")


if __name__ == "__main__":
    main()
