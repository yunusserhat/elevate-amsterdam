# -*- coding: utf-8 -*-
"""Step 0.2 smoke tests. Usage: python 00_smoke_tests.py vlm|embed"""
import sys
import time


def vlm():
    from geoai_vlm import check_model
    from geoai_vlm.chat import make_synthetic_street_image
    rep = check_model("Qwen/Qwen3.5-4B", backend="vllm", prompt_template="active_mobility_audit_v1",
                      image=make_synthetic_street_image(), max_new_tokens=2048, structured_output=True,
                      gpu_memory_utilization=0.5, max_model_len=8192, tensor_parallel_size=1, max_num_seqs=64)
    print(rep.summary())


def embed():
    import numpy as np
    from geoai_vlm import ImageEmbedder
    from geoai_vlm.chat import make_synthetic_street_image
    t = time.time()
    e = ImageEmbedder(backend="sentence-transformers", truncate_dim=256, config_kwargs={"audio_config": None})
    img = make_synthetic_street_image()
    v = e.embed_multimodal([{"image": img}, {"image": img, "text": "a street with a tree"}], instruction="document")
    q = e.embed_queries(["street with a tree"])
    print("EmbeddingGemma 2:", v.shape, q.shape, e.backend.dtype, "finite", bool(np.isfinite(v).all()), f"{time.time() - t:.1f}s")


if __name__ == "__main__":
    {"vlm": vlm, "embed": embed}[sys.argv[1]]()
