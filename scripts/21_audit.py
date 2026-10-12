# -*- coding: utf-8 -*-
"""Steps 2.1, 2.2, 2.7: active-mobility audit over view images with vLLM (JSON-schema decoding).

Usage:
  python 21_audit.py LIST OUT [--shard K/N] [--model Qwen/Qwen3.5-4B] [--util 0.85]
LIST: text file with one view path per line (relative to the project). With --shard K/N
the process takes every N-th line starting at K (run one process per GPU and set
CUDA_VISIBLE_DEVICES). Resume is GeoAI-VLM's own: finished (image_id, processing_id)
records in OUT are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from geoai_vlm import ImageDescriber

PROJ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_list")
    ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--template", default="active_mobility_audit_v2")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs per model)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="only for re-running records whose JSON-schema output ran into a whitespace loop")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    lines = [l.strip() for l in (PROJ / args.image_list).read_text().splitlines() if l.strip()]
    paths = [str(PROJ / l) for i, l in enumerate(lines) if i % n == k]
    describer = ImageDescriber(
        model_name=args.model, backend="vllm", prompt_template=args.template, structured_output=True,
        max_new_tokens=args.max_new_tokens, temperature=0.0, repetition_penalty=args.repetition_penalty, gpu_memory_utilization=args.util, max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp, max_num_seqs=args.max_num_seqs,
    )
    out = PROJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    df = describer.describe(image_paths=paths, output_path=str(out), batch_size=args.batch_size)
    states = [c for c in df.columns if c.startswith("audit_") and c.endswith("_state")]
    summary = {"model": args.model, "template": args.template, "shard": args.shard, "n": int(len(df)), "seconds": round(time.time() - t, 1),
               "s_per_image": round((time.time() - t) / max(len(df), 1), 3),
               "parse_failures": int(df["parse_error"].astype(bool).sum()),
               "invalid_or_failed_states": int(df[states].isin(["invalid", "failed", "not_assessed"]).sum().sum()),
               "decoding_mode": df["decoding_mode"].value_counts().to_dict(),
               "processing_id": df["processing_id"].unique().tolist(), "model_revision": df["model_revision"].unique().tolist()}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
