# -*- coding: utf-8 -*-
"""Step 2.1/2.7: reference audit with a larger model through Ollama (OpenAI-compatible backend).

Usage: python 22_reference.py LIST OUT [--template active_mobility_audit_v2] [--model qwen3.8:27b]
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
    ap.add_argument("--template", default="active_mobility_audit_v2")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    paths = [str(PROJ / l.strip()) for l in (PROJ / args.image_list).read_text().splitlines() if l.strip()]
    describer = ImageDescriber(model_name=args.model, backend="openai", base_url="http://127.0.0.1:11434/v1",
                               prompt_template=args.template, structured_output=True, max_new_tokens=4096,
                               temperature=0.0, max_concurrency=args.concurrency, timeout=900,
                               extra_body={"reasoning_effort": "none"})
    out = PROJ / args.out
    t = time.time()
    df = describer.describe(image_paths=paths, output_path=str(out), batch_size=16)
    summary = {"model": args.model, "template": args.template, "n": int(len(df)),
               "s_per_image": round((time.time() - t) / max(len(df), 1), 2),
               "parse_failures": int(df["parse_error"].astype(bool).sum()),
               "decoding_mode": df["decoding_mode"].value_counts().to_dict()}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
