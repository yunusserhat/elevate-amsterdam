# -*- coding: utf-8 -*-
"""Step 7.2: before/after visualisations with Qwen-Image-2.1-Turbo for the WP6 edit candidates.

An idea brought to the project, not part of the protocol (the protocol's Fig. 2 images are
illustrative, made with ChatGPT). Model: Qwen/Qwen-Image-2.1-Turbo (Qwen Research Licence,
non-commercial research only), 8-step schedule saved with the checkpoint, CFG 1, KV cache.
The ~32 GB of weights are spread over both GPUs (diffusers device_map "balanced").
Every output is an AI-generated visualisation, not a design proposal.

Usage:
  python 71_wp6_edit.py --smoke            # one candidate at two sizes: time and memory
  python 71_wp6_edit.py [--width 1024 --height 768]
Writes outputs/wp6/edits/<image_id>__<gap>.png and outputs/wp6/edits_log.csv (prompt, seed, size, seconds).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen-Image-2.1-Turbo"
REVISION = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
KEEP = ("Keep everything else exactly as it is: the buildings, facades, trees, parked cars and bicycles, people, "
        "road markings, sky, light, shadows, colours and the camera position and lens. Photorealistic, same time of day, "
        "same image framing.")
PROMPTS = {
    "crossing": "Edit this street photograph from Amsterdam. Add a marked pedestrian crossing, white zebra stripes painted "
                "on the road surface, across the street at the junction ahead, with lowered kerbs where it meets the sidewalks.",
    "trees": "Edit this street photograph from Amsterdam. Add young street trees in square tree pits along the sidewalk, "
             "about eight metres apart, with green summer foliage.",
    "seating": "Edit this street photograph from Amsterdam. Add one or two simple public benches with backrests on the "
               "sidewalk, placed along the kerb side so that the walking path stays free.",
    "greenery": "Edit this street photograph from Amsterdam. Add planters with low shrubs and flowers along the edge of the "
                "sidewalk, without blocking the walking path.",
    "cycle": "Edit this street photograph from Amsterdam. Add a Dutch-style separated cycle track of red asphalt between "
             "the carriageway and the sidewalk, separated from the traffic lanes by a low kerb, with white bicycle symbols.",
}


def load():
    return QwenImage21Pipeline.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="balanced")


def edit(pipe, row, width, height, seed):
    src = Image.open(PROJ / row.view_path).convert("RGBA")
    prompt = PROMPTS[row.gap_type] + " " + KEEP
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t = time.time()
    img = pipe(prompt=prompt, image=src, width=width, height=height, use_kv_cache=True,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    secs = time.time() - t
    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]
    return img, prompt, secs, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    args = ap.parse_args()
    cand = pd.read_csv(PROJ / "outputs/wp6/edit_candidates.csv")
    out = PROJ / "outputs/wp6/edits"
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    pipe = load()
    print(f"loaded in {time.time() - t:.0f}s; device map: {getattr(pipe, 'hf_device_map', None)}", flush=True)
    if args.smoke:
        row = cand.iloc[0]
        for w, h in [(1024, 768), (2400, 1792)]:
            img, _, secs, peak = edit(pipe, row, w, h, 20261011)
            img.save(out / f"smoke_{w}x{h}__{row.image_id}__{row.gap_type}.png")
            print(f"SMOKE {w}x{h}: {secs:.1f}s, peak GiB per GPU {peak}", flush=True)
        return
    log_path = PROJ / "outputs/wp6/edits_log.csv"
    done = set(pd.read_csv(log_path).image_id) if log_path.exists() else set()
    for k, row in enumerate(cand.itertuples(index=False)):
        if row.image_id in done:
            continue
        seed = 20261011 + k
        img, prompt, secs, peak = edit(pipe, row, args.width, args.height, seed)
        name = f"{row.image_id}__{row.gap_type}.png"
        img.save(out / name)
        rec = pd.DataFrame([{"image_id": row.image_id, "gap_type": row.gap_type, "edit_file": f"outputs/wp6/edits/{name}",
                             "prompt": prompt, "seed": seed, "width": args.width, "height": args.height, "seconds": round(secs, 1),
                             "peak_gib": str(peak), "model": f"{MODEL}@{REVISION[:8]}",
                             "label": "AI-generated visualisation, not a design proposal"}])
        rec.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        print(f"{k + 1}/{len(cand)} {row.gap_type} {secs:.1f}s", flush=True)
    print("EDITS_DONE", flush=True)


if __name__ == "__main__":
    main()
