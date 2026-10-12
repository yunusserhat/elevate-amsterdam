# -*- coding: utf-8 -*-
"""Step 2.5: EmbeddingGemma 2 search index over the audited views (GeoAI-VLM SceneIndex).

Each view is indexed as one joint image+text vector (modality "image+text", document role,
Matryoshka 256 d). The text is an observation summary written from the view's own audit
record (template v2): items reported present with their attribute and the model's evidence,
then items reported absent and not visible. It is model output, not a measurement.

Usage:
  python 27_index.py build [--shard K/N] [--chunk 5000] [--batch 32] [--limit N] [--parts DIR]
  python 27_index.py merge
Build writes outputs/wp3/index/parts/shardK_partNNN.{parquet,npy} (a rerun skips finished
parts); merge joins them into one SceneIndex at outputs/wp3/index/embeddinggemma2_256/.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from geoai_vlm import ImageEmbedder
from geoai_vlm.service import SceneIndex

PROJ = Path(__file__).resolve().parents[1]
MODEL = "google/embeddinggemma-2"
DIM = 256
PARTS = PROJ / "outputs/wp3/index/parts"
FINAL = PROJ / "outputs/wp3/index/embeddinggemma2_256"
ATTRS = ["width_class", "obstruction_type", "crossing_type", "calming_type", "cycle_type", "barrier_type"]
TEXT_VERSION = "obs_text_v1"


def observation_text(parsed: str) -> str:
    """'Present: item (attribute): evidence; ... Absent: ... Not visible: ...' from one audit record."""
    d = json.loads(parsed) if parsed else {}
    present, absent, unseen = [], [], []
    for item, rec in (d.get("items") or {}).items():
        name = item.replace("_", " ")
        state = rec.get("state")
        if state == "present":
            attr = next((str(rec[a]).replace("_", " ") for a in ATTRS if rec.get(a) not in (None, "", "none", "unknown")), None)
            ev = (rec.get("evidence") or "").strip().rstrip(".")
            present.append(f"{name}{f' ({attr})' if attr else ''}{f': {ev}' if ev else ''}")
        elif state == "absent":
            absent.append(name)
        elif state == "not_visible":
            unseen.append(name)
    parts = []
    if present:
        parts.append("Present: " + "; ".join(present) + ".")
    if absent:
        parts.append("Absent: " + ", ".join(absent) + ".")
    if unseen:
        parts.append("Not visible: " + ", ".join(unseen) + ".")
    issues = [i for i in ((d.get("image_quality") or {}).get("issues") or []) if i and i != "none"]
    if issues:
        parts.append("Image issues: " + ", ".join(i.replace("_", " ") for i in issues) + ".")
    return " ".join(parts)


def records() -> pd.DataFrame:
    """One row per audited view, in views_all.txt order, with location and observation text."""
    audit = pd.concat([pd.read_parquet(f, columns=["image_id", "parsed_json", "usable"])
                       for f in sorted(glob.glob(str(PROJ / "outputs/wp3/audit_full_qwen35_4b_v2_shard*.parquet")))])
    lines = [l.strip() for l in (PROJ / "data/derived/views_all.txt").read_text().splitlines() if l.strip()]
    df = pd.DataFrame({"path": lines})
    df["image_id"] = df.path.map(lambda p: Path(p).stem)
    df["pano_id"], df["view"] = df.image_id.str.rsplit("_", n=1).str[0], df.image_id.str.rsplit("_", n=1).str[1]
    sample = pd.read_parquet(PROJ / "data/derived/sample_25k.parquet")
    keep = ["pano_id", "timestamp", "lat", "lon", "segment_id", "area_id", "buurtnaam",
            "yaw_front", "yaw_right", "yaw_back", "yaw_left"]
    df = df.merge(sample[keep], on="pano_id", how="left", validate="many_to_one")
    views = ["front", "right", "back", "left"]
    df["heading_deg"] = np.select([df.view == v for v in views], [df[f"yaw_{v}"] for v in views], np.nan)
    df = df.drop(columns=[f"yaw_{v}" for v in views])
    df = df.merge(audit, on="image_id", how="left", validate="one_to_one")
    if df.parsed_json.isna().any() or df.lat.isna().any():
        raise ValueError("views without an audit record or a location")
    df["observation_text"] = df.parsed_json.map(observation_text)
    return df.drop(columns=["parsed_json"])


def build(args):
    k, n = (int(x) for x in args.shard.split("/"))
    df = records()
    df = df[np.arange(len(df)) % n == k].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
    emb = ImageEmbedder(backend="sentence-transformers", model_name=MODEL, truncate_dim=DIM,
                        config_kwargs={"audio_config": None})
    parts = Path(args.parts)
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for part, start in enumerate(range(0, len(df), args.chunk)):
        stem = parts / f"shard{k}_part{part:03d}"
        if stem.with_suffix(".npy").exists():
            continue
        chunk = df.iloc[start:start + args.chunk].assign(image_file=lambda d: [str(PROJ / p) for p in d.path])
        idx = SceneIndex.build(chunk, emb, text_column="observation_text", image_column="image_file",
                               modality="image+text", batch_size=args.batch)
        idx.records.drop(columns=["image_file"]).to_parquet(stem.with_suffix(".parquet"), index=False)
        np.save(stem.with_suffix(".npy"), idx.vectors.astype(np.float32))
        done = min(start + args.chunk, len(df))
        print(f"shard {k}: {done}/{len(df)} in {time.time() - t0:.0f}s", flush=True)
    print(f"INDEX_SHARD_{k}_DONE", flush=True)


def merge(args):
    stems = sorted(Path(p).with_suffix("") for p in glob.glob(str(Path(args.parts) / "shard*_part*.npy")))
    recs = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    vecs = np.vstack([np.load(s.with_suffix(".npy")) for s in stems])
    if recs.image_id.duplicated().any() or not np.isfinite(vecs).all():
        raise ValueError("duplicate views or non-finite vectors")
    revision = Path(snapshot_download(MODEL, local_files_only=True)).name
    meta = {"embedding_model": MODEL, "model_revision": revision, "truncate_dim": DIM, "embedded": "image+text",
            "document_prompt": "document", "text": TEXT_VERSION,
            "text_source": "GeoAI-VLM active_mobility_audit_v2, Qwen/Qwen3.5-4B (processing_id 9fa2547ea019)",
            "images": "Gemeente Amsterdam panoramas 2025, CC BY 4.0; perspective views 768x576",
            "note": "Observation text is model output, not a measurement.", "built": str(date.today()),
            "n": int(len(recs))}
    SceneIndex(recs, vecs, None, "image_id", "observation_text", meta).save(FINAL)
    print(f"merged {len(recs)} views, vectors {vecs.shape} -> {FINAL}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "merge"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N views of the shard only (timing test)")
    ap.add_argument("--parts", default=str(PARTS), help="directory for the part files")
    args = ap.parse_args()
    {"build": build, "merge": merge}[args.step](args)


if __name__ == "__main__":
    main()
