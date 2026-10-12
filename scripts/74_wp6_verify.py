# -*- coding: utf-8 -*-
"""Step 7.3: did the edit add what was asked, and what else did it change?

Each edited view (1024 x 768 from Qwen-Image) is resized to the original 768 x 576.
* Re-audit: the edited views are audited with the 4B model and with the 27B reference
  (template v2). Success = the target item is reported present in the edit (crossing ->
  pedestrian_crossing, trees -> street_trees, seating -> seating, greenery -> other_greenery,
  cycle -> cycle_infrastructure). Other items: how many of the remaining 18 change state between
  the original and the edit, against the 4B model's own run-to-run instability (3.8% of states).
* Segmentation (SegFormer-B5 Cityscapes): share of pixels whose class changed, and the share
  that changed between classes the edit should not touch (anything outside the expected classes
  for its type, e.g. building, car, person, sky for a crossing).
Failures are kept and reported.

Usage:
  python 74_wp6_verify.py prepare   # resized edits + list file for the audits
  python 74_wp6_verify.py measure   # after the audits: outputs/wp6/edit_verification.csv, verify_summary.json, gallery
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parents[1]
W6 = PROJ / "outputs/wp6"
TARGET = {"crossing": "pedestrian_crossing", "trees": "street_trees", "seating": "seating", "greenery": "other_greenery",
          "cycle": "cycle_infrastructure"}
EXPECTED = {"crossing": {"road", "sidewalk", "terrain"},
            "trees": {"vegetation", "sidewalk", "terrain", "pole"},
            "seating": {"sidewalk", "fence", "wall", "pole", "terrain"},
            "greenery": {"vegetation", "terrain", "sidewalk", "fence", "wall"},
            "cycle": {"road", "sidewalk", "terrain"}}
SIZE = (768, 576)


def prepare():
    log = pd.read_csv(W6 / "edits_log.csv")
    out = W6 / "edits_768"
    out.mkdir(exist_ok=True)
    lines = []
    for r in log.itertuples():
        p = out / f"{r.image_id}__{r.gap_type}.jpg"
        Image.open(PROJ / r.edit_file).convert("RGB").resize(SIZE, Image.LANCZOS).save(p, quality=95)
        lines.append(str(p.relative_to(PROJ)))
    (PROJ / "data/derived/wp6_edits_768.txt").write_text("\n".join(lines) + "\n")
    print(f"prepared {len(lines)}")


def states(path_glob: str) -> pd.DataFrame:
    from geoai_vlm import audit_table
    files = sorted(W6.glob(path_glob))
    if not files:
        return pd.DataFrame(columns=["key", "item", "state"])
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[~df.parse_error.astype(bool)].drop_duplicates("image_id", keep="last")
    t = audit_table(df)[["image_id", "item", "state"]]
    t["key"] = t.image_id.str.split("__").str[0]
    return t


def measure():
    from geoai_vlm import SemanticSegmenter
    log = pd.read_csv(W6 / "edits_log.csv")
    cand = pd.read_csv(W6 / "edit_candidates.csv")[["image_id", "buurtnaam", "road_group", "name"]]
    log = log.merge(cand, on="image_id", how="left")
    import glob
    orig4 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/audit_full_qwen35_4b_v2_shard*.parquet")))])
    from geoai_vlm import audit_table
    orig4 = audit_table(orig4[orig4.image_id.isin(log.image_id)])[["image_id", "item", "state"]].rename(columns={"image_id": "key"})
    orig27 = states("reference_pool_v2*.parquet").drop(columns=["image_id"])
    ed4 = states("audit_edits_qwen35_4b_v2*.parquet").drop(columns=["image_id"])
    ed27 = states("reference_edits_qwen38_27b_v2*.parquet").drop(columns=["image_id"])
    seg = SemanticSegmenter(model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    names = None
    rows = []
    for r in log.itertuples():
        tgt = TARGET[r.gap_type]
        row = {"image_id": r.image_id, "gap_type": r.gap_type, "buurt": r.buurtnaam, "street": r.name, "road_group": r.road_group}
        for tag, orig, ed in (("4b", orig4, ed4), ("27b", orig27, ed27)):
            o = orig[orig.key == r.image_id].set_index("item").state
            e = ed[ed.key == r.image_id].set_index("item").state
            row[f"target_before_{tag}"] = o.get(tgt)
            row[f"target_after_{tag}"] = e.get(tgt)
            row[f"success_{tag}"] = (e.get(tgt) == "present") if len(e) else None
            if tag == "4b" and len(e) and len(o):
                others = [i for i in o.index if i != tgt and i in e.index]
                row["other_items_changed_4b"] = int(sum(o[i] != e[i] for i in others))
                row["other_items_compared_4b"] = len(others)
        a = np.asarray(seg.segment(Image.open(PROJ / "data/views" / f"{r.image_id}.jpg").convert("RGB")))
        b = np.asarray(seg.segment(Image.open(W6 / "edits_768" / f"{r.image_id}__{r.gap_type}.jpg").convert("RGB")))
        if names is None:
            id2label = seg.model.config.id2label if hasattr(seg, "model") else {}
            names = {int(k): v.lower() for k, v in id2label.items()}
        changed = a != b
        ok_ids = [k for k, v in names.items() if v in EXPECTED[r.gap_type]]
        unintended = changed & ~np.isin(a, ok_ids) & ~np.isin(b, ok_ids)
        row["pixels_changed_share"] = float(changed.mean())
        row["pixels_changed_unexpected_share"] = float(unintended.mean())
        for cls in ("building", "car", "person", "sky", "vegetation", "road", "sidewalk"):
            cid = [k for k, v in names.items() if v == cls]
            if cid:
                row[f"delta_{cls}"] = float((b == cid[0]).mean() - (a == cid[0]).mean())
        rows.append(row)
    v = pd.DataFrame(rows)
    v.to_csv(W6 / "edit_verification.csv", index=False)
    by = v.groupby("gap_type").agg(n=("image_id", "size"), success_4b=("success_4b", "mean"), success_27b=("success_27b", "mean"),
                                   other_items_changed_4b=("other_items_changed_4b", "mean"),
                                   pixels_changed=("pixels_changed_share", "median"),
                                   pixels_unexpected=("pixels_changed_unexpected_share", "median"))
    summary = {"edits": int(len(v)), "success_4b": round(float(v.success_4b.mean()), 3), "success_27b": round(float(v.success_27b.mean()), 3),
               "other_items_changed_4b_mean": round(float(v.other_items_changed_4b.mean()), 2),
               "other_items_expected_by_instability": round(18 * 0.038, 2),
               "pixels_changed_median": round(float(v.pixels_changed_share.median()), 3),
               "pixels_unexpected_median": round(float(v.pixels_changed_unexpected_share.median()), 3),
               "pixels_unexpected_p90": round(float(v.pixels_changed_unexpected_share.quantile(0.9)), 3),
               "by_gap": by.round(3).reset_index().to_dict("records")}
    (W6 / "verify_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    gallery(v)
    print(json.dumps(summary, indent=1, default=str))


def gallery(v: pd.DataFrame):
    w, h = 384, 288
    rows = len(v)
    img = Image.new("RGB", (2 * w, rows * (h + 18)), "white")
    d = ImageDraw.Draw(img)
    for i, r in enumerate(v.itertuples()):
        y = i * (h + 18)
        img.paste(Image.open(PROJ / "data/views" / f"{r.image_id}.jpg").convert("RGB").resize((w, h)), (0, y + 18))
        img.paste(Image.open(W6 / "edits_768" / f"{r.image_id}__{r.gap_type}.jpg").convert("RGB").resize((w, h)), (w, y + 18))
        d.text((3, y + 3), f"{i + 1} {r.gap_type} | 4B {r.success_4b} 27B {r.success_27b} | unexpected px {r.pixels_changed_unexpected_share:.3f}", fill="black")
    img.save(W6 / "edits_gallery_qc.jpg", quality=80)


if __name__ == "__main__":
    {"prepare": prepare, "measure": measure}[sys.argv[1]]()
