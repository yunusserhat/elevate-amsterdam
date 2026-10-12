# -*- coding: utf-8 -*-
"""Step 7.1: evidence layer of street-level gaps for WP6, prioritised by equity.

A gap is defined on a single usable front or back view (the view an edit would start from),
from audit items of tier A unless stated, and the segmentation:
  crossing   intersection present and pedestrian crossing absent (protocol Fig. 2 example)
  trees      sidewalk present and street trees absent
  seating    sidewalk present, active frontage present (tier B) and seating absent
  greenery   sidewalk present and vegetation below 5% of the view (segmentation)
  cycle      main road, cycle infrastructure absent (tier B: the model over-reports cycle
             infrastructure, so absences are the safer side)
Priority: 2 = buurt in the WP5 equity-priority set (low green-view cluster and many low-income
households), 1 = other buurt in the two most low-income quintiles, 0 = otherwise or unknown.
Segment layer: share of a segment's panoramas with each gap. Candidate pool for the ~50 edits
(step 7.2): 2.5 x the quota per gap type, highest priority first, then seeded random, at most
three per buurt per gap type. 72_wp6_confirm.py keeps the candidates whose gap the larger
reference model also sees, up to the quota.
The gaps are candidate interventions to discuss in workshops, not recommendations.

Usage: python 70_wp6_gaps.py
Writes outputs/wp6/{gap_views.parquet, gap_segments.parquet, gap_by_quintile.csv, edit_pool.csv, gaps_summary.json}
and data/derived/wp6_pool_views.txt.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "outputs/wp6"
SEED = 20261011
QUOTA = {"crossing": 14, "trees": 13, "seating": 9, "greenery": 8, "cycle": 6}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "usable"] + [f"audit_{i}_state" for i in
                                     ["intersection", "pedestrian_crossing", "sidewalk", "street_trees", "seating", "active_frontage", "cycle_infrastructure"]]
    a = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/audit_full_qwen35_4b_v2_shard*.parquet")))])
    seg = pd.concat([pd.read_parquet(f, columns=["image_id", "vegetation_pixel_fraction"]) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/segmentation/*.parquet")))])
    a = a.merge(seg, on="image_id", validate="one_to_one")
    a["pano_id"], a["view"] = a.image_id.str.rsplit("_", n=1).str[0], a.image_id.str.rsplit("_", n=1).str[1]
    a = a[a.usable & a.view.isin(["front", "back"])]
    sample = pd.read_parquet(PROJ / "data/derived/sample_25k.parquet", columns=["pano_id", "lat", "lon", "segment_id", "area_id", "yaw_front", "yaw_back", "timestamp"])
    segs = pd.read_parquet(PROJ / "data/derived/segments.parquet", columns=["segment_id", "road_group", "name"])
    a = a.merge(sample, on="pano_id").merge(segs, on="segment_id", how="left")
    eq = gpd.read_parquet(PROJ / "outputs/wp5/buurt_equity.parquet")[["buurtcode", "buurtnaam", "low_income_quintile", "low_income_households", "priority_low_green_low_income"]]
    a = a.merge(eq.rename(columns={"buurtcode": "area_id"}), on="area_id", how="left")
    s = lambda item, state: a[f"audit_{item}_state"].eq(state)  # noqa: E731
    gaps = {
        "crossing": s("intersection", "present") & s("pedestrian_crossing", "absent"),
        "trees": s("sidewalk", "present") & s("street_trees", "absent"),
        "seating": s("sidewalk", "present") & s("active_frontage", "present") & s("seating", "absent"),
        "greenery": s("sidewalk", "present") & (a.vegetation_pixel_fraction < 0.05),
        "cycle": a.road_group.eq("main road") & s("cycle_infrastructure", "absent"),
    }
    for g, m in gaps.items():
        a[f"gap_{g}"] = m.to_numpy()
    a["priority"] = np.where(a.priority_low_green_low_income.eq(True), 2,
                             np.where(a.low_income_quintile.isin(["Q4", "Q5 (most)"]), 1, 0))
    a["heading_deg"] = np.where(a.view == "front", a.yaw_front, a.yaw_back)
    gcols = [f"gap_{g}" for g in gaps]
    views = a[a[gcols].any(axis=1)].drop(columns=["yaw_front", "yaw_back"])
    views.to_parquet(OUT / "gap_views.parquet", index=False)

    # segment layer: share of panoramas (either along-street view) with each gap
    pano = a.groupby(["pano_id", "segment_id"])[gcols].any().reset_index()
    seg_layer = pano.groupby("segment_id")[gcols].mean().join(pano.groupby("segment_id").size().rename("n_panoramas"))
    geo = gpd.read_parquet(PROJ / "data/derived/segments.parquet", columns=["segment_id", "road_group", "geometry"])
    seg_layer = geo.merge(seg_layer.reset_index(), on="segment_id")
    seg_layer.to_parquet(OUT / "gap_segments.parquet", index=False)

    # equity of gaps: share of along-street views with each gap, by low-income quintile
    q = a.groupby("low_income_quintile", observed=True)[gcols].mean()
    q.insert(0, "n_views", a.groupby("low_income_quintile", observed=True).size())
    q.round(4).to_csv(OUT / "gap_by_quintile.csv")

    # edit candidates
    rng = np.random.default_rng(SEED)
    picked = []
    used = set()
    for g, n in QUOTA.items():
        c = views[views[f"gap_{g}"] & ~views.image_id.isin(used)].copy()
        c["r"] = rng.random(len(c))
        c = c.sort_values(["priority", "r"], ascending=[False, True])
        c = c.groupby("area_id").head(3).head(int(2.5 * n))
        c["gap_type"] = g
        used.update(c.image_id)
        picked.append(c)
    cand = pd.concat(picked)
    cand["view_path"] = "data/views/" + cand.image_id + ".jpg"
    keep = ["gap_type", "image_id", "pano_id", "view", "heading_deg", "lat", "lon", "timestamp", "area_id", "buurtnaam",
            "low_income_quintile", "priority", "road_group", "name", "view_path"]
    cand[keep].to_csv(OUT / "edit_pool.csv", index=False)
    (PROJ / "data/derived/wp6_pool_views.txt").write_text("\n".join(cand.view_path) + "\n")

    summary = {"views_front_back_usable": int(len(a)), "gap_views": {g: int(a[f"gap_{g}"].sum()) for g in gaps},
               "gap_share": {g: round(float(a[f"gap_{g}"].mean()), 4) for g in gaps},
               "segments_with_any_gap": int((seg_layer[gcols].max(axis=1) > 0).sum()),
               "candidates": cand.gap_type.value_counts().to_dict(), "candidates_by_priority": cand.priority.value_counts().to_dict(),
               "candidate_buurten": int(cand.area_id.nunique())}
    (OUT / "gaps_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(q.round(3).to_string())


if __name__ == "__main__":
    main()
