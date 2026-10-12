# -*- coding: utf-8 -*-
"""Step 7.2 (selection): keep edit candidates whose gap the reference model also sees.

The pool (70_wp6_gaps.py) was chosen from the 4B audit. The same views are audited with
Qwen3.8-27B-FP8 and template v2 (32_reference_eval-style run, outputs/wp6/reference_pool_*.parquet).
A gap counts as confirmed when the reference reports, on the same view:
  crossing  intersection present and pedestrian crossing absent
  trees     sidewalk present and street trees absent
  seating   sidewalk present and seating absent
  greenery  sidewalk present (the gap itself, vegetation < 5% of the view, is from segmentation)
  cycle     cycle infrastructure absent
Confirmed candidates are taken in pool order (priority, then seeded random) up to the quota.
Usage: python 72_wp6_confirm.py   -> outputs/wp6/edit_candidates.csv, outputs/wp6/confirm_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geoai_vlm import audit_table

PROJ = Path(__file__).resolve().parents[1]
W6 = PROJ / "outputs/wp6"
QUOTA = {"crossing": 14, "trees": 13, "seating": 9, "greenery": 8, "cycle": 6}
RULES = {"crossing": {"intersection": "present", "pedestrian_crossing": "absent"},
         "trees": {"sidewalk": "present", "street_trees": "absent"},
         "seating": {"sidewalk": "present", "seating": "absent"},
         "greenery": {"sidewalk": "present"},
         "cycle": {"cycle_infrastructure": "absent"}}


def main():
    pool = pd.read_csv(W6 / "edit_pool.csv")
    ref = pd.concat([pd.read_parquet(p) for p in sorted(W6.glob("reference_pool_*.parquet"))])
    ref = ref[~ref.parse_error.astype(bool)].drop_duplicates("image_id", keep="last")
    st = audit_table(ref).pivot_table(index="image_id", columns="item", values="state", aggfunc="first")
    rows = []
    for r in pool.itertuples(index=False):
        if r.image_id not in st.index:
            rows.append("no reference answer")
            continue
        ok = all(st.loc[r.image_id].get(item) == state for item, state in RULES[r.gap_type].items())
        rows.append("confirmed" if ok else "not confirmed")
    pool["reference_check"] = rows
    pool.to_csv(W6 / "edit_pool.csv", index=False)
    chosen = pd.concat([pool[(pool.gap_type == g) & (pool.reference_check == "confirmed")].head(n) for g, n in QUOTA.items()])
    chosen.to_csv(W6 / "edit_candidates.csv", index=False)
    summary = {"pool": pool.gap_type.value_counts().to_dict(),
               "confirmed_share": pool.groupby("gap_type").reference_check.apply(lambda s: round(float((s == "confirmed").mean()), 3)).to_dict(),
               "chosen": chosen.gap_type.value_counts().to_dict(), "reference_failures": int((pool.reference_check == "no reference answer").sum())}
    (W6 / "confirm_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
