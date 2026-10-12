# -*- coding: utf-8 -*-
"""Step 2.7: seeded samples for evaluation, drawn from the 24,775 audited panoramas.

1. data/derived/eval_500.txt: 500 views for the reference model, one view per panorama, 125
   per direction. Panoramas are drawn across road groups in proportion to the sample. Pilot
   panoramas are excluded, because the v2 definitions were written after looking at them.
2. outputs/wp3/manual_label_sample.csv: views for expert labelling. For each of the 19 items,
   up to 10 views the 4B model reports present and 10 it reports absent or not visible
   (unique views, pilot and eval_500 excluded), so rare items get positives. One blank
   label column per item. Paths point to the local view files; the API url identifies the
   panorama at the City of Amsterdam.

Usage: python 30_eval_sample.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
SEED = 20261011
VIEWS = ["front", "right", "back", "left"]


def main():
    rng = np.random.default_rng(SEED)
    sample = pd.read_parquet(PROJ / "data/derived/sample_25k.parquet")
    segs = pd.read_parquet(PROJ / "data/derived/segments.parquet", columns=["segment_id", "road_group"])
    sample = sample.merge(segs, on="segment_id", how="left", validate="many_to_one")
    pilot = set((PROJ / "data/derived/pilot_500_panos.txt").read_text().split())
    pool = sample[~sample.pano_id.isin(pilot)]

    # 1. reference sample: 500 panoramas, proportional by road group, then 125 per direction
    share = pool.road_group.value_counts(normalize=True)
    quota = (share * 500).round().astype(int)
    quota.iloc[0] += 500 - quota.sum()
    picked = pd.concat([g.sample(int(quota[rg]), random_state=SEED) for rg, g in pool.groupby("road_group") if quota.get(rg, 0) > 0])
    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)
    picked["view"] = np.repeat(VIEWS, 125)[rng.permutation(500)]
    lines = [f"data/views/{p}_{v}.jpg" for p, v in zip(picked.pano_id, picked.view)]
    (PROJ / "data/derived/eval_500.txt").write_text("\n".join(lines) + "\n")
    picked[["pano_id", "view", "road_group", "area_id"]].to_csv(PROJ / "data/derived/eval_500_meta.csv", index=False)
    print("eval_500 by road group:", picked.road_group.value_counts().to_dict())
    print("eval_500 by view:", picked.view.value_counts().to_dict())

    # 2. manual labelling sample
    audit = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/audit_full_qwen35_4b_v2_shard*.parquet")))])
    states = sorted(c for c in audit.columns if c.startswith("audit_") and c.endswith("_state"))
    audit["pano_id"], audit["view"] = audit.image_id.str.rsplit("_", n=1).str[0], audit.image_id.str.rsplit("_", n=1).str[1]
    taken = set(f"{p}_{v}" for p, v in zip(picked.pano_id, picked.view))
    cand = audit[~audit.pano_id.isin(pilot) & ~audit.image_id.isin(taken) & audit.usable]
    chosen: list[str] = []
    reason: dict[str, str] = {}
    for c in states:
        item = c[6:-6]
        for label, mask in (("present", cand[c].eq("present")), ("not present", ~cand[c].eq("present"))):
            pool_ids = cand.loc[mask & ~cand.image_id.isin(chosen), "image_id"]
            ids = pool_ids.sample(min(10, len(pool_ids)), random_state=SEED).tolist()
            chosen += ids
            reason.update({i: f"{item}: {label}" for i in ids})
    lab = cand[cand.image_id.isin(chosen)].copy()
    lab["selected_for"] = lab.image_id.map(reason)
    lab = lab.merge(sample[["pano_id", "lat", "lon", "yaw_front", "yaw_right", "yaw_back", "yaw_left", "area_id", "road_group"]],
                    on="pano_id", validate="many_to_one")
    lab["heading_deg"] = [r[f"yaw_{r.view}"] for _, r in lab.iterrows()]
    lab["view_path"] = "data/views/" + lab.image_id + ".jpg"
    lab["panorama_api_url"] = "https://api.data.amsterdam.nl/panorama/panoramas/" + lab.pano_id + "/"
    out = lab[["image_id", "view", "heading_deg", "lat", "lon", "area_id", "road_group", "selected_for", "view_path", "panorama_api_url"]].copy()
    for c in states:
        out[f"model_{c[6:-6]}"] = lab[c].values
    for c in states:
        out[f"label_{c[6:-6]}"] = ""
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.insert(0, "row", range(1, len(out) + 1))
    out.to_csv(PROJ / "outputs/wp3/manual_label_sample.csv", index=False)
    print(f"manual_label_sample.csv: {len(out)} views; by view {out.view.value_counts().to_dict()}")
    print(out.selected_for.str.split(":").str[1].value_counts().to_dict())


if __name__ == "__main__":
    main()
