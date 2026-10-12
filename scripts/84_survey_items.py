# -*- coding: utf-8 -*-
"""Step 8.8: item pool for the survey layer (pilot).

* space/data/survey/wp3_views.json: 120 street views for pairwise perception questions (WP3):
  front and back views the audit marked usable, stratified by quintile of the stand-in model's
  safety score (24 per quintile), at most two per buurt and one per panorama, seeded. The page gets
  ids, the City's image link and the buurt name only; model scores stay here, in
  outputs/survey/wp3_pool_with_model_scores.parquet, for the later comparison.

Usage: python 84_survey_items.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
SEED, PER_QUINTILE, MAX_PER_BUURT = 2026, 24, 2

rec = pd.read_parquet(PROJ / "space/data/index/records.parquet",
                      columns=["image_id", "pano_id", "view", "buurtnaam", "usable", "city_image"])
perc = pd.concat(pd.read_parquet(f, columns=["image_id", "vit_safety", "vit_beautiful", "vit_lively"])
                 for f in sorted(glob.glob(str(PROJ / "outputs/wp3/perception/*.parquet"))))
d = rec.merge(perc, on="image_id")
d = d[d.usable.astype(bool) & d.view.isin(["front", "back"]) & d.buurtnaam.notna()].copy()
d["quintile"] = pd.qcut(d.vit_safety, 5, labels=False)

rng = np.random.default_rng(SEED)
picked, per_buurt, panos = [], {}, set()
for q in range(5):
    pool = d[d.quintile == q].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    n = 0
    for r in pool.itertuples(index=False):
        if r.pano_id in panos or per_buurt.get(r.buurtnaam, 0) >= MAX_PER_BUURT:
            continue
        picked.append(r); panos.add(r.pano_id); per_buurt[r.buurtnaam] = per_buurt.get(r.buurtnaam, 0) + 1
        n += 1
        if n == PER_QUINTILE:
            break
sel = pd.DataFrame(picked)
(PROJ / "outputs/survey").mkdir(parents=True, exist_ok=True)
sel.to_parquet(PROJ / "outputs/survey/wp3_pool_with_model_scores.parquet", index=False)
out = PROJ / "space/data/survey"
out.mkdir(parents=True, exist_ok=True)
(out / "wp3_views.json").write_text(json.dumps(
    [{"id": r.image_id, "img": r.city_image, "buurt": r.buurtnaam} for r in sel.itertuples(index=False)],
    ensure_ascii=False, separators=(",", ":")))
print("views", len(sel), "buurten", sel.buurtnaam.nunique(), "front/back", sel.view.value_counts().to_dict())
print("safety quintile counts", sel.quintile.value_counts().sort_index().to_dict())
