# -*- coding: utf-8 -*-
"""Step 2.4b QC: the ELEVATE team's perception model (svi_percept) set against independent measures.

The model scores walkability, bikeability, pleasantness, greenness and safety on the Percept survey's
1-5 scale (scripts/26b_svi_percept.py). Here it is compared, on the usable front and back views (the
views the aggregation uses) and on their per-panorama mean, with measures made without it:
* Spearman with the closest Place Pulse questions (ViT; the earlier stand-in, now a cross-check): safety - safety,
  pleasantness - beautiful, walkability - lively;
* Spearman of greenness with the segmentation's vegetation share (green view);
* AUC of each quality for the audit item closest to it, present vs absent (views where the item is
  not visible are left out): walkability - sidewalk, bikeability - cycle infrastructure, greenness -
  street trees, safety - street light pole;
* front vs back agreement and the mean difference between side and along-street views.
Nothing here is ground truth: agreement shows that two measures move together, not that either is right.

Usage: python 28b_svp_qc.py
Writes outputs/wp3/perception_qc/svp_summary.csv, svp_agreement.csv and svp_qc.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "outputs/wp3/perception_qc"
QUALITIES = ["walkability", "bikeability", "pleasantness", "greenness", "safety"]
PLACE_PULSE = {"safety": "safety", "pleasantness": "beautiful", "walkability": "lively"}
AUDIT = {"walkability": "sidewalk", "bikeability": "cycle_infrastructure", "greenness": "street_trees",
         "safety": "street_light_pole"}


def read(pattern: str, columns=None) -> pd.DataFrame:
    files = sorted(glob.glob(str(PROJ / "outputs/wp3" / pattern)))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def auc(score: pd.Series, positive: pd.Series) -> float:
    """Probability that a view with the item scores higher than one without it (Mann-Whitney)."""
    r = score.rank()
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    svp = read("perception/svipercept_shard*_part*.parquet", ["image_id", "svp_model", "svp_clip"] + [f"svp_{q}" for q in QUALITIES])
    pp = read("perception/placepulse_shard*_part*.parquet", ["image_id"] + [f"vit_{q}" for q in set(PLACE_PULSE.values())])
    seg = read("segmentation/segformer_b5_shard*_part*.parquet", ["image_id", "vegetation_pixel_fraction"])
    audit = read("audit_full_qwen35_4b_v2_shard*.parquet", ["image_id", "usable"] + [f"audit_{i}_state" for i in AUDIT.values()])
    v = svp.merge(pp, on="image_id", validate="one_to_one").merge(seg, on="image_id", validate="one_to_one") \
           .merge(audit, on="image_id", validate="one_to_one")
    if len(v) != 99100:
        raise ValueError(f"{len(v)} views after joins")
    v["pano_id"], v["view"] = v.image_id.str.rsplit("_", n=1).str[0], v.image_id.str.rsplit("_", n=1).str[1]
    usable = v[v.usable]
    along = usable[usable.view.isin(["front", "back"])]
    pano = along.groupby("pano_id").mean(numeric_only=True)

    summary = []
    for q in QUALITIES:
        c = f"svp_{q}"
        for level, d in (("view", along[c]), ("panorama", pano[c])):
            summary.append({"quality": q, "level": level, "n": int(d.notna().sum()), "mean": d.mean(), "sd": d.std(),
                            "p05": d.quantile(.05), "p50": d.median(), "p95": d.quantile(.95)})
    pd.DataFrame(summary).round(4).to_csv(OUT / "svp_summary.csv", index=False)

    rows = []
    for q, pq in PLACE_PULSE.items():
        rows.append({"quality": q, "compared_with": f"Place Pulse {pq} (cross-check)", "measure": "spearman",
                     "per_view": along[f"svp_{q}"].corr(along[f"vit_{pq}"], method="spearman"),
                     "per_panorama": pano[f"svp_{q}"].corr(pano[f"vit_{pq}"], method="spearman"), "n_views": int(len(along))})
    rows.append({"quality": "greenness", "compared_with": "green view (segmentation, vegetation share)", "measure": "spearman",
                 "per_view": along.svp_greenness.corr(along.vegetation_pixel_fraction, method="spearman"),
                 "per_panorama": pano.svp_greenness.corr(pano.vegetation_pixel_fraction, method="spearman"), "n_views": int(len(along))})
    for q, item in AUDIT.items():
        s = along[f"audit_{item}_state"]
        known = s.isin(["present", "absent"])
        rows.append({"quality": q, "compared_with": f"audit: {item.replace('_', ' ')} present vs absent", "measure": "auc",
                     "per_view": auc(along.loc[known, f"svp_{q}"], s[known].eq("present")), "per_panorama": np.nan,
                     "n_views": int(known.sum())})
    agree = pd.DataFrame(rows)
    agree.round(3).to_csv(OUT / "svp_agreement.csv", index=False)

    wide = along.pivot(index="pano_id", columns="view", values=[f"svp_{q}" for q in QUALITIES])
    side = usable[usable.view.isin(["left", "right"])]
    qc = {
        "model": sorted(v.svp_model.unique()), "clip": sorted(v.svp_clip.unique()),
        "views": int(len(v)), "usable_along_views": int(len(along)), "panoramas": int(len(pano)),
        "front_vs_back_spearman": {q: round(float(wide[(f"svp_{q}", "front")].corr(wide[(f"svp_{q}", "back")], method="spearman")), 3) for q in QUALITIES},
        "side_minus_along_mean": {q: round(float(side[f"svp_{q}"].mean() - along[f"svp_{q}"].mean()), 3) for q in QUALITIES},
        "between_qualities_spearman_view": along[[f"svp_{q}" for q in QUALITIES]].corr(method="spearman").round(3).to_dict(),
    }
    (OUT / "svp_qc.json").write_text(json.dumps(qc, indent=1))
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(agree.round(3).to_string(index=False))
    print(json.dumps({k: qc[k] for k in ("front_vs_back_spearman", "side_minus_along_mean")}, indent=1))


if __name__ == "__main__":
    main()
