# -*- coding: utf-8 -*-
"""Step 2.6: from views to panoramas, street segments and buurten.

View -> panorama (24,775 points):
* audit items: present if any usable view reports present; absent if none does and at least
  one usable view reports absent; otherwise not_visible. Views the audit marks unusable are
  ignored (they are 17-23% of side views and under 0.5% of front and back views);
* Cityscapes class fractions: mean of the four views (the usual multi-heading view index,
  e.g. green view = vegetation);
* perception (Place Pulse, ViT primary and ResNet second opinion): mean of the front and back
  views, the two along-street views (usable in > 99.5% of cases). Side views score
  systematically lower and are often unusable, so mixing them in would make the score depend
  on how many side views survived (docs/wp3_perception.md).
Panorama -> segment: GeoAI-VLM aggregate_segments, 25 m support centred on the snapped
position; states as present / (present + absent), numbers as means.
Segment -> buurt: segment values weighted by the observed (covered) street length that lies
inside the buurt, so a segment crossing a boundary counts in both by length.

Usage: python 29_aggregate.py
Writes outputs/wp3/indicators_{panorama,segment,buurt}.parquet and indicators_summary.json.
Values are summaries of model observations, not measurements.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoai_vlm.segments import aggregate_segments, write_segment_parquet

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "outputs/wp3"
METRIC = "EPSG:28992"
SEG_CLASSES = ["vegetation", "terrain", "sky", "building", "road", "sidewalk", "car", "bicycle", "person", "pole"]
QUESTIONS = ["safety", "lively", "wealthy", "beautiful", "boring", "depressing"]


def read(pattern: str, columns=None) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f, columns=columns) for f in sorted(glob.glob(str(OUT / pattern)))], ignore_index=True)


def panoramas() -> gpd.GeoDataFrame:
    audit = read("audit_full_qwen35_4b_v2_shard*.parquet")
    states = [c for c in audit.columns if c.startswith("audit_") and c.endswith("_state")]
    audit = audit[["image_id", "usable"] + states]
    seg = read("segmentation/segformer_b5_shard*_part*.parquet", ["image_id"] + [f"{c}_pixel_fraction" for c in SEG_CLASSES])
    perc = read("perception/placepulse_shard*_part*.parquet")
    perc = perc[["image_id"] + [f"{f}_{q}" for f in ("vit", "r50") for q in QUESTIONS]]
    v = audit.merge(seg, on="image_id", validate="one_to_one").merge(perc, on="image_id", validate="one_to_one")
    if len(v) != 99100:
        raise ValueError(f"{len(v)} views after joins")
    v["pano_id"], v["view"] = v.image_id.str.rsplit("_", n=1).str[0], v.image_id.str.rsplit("_", n=1).str[1]

    u = v[v.usable]
    out = pd.DataFrame(index=pd.Index(sorted(v.pano_id.unique()), name="pano_id"))
    out["n_views_usable"] = u.groupby("pano_id").size().reindex(out.index, fill_value=0)
    for c in states:
        has_p = u[c].eq("present").groupby(u.pano_id).any().reindex(out.index, fill_value=False)
        has_a = u[c].eq("absent").groupby(u.pano_id).any().reindex(out.index, fill_value=False)
        out[c] = np.where(has_p, "present", np.where(has_a, "absent", "not_visible"))
    seg_cols = [f"{c}_pixel_fraction" for c in SEG_CLASSES]
    out[seg_cols] = v.groupby("pano_id")[seg_cols].mean()
    along = u[u.view.isin(["front", "back"])]
    pcols = [f"{f}_{q}" for f in ("vit", "r50") for q in QUESTIONS]
    out[pcols] = along.groupby("pano_id")[pcols].mean()
    out["n_views_perception"] = along.groupby("pano_id").size().reindex(out.index, fill_value=0)

    sample = pd.read_parquet(PROJ / "data/derived/sample_25k.parquet")
    keep = ["pano_id", "timestamp", "lon", "lat", "bearing", "segment_id", "snap_distance_m", "snap_offset_m",
            "area_id", "buurtnaam", "wijkcode", "sequence_id"]
    out = sample[keep].merge(out.reset_index(), on="pano_id", validate="one_to_one")
    out["snap_status"] = "matched"  # the sample holds unambiguous matches within 20 m only (step 1.4)
    return gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out.lon, out.lat), crs=4326)


def buurt_table(seg: gpd.GeoDataFrame, value_cols: list[str]) -> gpd.GeoDataFrame:
    buurten = gpd.read_parquet(PROJ / "data/derived/buurten.parquet").to_crs(METRIC)
    net = gpd.read_parquet(PROJ / "data/derived/segments.parquet")[["segment_id", "geometry"]].to_crs(METRIC)
    net_len = gpd.overlay(net, buurten[["buurtcode", "geometry"]], how="intersection", keep_geom_type=True)
    net_km = (net_len.length / 1000).groupby(net_len.buurtcode).sum()

    obs = seg[seg.n_images > 0][["segment_id"] + value_cols + ["covered_geometry"]].copy()
    obs = gpd.GeoDataFrame(obs.drop(columns="covered_geometry"), geometry=obs["covered_geometry"], crs=4326).to_crs(METRIC)
    parts = gpd.overlay(obs, buurten[["buurtcode", "geometry"]], how="intersection", keep_geom_type=True)
    parts["w"] = parts.length
    rows = {}
    for code, g in parts.groupby("buurtcode"):
        r = {"covered_km": g.w.sum() / 1000, "n_segments_observed": g.segment_id.nunique()}
        for c in value_cols:
            ok = g[c].notna() & (g.w > 0)
            r[c] = np.average(g.loc[ok, c], weights=g.loc[ok, "w"]) if ok.any() else np.nan
        rows[code] = r
    t = pd.DataFrame.from_dict(rows, orient="index")
    out = buurten.set_index("buurtcode").join(t)
    out["network_km"] = net_km.reindex(out.index).fillna(0.0)
    out["covered_share"] = out.covered_km / out.network_km.replace(0, np.nan)
    return out.reset_index().to_crs(4326)


def main():
    pano = panoramas()
    states = [c for c in pano.columns if c.startswith("audit_") and c.endswith("_state")]
    numeric = [f"{c}_pixel_fraction" for c in SEG_CLASSES] + [f"{f}_{q}" for f in ("vit", "r50") for q in QUESTIONS]
    pano.to_parquet(OUT / "indicators_panorama.parquet", index=False)

    segments = gpd.read_parquet(PROJ / "data/derived/segments.parquet")
    seg = aggregate_segments(pano, segments, numeric_columns=numeric, state_columns=states, support_length_m=25,
                             sequence_column="sequence_id", time_column="timestamp", metric_crs=METRIC)
    meta = seg.attrs["geoai_vlm"]
    meta["rules"]["panorama_rules"] = {
        "audit_state": "present if any usable view reports present; absent if none and >= 1 usable view absent; else not_visible",
        "segmentation": "mean of the four views",
        "perception": "mean of the usable front and back views",
        "audit": "Qwen/Qwen3.5-4B, active_mobility_audit_v2, processing_id 9fa2547ea019",
    }
    write_segment_parquet(seg, OUT / "indicators_segment.parquet", meta)

    value_cols = [f"{c}__present_share" for c in states] + [f"{c}__mean" for c in numeric]
    buurt = buurt_table(seg, value_cols)
    buurt.to_parquet(OUT / "indicators_buurt.parquet", index=False)

    observed = seg[seg.n_images > 0]
    summary = {
        "panoramas": int(len(pano)), "panoramas_without_usable_view": int((pano.n_views_usable == 0).sum()),
        "panoramas_without_perception_view": int((pano.n_views_perception == 0).sum()),
        "segments_observed": int(len(observed)), "segments_total": int(len(seg)),
        "observed_km": round(float(observed.covered_length_m.sum()) / 1000, 1),
        "buurten_with_observations": int(buurt.covered_km.notna().sum()), "buurten_total": int(len(buurt)),
        "panorama_present_share": {c[6:-6]: round(float((pano[c] == "present").mean()), 3) for c in states},
    }
    (OUT / "indicators_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
