# -*- coding: utf-8 -*-
"""Step 1.4: the 25,000-panorama audit sample (seeded, spread over the network).

Eligible: matched to a segment (unambiguous), leaf-on months (June–October, so that trees
and greenery are comparable), daylight by solar elevation (GeoAI-VLM light_condition),
a known driving bearing. Spread: each segment is cut into 50 m pieces and at most one
panorama is kept per piece (the one closest to the piece centre). If more pieces than
the target remain, buurten are sampled in proportion to their eligible pieces with a
floor, so that every buurt with imagery is represented. Seed 2026.
Each sampled panorama gets four view headings: front (driving bearing), right, back, left.
Output: data/derived/sample_25k.parquet, outputs/wp3/sample_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoai_vlm import light_condition

PROJ = Path(__file__).resolve().parents[1]
DER = PROJ / "data" / "derived"
TARGET, PIECE_M, SEED, FLOOR = 25_000, 50.0, 2026, 5


def main():
    rng = np.random.default_rng(SEED)
    p = gpd.read_parquet(DER / "panoramas_2025_snapped.parquet")
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    p["light"] = light_condition(p["timestamp"], p["lat"], p["lon"])
    elig = p[(p["snap_status"] == "matched") & p["timestamp"].dt.month.between(6, 10)
             & (p["light"] == "day") & p["bearing"].notna()].copy()
    elig["piece"] = (elig["snap_offset_m"] // PIECE_M).astype(int)
    centre = (elig["piece"] + 0.5) * PIECE_M
    elig["dist_to_piece_centre"] = (elig["snap_offset_m"] - centre).abs()
    elig = elig.sort_values(["segment_id", "piece", "dist_to_piece_centre", "pano_id"])
    pieces = elig.drop_duplicates(["segment_id", "piece"]).copy()
    pieces["area_id"] = pieces["area_id"].fillna("(outside buurten)")

    if len(pieces) > TARGET:
        counts = pieces["area_id"].value_counts()
        quota = np.maximum(FLOOR, np.floor(counts / counts.sum() * TARGET)).astype(int)
        quota = np.minimum(quota, counts)
        while quota.sum() > TARGET:          # trim the largest quotas until the total fits
            quota[quota.idxmax()] -= 1
        chosen = []
        for area, q in quota.items():
            rows = pieces.index[pieces["area_id"] == area]
            chosen.extend(rng.choice(rows, size=int(q), replace=False))
        sample = pieces.loc[sorted(chosen)].copy()
    else:
        sample = pieces.copy()
    for name, add in (("front", 0), ("right", 90), ("back", 180), ("left", 270)):
        sample[f"yaw_{name}"] = (sample["bearing"] + add) % 360
    keep = ["pano_id", "timestamp", "lon", "lat", "bearing", "yaw_front", "yaw_right", "yaw_back", "yaw_left",
            "segment_id", "snap_distance_m", "snap_offset_m", "area_id", "buurtnaam", "wijkcode", "light",
            "cubic_img_baseurl", "sequence_id"]
    out = pd.DataFrame(sample[keep]).reset_index(drop=True)
    out["sample_seed"] = SEED
    out.to_parquet(DER / "sample_25k.parquet", index=False)
    segs = gpd.read_parquet(DER / "segments.parquet")[["segment_id", "road_group"]]
    out = out.merge(segs, on="segment_id", how="left")
    info = {
        "panoramas": int(len(p)), "eligible": int(len(elig)), "pieces_50m": int(len(pieces)),
        "sampled": int(len(out)), "views": int(4 * len(out)), "segments": int(out["segment_id"].nunique()),
        "buurten": int(out["area_id"].nunique()), "by_road_group": out["road_group"].value_counts().to_dict(),
        "by_month": pd.to_datetime(out["timestamp"], utc=True).dt.month.value_counts().sort_index().to_dict(),
        "seed": SEED, "piece_m": PIECE_M, "floor_per_buurt": FLOOR,
    }
    (PROJ / "outputs" / "wp3" / "sample_summary.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
