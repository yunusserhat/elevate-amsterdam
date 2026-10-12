# -*- coding: utf-8 -*-
"""Step 1.3: driving direction, segment match and coverage of the 2025 panoramas.

* Driving bearing: panoramas are north-aligned (tested: yaw 0 = north). The recording id
  and frame number in `pano_id` (recording_<date>_<time>_<frame>) order the frames of one
  drive; the bearing of a frame is taken from the previous to the next frame of the same
  recording, if they are 2–40 m apart.
* Segment match: GeoAI-VLM snap_images_to_segments (20 m, 2 m ambiguity margin).
* Coverage: aggregate_segments (25 m support), network_coverage by road group,
  coverage_by_area by buurt (CBS 2023).
Outputs: data/derived/panoramas_2025_snapped.parquet, outputs/wp3/coverage_*.csv,
outputs/wp3/coverage_segments_2025.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoai_vlm import (aggregate_segments, coverage_by_area, network_coverage, snap_images_to_segments,
                       write_segment_parquet)

PROJ = Path(__file__).resolve().parents[1]
DER, OUT = PROJ / "data" / "derived", PROJ / "outputs" / "wp3"
METRIC = "EPSG:28992"


def bearings(df: pd.DataFrame) -> pd.Series:
    parts = df["pano_id"].str.extract(r"^(recording_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$")
    df = df.assign(rec=parts[0], frame=pd.to_numeric(parts[1], errors="coerce"))
    pts = gpd.GeoSeries(gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(METRIC)
    df["x"], df["y"] = pts.x.values, pts.y.values
    df = df.sort_values(["rec", "frame"])
    g = df.groupby("rec")
    dx = g["x"].shift(-1).fillna(df["x"]) - g["x"].shift(1).fillna(df["x"])
    dy = g["y"].shift(-1).fillna(df["y"]) - g["y"].shift(1).fillna(df["y"])
    dist = np.hypot(dx, dy)
    b = (np.degrees(np.arctan2(dx, dy)) + 360) % 360   # RD New is close to grid north here
    b[(dist < 2) | (dist > 40) | df["rec"].isna()] = np.nan
    return b.reindex(df.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pano = pd.read_parquet(PROJ / "data" / "raw" / "panoramas_2025.parquet")
    pano["bearing"] = bearings(pano)
    segs = gpd.read_parquet(DER / "segments.parquet")
    buurten = gpd.read_parquet(DER / "buurten.parquet").rename(columns={"buurtcode": "area_id"})
    gdf = gpd.GeoDataFrame(pano, geometry=gpd.points_from_xy(pano.lon, pano.lat), crs=4326)
    gdf["captured_at"] = gdf["timestamp"]
    gdf["sequence_id"] = gdf["pano_id"].str.extract(r"^(recording_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_")[0]
    snapped = snap_images_to_segments(gdf, segs, max_distance_m=20, ambiguity_margin_m=2, metric_crs=METRIC)
    snapped = gpd.sjoin(snapped, buurten[["area_id", "buurtnaam", "wijkcode", "geometry"]], how="left",
                        predicate="within").drop(columns="index_right")
    snapped = snapped[~snapped.index.duplicated()]
    snapped.to_parquet(DER / "panoramas_2025_snapped.parquet", index=False)

    summary = aggregate_segments(snapped, segs, support_length_m=25, metric_crs=METRIC)
    write_segment_parquet(summary, OUT / "coverage_segments_2025.parquet")
    network_coverage(summary).to_csv(OUT / "coverage_overall_2025.csv", index=False)
    network_coverage(summary, group_column="road_group").to_csv(OUT / "coverage_by_road_group_2025.csv", index=False)
    cov_area = coverage_by_area(summary, buurten, area_id_column="area_id", images=snapped, metric_crs=METRIC)
    cov_area.merge(buurten.drop(columns="geometry"), on="area_id", how="left").to_csv(
        OUT / "coverage_by_buurt_2025.csv", index=False)

    info = {
        "panoramas": int(len(pano)),
        "with_bearing": int(pano["bearing"].notna().sum()),
        "snap_status": snapped["snap_status"].value_counts().to_dict(),
        "in_a_buurt": int(snapped["area_id"].notna().sum()),
        "coverage_overall": network_coverage(summary).drop(columns="group").iloc[0].round(4).to_dict(),
    }
    (OUT / "coverage_2025_summary.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))
    print(network_coverage(summary, group_column="road_group")[
        ["road_group", "network_length_m", "covered_length_share", "n_images", "images_per_km"]].round(3).to_string())


if __name__ == "__main__":
    main()
