# -*- coding: utf-8 -*-
"""Step 3.2: inputs for the synthetic WP2 data (docs/wp2_synthetic_design.md).

* Points of interest from OpenStreetMap (ODbL) inside the municipality, one point per feature
  (polygons by their representative point), in destination classes:
  daily_shop, other_shop, school, park, sport, health, transit_stop, social, office.
* Buurten with population and the CBS 2023 share of low-income households, cut into
  tertiles of buurten for the synthetic stratified sample (buurten without the share or
  without inhabitants are left out).
* The routing network: WP3 segments with their end nodes, length (EPSG:28992), road group
  and which modes may use them (walk, bike, drive). OSM access tags beyond the highway type
  are not used; this is a simplification for synthetic routes.

Usage: python 40_wp2_inputs.py
Writes data/derived/wp2_pois.parquet, wp2_buurten.parquet, wp2_network.parquet, wp2_inputs_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DER = PROJ / "data/derived"
METRIC = "EPSG:28992"
ox.settings.use_cache = True
ox.settings.cache_folder = str(PROJ / "data/raw/osmnx_cache")
ox.settings.requests_timeout = 600

TAGS = {"shop": True, "amenity": ["school", "kindergarten", "college", "university", "doctors", "clinic", "pharmacy",
                                  "dentist", "hospital", "cafe", "restaurant", "bar", "library", "community_centre",
                                  "theatre", "cinema", "place_of_worship", "townhall"],
        "leisure": ["park", "sports_centre", "fitness_centre", "pitch", "swimming_pool", "sports_hall", "playground"],
        "office": True, "highway": ["bus_stop"], "railway": ["station", "tram_stop", "halt"]}
INCOME = "percentageHuishoudensMetLaagInkomen"
DAILY_SHOPS = {"supermarket", "convenience", "bakery", "greengrocer", "butcher", "deli", "cheese", "seafood"}


def poi_class(r) -> str | None:
    shop, amenity, leisure = r.get("shop"), r.get("amenity"), r.get("leisure")
    if isinstance(shop, str):
        return "daily_shop" if shop in DAILY_SHOPS else "other_shop"
    if amenity in ("school", "kindergarten", "college", "university"):
        return "school"
    if amenity in ("doctors", "clinic", "pharmacy", "dentist", "hospital"):
        return "health"
    if amenity in ("cafe", "restaurant", "bar", "library", "community_centre", "theatre", "cinema", "place_of_worship"):
        return "social"
    if leisure == "park":
        return "park"
    if leisure in ("sports_centre", "fitness_centre", "pitch", "swimming_pool", "sports_hall", "playground"):
        return "sport"
    if isinstance(r.get("office"), str) or amenity == "townhall":
        return "office"
    if r.get("highway") == "bus_stop" or r.get("railway") in ("station", "tram_stop", "halt"):
        return "transit_stop"
    return None


def main():
    gemeente = gpd.read_parquet(DER / "gemeente.parquet")
    feats = ox.features_from_polygon(gemeente.geometry.iloc[0], TAGS).reset_index()
    feats["poi_class"] = feats.apply(poi_class, axis=1)
    feats = feats[feats.poi_class.notna()].copy()
    pts = feats.to_crs(METRIC)
    pts["geometry"] = pts.geometry.representative_point()
    if "park" in set(pts.poi_class):  # small parks are pocket greens, keep parks >= 0.5 ha
        area = feats.to_crs(METRIC).geometry.area
        pts = pts[(pts.poi_class != "park") | (area >= 5000)]
    pois = gpd.GeoDataFrame(pts[["element", "id", "poi_class", "name", "geometry"]].rename(columns={"id": "osm_id"}),
                            crs=METRIC).to_crs(4326)
    pois.to_parquet(DER / "wp2_pois.parquet", index=False)

    b = gpd.read_parquet(DER / "buurten.parquet")
    # CBS mean income per inhabitant is filled for only 147 buurten (58% of inhabitants); the share
    # of low-income households for 420 (99%), so the strata use that share (high share = low stratum)
    b = b[(b.aantalInwoners > 0) & b[INCOME].notna()].copy()
    b["income_tertile"] = pd.qcut(b[INCOME], 3, labels=["high", "middle", "low"]).astype(str)
    b.to_parquet(DER / "wp2_buurten.parquet", index=False)

    s = gpd.read_parquet(DER / "segments.parquet").to_crs(METRIC)
    s["length_m"] = s.length
    hw = s.highway.fillna("")
    # walkers keep off ways tagged only as cycleway or busway; cyclists off ways tagged only as
    # footway or pedestrian (shared paths, highway=path/track, stay open to both)
    s["walk"] = ~hw.str.fullmatch(r"(cycleway|busway)(\|(cycleway|busway))*")
    s["bike"] = ~hw.str.fullmatch(r"(footway|pedestrian)(\|(footway|pedestrian))*")
    s["drive"] = s.road_group.isin(["main road", "local street", "service road", "bus corridor"])
    net = s[["segment_id", "u", "v", "length_m", "road_group", "walk", "bike", "drive", "geometry"]].to_crs(4326)
    net.to_parquet(DER / "wp2_network.parquet", index=False)

    summary = {"pois_by_class": pois.poi_class.value_counts().to_dict(),
               "buurten_used": int(len(b)), "income_field": INCOME,
               "tertile_range": {k: [float(g[INCOME].min()), float(g[INCOME].max())] for k, g in b.groupby("income_tertile")},
               "population_by_tertile": b.groupby("income_tertile").aantalInwoners.sum().astype(int).to_dict(),
               "network_km": {m: round(float(s.loc[s[m], "length_m"].sum()) / 1000, 1) for m in ("walk", "bike", "drive")}}
    (DER / "wp2_inputs_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
