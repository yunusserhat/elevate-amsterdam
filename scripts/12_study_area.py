# -*- coding: utf-8 -*-
"""Step 1.2: study area, neighbourhoods and street segments for Amsterdam.

* Municipality boundary (CBS gemeente GM0363) and buurten from the CBS Wijk- en
  Buurtkaart 2023 (PDOK WFS, CC0). 2023 is the latest edition with income filled.
* Street segments from OpenStreetMap: every way people can walk or cycle along
  (no motorways or trunk roads), separately mapped sidewalks and crossings left out,
  parking aisles, driveways and private ways left out. The graph is downloaded for the
  boundary buffered by 500 m (OSMnx drops edges whose nodes fall outside the query
  polygon), then clipped to the boundary.
Outputs (data/derived/): gemeente.parquet, buurten.parquet, segments.parquet,
motorways.parquet, study_area_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests

from geoai_vlm import prepare_segments, segments_from_graph

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data" / "derived"
METRIC = "EPSG:28992"
WFS = "https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0"
ox.settings.use_cache = True
ox.settings.cache_folder = str(PROJ / "data" / "raw" / "osmnx_cache")
ox.settings.requests_timeout = 600
ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + [
    "footway", "service", "access", "foot", "bicycle", "cycleway", "sidewalk", "surface", "busway"]

DROP_HIGHWAY = {"motorway", "motorway_link", "trunk", "trunk_link", "steps", "corridor", "elevator",
                "platform", "proposed", "construction", "raceway", "bridleway", "escape", "rest_area"}
DROP_FOOTWAY = {"sidewalk", "crossing", "traffic_island", "access_aisle"}
DROP_SERVICE = {"parking_aisle", "driveway", "drive-through"}
BUURT_FIELDS = ["buurtcode", "buurtnaam", "wijkcode", "wijknaam", "gemeentecode", "water", "aantalInwoners",
                "bevolkingsdichtheidInwonersPerKm2", "aantalHuishoudens", "omgevingsadressendichtheid",
                "percentageHuishoudensMetLaagInkomen", "percentageHuishoudensOnderOfRondSociaalMinimum",
                "percentagePersonenMetLaagInkomen", "gemiddeldInkomenPerInwoner", "personenautosPerHuishouden"]


def wfs(layer: str) -> gpd.GeoDataFrame:
    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": f"wijkenbuurten:{layer}",
              "outputFormat": "application/json", "srsName": "EPSG:4326", "count": 5000,
              "filter": "<Filter><PropertyIsEqualTo><PropertyName>gemeentecode</PropertyName>"
                        "<Literal>GM0363</Literal></PropertyIsEqualTo></Filter>"}
    gj = requests.get(WFS, params=params, timeout=300).json()
    return gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")


def _flat(v):
    if isinstance(v, (list, tuple)):
        return "|".join(sorted(str(x) for x in v))
    return v if isinstance(v, str) else None


def _s(row, key) -> str:
    v = row.get(key)
    return v if isinstance(v, str) else ""


def keep(row) -> bool:
    hw = set(_s(row, "highway").split("|"))
    if hw & DROP_HIGHWAY and not hw - DROP_HIGHWAY:
        return False
    if set(_s(row, "footway").split("|")) & DROP_FOOTWAY:
        return False
    if _s(row, "cycleway") == "crossing" and hw <= {"cycleway"}:
        return False
    if set(_s(row, "service").split("|")) & DROP_SERVICE:
        return False
    access = set(_s(row, "access").split("|"))
    if access & {"private", "no"} and _s(row, "foot") not in ("yes", "designated") \
            and _s(row, "bicycle") not in ("yes", "designated"):
        return False
    return True


def road_group(hw: str) -> str:
    parts = set((hw or "").split("|"))
    if parts & {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}:
        return "main road"
    if parts & {"residential", "unclassified", "living_street", "road"}:
        return "local street"
    if parts & {"pedestrian"}:
        return "pedestrian street"
    if parts & {"busway"}:
        return "bus corridor"
    if parts & {"cycleway"}:
        return "cycleway"
    if parts & {"footway", "path", "track"}:
        return "footway / path"
    if parts & {"service"}:
        return "service road"
    return "other"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gemeente = wfs("gemeenten")
    boundary = gemeente.to_crs(METRIC).geometry.union_all()
    gpd.GeoDataFrame(geometry=[boundary], crs=METRIC).to_crs(4326).to_parquet(OUT / "gemeente.parquet")

    buurten = wfs("buurten")
    buurten = buurten[[c for c in BUURT_FIELDS if c in buurten.columns] + ["geometry"]].copy()
    for c in buurten.columns:
        if c not in ("geometry",) and pd.api.types.is_numeric_dtype(buurten[c]):
            buurten.loc[buurten[c] <= -99990, c] = np.nan   # CBS codes for suppressed / not available
    buurten.to_parquet(OUT / "buurten.parquet", index=False)

    poly = gpd.GeoSeries([boundary.buffer(500)], crs=METRIC).to_crs(4326).iloc[0]
    graph = ox.graph_from_polygon(poly, network_type="all", simplify=True, retain_all=True)
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
    for col in ["highway", "footway", "service", "access", "foot", "bicycle", "cycleway", "name", "busway"]:
        if col not in edges.columns:
            edges[col] = None
        edges[col] = edges[col].map(_flat)
    edges[edges["highway"].fillna("").str.contains("motorway|trunk")][["highway", "name", "geometry"]] \
        .to_parquet(OUT / "motorways.parquet", index=False)
    kept = edges[edges.apply(keep, axis=1)]

    segs = segments_from_graph(graph)
    segs = segs.merge(kept[["u", "v", "key", "highway", "name", "footway", "cycleway", "busway"]]
                      .drop_duplicates(["u", "v", "key"]), on=["u", "v", "key"], how="inner")
    segs["road_class"] = segs["highway"]
    segs["road_group"] = segs["highway"].map(road_group)
    segs = segs.to_crs(METRIC)
    segs = segs[segs.intersects(boundary)].copy()
    segs["geometry"] = segs.geometry.intersection(boundary)
    segs = segs[~segs.geometry.is_empty].explode(index_parts=True)
    segs = segs[segs.geom_type == "LineString"].copy()
    part = segs.index.get_level_values(-1)
    segs["segment_id"] = [sid if k == 0 else f"{sid}#{k}" for sid, k in zip(segs["segment_id"], part)]
    segs = prepare_segments(segs.reset_index(drop=True).to_crs(4326), id_column="segment_id")
    segs.to_parquet(OUT / "segments.parquet", index=False)

    seg_m = segs.to_crs(METRIC)
    summary = {
        "gemeente_km2": round(boundary.area / 1e6, 1),
        "n_buurten": int(len(buurten)),
        "buurten_with_low_income_share": int(buurten["percentageHuishoudensMetLaagInkomen"].notna().sum()),
        "n_segments": int(len(segs)),
        "network_km": round(float(seg_m.length.sum()) / 1000, 1),
        "km_by_group": seg_m.assign(km=seg_m.length / 1000).groupby("road_group").km.sum().round(1).to_dict(),
    }
    (OUT / "study_area_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
