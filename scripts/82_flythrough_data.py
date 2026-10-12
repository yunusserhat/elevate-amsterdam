# -*- coding: utf-8 -*-
"""Step 8 (site): data for the 3D walking flythrough, from the SYNTHETIC WP2 GNSS data.

Picks one synthetic walking trip of 1.2-2.9 km whose path crosses the historic centre and runs
mostly along street segments the WP3 sample observed, then writes:
* route: the true path snapped to the walking network (shortest walk path between consecutive
  true positions), with cumulative distance and the green view of observed segments within 25 m;
* fixes: the raw synthetic GNSS fixes of the trip (with reported accuracy) and the cleaned ones;
* views: sampled street views within 25 m of the path, with their observations and City links.
Usage: python 82_flythrough_data.py [--person S... --day N --start HH:MM]
Writes space/data/layers/flythrough_synthetic.json. SYNTHETIC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wp2_common import METRIC, Exposure, ModeGraph, mode_masks, osm_graph_tables  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
CENTRE = (52.3600, 52.3850, 4.8750, 4.9150)  # lat min/max, lon min/max: canal belt and around
TO_M = Transformer.from_crs(4326, METRIC, always_xy=True)
TO_W = Transformer.from_crs(METRIC, 4326, always_xy=True)


def walking_runs(gps: pd.DataFrame) -> pd.DataFrame:
    g = gps.sort_values(["person_id", "day", "time"]).copy()
    walk = g.true_state.eq("walk")
    brk = (~walk) | (g.person_id != g.person_id.shift()) | (g.day != g.day.shift()) | ((g.time - g.time.shift()).dt.total_seconds() > 120)
    g["run"] = brk.cumsum()
    return g[walk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person")
    ap.add_argument("--day", type=int)
    args = ap.parse_args()
    gps = pd.read_parquet(PROJ / "outputs/wp2/synthetic_gps.parquet")
    runs = walking_runs(gps)
    runs["x"], runs["y"] = TO_M.transform(runs.true_lon.to_numpy(), runs.true_lat.to_numpy())
    expo = Exposure()
    cand = []
    for r, q in runs.groupby("run"):
        if len(q) < 25:
            continue
        L = float(np.hypot(np.diff(q.x), np.diff(q.y)).sum())
        inside = q.true_lat.between(*CENTRE[:2]) & q.true_lon.between(*CENTRE[2:])
        if not (1200 <= L <= 2900) or inside.mean() < 0.6:
            continue
        cov = float(np.isfinite(expo.at(np.c_[q.x, q.y])).mean())
        cand.append((r, q.person_id.iloc[0], int(q.day.iloc[0]), L, cov))
    c = pd.DataFrame(cand, columns=["run", "person", "day", "length_m", "observed_share"])
    if args.person:
        c = c[(c.person == args.person) & (c.day == args.day)]
    c = c.sort_values(["observed_share", "length_m"], ascending=False)
    print(c.head(8).to_string(index=False))
    pick = c.iloc[0]
    q = runs[runs.run == pick.run].sort_values("time")

    nodes, edges = osm_graph_tables()
    walk = ModeGraph(nodes, edges, mode_masks(edges)["walk"])
    ids = walk.nearest(np.c_[q.x, q.y])
    ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    parts = []
    for a, b in zip(ids[:-1], ids[1:]):
        # plain shortest walk between consecutive true positions (no jitter)
        dist, pred = dijkstra(walk.csr, indices=int(a), return_predecessors=True, limit=2000.0)
        if not np.isfinite(dist[b]):
            continue
        path = [int(b)]
        while path[-1] != int(a):
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        for u, v in zip(path[:-1], path[1:]):
            eid, _ = walk._edge[u * walk.n + v]
            cc = walk._coords(eid)
            if np.hypot(*(cc[0] - walk.xy[u])) > np.hypot(*(cc[-1] - walk.xy[u])):
                cc = cc[::-1]
            parts.append(cc if not parts else cc[1:])
    line = np.vstack(parts)
    # densify to <= 5 m steps for a smooth camera
    out = [line[0]]
    for p0, p1 in zip(line[:-1], line[1:]):
        d = float(np.hypot(*(p1 - p0)))
        n = max(1, int(np.ceil(d / 5.0)))
        for k in range(1, n + 1):
            out.append(p0 + (p1 - p0) * k / n)
    line = np.array(out)
    keep = np.r_[True, np.hypot(*np.diff(line, axis=0).T) > 0.01]
    line = line[keep]
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    gv = expo.at(line)
    lon, lat = TO_W.transform(line[:, 0], line[:, 1])

    raw = gps[(gps.person_id == pick.person) & (gps.day == pick.day) & (gps.time >= q.time.min()) & (gps.time <= q.time.max())]
    fx = pd.read_parquet(PROJ / "outputs/wp2/pipeline_fixes.parquet")
    t0 = (q.time.min() - q.time.min().normalize()).total_seconds() - 7 * 3600
    t1 = (q.time.max() - q.time.max().normalize()).total_seconds() - 7 * 3600
    clean = fx[(fx.person_id == pick.person) & (fx.day == pick.day) & (fx.t >= t0) & (fx.t <= t1)]

    rec = pd.read_parquet(PROJ / "space/data/index/records.parquet")
    rx, ry = TO_M.transform(rec.lon.to_numpy(), rec.lat.to_numpy())
    from scipy.spatial import cKDTree
    d, i = cKDTree(line).query(np.c_[rx, ry])
    near = rec[(d <= 25) & rec.view.isin(["front", "back"]).to_numpy()].copy()
    near["along_m"] = cum[i[(d <= 25) & rec.view.isin(["front", "back"]).to_numpy()]]
    seg = pd.concat([pd.read_parquet(f, columns=["image_id", "vegetation_pixel_fraction"]) for f in sorted((PROJ / "outputs/wp3/segmentation").glob("*.parquet"))])
    near = near.merge(seg, on="image_id", how="left").sort_values("along_m")

    data = {
        "SYNTHETIC": True,
        "note": "A synthetic person's walk (WP2); the path is the synthetic truth snapped to the walking network.",
        "person": pick.person, "day": int(pick.day), "start": str(q.time.min()), "end": str(q.time.max()),
        "length_m": round(float(cum[-1]), 1), "observed_share": round(float(np.isfinite(gv).mean()), 3),
        "route": [[round(a, 6), round(b, 6), round(float(c), 1), None if not np.isfinite(g) else round(float(g), 3)]
                  for a, b, c, g in zip(lon, lat, cum, gv)],
        "raw_fixes": [[round(a, 6), round(b, 6), float(acc)] for a, b, acc in zip(raw.lon, raw.lat, raw.accuracy_m)],
        "clean_fixes": [[round(a, 6), round(b, 6), m] for a, b, m in zip(clean.lon, clean.lat, clean["mode"])],
        "views": [{"image_id": r.image_id, "pano_id": r.pano_id, "view": r.view, "lon": round(r.lon, 6), "lat": round(r.lat, 6),
                   "heading_deg": round(float(r.heading_deg), 1), "along_m": round(float(r.along_m), 1), "buurt": r.buurtnaam,
                   "green_view": None if pd.isna(r.vegetation_pixel_fraction) else round(float(r.vegetation_pixel_fraction), 3),
                   "observation_text": r.observation_text, "city_image": r.city_image} for r in near.itertuples()],
    }
    path = PROJ / "space/data/layers/flythrough_synthetic.json"
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"{pick.person} day {pick.day}: {cum[-1]:.0f} m, {len(line)} route points, {len(raw)} raw fixes, "
          f"{len(clean)} cleaned, {len(near)} views, observed share {np.isfinite(gv).mean():.2f} -> {path}")


if __name__ == "__main__":
    main()
