# -*- coding: utf-8 -*-
"""Step 8.7: data for the v2 story pages (site redesign of 2026-10-11).

* layers/wp2_spikes_synthetic.json: SYNTHETIC walking and cycling fixes per 250 m cell and
  half-hour (07:00-19:00), as minutes recorded on an average weekday and weekend day.
* layers/wp2_exposure_synthetic.json: SYNTHETIC cleaned walking and cycling fixes with the green
  view of the observed segment within 25 m (a seeded sample).
* wp3_example/pano_equirect.jpg: the example panorama (Gemeente Amsterdam, CC BY 4.0), rolled so
  the four views (front, right, back, left) are the four quarters left to right; 2400 x 1200.
* home/chapters.json: places the home page flies to, with the numbers its cards show.
* home/poster.jpg: PDOK 2025 aerial photo of the centre (CC BY 4.0), shown while the 3D view loads.
* tables/wp4_buurt_model.json: per buurt, the model indicators WP4 sets against residents' reports.
* tables/wp6_confirm_summary.json: copy of outputs/wp6/confirm_summary.json (how often the larger model confirms a gap).

Usage: python 83_space_story_data.py
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "space/data"
SEED = 2026
CELL_M = 250.0
LAT0, LON0 = 52.37, 4.89
M_LAT = 111_320.0
M_LON = M_LAT * math.cos(math.radians(LAT0))
SLOTS = [f"{7 + i // 2:02d}:{30 * (i % 2):02d}" for i in range(24)]
PDOK = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/2025_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
EQUI = "https://t1.data.amsterdam.nl/panorama/2025/360geo/{pano}/equirectangular/panorama_4000.jpg"


def spikes() -> dict:
    g = pd.read_parquet(PROJ / "outputs/wp2/synthetic_gps.parquet",
                        columns=["person_id", "day", "time", "lon", "lat", "true_state", "is_spike"])
    g = g[~g.is_spike & g.true_state.isin(["walk", "bike"])].copy()
    minute = g.time.dt.hour * 60 + g.time.dt.minute
    g["slot"] = (minute - 7 * 60) // 30
    g = g[(g.slot >= 0) & (g.slot < 24)]
    g["weekend"] = g.time.dt.dayofweek >= 5
    g["cx"] = np.floor((g.lon - LON0) * M_LON / CELL_M).astype(int)
    g["cy"] = np.floor((g.lat - LAT0) * M_LAT / CELL_M).astype(int)
    days = g.groupby("weekend").day.nunique().to_dict()
    out = {"SYNTHETIC": True, "cell_m": CELL_M, "slots": SLOTS,
           "unit": "minutes of walking or cycling recorded (30 s fixes) per 250 m cell and half-hour, average day",
           "source": "synthetic WP2 GPS (outputs/wp2/synthetic_gps.parquet), true state, spikes removed"}
    for weekend, name in [(False, "weekday"), (True, "weekend")]:
        d = g[g.weekend == weekend]
        n = days[weekend]
        counts = d.groupby(["cx", "cy", "true_state", "slot"]).size().rename("n").reset_index()
        cells = counts[["cx", "cy"]].drop_duplicates().sort_values(["cy", "cx"], ascending=[False, True]).reset_index(drop=True)
        cells["i"] = np.arange(len(cells))
        counts = counts.merge(cells, on=["cx", "cy"])
        arr = {m: np.zeros((len(cells), 24)) for m in ["walk", "bike"]}
        for m in arr:
            c = counts[counts.true_state == m]
            arr[m][c.i.to_numpy(), c.slot.to_numpy()] = c.n.to_numpy() * 0.5 / n
        persons = d.groupby(["true_state", "slot"]).apply(lambda x: x[["person_id", "day"]].drop_duplicates().shape[0] / n,
                                                         include_groups=False)
        lon = LON0 + (cells.cx + 0.5) * CELL_M / M_LON
        lat = LAT0 + (cells.cy + 0.5) * CELL_M / M_LAT
        out[name] = {
            "days": int(n),
            "cells": [[round(a, 5), round(b, 5)] for a, b in zip(lon, lat)],
            "walk": np.round(arr["walk"], 1).tolist(),
            "bike": np.round(arr["bike"], 1).tolist(),
            "persons_walk": [round(float(persons.get(("walk", s), 0)), 1) for s in range(24)],
            "persons_bike": [round(float(persons.get(("bike", s), 0)), 1) for s in range(24)],
            "max_minutes": round(float(max(arr["walk"].max(), arr["bike"].max())), 1),
        }
        print(name, "days", n, "cells", len(cells), "max", out[name]["max_minutes"],
              "peak walkers", max(out[name]["persons_walk"]), "peak cyclists", max(out[name]["persons_bike"]))
    return out


def exposure() -> dict:
    f = pd.read_parquet(PROJ / "outputs/wp2/pipeline_fixes.parquet", columns=["mode", "lon", "lat", "x_green_view"])
    f = f[f["mode"].isin(["walk", "bike"]) & f.x_green_view.notna()]
    s = f.sample(n=min(20_000, len(f)), random_state=SEED)
    print("exposure fixes", len(f), "sampled", len(s))
    return {"SYNTHETIC": True, "fixes_with_exposure": int(len(f)), "sampled": int(len(s)),
            "unit": "green view of the observed WP3 segment within 25 m (mean of its panoramas)",
            "points": [[round(a, 5), round(b, 5), round(c, 3)] for a, b, c in zip(s.lon, s.lat, s.x_green_view)]}


def panorama() -> dict:
    ex = json.loads((OUT / "wp3_example/example.json").read_text())
    front = next(v for v in ex["views"] if v["view"] == "front")["heading_deg"]
    r = requests.get(EQUI.format(pano=ex["pano_id"]), timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    w, h = img.size
    start = (front - 45) % 360                       # left edge of the front view
    col = int(round(((0.5 + start / 360) % 1) * w))   # north-aligned: centre column = 0 degrees
    arr = np.roll(np.asarray(img), -col, axis=1)
    out = Image.fromarray(arr).resize((2400, 1200), Image.LANCZOS)
    out.save(OUT / "wp3_example/pano_equirect.jpg", quality=80, optimize=True, progressive=True)
    info = {"source": EQUI.format(pano=ex["pano_id"]), "original_size": [w, h], "left_edge_heading_deg": round(start, 1),
            "quarters": ["front", "right", "back", "left"],
            "view_band": {"top": 0.25, "bottom": round((90 + math.degrees(math.atan(192 / 384))) / 180, 4)},
            "credit": "Panorama: Gemeente Amsterdam, CC BY 4.0 (rolled and resized here)"}
    print("panorama", w, h, "rolled by", col)
    return info


def chapters() -> dict:
    b = gpd.read_file(OUT / "layers/buurten.geojson")
    ind = pd.read_parquet(PROJ / "outputs/wp3/indicators_buurt.parquet")
    col = [c for c in ind.columns if c.startswith("audit_sidewalk_obstruction") and c.endswith("present_share")][0]
    key = "buurtcode" if "buurtcode" in ind.columns else ind.index.name
    ind = ind.reset_index() if key == ind.index.name else ind
    b = b.merge(ind[["buurtcode", col]], on="buurtcode", how="left")
    b["r_rep"] = b.reports_obstruction_per_km.rank(pct=True)
    b["r_mod"] = b[col].rank(pct=True)
    b["both"] = b[["r_rep", "r_mod"]].min(axis=1)
    top = b.sort_values("both", ascending=False).iloc[0]
    c4 = top.geometry.representative_point()
    print("wp4", top.buurtnaam, round(top.reports_obstruction_per_km, 1), round(top[col], 3), round(top.both, 3))

    pr = b[b.priority_low_green_low_income]
    c5 = pr.to_crs(28992).union_all().centroid
    c5 = gpd.GeoSeries([c5], crs=28992).to_crs(4326).iloc[0]

    v = pd.DataFrame(json.loads((OUT / "tables/wp6_edit_verification.json").read_text()))
    rec = pd.read_parquet(OUT / "index/records.parquet", columns=["image_id", "lat", "lon", "heading_deg"])
    v = v.merge(rec, on="image_id")
    v = v[(v.gap_type == "trees") & v.success_4b & v.success_27b]
    v = v.sort_values("pixels_changed_unexpected_share")
    print(v[["image_id", "gap_type", "buurt", "street", "pixels_changed_share", "pixels_changed_unexpected_share"]].head(8).to_string())
    pick = v.iloc[0]
    ex = json.loads((OUT / "wp3_example/example.json").read_text())
    return {
        "wp3": {"lat": ex["lat"], "lon": ex["lon"], "pano_id": ex["pano_id"], "buurt": ex["buurt"],
                "front_heading_deg": next(x for x in ex["views"] if x["view"] == "front")["heading_deg"]},
        "wp4": {"buurtcode": top.buurtcode, "buurt": top.buurtnaam, "lat": round(c4.y, 6), "lon": round(c4.x, 6),
                "obstruction_reports_per_km": round(float(top.reports_obstruction_per_km), 1),
                "model_obstruction_share": round(float(top[col]), 3),
                "rule": "highest of min(percentile of obstruction reports per km, percentile of the model's obstructed-sidewalk share)"},
        "wp5": {"lat": round(c5.y, 6), "lon": round(c5.x, 6), "priority_buurten": int(len(pr))},
        "wp6": {"image_id": pick.image_id, "gap_type": pick.gap_type, "buurt": pick.buurt, "street": pick.street,
                "lat": round(float(pick.lat), 6), "lon": round(float(pick.lon), 6), "heading_deg": round(float(pick.heading_deg), 1),
                "rule": "confirmed street-tree edit with the smallest share of unexpected pixel changes"},
    }


def wp4_model() -> dict:
    ind = pd.read_parquet(PROJ / "outputs/wp3/indicators_buurt.parquet")
    ind = ind.reset_index() if "buurtcode" not in ind.columns else ind
    cols = {"litter": "audit_litter_state__present_share", "obstruction": "audit_sidewalk_obstruction_state__present_share",
            "parking": "audit_sidewalk_parking_state__present_share", "pavement": "audit_surface_damage_state__present_share",
            "street_furniture": "audit_seating_state__present_share", "street_lighting": "audit_street_light_pole_state__present_share",
            "trees": "vegetation_pixel_fraction__mean"}
    out = {r.buurtcode: {k: (None if pd.isna(getattr(r, c)) else round(float(getattr(r, c)), 3)) for k, c in cols.items()}
           for r in ind[["buurtcode", *cols.values()]].itertuples(index=False)}
    print("wp4 model buurten", len(out))
    return {"columns": cols, "buurten": out}


def poster():
    z, lat, lon = 15, 52.3725, 4.8935
    n = 2 ** z
    xt = (lon + 180) / 360 * n
    yt = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    nx, ny = 7, 5
    x0, y0 = int(xt - nx / 2), int(yt - ny / 2)
    img = Image.new("RGB", (256 * nx, 256 * ny))
    s = requests.Session()
    for i in range(nx):
        for j in range(ny):
            r = s.get(PDOK.format(z=z, x=x0 + i, y=y0 + j), timeout=30)
            r.raise_for_status()
            img.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), (256 * i, 256 * j))
    (OUT / "home").mkdir(exist_ok=True)
    img.save(OUT / "home/poster.jpg", quality=72, optimize=True, progressive=True)
    # the page shows the poster darkened behind the title while the 3D city loads: a WebP at 1280 px, quality 60, is enough (226 KB against 543 KB)
    img.resize((1280, round(img.height * 1280 / img.width)), Image.LANCZOS).save(OUT / "home/poster.webp", quality=60, method=6)
    print("poster", img.size)


if __name__ == "__main__":
    (OUT / "layers/wp2_spikes_synthetic.json").write_text(json.dumps(spikes(), separators=(",", ":")))
    (OUT / "layers/wp2_exposure_synthetic.json").write_text(json.dumps(exposure(), separators=(",", ":")))
    (OUT / "wp3_example/pano.json").write_text(json.dumps(panorama(), indent=1))
    (OUT / "home").mkdir(exist_ok=True)
    (OUT / "home/chapters.json").write_text(json.dumps(chapters(), indent=1, ensure_ascii=False))
    (OUT / "tables/wp4_buurt_model.json").write_text(json.dumps(wp4_model(), separators=(",", ":")))
    (OUT / "tables/wp6_confirm_summary.json").write_text((PROJ / "outputs/wp6/confirm_summary.json").read_text())
    poster()
    for p in ["layers/wp2_spikes_synthetic.json", "layers/wp2_exposure_synthetic.json", "wp3_example/pano_equirect.jpg",
              "home/poster.jpg"]:
        print(p, round((OUT / p).stat().st_size / 1e6, 2), "MB")
