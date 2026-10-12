# -*- coding: utf-8 -*-
"""Step 1.1: inventory of the City of Amsterdam panoramas for one year (metadata only).

Pages through https://api.data.amsterdam.nl/panorama/panoramas/ (Kernregistratie
Panoramabeelden, Gemeente Amsterdam, CC BY 4.0) and writes one row per panorama.
Each page is cached in data/raw/panorama_pages_<year>/, so a rerun resumes.
Usage: python 11_inventory_panoramas.py 2025
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
API = "https://api.data.amsterdam.nl/panorama/panoramas/"
PAGE_SIZE = 10000


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    cache = PROJ / "data" / "raw" / f"panorama_pages_{year}"
    cache.mkdir(parents=True, exist_ok=True)
    params = {"format": "json", "page_size": PAGE_SIZE,
              "timestamp_after": f"{year}-01-01", "timestamp_before": f"{year + 1}-01-01"}
    page, count = 1, None
    while True:
        path = cache / f"page_{page:04d}.json"
        if not path.exists():
            for attempt in range(6):
                try:
                    r = requests.get(API, params=dict(params, page=page), timeout=180,
                                     headers={"User-Agent": "elevate-amsterdam-demo (research; worf)"})
                    if r.status_code == 200:
                        path.write_text(r.text)
                        break
                except requests.RequestException:
                    pass
                time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"page {page} failed")
            time.sleep(1)  # be polite to the city's API
        data = json.loads(path.read_text())
        count = data.get("count", count)
        if not data["_links"].get("next", {}).get("href"):
            break
        page += 1
        if page % 10 == 0:
            print(f"page {page}, ~{page * PAGE_SIZE}/{count}", flush=True)
    rows = []
    for p in sorted(cache.glob("page_*.json")):
        for x in json.loads(p.read_text())["_embedded"]["panoramas"]:
            lon, lat, alt = (x["geometry"]["coordinates"] + [None])[:3]
            rows.append({"pano_id": x["pano_id"], "timestamp": x["timestamp"], "lon": lon, "lat": lat, "alt": alt,
                         "roll": x.get("roll"), "pitch": x.get("pitch"), "heading": x.get("heading"),
                         "surface_type": x.get("surface_type"), "mission_type": x.get("mission_type"),
                         "mission_distance": x.get("mission_distance"), "mission_year": x.get("mission_year"),
                         "cubic_img_baseurl": x.get("cubic_img_baseurl")})
    df = pd.DataFrame(rows).drop_duplicates("pano_id")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = PROJ / "data" / "raw" / f"panoramas_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} panoramas (API count {count}) -> {out}")
    print("months:", df["timestamp"].dt.month.value_counts().sort_index().to_dict())
    for col in ("mission_type", "surface_type", "mission_distance"):
        print(col, df[col].value_counts().head(8).to_dict())
    print("extent lon", round(df.lon.min(), 4), round(df.lon.max(), 4), "lat", round(df.lat.min(), 4), round(df.lat.max(), 4))
    print("hour (UTC):", df["timestamp"].dt.hour.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
