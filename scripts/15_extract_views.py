# -*- coding: utf-8 -*-
"""Step 1.5: four perspective views per sampled panorama (front, right, back, left).

For each panorama, one north-aligned equirectangular image (4000 x 2000) is downloaded
from the City of Amsterdam (CC BY 4.0), four views with a 90 degree horizontal field of
view are rendered around the driving bearing, and the panorama is discarded. Views are
768 x 576: the square 90 degree render cropped below about -27 degrees, where the
recording car's roof starts. A manifest records path, size and SHA-256 per view.
Resumable: panoramas whose four views exist are skipped.
Usage: python 15_extract_views.py [--limit N] [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
from pano_views import perspective  # noqa: E402

VIEWS = PROJ / "data" / "views"
SIZE, CROP_H = 768, 576
DIRS = ("front", "right", "back", "left")


def equirect_url(cubic_base: str) -> str:
    return cubic_base.rstrip("/").rsplit("/cubic", 1)[0] + "/equirectangular/panorama_4000.jpg"


def work(row: dict) -> list:
    paths = [VIEWS / f"{row['pano_id']}_{d}.jpg" for d in DIRS]
    if all(p.exists() for p in paths):
        return [dict(pano_id=row["pano_id"], view=d, path=str(p), status="exists") for d, p in zip(DIRS, paths)]
    for attempt in range(5):
        try:
            r = requests.get(equirect_url(row["cubic_img_baseurl"]), timeout=120,
                             headers={"User-Agent": "elevate-amsterdam-demo (research; worf)"})
            if r.status_code == 200:
                equi = Image.open(io.BytesIO(r.content))
                equi.load()
                break
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    else:
        return [dict(pano_id=row["pano_id"], view=d, path=None, status="download_failed") for d in DIRS]
    out = []
    for d, p in zip(DIRS, paths):
        view = perspective(equi, row[f"yaw_{d}"], size=SIZE).crop((0, 0, SIZE, CROP_H))
        buf = io.BytesIO()
        view.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()
        p.write_bytes(data)
        out.append(dict(pano_id=row["pano_id"], view=d, yaw=round(float(row[f"yaw_{d}"]), 1), path=str(p),
                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), status="ok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", default=str(PROJ / "data" / "derived" / "sample_25k.parquet"))
    args = ap.parse_args()
    VIEWS.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(args.sample)
    if args.limit:
        sample = sample.head(args.limit)
    rows = sample.to_dict(orient="records")
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, r) for r in rows]
        for i, f in enumerate(as_completed(futures), 1):
            results.extend(f.result())
            if i % 500 == 0:
                print(f"{i}/{len(rows)} panoramas, {time.time() - t0:.0f}s", flush=True)
    man = pd.DataFrame(results)
    path = PROJ / "data" / "derived" / ("views_manifest.parquet" if not args.limit else f"views_manifest_first{args.limit}.parquet")
    man.to_parquet(path, index=False)
    print(man["status"].value_counts().to_dict(), f"in {time.time() - t0:.0f}s ->", path)


if __name__ == "__main__":
    main()
