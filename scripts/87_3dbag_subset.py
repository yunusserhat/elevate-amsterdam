# -*- coding: utf-8 -*-
"""Step 10.5: a light, self-hosted 3DBAG subset for the home page's 3D city.

3DBAG (v20250903, Cesium 3D Tiles, CC BY 4.0) tiles are already meshopt-compressed, but each also carries a
per-building feature id inside its vertex stream and a 42-column attribute table; the home page only draws the
shapes. This script keeps the tiles whose boxes touch the Amsterdam municipality (plus a margin), strips the ids
and the table and re-encodes the geometry with Draco (tools/gltf/strip.mjs), and writes one tileset.json with
the external tilesets inlined, so the page loads every building from the site's own server.

The home page's camera follows the story, so the tiles it can show are known: --tiles takes the list recorded
while visiting every stop (8 headings each) and the WP2 walk at several screen sizes (step 10.5), and keeps those
plus their neighbours on the same level, so a slightly different screen still finds its tiles.

3DBAG keeps buildings only in the leaves; the inner tiles are empty. From a distance Cesium stops at an inner tile
whose error is small enough for the screen and draws nothing there, so the dense centre (deep leaves) vanished
from the home view on most screens (step 11.8). --crop keeps every leaf that touches a box instead, and
--refine-all gives the empty inner tiles a huge geometric error, so the page always goes down to the leaves and
every building in the crop is drawn at any distance. --no-normals drops the normals (strip.mjs; the page lights
the flat faces itself).

Usage: python 87_3dbag_subset.py [--lod lod22] [--margin-km 1.0] [--tiles recorded.json] [--dry-run]
       python 87_3dbag_subset.py --crop 4.835,52.333,4.955,52.402 --refine-all --no-normals
Writes space/data/3dbag/<lod>/tileset.json and t/<z>/<x>/<y>.glb; originals are cached in data/raw/3dbag/<lod>/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from pyproj import Transformer

PROJ = Path(__file__).resolve().parents[1]
VERSION = "v20250903"
NODE = Path("/scratch/ybicakci/tools/node/bin/node")
ECEF_TO_LONLAT = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
session = requests.Session()


def get(url: str, path: Path) -> Path:
    """Download once into the raw cache (three tries)."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.rename(path)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    return path


def box_lonlat(box: list[float]) -> tuple[float, float, float, float]:
    """Lon/lat bounds of an oriented box given as centre and three half-axis vectors in ECEF."""
    c, a, b, d = np.array(box[:3]), np.array(box[3:6]), np.array(box[6:9]), np.array(box[9:12])
    corners = np.array([c + i * a + j * b + k * d for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    lon, lat, _ = ECEF_TO_LONLAT.transform(corners[:, 0], corners[:, 1], corners[:, 2])
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lod", default="lod22")
    ap.add_argument("--margin-km", type=float, default=1.0)
    ap.add_argument("--tiles", nargs="+", help="JSON lists of tile uris (t/z/x/y.glb) recorded on the home page")
    ap.add_argument("--complete", choices=["none", "leaves", "full"], default="leaves",
                    help="finish the families of kept tiles: a refined parent is replaced by all its children, so a missing sibling is a hole")
    ap.add_argument("--crop", help="w,s,e,n in degrees: keep every leaf touching this box (instead of the municipality and --tiles)")
    ap.add_argument("--refine-all", action="store_true", help="empty inner tiles always refine: every leaf in view is drawn")
    ap.add_argument("--no-normals", action="store_true", help="drop the normals (the page computes flat ones)")
    ap.add_argument("--dry-run", action="store_true", help="select tiles and report, without downloading them")
    args = ap.parse_args()
    base = f"https://data.3dbag.nl/{VERSION}/cesium3dtiles/{args.lod}/"
    raw = PROJ / "data/raw/3dbag" / args.lod
    out = PROJ / "space/data/3dbag" / args.lod

    if args.crop:
        aoi = tuple(float(v) for v in args.crop.split(","))
    else:
        gem = gpd.read_parquet(PROJ / "data/derived/gemeente.parquet").to_crs(4326)
        x0, y0, x1, y1 = gem.total_bounds
        mlat = args.margin_km / 111.0
        mlon = args.margin_km / (111.0 * np.cos(np.radians((y0 + y1) / 2)))
        aoi = (x0 - mlon, y0 - mlat, x1 + mlon, y1 + mlat)

    def touches(node) -> bool:
        bv = node["boundingVolume"]
        if "box" in bv:
            w, s, e, n = box_lonlat(bv["box"])
        elif "region" in bv:
            w, s, e, n = (np.degrees(v) for v in bv["region"][:4])
        else:
            return True
        return not (e < aoi[0] or w > aoi[2] or n < aoi[1] or s > aoi[3])

    tiles = []   # (remote uri, local relative uri)

    def walk(node, url_base: str):
        if not touches(node):
            return None
        new = {k: node[k] for k in ("boundingVolume", "geometricError", "refine", "transform") if k in node}
        children = []
        uri = (node.get("content") or {}).get("uri") or (node.get("content") or {}).get("url")
        if uri and uri.endswith(".json"):
            ext = json.loads(get(url_base + uri, raw / uri).read_text())
            child = walk(ext["root"], url_base + uri.rsplit("/", 1)[0] + "/" if "/" in uri else url_base)
            if child:
                children.append(child)
        elif uri:
            rel = (url_base + uri)[len(base):]
            tiles.append(rel)
            new["content"] = {"uri": rel}
        for ch in node.get("children", []):
            kept = walk(ch, url_base)
            if kept:
                children.append(kept)
        if children:
            new["children"] = children
        return new if ("content" in new or children) else None

    root_json = json.loads(get(base + "tileset.json", raw / "tileset.json").read_text())
    root = walk(root_json["root"], base)
    print(f"AOI {tuple(round(v, 4) for v in aoi)}: {len(tiles)} content tiles")
    if args.tiles and not args.crop:
        wanted = set()
        for f in args.tiles:
            d = json.loads(Path(f).read_text())
            wanted |= set(d if isinstance(d, list) else d.get("tiles", []))
        present = set(tiles)
        keep = set(wanted & present)
        for t in wanted & present:   # neighbours on the same level (the grid step is 2 ** (10 - z))
            _, z, x, y = t[:-4].split("/")
            z, x, y = int(z), int(x), int(y)
            step = 2 ** (10 - z)
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    n = f"t/{z}/{x + dx}/{y + dy}.glb"
                    if n in present:
                        keep.add(n)

        def complete(node, depth_left):
            """A sibling added to finish a family: its content, or, if it is an inner node, its children."""
            new = {k: v for k, v in node.items() if k not in ("children", "content")}
            if node.get("content"):
                keep.add(node["content"]["uri"])
                new["content"] = node["content"]
                return new
            if depth_left == 0:
                return None
            kids = [k for k in (complete(c, depth_left - 1) for c in node.get("children", [])) if k]
            if kids:
                new["children"] = kids
            return new if kids else None

        def prune_tree(node):
            pruned = [prune_tree(c) for c in node.get("children", [])]
            if args.complete != "none" and any(pruned):
                depth = 99 if args.complete == "full" else 0
                pruned = [p if p else complete(c, depth) for p, c in zip(pruned, node.get("children", []))]
            kids = [k for k in pruned if k]
            content = node.get("content")
            if content and content["uri"] not in keep:
                content = None
            if not kids and not content:
                return None
            new = {k: v for k, v in node.items() if k not in ("children", "content")}
            if content:
                new["content"] = content
            if kids:
                new["children"] = kids
            return new

        root = prune_tree(root)
        tiles = sorted(keep)
        print(f"recorded {len(wanted)}, of which in the AOI {len(wanted & present)}; with neighbours: {len(tiles)}")
    if args.dry_run:
        return

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda rel: get(base + rel, raw / rel), tiles))
    pairs = [[str(raw / rel), str(out / rel)] for rel in tiles]
    (raw / "strip_pairs.json").write_text(json.dumps(pairs))
    env = {**os.environ, "PATH": f"{NODE.parent}:{os.environ['PATH']}"}
    opts = {"normals": not args.no_normals, "force": True}
    subprocess.run([str(NODE), "strip.mjs", str(raw / "strip_pairs.json"), json.dumps(opts)], cwd=PROJ / "tools/gltf", env=env, check=True)

    out.mkdir(parents=True, exist_ok=True)
    for f in out.rglob("*.glb"):   # tiles left out of this selection must not reach the site
        if f.relative_to(out).as_posix() not in set(tiles):
            f.unlink()
    geometric_error = root_json["geometricError"]
    if args.refine_all:
        def refine(node):
            if "content" not in node:
                node["geometricError"] = 1e6
            for ch in node.get("children", []):
                refine(ch)
        refine(root)
        geometric_error = 1e6
    what = f"crop {args.crop}" if args.crop else "Amsterdam subset"
    version = f"3DBAG {VERSION} {args.lod}, {what}, metadata removed{', no normals' if args.no_normals else ''}, Draco"
    tileset = {"asset": {**root_json["asset"], "tilesetVersion": version}, "geometricError": geometric_error, "root": root}
    (out / "tileset.json").write_text(json.dumps(tileset, separators=(",", ":")))
    raw_mb = sum((raw / rel).stat().st_size for rel in tiles) / 1e6
    out_mb = sum((out / rel).stat().st_size for rel in tiles) / 1e6
    print(json.dumps({"lod": args.lod, "tiles": len(tiles), "original_mb": round(raw_mb, 1), "stripped_mb": round(out_mb, 1),
                      "tileset_kb": round((out / "tileset.json").stat().st_size / 1024, 1)}))


if __name__ == "__main__":
    main()
