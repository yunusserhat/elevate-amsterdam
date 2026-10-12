# -*- coding: utf-8 -*-
"""Step 8.2 (data): the small files the Hugging Face Space serves, written to space/data/.

* index/: view records (no image paths) and the 256-d EmbeddingGemma 2 vectors as float16;
* layers/: observed street segments and buurten as simplified GeoJSON (WGS84, 6 decimals);
* tables/: WP1, WP2 (synthetic), WP3 evaluation, WP4, WP5 and WP6 tables as CSV/JSON;
* provenance.json: model ids, revisions, processing ids, dates and licences.
Street images are not copied, except the curated WP6 before/after pairs (step 7.3) and the
synthetic person-day figure. Every view carries the City's own image links.

Usage: python 80_space_data.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "space/data"
THUMB = "https://api.data.amsterdam.nl/panorama/thumbnail/{pano}/?heading={h:.0f}"
TIERS = {"sidewalk": "A", "pedestrian_crossing": "A", "seating": "A", "street_trees": "A", "intersection": "A",
         "street_light_pole": "A", "shade_on_walkway": "A", "view_obstruction": "A", "curb_ramp": "B",
         "cycle_infrastructure": "B", "sidewalk_obstruction": "B", "active_frontage": "B", "other_greenery": "B",
         "litter": "C", "sidewalk_parking": "C", "accessibility_barrier": "C",
         "surface_damage": "rare", "traffic_calming": "rare", "sidewalk_gap": "rare"}
# street-segment layer: what the pages map. Walkability comes from the ELEVATE team's own Amsterdam
# perception model (svi_percept, 1-5); the Place Pulse cross-checks are compared in tables, not mapped.
SEG_INDICATORS = {"green_view": "vegetation_pixel_fraction__mean", "perceived_walkable": "svp_walkability__mean"}
SEG_INDICATORS.update({k: f"audit_{k}_state__present_share" for k, t in TIERS.items() if t in ("A", "B")})


def geojson(gdf: gpd.GeoDataFrame, path: Path, tolerance_m: float):
    g = gdf.to_crs("EPSG:28992")
    g["geometry"] = g.geometry.simplify(tolerance_m, preserve_topology=True)
    g = g.to_crs(4326)
    for c in g.columns:
        if c != "geometry" and pd.api.types.is_float_dtype(g[c]):
            g[c] = g[c].round(4)
    path.write_text(g.to_json(drop_id=True, to_wgs84=True).replace("\n", ""))
    js = json.loads(path.read_text())
    for f in js["features"]:  # 6 decimals is ~0.1 m
        f["geometry"] = json.loads(json.dumps(f["geometry"]), parse_float=lambda x: round(float(x), 6))
    path.write_text(json.dumps(js, separators=(",", ":")))


def main():
    for d in ("index", "layers", "tables", "wp6_pairs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # index
    idx = PROJ / "outputs/wp3/index/embeddinggemma2_256"
    rec = pd.read_parquet(idx / "records.parquet")
    rec["city_image"] = [THUMB.format(pano=p, h=h) for p, h in zip(rec.pano_id, rec.heading_deg)]
    keep = ["image_id", "pano_id", "view", "lat", "lon", "heading_deg", "timestamp", "segment_id", "area_id", "buurtnaam",
            "usable", "observation_text", "city_image"]
    rec[keep].to_parquet(OUT / "index/records.parquet", index=False)
    vec = np.load(idx / "vectors.npy").astype(np.float16)
    np.save(OUT / "index/vectors_f16.npy", vec)
    shutil.copy(idx / "index.json", OUT / "index/index.json")

    # layers
    seg = gpd.read_parquet(PROJ / "outputs/wp3/indicators_segment.parquet")
    seg = seg[seg.n_images > 0]
    cols = {"segment_id": "segment_id", "road_group": "road_group", "name": "name", "n_images": "n_panoramas",
            "covered_length_share": "covered_share"}
    cols.update({v: k for k, v in SEG_INDICATORS.items()})
    seg = seg[list(cols) + ["geometry"]].rename(columns=cols)
    geojson(seg, OUT / "layers/segments.geojson", 1.0)
    eq = gpd.read_parquet(PROJ / "outputs/wp5/buurt_equity.parquet")
    sig = gpd.read_parquet(PROJ / "outputs/wp4/signalen_by_buurt.parquet")
    sig = sig[["buurtcode"] + [c for c in sig.columns if c.startswith("reports_") and c.endswith("_per_km")]]
    eq = eq.merge(sig, on="buurtcode", how="left")
    eq["low_income_quintile"] = eq.low_income_quintile.astype(str)
    geojson(eq, OUT / "layers/buurten.geojson", 5.0)
    gaps = gpd.read_parquet(PROJ / "outputs/wp6/gap_segments.parquet")
    geojson(gaps, OUT / "layers/gap_segments.geojson", 1.0)

    # tables
    copies = {
        "outputs/wp1/stakeholder_matrix.csv": "wp1_stakeholder_matrix.csv",
        "outputs/wp1/indicator_longlist.csv": "wp1_indicator_longlist.csv",
        "outputs/wp2/pipeline_qc.json": "wp2_pipeline_qc_synthetic.json",
        "outputs/wp2/synthetic_summary.json": "wp2_summary_synthetic.json",
        "outputs/wp3/eval/item_tiers.csv": "wp3_item_tiers.csv",
        "outputs/wp3/eval/reference_agreement.csv": "wp3_reference_agreement.csv",
        "outputs/wp3/eval/stability.csv": "wp3_stability.csv",
        "outputs/wp3/eval/segmentation_auc.csv": "wp3_segmentation_auc.csv",
        "outputs/wp3/perception_qc/agreement.csv": "wp3_perception_agreement.csv",
        "outputs/wp3/perception_qc/svp_agreement.csv": "wp3_svp_agreement.csv",
        "outputs/wp3/perception_qc/svp_summary.csv": "wp3_svp_summary.csv",
        "outputs/wp3/indicators_summary.json": "wp3_indicators_summary.json",
        "outputs/wp3/coverage_by_road_group_2025.csv": "wp3_coverage_by_road_group.csv",
        "outputs/wp4/signalen_triangulation.csv": "wp4_signalen_triangulation.csv",
        "outputs/wp5/equity_spearman.csv": "wp5_equity_spearman.csv",
        "outputs/wp5/equity_by_low_income_quintile.csv": "wp5_equity_by_quintile.csv",
        "outputs/wp5/moran_global.csv": "wp5_moran_global.csv",
        "outputs/wp5/nbmm_results_synthetic.csv": "wp5_nbmm_results_synthetic.csv",
        "outputs/wp5/nbmm_subsample_summary_synthetic.csv": "wp5_nbmm_subsamples_synthetic.csv",
        "outputs/wp5/nbmm_moderation_synthetic.csv": "wp5_nbmm_moderation_synthetic.csv",
        "outputs/wp5/models_python_synthetic.json": "wp5_models_python_synthetic.json",
        "outputs/wp6/gaps_summary.json": "wp6_gaps_summary.json",
        "outputs/wp6/gap_by_quintile.csv": "wp6_gap_by_quintile.csv",
        "outputs/wp6/edit_verification.csv": "wp6_edit_verification.csv",
        "outputs/wp6/verify_summary.json": "wp6_verify_summary.json",
        "outputs/wp6/qa_precomputed.json": "wp6_qa_precomputed.json",
    }
    missing = []
    for src, dst in copies.items():
        if (PROJ / src).exists():
            shutil.copy(PROJ / src, OUT / "tables" / dst)
        else:
            missing.append(src)

    # JSON copies of the CSV tables, for the pages
    for f in (OUT / "tables").glob("*.csv"):
        pd.read_csv(f).to_json(f.with_suffix(".json"), orient="records", force_ascii=False, indent=0)

    # WP3 story: one example panorama (four views, CC BY 4.0, Gemeente Amsterdam) with its audit,
    # segmentation and perception records, and the sampled panorama points
    import glob
    from PIL import Image
    ex = "recording_2025-07-03_08-35-58_00239"
    (OUT / "wp3_example").mkdir(exist_ok=True)
    audit = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/audit_full_qwen35_4b_v2_shard*.parquet")))])
    audit = audit[audit.image_id.str.startswith(ex)]
    segf = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/segmentation/*.parquet")))])
    segf = segf[segf.image_id.str.startswith(ex)]
    perc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/perception/placepulse_shard*_part*.parquet")))])
    perc = perc[perc.image_id.str.startswith(ex)]
    svpf = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/perception/svipercept_shard*_part*.parquet")))])
    svp_ids = {"model": svpf.svp_model.iloc[0], "clip": svpf.svp_clip.iloc[0]}
    svpf = svpf[svpf.image_id.str.startswith(ex)]
    views = []
    for v in ("front", "right", "back", "left"):
        iid = f"{ex}_{v}"
        Image.open(PROJ / "data/views" / f"{iid}.jpg").convert("RGB").save(OUT / "wp3_example" / f"{v}.jpg", quality=86)
        a = audit[audit.image_id == iid].iloc[0]
        items = json.loads(a.parsed_json)["items"]
        s = segf[segf.image_id == iid].iloc[0]
        q = perc[perc.image_id == iid].iloc[0]
        sv = svpf[svpf.image_id == iid].iloc[0]
        r = rec[rec.image_id == iid].iloc[0]
        views.append({"view": v, "image": f"data/wp3_example/{v}.jpg", "heading_deg": round(float(r.heading_deg), 1),
                      "city_image": r.city_image, "items": items,
                      "segmentation": {c: round(float(s[f"{c}_pixel_fraction"]), 3) for c in ["vegetation", "building", "sky", "road", "sidewalk", "car", "bicycle"]},
                      "perception": {k: round(float(q[f"vit_{k}"]), 2) for k in ["safety", "lively", "wealthy", "beautiful", "boring", "depressing"]},
                      "perception_elevate": {k: round(float(sv[f"svp_{k}"]), 2) for k in ["walkability", "bikeability", "pleasantness", "greenness", "safety"]}})
    r0 = rec[rec.image_id == f"{ex}_front"].iloc[0]
    (OUT / "wp3_example/example.json").write_text(json.dumps({"pano_id": ex, "buurt": r0.buurtnaam, "lat": r0.lat, "lon": r0.lon,
        "timestamp": str(r0.timestamp), "model": "Qwen/Qwen3.5-4B, template active_mobility_audit_v2", "views": views,
        "credit": "Panorama: Gemeente Amsterdam, CC BY 4.0 (perspective views rendered here)"}, ensure_ascii=False))
    pts = rec.drop_duplicates("pano_id")
    (OUT / "layers/wp3_panoramas.json").write_text(json.dumps([[round(a, 5), round(b, 5)] for a, b in zip(pts.lon, pts.lat)], separators=(",", ":")))

    # WP2: two SYNTHETIC person-days as GeoJSON (true path by state, observed fixes, detected mode)
    gps = pd.read_parquet(PROJ / "outputs/wp2/synthetic_gps.parquet")
    fx = pd.read_parquet(PROJ / "outputs/wp2/pipeline_fixes.parquet")
    feats = []
    for pid, day in (("S010", 2), ("S150", 6)):
        g = gps[(gps.person_id == pid) & (gps.day == day)].sort_values("time")
        f = fx[(fx.person_id == pid) & (fx.day == day) & ~fx.interpolated].sort_values("t")
        key = f"{pid} day {day}"
        moving = g[g.true_state.isin(["walk", "bike", "car", "bus"])]
        for state, part in moving.groupby((moving.true_state != moving.true_state.shift()).cumsum()):
            if len(part) > 1:
                feats.append({"type": "Feature", "properties": {"case": key, "layer": "true path", "state": part.true_state.iloc[0]},
                              "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in zip(part.true_lon, part.true_lat)]}})
        for r in g.itertuples():
            feats.append({"type": "Feature", "properties": {"case": key, "layer": "raw fix", "accuracy_m": r.accuracy_m},
                          "geometry": {"type": "Point", "coordinates": [round(r.lon, 6), round(r.lat, 6)]}})
        for r in f.itertuples():
            feats.append({"type": "Feature", "properties": {"case": key, "layer": "cleaned fix", "mode": r.mode},
                          "geometry": {"type": "Point", "coordinates": [round(r.lon, 6), round(r.lat, 6)]}})
    (OUT / "layers/wp2_example_synthetic.geojson").write_text(json.dumps({"type": "FeatureCollection", "synthetic": True, "features": feats}))

    # WP6 pairs: original view (City, CC BY 4.0) and edit, both 768 x 576 JPEG
    ver = PROJ / "outputs/wp6/edit_verification.csv"
    if ver.exists():
        from PIL import Image
        for r in pd.read_csv(ver).itertuples():
            Image.open(PROJ / "data/views" / f"{r.image_id}.jpg").convert("RGB").save(OUT / "wp6_pairs" / f"{r.image_id}__before.jpg", quality=85)
            Image.open(PROJ / "outputs/wp6/edits_768" / f"{r.image_id}__{r.gap_type}.jpg").convert("RGB").save(
                OUT / "wp6_pairs" / f"{r.image_id}__{r.gap_type}__after.jpg", quality=85)

    # provenance
    audit = json.loads((PROJ / "outputs/wp3/audit_full_qwen35_4b_v2_shard0.summary.json").read_text())
    prov = {
        "built": pd.Timestamp.now(tz="Europe/Amsterdam").isoformat(timespec="minutes"),
        "imagery": {"source": "Gemeente Amsterdam, Kernregistratie Panoramabeelden, 2025 recordings", "licence": "CC BY 4.0",
                    "panoramas": int(rec.pano_id.nunique()), "views": int(len(rec))},
        "audit": {"model": audit["model"], "revision": audit["model_revision"], "template": audit["template"],
                  "processing_id": audit["processing_id"], "package": "GeoAI-VLM"},
        "reference": {"model": "Qwen/Qwen3.8-27B-FP8", "use": "agreement checks, WP6 candidate confirmation and edit checks"},
        "segmentation": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "perception": {"elevate": f"{svp_ids['model']} (GPL-3.0); CLIP {svp_ids['clip']} (apple/DFN5B-CLIP-ViT-H-14-378, Apple ML research licence)",
                       "primary": "Jiani11/human-perception-place-pulse@402164b5", "second": "seshing/placepulse@1d7f85c4",
                       "note": "the ELEVATE team's own Amsterdam model (Percept survey; Danish et al. 2025); the two Place Pulse models are cross-checks"},
        "embedding": json.loads((idx / "index.json").read_text())["metadata"],
        "image_edits": {"model": "Qwen/Qwen-Image-2.1-Turbo@d65dbc9a", "licence": "Qwen Research Licence (non-commercial)",
                        "label": "AI-generated visualisation, not a design proposal"},
        "income": "CBS StatLine 85618NED (2023), CC BY 4.0",
        "reports": "Meldingen openbare ruimte Amsterdam (SIA open dataset), CC BY",
        "network": "OpenStreetMap contributors, ODbL",
        "aerial_3d": "PDOK Luchtfoto 2025 orthoHR (Beeldmateriaal Nederland), CC BY 4.0, home page 3D view",
        "buildings_3d": "3DBAG v20250903 LoD2.2 3D Tiles (TU Delft 3D geoinformation and 3DGI), CC BY 4.0, recoloured and lowered 43 m to the ellipsoid",
        "synthetic": "WP2 persons, GPS and steps are synthetic (docs/wp2_synthetic_design.md)",
        "missing_at_build": missing,
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    sizes = {p.relative_to(OUT).as_posix(): round(p.stat().st_size / 1e6, 2) for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > 500_000}
    print(json.dumps({"missing": missing, "large_files_MB": sizes,
                      "total_MB": round(sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6, 1)}, indent=1))


if __name__ == "__main__":
    main()
