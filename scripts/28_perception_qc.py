# -*- coding: utf-8 -*-
"""Step 2.4 check: Place Pulse scores, two model families, sanity checks.

* score distribution per question and view direction;
* agreement of the two families (Spearman) per view and per panorama (mean of four views);
* agreement between the front and back views of the same panorama;
* Spearman with Cityscapes class fractions (vegetation, sky, building, road, car);
* contact sheets with the 8 highest and 8 lowest views per question (ViT), for a visual check.

Usage: python 28_perception_qc.py
Writes outputs/wp3/perception_qc/{summary.csv, agreement.csv, seg_correlation.csv, sheet_*.jpg}.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "outputs/wp3/perception_qc"
QUESTIONS = ["safety", "lively", "wealthy", "beautiful", "boring", "depressing"]
SEG = ["vegetation", "sky", "building", "road", "sidewalk", "car"]


def load() -> pd.DataFrame:
    p = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/perception/placepulse_shard*_part*.parquet")))],
                  ignore_index=True)
    if p.image_id.duplicated().any() or len(p) != 99100:
        raise ValueError(f"{len(p)} rows, duplicates: {p.image_id.duplicated().sum()}")
    p["pano_id"], p["view"] = p.image_id.str.rsplit("_", n=1).str[0], p.image_id.str.rsplit("_", n=1).str[1]
    return p


def sheet(df: pd.DataFrame, col: str, path: Path, n: int = 8, w: int = 256):
    """Two rows: n highest, n lowest views by col, with their scores."""
    h = w * 3 // 4
    img = Image.new("RGB", (n * w, 2 * (h + 18)), "white")
    draw = ImageDraw.Draw(img)
    for r, part in enumerate([df.nlargest(n, col), df.nsmallest(n, col)]):
        for c, (_, row) in enumerate(part.iterrows()):
            tile = Image.open(PROJ / "data/views" / f"{row.image_id}.jpg").convert("RGB").resize((w, h))
            img.paste(tile, (c * w, r * (h + 18) + 18))
            draw.text((c * w + 4, r * (h + 18) + 3), f"{row[col]:.2f} {row.view}", fill="black")
    img.save(path, quality=85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = load()
    rows = []
    for fam in ("vit", "r50"):
        for q in QUESTIONS:
            col = f"{fam}_{q}"
            for view, g in [("all", p)] + list(p.groupby("view")):
                rows.append({"family": fam, "question": q, "view": view, "mean": g[col].mean(), "sd": g[col].std(),
                             "p05": g[col].quantile(0.05), "p50": g[col].median(), "p95": g[col].quantile(0.95)})
    pd.DataFrame(rows).round(3).to_csv(OUT / "summary.csv", index=False)

    pano = p.groupby("pano_id")[[f"{f}_{q}" for f in ("vit", "r50") for q in QUESTIONS]].mean()
    wide = p.pivot(index="pano_id", columns="view", values=[f"vit_{q}" for q in QUESTIONS])
    agree = []
    for q in QUESTIONS:
        agree.append({"question": q,
                      "vit_vs_r50_view": p[f"vit_{q}"].corr(p[f"r50_{q}"], method="spearman"),
                      "vit_vs_r50_pano": pano[f"vit_{q}"].corr(pano[f"r50_{q}"], method="spearman"),
                      "vit_front_vs_back": wide[(f"vit_{q}", "front")].corr(wide[(f"vit_{q}", "back")], method="spearman"),
                      "vit_left_vs_right": wide[(f"vit_{q}", "left")].corr(wide[(f"vit_{q}", "right")], method="spearman")})
    pd.DataFrame(agree).round(3).to_csv(OUT / "agreement.csv", index=False)
    vit_corr = p[[f"vit_{q}" for q in QUESTIONS]].corr(method="spearman").round(3)
    vit_corr.to_csv(OUT / "vit_between_questions.csv")

    seg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROJ / "outputs/wp3/segmentation/*.parquet")))])
    m = p.merge(seg[["image_id"] + [f"{s}_pixel_fraction" for s in SEG]], on="image_id", how="inner", validate="one_to_one")
    sc = pd.DataFrame({s: [m[f"{fam}_{q}"].corr(m[f"{s}_pixel_fraction"], method="spearman") for fam in ("vit", "r50") for q in QUESTIONS]
                       for s in SEG}, index=[f"{fam}_{q}" for fam in ("vit", "r50") for q in QUESTIONS])
    sc.round(3).to_csv(OUT / "seg_correlation.csv")

    for q in QUESTIONS:
        sheet(p, f"vit_{q}", OUT / f"sheet_vit_{q}.jpg")
    print(pd.DataFrame(agree).round(3).to_string(index=False))
    print(vit_corr.to_string())
    print(sc.round(2).to_string())
    print(pd.DataFrame(rows).query("view == 'all'").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
