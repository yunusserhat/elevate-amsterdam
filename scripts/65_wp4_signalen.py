# -*- coding: utf-8 -*-
"""Step 6.1: triangulating model observations with residents' public-space reports (WP4, light).

Data: "Meldingen over de Openbare Ruimte in Amsterdam" (SIA / Signalen open dataset,
api.data.amsterdam.nl/v1/meldingen, licence "Creative Commons, Naamsvermelding"). It holds a
subset of all reports since mid-2018 for ten main categories. Report locations are
"coordinates optimised for visualisation", so this step works at buurt level only and never
matches single reports to single images.

Window: reports made June-October 2025, the months of the sampled panoramas. Category groups
are matched to audit items; reports per km of walk/cycle network are compared with the buurt
share of views where the 4B model reports the item (Spearman, bootstrap 95%). Report rates are
also compared with the CBS low-income share, because reporting is a behaviour as well as a
signal of conditions. Dense central buurten have both more reports and more of most observed
items, so a partial Spearman correlation given address density (CBS omgevingsadressendichtheid)
is reported next to the plain one.

Usage: python 65_wp4_signalen.py
Writes data/raw/sia_meldingen_2025_jun_oct.parquet and outputs/wp4/{signalen_by_buurt.parquet,
signalen_triangulation.csv, signalen_summary.json}.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
API = "https://api.data.amsterdam.nl/v1/meldingen/meldingen/"
FIELDS = "id,hoofdcategorie,subcategorie,datumMelding,externeStatus,gbdBuurtCode,gbdBuurtNaam,latitudeVisualisatie,longitudeVisualisatie"
GROUPS = {  # report subcategories (names as in the open dataset) -> the audit item or indicator they speak to
    "litter": (["Veeg- / zwerfvuil", "Grof afval", "Container bijplaatsing", "Huisafval", "Gedumpt gevaarlijk afval",
                "Puin- / sloopafval", "Bruin- en witgoed"], "audit_litter_state__present_share"),
    "obstruction": (["Hinderlijk geplaatst object", "Fietswrak", "Auto- / scooter- / bromfiets(wrak)", "Deelscooters en -(bak)fietsen"],
                    "audit_sidewalk_obstruction_state__present_share"),
    "parking": (["Parkeeroverlast"], "audit_sidewalk_parking_state__present_share"),
    "pavement": (["Onderhoud stoep, rijweg en parkeerplaats", "Onderhoud fietspad"], "audit_surface_damage_state__present_share"),
    "trees": (["Boom - ontbreekt", "Boom - dood", "Boom - overig", "Boom - gevaarlijke situatie", "Boom - verzoek inspectie"],
              "vegetation_pixel_fraction__mean"),
    "street_furniture": (["Straatmeubilair"], "audit_seating_state__present_share"),
    "street_lighting": (["Straatverlichting"], "audit_street_light_pole_state__present_share"),
}
SEED = 20261011


def download() -> pd.DataFrame:
    out = PROJ / "data/raw/sia_meldingen_2025_jun_oct.parquet"
    if out.exists():
        return pd.read_parquet(out)
    frames, page = [], 1
    while True:
        r = requests.get(API, params={"datumMelding[gte]": "2025-06-01", "datumMelding[lt]": "2025-11-01", "_format": "csv",
                                      "_fields": FIELDS, "_pageSize": 20000, "page": page}, timeout=600)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            break
        frames.append(df)
        if len(df) < 20000:
            break
        page += 1
    d = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    d.to_parquet(out, index=False)
    return d


def spearman_ci(x, y, rng, b=2000):
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    boots = [pd.Series(x[i]).corr(pd.Series(y[i]), method="spearman") for i in (rng.integers(0, len(x), len(x)) for _ in range(b))]
    return rho, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5), int(len(x))


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Spearman correlation of x and y after removing a linear effect of z from both ranks."""
    ok = x.notna() & y.notna() & z.notna()
    rx, ry, rz = (s[ok].rank().to_numpy() for s in (x, y, z))
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main():
    rng = np.random.default_rng(SEED)
    d = download()
    d["buurtcode"] = "BU0363" + d.gbdBuurtCode.astype(str)
    eq = gpd.read_parquet(PROJ / "outputs/wp5/buurt_equity.parquet")
    ind = gpd.read_parquet(PROJ / "outputs/wp3/indicators_buurt.parquet")
    dens = gpd.read_parquet(PROJ / "data/derived/buurten.parquet")[["buurtcode", "omgevingsadressendichtheid"]]
    b = eq[["buurtcode", "buurtnaam", "network_km", "covered_km", "n_panoramas", "low_income_households", "geometry"]].merge(dens, on="buurtcode", how="left").merge(
        ind[["buurtcode"] + sorted({c for _, c in GROUPS.values()})], on="buurtcode", how="left")
    names = d.drop_duplicates("buurtcode").set_index("buurtcode").gbdBuurtNaam
    matched = b.buurtcode.map(names)
    name_match = float((matched.str.lower() == b.buurtnaam.str.lower()).mean())
    rows = []
    for g, (subs, col) in GROUPS.items():
        n = d[d.subcategorie.isin(subs)].groupby("buurtcode").size()
        b[f"reports_{g}"] = b.buurtcode.map(n).fillna(0).astype(int)
        b[f"reports_{g}_per_km"] = b[f"reports_{g}"] / b.network_km.replace(0, np.nan)
        rho, lo, hi, k = spearman_ci(b[f"reports_{g}_per_km"], b[col], rng)
        rho2, lo2, hi2, _ = spearman_ci(b[f"reports_{g}_per_km"], b.low_income_households, rng)
        rows.append({"report_group": g, "subcategories": "; ".join(subs), "reports_jun_oct_2025": int(d.subcategorie.isin(subs).sum()),
                     "model_indicator": col, "spearman_with_model": rho, "ci_lo": lo, "ci_hi": hi,
                     "partial_given_address_density": partial_spearman(b[f"reports_{g}_per_km"], b[col], b.omgevingsadressendichtheid),
                     "spearman_with_low_income": rho2, "ci_lo_income": lo2, "ci_hi_income": hi2, "n_buurten": k})
    tri = pd.DataFrame(rows)
    out = PROJ / "outputs/wp4"
    out.mkdir(parents=True, exist_ok=True)
    tri.round(4).to_csv(out / "signalen_triangulation.csv", index=False)
    b.to_parquet(out / "signalen_by_buurt.parquet", index=False)
    summary = {"reports_jun_oct_2025": int(len(d)), "buurten": int(len(b)), "buurt_name_match_share": round(name_match, 3),
               "status_counts": d.externeStatus.value_counts().to_dict(),
               "top_subcategories": d.subcategorie.value_counts().head(15).to_dict(),
               "source": "api.data.amsterdam.nl/v1/meldingen (SIA open dataset), CC BY ('Creative Commons, Naamsvermelding')"}
    (out / "signalen_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(tri.drop(columns=["subcategories"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
