# -*- coding: utf-8 -*-
"""Steps 4.1-4.2: equity of street-level indicators across Amsterdam buurten, and spatial clustering.

Income: CBS StatLine table 85618NED "Kerncijfers wijken en buurten 2023" (CC BY 4.0, source:
Statistics Netherlands), fetched from the OData API:
* low40_households: share of households in the national lowest 40% of disposable household
  income (k_40HuishoudensMetLaagsteInkomen_85); this is the variable behind PDOK's
  percentageHuishoudensMetLaagInkomen, which is rounded there;
* low_income_households: share of households below the low-income threshold
  (HuishoudensMetEenLaagInkomen_87);
* mean income per inhabitant (GemiddeldInkomenPerInwoner_81, x 1,000 euro).
CBS leaves small buurten empty; they stay missing.

Indicators: WP3 buurt values (step 2.6), audit items of tier A or B only (step 2.7), the
green view (vegetation fraction), perceived qualities (Place Pulse ViT, a stand-in), and
data coverage (share of the walk/cycle network observed in the sample; share covered by the
2025 archive). Buurten need >= 10 sampled panoramas.

4.1: indicator means by quintile of low_income_households, the primary deprivation measure
     (it leaves out student households and households without a full year of income; the
     lowest-40% share is inflated in the centre by small and student households: Begijnhof
     60% lowest-40% against 7.4% low income); Spearman correlations with all three income
     measures, with bootstrap 95% intervals (2,000 resamples of buurten).
4.2: global Moran's I (999 permutations) and LISA clusters, k = 6 nearest buurt centroids,
     row-standardised; LISA at p < 0.05, without multiple-testing correction (exploratory).
Usage: python 50_wp5_equity.py
Writes outputs/wp5/*.csv, buurt_equity.parquet and maps (PNG). Model observations, not measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from esda.moran import Moran, Moran_Local  # noqa: E402
from libpysal.weights import KNN  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "outputs/wp5"
CBS = "https://opendata.cbs.nl/ODataApi/odata/85618NED/TypedDataSet"
CBS_FIELDS = {"k_40HuishoudensMetLaagsteInkomen_85": "low40_households", "HuishoudensMetEenLaagInkomen_87": "low_income_households",
              "GemiddeldInkomenPerInwoner_81": "mean_income_per_inhabitant", "AantalInwoners_5": "inhabitants_cbs"}
INDICATORS = {
    "green_view": ("vegetation_pixel_fraction__mean", "green view (vegetation share of the view)"),
    "street_trees": ("audit_street_trees_state__present_share", "street trees reported (A)"),
    "sidewalk": ("audit_sidewalk_state__present_share", "sidewalk reported (A)"),
    "pedestrian_crossing": ("audit_pedestrian_crossing_state__present_share", "pedestrian crossing reported (A)"),
    "seating": ("audit_seating_state__present_share", "seating reported (A)"),
    "shade_on_walkway": ("audit_shade_on_walkway_state__present_share", "shade on walkway reported (A)"),
    "street_light_pole": ("audit_street_light_pole_state__present_share", "street light reported (A)"),
    "cycle_infrastructure": ("audit_cycle_infrastructure_state__present_share", "cycle infrastructure reported (B)"),
    "perceived_beautiful": ("vit_beautiful__mean", "perceived beautiful (Place Pulse model)"),
    "perceived_safe": ("vit_safety__mean", "perceived safe (Place Pulse model)"),
    "perceived_lively": ("vit_lively__mean", "perceived lively (Place Pulse model)"),
    "sample_coverage": ("covered_share", "network share observed in the sample"),
    "archive_coverage": ("archive_covered_share", "network share covered by the 2025 archive"),
    # the ELEVATE team's own Amsterdam perception model (svi_percept, 1-5); appended last so that the
    # bootstrap draws of the indicators above stay as they were
    "perceived_walkable": ("svp_walkability__mean", "walkability (ELEVATE team's svi_percept model)"),
    "perceived_bikeable": ("svp_bikeability__mean", "bikeability (ELEVATE team's svi_percept model)"),
    "perceived_pleasant": ("svp_pleasantness__mean", "pleasantness (ELEVATE team's svi_percept model)"),
    "perceived_green": ("svp_greenness__mean", "greenness (ELEVATE team's svi_percept model)"),
    "perceived_safe_elevate": ("svp_safety__mean", "safety (ELEVATE team's svi_percept model)"),
}
SEED = 20261011
LISA_LABELS = {1: "high-high", 2: "low-high", 3: "low-low", 4: "high-low"}


def fetch_cbs(refresh: bool = False) -> pd.DataFrame:
    cached = PROJ / "data/raw/cbs_85618NED_amsterdam_buurten.parquet"
    if cached.exists() and not refresh:  # the copy fetched for the first run, so reruns stay reproducible
        return pd.read_parquet(cached)
    params = {"$format": "json", "$filter": "startswith(WijkenEnBuurten,'BU0363')",
              "$select": ",".join(["WijkenEnBuurten"] + list(CBS_FIELDS))}
    rows = requests.get(CBS, params=params, timeout=120).json()["value"]
    df = pd.DataFrame(rows).rename(columns=CBS_FIELDS)
    df["buurtcode"] = df.WijkenEnBuurten.str.strip()
    out = PROJ / "data/raw/cbs_85618NED_amsterdam_buurten.parquet"
    df.drop(columns=["WijkenEnBuurten"]).to_parquet(out, index=False)
    return df.drop(columns=["WijkenEnBuurten"])


def boot_spearman(x: pd.Series, y: pd.Series, rng, b: int = 2000):
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    boots = []
    for _ in range(b):
        i = rng.integers(0, len(x), len(x))
        boots.append(np.corrcoef(pd.Series(x[i]).rank(), pd.Series(y[i]).rank())[0, 1])
    return rho, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5), int(len(x))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cbs = fetch_cbs()
    b = gpd.read_parquet(PROJ / "outputs/wp3/indicators_buurt.parquet")
    pano = pd.read_parquet(PROJ / "outputs/wp3/indicators_panorama.parquet", columns=["area_id"])
    b["n_panoramas"] = b.buurtcode.map(pano.area_id.value_counts()).fillna(0).astype(int)
    arch = pd.read_csv(PROJ / "outputs/wp3/coverage_by_buurt_2025.csv")[["area_id", "covered_length_share"]]
    b["archive_covered_share"] = b.buurtcode.map(arch.set_index("area_id").covered_length_share)
    b = b.merge(cbs, on="buurtcode", how="left")
    use = b[(b.n_panoramas >= 10) & b.low_income_households.notna()].copy()
    use["low_income_quintile"] = pd.qcut(use.low_income_households, 5, labels=["Q1 (fewest)", "Q2", "Q3", "Q4", "Q5 (most)"])
    for k, (col, _) in INDICATORS.items():
        use[k] = use[col]
    keep = ["buurtcode", "buurtnaam", "wijkcode", "n_panoramas", "covered_km", "network_km", "low40_households",
            "low_income_households", "mean_income_per_inhabitant", "low_income_quintile"] + list(INDICATORS) + ["geometry"]
    use = gpd.GeoDataFrame(use[keep], geometry="geometry", crs=b.crs)

    # 4.1 quintile table and correlations
    q = use.groupby("low_income_quintile", observed=True)[list(INDICATORS)].mean()
    q.insert(0, "n_buurten", use.groupby("low_income_quintile", observed=True).size())
    q.insert(1, "low_income_range", use.groupby("low_income_quintile", observed=True).low_income_households.agg(lambda s: f"{s.min():.1f}-{s.max():.1f}"))
    q.round(4).to_csv(OUT / "equity_by_low_income_quintile.csv")
    rows = []
    for k, (col, label) in INDICATORS.items():
        for inc in ("low40_households", "low_income_households", "mean_income_per_inhabitant"):
            rho, lo, hi, n = boot_spearman(use[k], use[inc], rng)
            rows.append({"indicator": k, "label": label, "income_measure": inc, "spearman": rho, "ci_lo": lo, "ci_hi": hi, "n_buurten": n})
    sp = pd.DataFrame(rows)
    sp.round(4).to_csv(OUT / "equity_spearman.csv", index=False)

    # 4.2 Moran's I and LISA (k = 6 nearest centroids)
    pts = use.to_crs("EPSG:28992").copy()
    pts["geometry"] = pts.geometry.centroid
    w = KNN.from_dataframe(pts, k=6)
    w.transform = "r"
    mor, lisa_cols = [], {}
    for k in list(INDICATORS) + ["low_income_households", "low40_households"]:
        y = use[k].to_numpy(float)
        if np.isnan(y).any():
            y = np.where(np.isnan(y), np.nanmean(y), y)
        np.random.seed(SEED)
        m = Moran(y, w, permutations=999)
        mor.append({"variable": k, "moran_I": m.I, "expected_I": m.EI, "p_sim": m.p_sim, "z_sim": m.z_sim})
        np.random.seed(SEED)
        lm = Moran_Local(y, w, permutations=999)
        lisa_cols[f"lisa_{k}"] = np.where(lm.p_sim < 0.05, pd.Series(lm.q).map(LISA_LABELS).to_numpy(), "not significant")
    pd.DataFrame(mor).round(4).to_csv(OUT / "moran_global.csv", index=False)
    for c, v in lisa_cols.items():
        use[c] = v
    # equity-priority flag for WP6: low-low green view cluster in the two most low-income quintiles
    use["priority_low_green_low_income"] = (use.lisa_green_view == "low-low") & use.low_income_quintile.isin(["Q4", "Q5 (most)"])
    use["SOURCE_NOTE"] = "model observations of 2025 street views; CBS 2023 (CC BY 4.0)"
    use.to_parquet(OUT / "buurt_equity.parquet", index=False)

    # maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    base = b.to_crs("EPSG:28992")
    shown = use.to_crs("EPSG:28992")
    for ax, (col, title, cmap) in zip(axes.ravel(), [
            ("low_income_households", "Low-income households (%), CBS 2023", "Purples"),
            ("green_view", "Green view: vegetation share of street views (model)", "Greens"),
            ("seating", "Share of views with seating reported (model, tier A)", "Oranges"),
            ("perceived_walkable", "Walkability, ELEVATE team's model (svi_percept)", "Blues"),
            ("sample_coverage", "Share of walk/cycle network observed (sample)", "Greys"),
            ("lisa_green_view", "LISA clusters of green view (p < 0.05, exploratory)", None)]):
        base.plot(ax=ax, color="0.93", edgecolor="white", linewidth=0.2)
        if cmap:
            shown.plot(ax=ax, column=col, cmap=cmap, legend=True, linewidth=0.1, edgecolor="white",
                       legend_kwds={"shrink": 0.6})
        else:
            colours = {"high-high": "#1b7837", "low-low": "#762a83", "high-low": "#a6dba0", "low-high": "#c2a5cf", "not significant": "0.85"}
            for lab, g in shown.groupby(col):
                g.plot(ax=ax, color=colours[lab], linewidth=0.1, edgecolor="white", label=lab)
            pr = shown[shown.priority_low_green_low_income]
            pr.boundary.plot(ax=ax, color="black", linewidth=0.8)
            ax.legend(fontsize=7, loc="lower left")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.suptitle("Amsterdam buurten with >= 10 sampled panoramas. Street-level values are model observations, not measurements.", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "equity_maps.png", dpi=120)

    summary = {"buurten_used": int(len(use)), "buurten_total": int(len(b)),
               "priority_buurten": use.loc[use.priority_low_green_low_income, ["buurtcode", "buurtnaam"]].to_dict("records"),
               "cbs_table": "85618NED (Kerncijfers wijken en buurten 2023), fetched via OData"}
    (OUT / "equity_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(q.round(3).to_string())
    print(sp.pivot(index="indicator", columns="income_measure", values="spearman").round(3).to_string())
    print(pd.DataFrame(mor).round(3).to_string(index=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
