# -*- coding: utf-8 -*-
"""Step 4.3 (Python part): the protocol's WP5 model chain on the SYNTHETIC WP2 data. A pipeline
test, not findings.

1. Model input: valid half-hours (valid day, person with >= 4 valid days, phone worn the whole
   half-hour) with observed walking/cycling exposure (>= 2 exposed fixes). Green view is
   standardised over these half-hours. Walking and cycling minutes per half-hour come from the
   detected modes; they are the confounders a DAG would point to (half-hours spent walking have
   more steps and a different green view than half-hours spent cycling). Park-stay minutes
   (stay fixes within 50 m of an OSM park point) are a third: a park visit brings both green
   view on the way and many steps on the spot. A first fit without them showed a spurious
   green-view effect in variant A (+0.04), which this covariate is meant to remove.
   -> outputs/wp5/model_input_halfhours.csv (read by 52_wp5_nbmm.R).
2. ST-DBSCAN (Birant & Kut 2007) per person on cleaned, observed fixes: spatial eps 50 m,
   temporal eps 10 min, min 10 fixes. Clusters are stationary places, noise is movement.
   Scored against the synthetic truth and compared with the stay-point rule of step 3.4;
   steps per 30 s inside clusters and on paths; the most and least active clusters.
3. XGBoost (Poisson objective) with person-grouped 5-fold CV, then SHAP (TreeExplainer) for
   steps per half-hour in variants A (no effect) and B (planted green-view effect).
Usage: envs/stats/bin/python 51_wp5_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
W2, OUT = PROJ / "outputs/wp2", PROJ / "outputs/wp5"
SEED = 20261011
R_EARTH = 6371000.0


def model_input() -> pd.DataFrame:
    hh = pd.read_parquet(W2 / "pipeline_halfhours.parquet")
    fx = pd.read_parquet(W2 / "pipeline_fixes.parquet", columns=["person_id", "day", "halfhour", "mode", "stay", "lon", "lat"])
    mins = fx[fx["mode"].isin(["walk", "bike"])].groupby(["person_id", "day", "halfhour", "mode"]).size().unstack(fill_value=0) * 0.5
    mins = mins.rename(columns={"walk": "walk_min", "bike": "bike_min"})
    parks = pd.read_parquet(PROJ / "data/derived/wp2_pois.parquet", columns=["poi_class", "geometry"])
    parks = parks[parks.poi_class == "park"]
    from shapely import wkb
    pxy = np.array([wkb.loads(g).coords[0] for g in parks.geometry])
    lat0 = np.radians(52.37)
    to_m = lambda lon, lat: np.c_[np.radians(lon) * R_EARTH * np.cos(lat0), np.radians(lat) * R_EARTH]  # noqa: E731
    st = fx[fx.stay]
    dist, _ = cKDTree(to_m(pxy[:, 0], pxy[:, 1])).query(to_m(st.lon.to_numpy(), st.lat.to_numpy()))
    park_min = st[dist <= 50].groupby(["person_id", "day", "halfhour"]).size().rename("park_min") * 0.5
    mins = mins.join(park_min, how="outer")
    persons = pd.read_parquet(W2 / "synthetic_persons.parquet")[["person_id", "stratum", "gender", "age_group", "employed", "car"]]
    d = hh[hh.valid_day & hh.valid_person & (hh.wear_epochs == 60) & (hh.n_exposed_fixes >= 2)]
    d = d.merge(mins.reset_index(), on=["person_id", "day", "halfhour"], how="left").merge(persons, on="person_id")
    d[["walk_min", "bike_min", "park_min"]] = d[["walk_min", "bike_min", "park_min"]].fillna(0.0)
    d["hour"] = 7 + d.halfhour // 2
    d["weekend"] = (d.day >= 6).astype(int)
    d["z_green"] = (d.x_green_view - d.x_green_view.mean()) / d.x_green_view.std()
    cols = ["person_id", "day", "halfhour", "hour", "weekend", "stratum", "gender", "age_group", "employed", "car",
            "steps_A_null", "steps_B_planted", "walk_min", "bike_min", "park_min", "x_green_view", "z_green", "z_true",
            "x_street_trees", "x_sidewalk", "x_beautiful", "x_safety", "n_exposed_fixes"]
    d = d[cols]
    d.to_csv(OUT / "model_input_halfhours.csv", index=False)
    return d


def st_dbscan(x, y, t, eps_s=50.0, eps_t=600.0, min_pts=10) -> np.ndarray:
    """Labels (-1 = noise). Neighbours: within eps_s metres and eps_t seconds; t sorted."""
    n = len(t)
    labels = np.full(n, -2)
    lo = np.searchsorted(t, t - eps_t, side="left")
    hi = np.searchsorted(t, t + eps_t, side="right")

    def neigh(i):
        j = np.arange(lo[i], hi[i])
        return j[np.hypot(x[j] - x[i], y[j] - y[i]) <= eps_s]

    c = 0
    for i in range(n):
        if labels[i] != -2:
            continue
        nb = neigh(i)
        if len(nb) < min_pts:
            labels[i] = -1
            continue
        labels[i] = c
        seeds = list(nb)
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if labels[j] == -1:
                labels[j] = c
            if labels[j] != -2:
                continue
            labels[j] = c
            nj = neigh(j)
            if len(nj) >= min_pts:
                seeds.extend(nj.tolist())
        c += 1
    return labels


def run_st_dbscan() -> dict:
    fx = pd.read_parquet(W2 / "pipeline_fixes.parquet")
    fx = fx[~fx.interpolated].copy()
    lat0 = np.radians(fx.lat.mean())
    fx["xm"] = np.radians(fx.lon) * R_EARTH * np.cos(lat0)
    fx["ym"] = np.radians(fx.lat) * R_EARTH
    fx["tg"] = (fx.day - 1) * 86400 + fx.t
    fx = fx.sort_values(["person_id", "tg"])
    lab = np.empty(len(fx), dtype=np.int64)
    pos = 0
    for _, g in fx.groupby("person_id", sort=False):
        lab[pos:pos + len(g)] = st_dbscan(g.xm.to_numpy(), g.ym.to_numpy(), g.tg.to_numpy())
        pos += len(g)
    fx["cluster"] = lab
    gps = pd.read_parquet(W2 / "synthetic_gps.parquet", columns=["person_id", "day", "time", "true_state"])
    gps["t"] = (gps.time - gps.time.dt.normalize()).dt.total_seconds() - 7 * 3600
    fx = fx.merge(gps[["person_id", "day", "t", "true_state"]], on=["person_id", "day", "t"], how="left")
    true_stat = ~fx.true_state.isin(["walk", "bike", "car", "bus"])
    pred_st = fx.cluster >= 0
    steps = pd.read_parquet(W2 / "synthetic_steps_A_null.parquet", columns=["person_id", "day", "epoch", "steps"])
    fx = fx.merge(steps, on=["person_id", "day", "epoch"], how="left")
    fx["cid"] = fx.person_id + "_" + fx.cluster.astype(str)
    cl = fx[pred_st].groupby("cid").agg(n=("t", "size"), steps_per_30s=("steps", "mean"), lon=("lon", "mean"), lat=("lat", "mean"),
                                        true_activity=("true_state", lambda s: s.mode().iat[0] if len(s.mode()) else None))
    cl = cl[cl.n >= 20]
    top = cl.nlargest(max(1, len(cl) // 10), "steps_per_30s").true_activity.value_counts(normalize=True).round(3).to_dict()
    bottom = cl.nsmallest(max(1, len(cl) // 10), "steps_per_30s").true_activity.value_counts(normalize=True).round(3).to_dict()
    cl.reset_index().assign(SYNTHETIC=True).to_parquet(OUT / "stdbscan_clusters.parquet", index=False)
    res = {
        "fixes": int(len(fx)), "clusters": int(fx.loc[pred_st, "cid"].nunique()),
        "stationary_agreement_with_truth": round(float((pred_st == true_stat).mean()), 3),
        "stationary_precision": round(float(true_stat[pred_st].mean()), 3),
        "stationary_recall": round(float(pred_st[true_stat].mean()), 3),
        "agreement_with_stay_rule_3_4": round(float((pred_st == fx.stay).mean()), 3),
        "steps_per_30s_in_clusters": round(float(fx.loc[pred_st, "steps"].mean()), 2),
        "steps_per_30s_on_paths": round(float(fx.loc[~pred_st, "steps"].mean()), 2),
        "most_active_decile_true_activity": top, "least_active_decile_true_activity": bottom,
    }
    return res


def run_xgb_shap(d: pd.DataFrame) -> dict:
    feats = ["x_green_view", "x_street_trees", "x_sidewalk", "x_beautiful", "x_safety", "walk_min", "bike_min", "park_min", "hour", "weekend"]
    X = d[feats].assign(stratum=d.stratum.map({"low": 0, "middle": 1, "high": 2}))
    groups = d.person_id.to_numpy()
    params = dict(objective="count:poisson", n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, random_state=SEED, n_jobs=8)
    out, shap_curves = {}, {}
    for v in ("A_null", "B_planted"):
        y = d[f"steps_{v}"].to_numpy()
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            m = xgb.XGBRegressor(**params).fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
        dev = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1) / pred), 0) - (y - pred))
        null = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1) / y.mean()), 0) - (y - y.mean()))
        m = xgb.XGBRegressor(**params).fit(X, y)
        sv = shap.TreeExplainer(m).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
        bins = pd.qcut(X.x_green_view, 10, duplicates="drop")
        curve = pd.Series(sv[:, X.columns.get_loc("x_green_view")]).groupby(bins.to_numpy(), observed=True).mean()
        shap_curves[v] = (X.x_green_view.groupby(bins, observed=True).median().to_numpy(), curve.to_numpy())
        out[v] = {"cv_deviance_explained": round(float(1 - dev / null), 3),
                  "mean_abs_shap": imp.round(4).to_dict(),
                  "green_view_shap_by_decile": [round(float(c), 4) for c in curve.to_numpy()]}
    fig, ax = plt.subplots(figsize=(6, 4))
    for v, (xv, yv) in shap_curves.items():
        ax.plot(xv, yv, marker="o", label={"A_null": "A: no effect", "B_planted": "B: planted effect"}[v])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("green view during walking/cycling in the half-hour (vegetation share)")
    ax.set_ylabel("mean SHAP value (log steps)")
    ax.set_title("SYNTHETIC data: XGBoost + SHAP, green view", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "xgb_shap_green_view_synthetic.png", dpi=120)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = model_input()
    res = {"SYNTHETIC": True, "model_rows": int(len(d)), "persons": int(d.person_id.nunique()),
           "green_view_mean_sd": [round(float(d.x_green_view.mean()), 4), round(float(d.x_green_view.std()), 4)],
           "corr_green_view_walk_min": round(float(d.x_green_view.corr(d.walk_min)), 3),
           "corr_green_view_bike_min": round(float(d.x_green_view.corr(d.bike_min)), 3),
           "corr_green_view_park_min": round(float(d.x_green_view.corr(d.park_min)), 3),
           "halfhours_with_park_stay": int((d.park_min > 0).sum())}
    res["st_dbscan"] = run_st_dbscan()
    res["xgboost_shap"] = run_xgb_shap(d)
    (OUT / "models_python_synthetic.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
