# -*- coding: utf-8 -*-
"""Step 3.4: the WP2 analysis chain on the SYNTHETIC data, scored against the synthetic truth.

Protocol steps (docs/protocol_map.md, WP2), with the rules used here:
1. Valid days: >= 8 h wear; persons need >= 4 valid days.
2. Cleaning: drop fixes with reported horizontal error > 5 m; drop implausible speeds (a fix
   reached and left at > 15 m/s, or reached at > 50 m/s), twice.
3. Short gaps (<= 2 min) are filled by linear interpolation on the 30 s grid.
4. Stops: stay points (Li et al. 2008): fixes within 50 m of an anchor for >= 5 min. Other
   fixes form trips.
5. Mode per trip from its median speed: walk < 2 m/s, bicycle 2-6 m/s, vehicle > 6 m/s.
6. Map matching and exposure for walking and cycling fixes: nearest WP3 street segment within
   25 m; exposure = mean indicator of the observed WP3 segments within 25 m.
7. Half-hour exposure: time-weighted mean (equal 30 s fixes) over walking and cycling fixes.
Exposure indicators: green view (vegetation fraction), street trees and sidewalk (audit
tier A, present share), perceived beauty and safety (Place Pulse ViT).

Usage: python 42_wp2_pipeline.py [--workers 24]
Writes outputs/wp2/pipeline_{fixes,halfhours,persons}.parquet and pipeline_qc.json. SYNTHETIC.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from wp2_common import METRIC, Exposure

PROJ = Path(__file__).resolve().parents[1]
W2 = PROJ / "outputs/wp2"
EXPOSURES = {"green_view": "vegetation_pixel_fraction__mean", "street_trees": "audit_street_trees_state__present_share",
             "sidewalk": "audit_sidewalk_state__present_share", "beautiful": "vit_beautiful__mean", "safety": "vit_safety__mean"}
TO_M = Transformer.from_crs(4326, METRIC, always_xy=True)
STATE_TRUE = {"walk": "walk", "bike": "bike", "car": "vehicle", "bus": "vehicle"}  # anything else is a stay
G: dict = {}


def speeds(x, y, t):
    d = np.hypot(np.diff(x), np.diff(y))
    dt = np.maximum(np.diff(t), 1e-3)
    v = d / dt
    return np.r_[np.nan, v], np.r_[v, np.nan]


def clean_day(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    df = df[df.accuracy_m <= 5].copy()
    n_acc = len(df)
    for _ in range(2):
        if len(df) < 3:
            break
        t = df.t.to_numpy()
        vin, vout = speeds(df.x.to_numpy(), df.y.to_numpy(), t)
        bad = ((vin > 15) & (vout > 15)) | (vin > 50)
        df = df[~np.nan_to_num(bad, nan=False).astype(bool)]
    return df, {"raw": n0, "after_accuracy": n_acc, "after_speed": len(df)}


def fill_gaps(df: pd.DataFrame, max_gap: float = 120.0) -> pd.DataFrame:
    if len(df) < 2:
        return df.assign(interpolated=False)
    t = df.t.to_numpy()
    rows = [df.assign(interpolated=False)]
    gaps = np.flatnonzero((np.diff(t) > 45) & (np.diff(t) <= max_gap + 15))
    for i in gaps:
        new_t = np.arange(t[i] + 30, t[i + 1] - 15, 30.0)
        if len(new_t):
            f = (new_t - t[i]) / (t[i + 1] - t[i])
            a, b = df.iloc[i], df.iloc[i + 1]
            rows.append(pd.DataFrame({"t": new_t, "x": a.x + f * (b.x - a.x), "y": a.y + f * (b.y - a.y), "interpolated": True}))
    out = pd.concat(rows, ignore_index=True).sort_values("t").reset_index(drop=True)
    for c in ("person_id", "day"):
        out[c] = df[c].iloc[0]
    return out


def stays(x, y, t, d_max=50.0, t_min=300.0) -> np.ndarray:
    n, i, flag = len(t), 0, np.zeros(len(t), bool)
    while i < n:
        j = i + 1
        while j < n and np.hypot(x[j] - x[i], y[j] - y[i]) <= d_max:
            j += 1
        if t[j - 1] - t[i] >= t_min:
            flag[i:j] = True
            i = j
        else:
            i += 1
    return flag


def modes(x, y, t, stay) -> np.ndarray:
    out = np.where(stay, "stay", "").astype(object)
    vin, vout = speeds(x, y, t)
    v = np.nanmean(np.c_[vin, vout], axis=1)
    run = np.cumsum(np.r_[True, (stay[1:] != stay[:-1]) | (np.diff(t) > 600)])
    for r in np.unique(run[~stay]):
        sel = (run == r) & ~stay
        med = np.nanmedian(v[sel]) if np.isfinite(v[sel]).any() else 0.0
        out[sel] = "walk" if med < 2.0 else ("bike" if med < 6.0 else "vehicle")
    return out


def run_day(args):
    df = args
    df = df.sort_values("t")
    clean, counts = clean_day(df)
    filled = fill_gaps(clean)
    if len(filled) == 0:
        return filled, counts
    x, y, t = filled.x.to_numpy(), filled.y.to_numpy(), filled.t.to_numpy()
    filled["stay"] = stays(x, y, t)
    filled["mode"] = modes(x, y, t, filled.stay.to_numpy())
    active = filled["mode"].isin(["walk", "bike"]).to_numpy()
    pts = shapely.points(np.c_[x, y])
    seg_ids = np.full(len(filled), None, dtype=object)
    if active.any():
        ip, iseg = G["seg_tree"].query_nearest(pts[active], max_distance=25.0, return_distance=False, all_matches=False)
        seg_ids[np.flatnonzero(active)[ip]] = G["seg_ids"][iseg]
    filled["segment_id"] = seg_ids
    for name, ex in G["exposure"].items():
        v = np.full(len(filled), np.nan)
        if active.any():
            v[active] = ex.at(np.c_[x, y][active])
        filled[f"x_{name}"] = v
    return filled, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    seg = gpd.read_parquet(PROJ / "data/derived/segments.parquet", columns=["segment_id", "road_group", "geometry"]).to_crs(METRIC)
    G["seg_tree"], G["seg_ids"] = shapely.STRtree(seg.geometry.to_numpy()), seg.segment_id.to_numpy()
    G["exposure"] = {k: Exposure(c) for k, c in EXPOSURES.items()}

    gps = pd.read_parquet(W2 / "synthetic_gps.parquet")
    gps["x"], gps["y"] = TO_M.transform(gps.lon.to_numpy(), gps.lat.to_numpy())
    gps["t"] = (gps.time - gps.time.dt.normalize()).dt.total_seconds() - 7 * 3600
    obs = gps[["person_id", "day", "t", "x", "y", "accuracy_m"]]
    with Pool(args.workers) as pool:
        res = pool.map(run_day, [g for _, g in obs.groupby(["person_id", "day"])])
    fixes = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    counts = pd.DataFrame([r[1] for r in res]).sum().to_dict()
    fixes["epoch"] = np.clip(np.round(fixes.t / 30).astype(int), 0, 1439)
    fixes["halfhour"] = fixes.epoch // 60
    fixes = fixes.merge(seg[["segment_id", "road_group"]], on="segment_id", how="left")

    # --- scoring against the synthetic truth (observed, non-interpolated fixes only)
    truth = gps[["person_id", "day", "t", "true_state", "is_spike", "accuracy_m"]]
    acc_ok = truth[truth.accuracy_m <= 5]
    kept = fixes[~fixes.interpolated][["person_id", "day", "t"]].assign(kept=True)
    sc = acc_ok.merge(kept, on=["person_id", "day", "t"], how="left")
    sc["kept"] = sc.kept.fillna(False).astype(bool)
    spike_recall = float((~sc.kept[sc.is_spike]).mean()) if sc.is_spike.any() else np.nan
    false_drop = float((~sc.kept[~sc.is_spike]).mean())
    m = fixes[~fixes.interpolated].merge(truth[["person_id", "day", "t", "true_state"]], on=["person_id", "day", "t"])
    m["true_class"] = m.true_state.map(STATE_TRUE).fillna("stay")
    m["pred_class"] = m["mode"]
    confusion = pd.crosstab(m.true_class, m.pred_class, normalize="index").round(3)

    # --- half-hours
    steps = {}
    for v in ("A_null", "B_planted"):
        s = pd.read_parquet(W2 / f"synthetic_steps_{v}.parquet")
        s["halfhour"] = s.epoch // 60
        steps[v] = s.groupby(["person_id", "day", "halfhour"]).agg(steps=("steps", "sum"), wear_epochs=("wear", "sum"))
    hh = steps["A_null"].rename(columns={"steps": "steps_A_null"}).join(steps["B_planted"][["steps"]].rename(columns={"steps": "steps_B_planted"}))
    wear_h = hh.groupby(level=[0, 1]).wear_epochs.transform("sum") * 30 / 3600
    hh["valid_day"] = wear_h >= 8
    act = fixes[fixes["mode"].isin(["walk", "bike"])]
    ex_cols = [f"x_{k}" for k in EXPOSURES]
    agg = act.groupby(["person_id", "day", "halfhour"]).agg(n_active_fixes=("t", "size"),
                                                           n_exposed_fixes=("x_green_view", lambda s: int(s.notna().sum())),
                                                           **{c: (c, "mean") for c in ex_cols})
    hh = hh.join(agg)
    hh["n_active_fixes"] = hh.n_active_fixes.fillna(0).astype(int)
    hh["n_exposed_fixes"] = hh.n_exposed_fixes.fillna(0).astype(int)
    tru = pd.read_parquet(W2 / "synthetic_truth_halfhours.parquet").set_index(["person_id", "day", "halfhour"])
    hh = hh.join(tru[["true_exposure", "z_true"]])
    hh = hh.reset_index()
    valid_days = hh.groupby(["person_id", "day"]).valid_day.first()
    persons_ok = valid_days.groupby(level=0).sum() >= 4
    hh["valid_person"] = hh.person_id.map(persons_ok)
    hh["SYNTHETIC"] = True

    # --- person profiles
    pp = act.groupby("person_id").agg(active_fixes=("t", "size"), exposed_share=("x_green_view", lambda s: float(s.notna().mean())),
                                      green_view=("x_green_view", "mean"), street_trees=("x_street_trees", "mean"),
                                      beautiful=("x_beautiful", "mean"))
    pp["active_minutes_per_day"] = pp.active_fixes * 0.5 / 7
    rg = act.dropna(subset=["road_group"]).groupby(["person_id", "road_group"]).size().unstack(fill_value=0)
    pp = pp.join(rg.div(rg.sum(axis=1), axis=0).add_prefix("share_on_"))
    daily = hh[hh.valid_day].groupby(["person_id", "day"]).steps_A_null.sum().groupby(level=0).mean().rename("mean_daily_steps_A")
    pp = pp.join(daily).join(valid_days.groupby(level=0).sum().rename("valid_days"))
    pp["SYNTHETIC"] = True

    to_wgs = Transformer.from_crs(METRIC, 4326, always_xy=True)
    fixes["lon"], fixes["lat"] = to_wgs.transform(fixes.x.to_numpy(), fixes.y.to_numpy())
    fixes.drop(columns=["x", "y"]).to_parquet(W2 / "pipeline_fixes.parquet", index=False)
    hh.to_parquet(W2 / "pipeline_halfhours.parquet", index=False)
    pp.reset_index().to_parquet(W2 / "pipeline_persons.parquet", index=False)

    with_true = hh.dropna(subset=["x_green_view", "true_exposure"])
    qc = {
        "SYNTHETIC": True, "fixes": counts, "fixes_after_gap_filling": int(len(fixes)),
        "interpolated": int(fixes.interpolated.sum()),
        "spike_recall": round(spike_recall, 3), "non_spike_dropped_by_speed_rule": round(false_drop, 4),
        "mode_confusion_rows_true": confusion.to_dict(orient="index"),
        "valid_person_days": int(valid_days.sum()), "valid_persons": int(persons_ok.sum()),
        "halfhours_valid": int(hh[hh.valid_day & hh.valid_person].shape[0]),
        "halfhours_with_observed_exposure": int(hh[hh.valid_day & hh.valid_person].x_green_view.notna().sum()),
        "active_fixes_with_exposure_share": round(float(act.x_green_view.notna().mean()), 3),
        "observed_vs_true_halfhour_green_view_r": round(float(with_true.x_green_view.corr(with_true.true_exposure)), 3),
    }
    (W2 / "pipeline_qc.json").write_text(json.dumps(qc, indent=1, default=str))
    print(json.dumps(qc, indent=1, default=str))
    print(confusion.to_string())


if __name__ == "__main__":
    main()
