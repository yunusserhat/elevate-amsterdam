# -*- coding: utf-8 -*-
"""Step 3.3: generate the SYNTHETIC WP2 data (design: docs/wp2_synthetic_design.md).

200 synthetic persons, 7 days (Mon 15 - Sun 21 September 2025), 07:00-19:00 local time, one
30 s epoch grid per person-day. Persons, programmes, routes and GPS fixes are generated once;
steps twice from the same movement: A without any environmental effect, B with a planted
effect exp(beta * z) on the half-hour's step rate, z being the standardised true green-view
exposure while walking or cycling in that half-hour, beta = log(1.15) (protocol power analysis).
Revision 2026-10-11 (2): lunch walks only to places within 1 km; walks longer than 3 km on the
network become bicycle trips, and so do walks to or from a stop longer than 2 km; the City's
ferries are part of the walk and bicycle network.

Nothing here describes real people. Every output carries "synthetic" in its name.

Usage: python 41_wp2_generate.py [--persons 200] [--workers 24]
Writes outputs/wp2/synthetic_{persons,gps,steps_A_null,steps_B_planted,truth_events,truth_halfhours}.parquet
and synthetic_summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from wp2_common import METRIC, Exposure, ModeGraph, mode_masks, osm_graph_tables

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "outputs/wp2"
SEED = 2026
BETA = float(np.log(1.15))
DAYS = pd.date_range("2025-09-15", periods=7, freq="D")
WINDOW = 12 * 3600          # 07:00-19:00
EPOCH = 30
N_EPOCH = WINDOW // EPOCH   # 1440
ACT_TYPES = ["daily_shop", "park", "sport", "health", "social", "other_shop"]
ACT_P = [0.35, 0.2, 0.1, 0.05, 0.2, 0.1]
ACT_SCALE_KM = {"daily_shop": 1.0, "park": 2.0, "sport": 2.0, "health": 3.0, "social": 2.0, "other_shop": 3.0}
ACT_MIN = {"daily_shop": (10, 40), "park": (20, 90), "sport": (45, 90), "health": (20, 60), "social": (60, 180),
           "other_shop": (10, 40)}
STAY_RATE = {"home": 2.0, "work": 1.5, "daily_shop": 12.0, "other_shop": 12.0, "park": 15.0, "sport": 20.0,
             "health": 3.0, "social": 3.0, "stop": 3.0}   # steps per 30 s before the person factor
INDOOR = {"home", "work", "daily_shop", "other_shop", "sport", "health", "social"}
TO_WGS = Transformer.from_crs(METRIC, 4326, always_xy=True)

G: dict = {}  # filled in the parent before forking


def setup():
    nodes, edges = osm_graph_tables()
    masks = mode_masks(edges)
    G["graph"] = {m: ModeGraph(nodes, edges, masks[m]) for m in ("walk", "bike", "drive")}
    pois = gpd.read_parquet(PROJ / "data/derived/wp2_pois.parquet").to_crs(METRIC)
    pois["x"], pois["y"] = pois.geometry.x, pois.geometry.y
    G["pois"] = {c: g[["x", "y"]].to_numpy() for c, g in pois.groupby("poi_class")}
    work = pois[pois.poi_class.isin(["office", "other_shop", "school", "health", "social"])]
    G["work_xy"], G["work_tree"] = work[["x", "y"]].to_numpy(), cKDTree(work[["x", "y"]].to_numpy())
    G["stops_xy"] = G["pois"]["transit_stop"]
    G["stops_tree"] = cKDTree(G["stops_xy"])
    G["exposure"] = Exposure()


# --------------------------------------------------------------------------- persons
def make_persons(n: int, rng) -> pd.DataFrame:
    b = gpd.read_parquet(PROJ / "data/derived/wp2_buurten.parquet").to_crs(METRIC)
    walk = G["graph"]["walk"]
    main_xy = walk.xy[walk.main]
    quotas = {"low": n - 2 * (n // 3), "middle": n // 3, "high": n // 3}
    rows = []
    for stratum, q in quotas.items():
        pool = b[b.income_tertile == stratum]
        picks = rng.choice(len(pool), size=q, p=(pool.aantalInwoners / pool.aantalInwoners.sum()).to_numpy())
        for i in picks:
            bu = pool.iloc[i]
            minx, miny, maxx, maxy = bu.geometry.bounds
            box = np.flatnonzero((main_xy[:, 0] >= minx) & (main_xy[:, 0] <= maxx) & (main_xy[:, 1] >= miny) & (main_xy[:, 1] <= maxy))
            inside = box[shapely.contains_xy(bu.geometry, main_xy[box, 0], main_xy[box, 1])]
            node = walk.main[rng.choice(inside)] if len(inside) else walk.nearest(np.array(bu.geometry.representative_point().coords[0]))[0]
            rows.append({"stratum": stratum, "buurtcode": bu.buurtcode, "home_x": walk.xy[node, 0], "home_y": walk.xy[node, 1],
                         "cars_per_household": bu.personenautosPerHuishouden})
    p = pd.DataFrame(rows)
    p.insert(0, "person_id", [f"S{i:03d}" for i in range(1, len(p) + 1)])
    p["gender"] = rng.choice(["female", "male"], size=len(p))
    p["age_group"] = rng.choice(["18-34", "35-54", "55-75"], size=len(p))
    p["employed"] = rng.random(len(p)) < np.where(p.age_group == "55-75", 0.40, 0.65)
    p["car"] = rng.random(len(p)) < np.clip(p.cars_per_household.fillna(0.4), 0, 0.9)
    p["cadence_spm"] = np.clip(rng.normal(105, 8, len(p)), 80, 130)
    p["walk_speed"] = np.clip(rng.normal(1.3, 0.15, len(p)), 0.9, 1.8)
    p["bike_speed"] = np.clip(rng.normal(4.3, 0.6, len(p)), 2.8, 6.0)
    p["daily_steps_target"] = np.maximum(rng.normal(9448, 2218, len(p)), 2000)
    wx, wy = np.full(len(p), np.nan), np.full(len(p), np.nan)
    for i in np.flatnonzero(p.employed):
        d = np.exp(rng.normal(np.log(4000), 0.7))
        a = rng.uniform(0, 2 * np.pi)
        target = np.array([p.home_x[i] + d * np.cos(a), p.home_y[i] + d * np.sin(a)])
        _, j = G["work_tree"].query(target)
        wx[i], wy[i] = G["work_xy"][j]
    p["work_x"], p["work_y"] = wx, wy
    return p


# --------------------------------------------------------------------------- programme
def choose_poi(kind: str, origin: np.ndarray, rng, scale_km: float | None = None) -> np.ndarray:
    xy = G["pois"][kind]
    d = np.hypot(*(xy - origin).T) / 1000
    s = scale_km or ACT_SCALE_KM.get(kind, 2.0)
    w = np.exp(-d / s)
    w[d > max(4 * s, np.sort(d)[min(4, len(d) - 1)])] = 0.0  # no destinations beyond 4 scales (keep the 5 nearest)
    return xy[rng.choice(len(xy), p=w / w.sum())]


def tour_mode(dist_km: float, car: bool, rng) -> str:
    if rng.random() < 1 / (1 + np.exp(5 * (dist_km - 1.2))):
        return "walk"
    if dist_km < 7.5 and rng.random() < 0.85:
        return "bike"
    return "car" if (car and rng.random() < 0.7) else "pt"


def programme(person, day_idx: int, rng) -> list:
    """List of tours; a tour is (planned departure s, mode or None, [(activity, xy, minutes)])."""
    home = np.array([person.home_x, person.home_y])
    weekday = day_idx < 5
    tours = []
    if person.employed and weekday:
        work = np.array([person.work_x, person.work_y])
        dep = rng.uniform(1800, 7200)
        back = rng.uniform(34200, 39600)
        stops = [("work", work, (back - dep) / 60)]
        tours.append((dep, None, stops, "commute"))
        if rng.random() < 0.4:
            kind = rng.choice(["daily_shop", "sport", "social"], p=[0.6, 0.2, 0.2])
            tours.append((back + rng.uniform(900, 2700), None, [(kind, choose_poi(kind, home, rng), rng.uniform(*ACT_MIN[kind]))], "evening"))
        lunch = rng.random() < 0.3
        return [(t, m, s, k, lunch) for t, m, s, k in tours]
    n_tours = rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])
    starts = np.sort(rng.uniform(3600, 34000, n_tours))
    for st in starts:
        stops, here = [], home
        for _ in range(1 if rng.random() < 0.7 else 2):
            kind = rng.choice(ACT_TYPES, p=ACT_P)
            here = choose_poi(kind, here, rng)
            stops.append((kind, here, rng.uniform(*ACT_MIN[kind])))
        tours.append((st, None, stops, "errand"))
    return [(t, m, s, k, False) for t, m, s, k in tours]


# --------------------------------------------------------------------------- movement
def legs_for_trip(a: np.ndarray, b: np.ndarray, mode: str, person, rng) -> list:
    """[(mode, coords, speed)] for one trip, coords in EPSG:28992."""
    def path(m, p, q):
        g = G["graph"][m]
        r = g.route(g.nearest(p)[0], g.nearest(q)[0], rng)
        return np.vstack([p, r, q])
    if mode == "walk":
        p = path("walk", a, b)
        if np.hypot(*np.diff(p, axis=0).T).sum() <= 3000:
            return [("walk", p, person.walk_speed)]
        mode = "bike"  # a walk that would exceed 3 km on the network is made by bicycle
    if mode == "bike":
        return [("bike", path("bike", a, b), person.bike_speed)]
    if mode == "car":
        return [("car", path("drive", a, b), 7.0)]
    sa, sb = G["stops_xy"][G["stops_tree"].query(a)[1]], G["stops_xy"][G["stops_tree"].query(b)[1]]

    def access(p, q):  # walk to or from the stop, by bicycle when the walk would exceed 2 km
        w = path("walk", p, q)
        if np.hypot(*np.diff(w, axis=0).T).sum() <= 2000:
            return ("walk", w, person.walk_speed)
        return ("bike", path("bike", p, q), person.bike_speed)
    return [access(a, sa), ("wait", np.vstack([sa, sa]), rng.uniform(180, 480)),
            ("bus", path("drive", sa, sb), 7.0), access(sb, b)]


def simulate_day(person, day_idx: int, rng) -> list:
    """Events covering [0, WINDOW): dicts with kind stay/move, t0, t1, mode/activity, coords."""
    home = np.array([person.home_x, person.home_y])
    events, t, here, place = [], 0.0, home, "home"

    def stay_until(t_end, where, what):
        nonlocal t
        if t_end > t:
            events.append({"kind": "stay", "t0": t, "t1": t_end, "activity": what, "xy": where})
            t = t_end

    def travel(dest, mode):
        nonlocal t, here
        for m, coords, speed in legs_for_trip(here, dest, mode, person, rng):
            if m == "wait":
                events.append({"kind": "stay", "t0": t, "t1": t + speed, "activity": "stop", "xy": coords[0]})
                t += speed
                continue
            length = float(np.hypot(*np.diff(coords, axis=0).T).sum())
            dur = max(length / speed, 1.0)
            events.append({"kind": "move", "t0": t, "t1": t + dur, "mode": m, "coords": coords, "length_m": length})
            t += dur
        here = dest

    for dep, _, stops, kind, lunch in programme(person, day_idx, rng):
        if t >= WINDOW:
            break
        stay_until(max(t, dep), here, place)
        far = max(np.hypot(*(s[1] - home)) for s in stops) / 1000 * 1.3
        mode = tour_mode(far, bool(person.car), rng)
        for act, xy, minutes in stops:
            travel(xy, mode)
            place = act
            end = t + 60 * minutes
            if act == "work" and lunch:
                noon = rng.uniform(18000, 23400)
                if t < noon < end - 3600:
                    stay_until(noon, here, "work")
                    work_xy = here
                    kind_l = "park" if rng.random() < 0.5 else "daily_shop"
                    near = G["pois"][kind_l][np.hypot(*(G["pois"][kind_l] - work_xy).T) <= 1000]
                    if len(near):  # a lunch walk only when such a place lies within 1 km
                        target = near[rng.integers(len(near))]
                        travel(target, "walk")
                        stay_until(t + 60 * rng.uniform(5, 20), here, kind_l)
                        travel(work_xy, "walk")
            stay_until(end, here, act)
        travel(home, mode)
        place = "home"
    stay_until(max(t, WINDOW), here, place)
    return events


def position_at(events: list, ts: np.ndarray):
    """True xy, state ('walk','bike','car','bus' or a stay activity) and event index per epoch."""
    t0 = np.array([e["t0"] for e in events])
    k = np.clip(np.searchsorted(t0, ts, side="right") - 1, 0, len(events) - 1)
    xy = np.full((len(ts), 2), np.nan)
    state = np.empty(len(ts), dtype=object)
    for j in np.unique(k):
        e, sel = events[j], np.flatnonzero(k == j)
        if e["kind"] == "stay":
            xy[sel] = e["xy"]
            state[sel] = e["activity"]
        else:
            c = e["coords"]
            cum = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
            d = np.clip((ts[sel] - e["t0"]) / (e["t1"] - e["t0"]), 0, 1) * cum[-1]
            xy[sel, 0], xy[sel, 1] = np.interp(d, cum, c[:, 0]), np.interp(d, cum, c[:, 1])
            state[sel] = e["mode"]
    return xy, state, k


def person_run(args):
    person, seed = args
    rng = np.random.default_rng(seed)
    gps, epochs, truth = [], [], []
    for d, day in enumerate(DAYS):
        events = simulate_day(person, d, rng)
        ts = np.arange(N_EPOCH) * EPOCH + rng.uniform(0, 2, N_EPOCH)
        xy, state, k = position_at(events, ts)
        # wear: short non-wear blocks; on about 10% of days a long one (day becomes invalid)
        wear = np.ones(N_EPOCH, bool)
        blocks = [rng.uniform(15, 90) * 60 for _ in range(rng.poisson(1))]
        if rng.random() < 0.10:
            blocks.append(rng.uniform(4.5, 8) * 3600)
        for length in blocks:
            s = rng.uniform(0, WINDOW - length)
            wear[(ts >= s) & (ts < s + length)] = False
        # GPS
        has_fix = wear.copy()
        for _ in range(rng.poisson(2)):
            s, length = rng.uniform(0, WINDOW), rng.uniform(60, 600)
            has_fix[(ts >= s) & (ts < s + length)] = False
        indoor = np.isin(state.astype(str), list(INDOOR))
        lost_stays = {j for j, e in enumerate(events) if e["kind"] == "stay" and e["activity"] in INDOOR and rng.random() < 0.7}
        has_fix &= ~np.isin(k, list(lost_stays))
        sigma = np.where(indoor, rng.uniform(20, 60, N_EPOCH), np.where(rng.random(N_EPOCH) < 0.05, 15.0, 4.0))
        err = rng.normal(0, 1, (N_EPOCH, 2)) * sigma[:, None]
        spike = rng.random(N_EPOCH) < 0.002
        ang, dist = rng.uniform(0, 2 * np.pi, N_EPOCH), rng.uniform(300, 2000, N_EPOCH)
        err[spike] += np.c_[dist[spike] * np.cos(ang[spike]), dist[spike] * np.sin(ang[spike])]
        obs = xy + err
        acc = sigma * rng.uniform(0.7, 1.3, N_EPOCH)
        lon, lat = TO_WGS.transform(obs[:, 0], obs[:, 1])
        tlon, tlat = TO_WGS.transform(xy[:, 0], xy[:, 1])
        stamp = day + pd.Timedelta(hours=7) + pd.to_timedelta(ts, unit="s")
        f = np.flatnonzero(has_fix)
        gps.append(pd.DataFrame({"person_id": person.person_id, "day": d + 1, "time": stamp[f], "lon": lon[f], "lat": lat[f],
                                 "accuracy_m": acc[f].round(1), "true_state": state[f].astype(str), "true_lon": tlon[f],
                                 "true_lat": tlat[f], "is_spike": spike[f]}))
        # step rates (person factor applied later) and true exposure while walking or cycling
        walk = state == "walk"
        rate_walk = np.where(walk, 0.5 * person.cadence_spm, 0.0)
        rate_vehicle = np.where(np.isin(state.astype(str), ["bike", "car", "bus"]), 0.5, 0.0)
        rate_stay = np.array([STAY_RATE.get(s, 0.0) if s not in ("walk", "bike", "car", "bus") else 0.0 for s in state.astype(str)])
        active = np.isin(state.astype(str), ["walk", "bike"])
        expo = np.full(N_EPOCH, np.nan)
        if active.any():
            expo[active] = G["exposure"].at(xy[active])
        epochs.append(pd.DataFrame({"person_id": person.person_id, "day": d + 1, "epoch": np.arange(N_EPOCH),
                                    "time": stamp, "wear": wear, "state": state.astype(str), "rate_walk": rate_walk,
                                    "rate_vehicle": rate_vehicle, "rate_stay": rate_stay, "true_exposure": expo}))
        for j, e in enumerate(events):
            truth.append({"person_id": person.person_id, "day": d + 1, "event": j, "kind": e["kind"],
                          "start": day + pd.Timedelta(hours=7) + pd.Timedelta(seconds=float(e["t0"])),
                          "end": day + pd.Timedelta(hours=7) + pd.Timedelta(seconds=float(min(e["t1"], WINDOW))),
                          "activity_or_mode": e.get("activity", e.get("mode")), "length_m": e.get("length_m", 0.0)})
    return pd.concat(gps), pd.concat(epochs), pd.DataFrame(truth)


# --------------------------------------------------------------------------- steps
def draw_steps(ep: pd.DataFrame, persons: pd.DataFrame, seed: int):
    p = persons.set_index("person_id")
    w = ep.wear.to_numpy()
    walk_v = (ep.rate_walk + ep.rate_vehicle).to_numpy() * w
    stay = ep.rate_stay.to_numpy() * w
    # person factor on stay rates so that the expected mean total of the valid days (>= 8 h wear)
    # matches the person's target
    hours = ep.groupby(["person_id", "day"]).wear.transform("sum").to_numpy() * EPOCH / 3600
    valid = hours >= 8
    g = pd.DataFrame({"person_id": ep.person_id, "walk_v": walk_v * valid, "stay": stay * valid}).groupby("person_id").sum()
    days = ep[valid].groupby("person_id").day.nunique().reindex(g.index).fillna(0)
    target = p.daily_steps_target.reindex(g.index) * days
    k = ((target - g.walk_v) / g.stay).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.2)
    persons_k = k.rename("stay_factor")
    lam = walk_v + stay * ep.person_id.map(k).to_numpy()

    # true half-hour exposure (walking or cycling epochs), standardised over person-half-hours
    hh = ep.epoch.to_numpy() // 60
    key = pd.MultiIndex.from_arrays([ep.person_id, ep.day, hh])
    x = pd.Series(ep.true_exposure.to_numpy(), index=key).groupby(level=[0, 1, 2]).mean()
    mu, sd = x.mean(), x.std()
    z = ((x - mu) / sd).fillna(0.0)
    mult = np.exp(BETA * z.reindex(key).to_numpy())
    truth_hh = pd.DataFrame({"true_exposure": x, "z_true": ((x - mu) / sd), "multiplier_B": np.exp(BETA * z)}).reset_index()
    truth_hh.columns = ["person_id", "day", "halfhour", "true_exposure", "z_true", "multiplier_B"]
    truth_hh.attrs["standardisation"] = {"mean": float(mu), "sd": float(sd)}

    out = {}
    for name, lam_v in (("A_null", lam), ("B_planted", lam * mult)):
        rng = np.random.default_rng(seed)  # same stream for both variants
        steps = np.where(w, rng.poisson(lam_v), 0)
        df = ep[["person_id", "day", "epoch", "time", "wear"]].copy()
        df["steps"] = steps.astype(np.int32)
        if name == "B_planted":
            df["planted_multiplier"] = mult.astype(np.float32)
        out[name] = df
    return out["A_null"], out["B_planted"], truth_hh, persons_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    setup()
    print(f"graphs ready in {time.time() - t0:.0f}s", flush=True)
    ss = np.random.SeedSequence(SEED)
    rng = np.random.default_rng(ss.spawn(1)[0])
    persons = make_persons(args.persons, rng)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(persons) + 1)[1:]]
    with Pool(args.workers) as pool:
        res = pool.map(person_run, [(SimpleNamespace(**row), s) for row, s in zip(persons.to_dict("records"), seeds)])
    print(f"movement done in {time.time() - t0:.0f}s", flush=True)
    gps = pd.concat([r[0] for r in res], ignore_index=True)
    ep = pd.concat([r[1] for r in res], ignore_index=True)
    truth = pd.concat([r[2] for r in res], ignore_index=True)
    steps_a, steps_b, truth_hh, k = draw_steps(ep, persons, SEED + 1)
    persons = persons.merge(k, left_on="person_id", right_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pub = persons.drop(columns=["home_x", "home_y", "work_x", "work_y", "cars_per_household"])
    pub.to_parquet(OUT / "synthetic_persons.parquet", index=False)
    persons.to_parquet(OUT / "synthetic_truth_persons.parquet", index=False)
    gps.to_parquet(OUT / "synthetic_gps.parquet", index=False)
    steps_a.to_parquet(OUT / "synthetic_steps_A_null.parquet", index=False)
    steps_b.to_parquet(OUT / "synthetic_steps_B_planted.parquet", index=False)
    truth.to_parquet(OUT / "synthetic_truth_events.parquet", index=False)
    truth_hh.to_parquet(OUT / "synthetic_truth_halfhours.parquet", index=False)

    valid = steps_a.groupby(["person_id", "day"]).wear.sum() * EPOCH / 3600 >= 8
    daily = {n: s.groupby(["person_id", "day"]).steps.sum()[valid] for n, s in (("A", steps_a), ("B", steps_b))}
    moves = truth[truth.kind == "move"]
    summary = {
        "SYNTHETIC": True, "persons": int(len(persons)), "person_days": int(len(valid)),
        "valid_days_share": round(float(valid.mean()), 3),
        "persons_with_4_valid_days": int((valid.groupby(level=0).sum() >= 4).sum()),
        "gps_fixes": int(len(gps)), "fix_share_of_epochs": round(len(gps) / len(ep), 3),
        "spikes": int(gps.is_spike.sum()),
        "trips_by_mode": moves.activity_or_mode.value_counts().to_dict(),
        "km_by_mode": (moves.groupby("activity_or_mode").length_m.sum() / 1000).round(0).to_dict(),
        "daily_steps_valid_days": {n: {"mean": round(float(v.mean())), "sd": round(float(v.std()))} for n, v in daily.items()},
        "halfhours_with_true_exposure_share": round(float(truth_hh.true_exposure.notna().mean()), 3),
        "exposure_standardisation": truth_hh.attrs["standardisation"], "beta_planted": BETA,
        "seconds": round(time.time() - t0),
    }
    (OUT / "synthetic_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
