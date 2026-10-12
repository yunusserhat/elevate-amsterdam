# -*- coding: utf-8 -*-
"""Step 2.7: how far the 4B audit can be trusted, item by item.

(a) Agreement with a larger model (Qwen3.8-27B-FP8, same template v2) on eval_500: 500 views,
    one per panorama, 125 per direction, pilot excluded. Percent agreement and Cohen's kappa
    over the states, with 95% bootstrap intervals (2,000 resamples of views). Reference
    records whose JSON-schema output looped on whitespace are re-run once with
    repetition_penalty 1.05 and counted; those still failing are dropped and counted.
    The reference is a larger model, not ground truth.
(b) Stability: the same 2,000 pilot views were audited twice with identical settings (the
    v2 pilot run and the full run, different batches).
(c) Front against back of the same panorama (all 24,775 panoramas), for items a street
    view along the street should see in both directions.
(d) Segmentation cross-check: AUC of a Cityscapes class fraction for views the 4B model
    reports present against absent.
Tiers: A kappa >= 0.6, B 0.4-0.6, C < 0.4, on presence (present vs not present), because the
indicators use presence only. The first version tiered on kappa over all states; that rewards
agreeing on absent vs not visible and rated near-absent items highly, so it was replaced
after the first look at (a); both kappas are reported. When the reference reports an item in
fewer than 3% or more than 97% of views, kappa is not informative and the item is tiered
"rare" or "near-universal". Stability below 90% is flagged.

Usage: python 31_eval.py
Writes outputs/wp3/eval/{reference_agreement,stability,front_back,segmentation_auc,item_tiers}.csv
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from geoai_vlm import audit_table, cohen_kappa, percent_agreement

PROJ = Path(__file__).resolve().parents[1]
W = PROJ / "outputs/wp3"
OUT = W / "eval"
STATES = ["present", "absent", "not_visible", "uncertain"]
REF = "reference_eval500_qwen38_27b_fp8_v2"
B = 2000
SEED = 20261011
SEG_PAIRS = {"street_trees": ["vegetation"], "other_greenery": ["vegetation", "terrain"], "sidewalk": ["sidewalk"],
             "sidewalk_parking": ["car"], "cycle_infrastructure": ["bicycle"], "street_light_pole": ["pole"],
             "active_frontage": ["building"], "shade_on_walkway": ["vegetation"]}


def states(path_or_df) -> pd.DataFrame:
    df = pd.read_parquet(path_or_df) if isinstance(path_or_df, (str, Path)) else path_or_df
    t = audit_table(df)
    return t[["image_id", "item", "state"]]


def kappa_fast(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Cohen's kappa for integer-coded labels 0..k-1."""
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def reference_agreement(full: pd.DataFrame) -> pd.DataFrame | None:
    main = W / f"{REF}.parquet"
    if not main.exists():
        print("no reference run yet; skipping (a)")
        return None
    ref = pd.read_parquet(main)
    n_fail_first = int(ref.parse_error.astype(bool).sum())
    fb_path = W / f"{REF}_rp105.parquet"
    if fb_path.exists():
        fb = pd.read_parquet(fb_path)
        fb = fb[~fb.parse_error.astype(bool)]
        ref = pd.concat([ref[~ref.image_id.isin(fb.image_id)], fb])
    ok = ref[~ref.parse_error.astype(bool)]
    print(f"reference: {len(ref)} views, first-pass failures {n_fail_first}, used {len(ok)}")
    r = states(ok)
    m = full.merge(r, on=["image_id", "item"], suffixes=("", "_ref"))
    m = m[m.state.isin(STATES) & m.state_ref.isin(STATES)]
    m["view"] = m.image_id.str.rsplit("_", n=1).str[1]
    code = {s: i for i, s in enumerate(STATES)}
    rng = np.random.default_rng(SEED)
    rows = []
    for item, g in [("ALL", m)] + list(m.groupby("item")):
        a, b = g.state.map(code).to_numpy(), g.state_ref.map(code).to_numpy()
        k_point = cohen_kappa(g.state, g.state_ref, labels=STATES)
        if item != "ALL":
            assert abs(k_point - kappa_fast(a, b, 4)) < 1e-6 or np.isnan(k_point)
        pa, pb = (a == 0).astype(int), (b == 0).astype(int)  # present (code 0) vs not present
        boots_k, boots_a, boots_p = [], [], []
        for _ in range(B):
            i = rng.integers(0, len(a), len(a))
            boots_k.append(kappa_fast(a[i], b[i], 4))
            boots_a.append((a[i] == b[i]).mean())
            boots_p.append(kappa_fast(pa[i], pb[i], 2))
        row = {"item": item, "n": len(g), "agree": percent_agreement(g.state, g.state_ref),
               "agree_lo": np.nanpercentile(boots_a, 2.5), "agree_hi": np.nanpercentile(boots_a, 97.5),
               "kappa": k_point, "kappa_lo": np.nanpercentile(boots_k, 2.5), "kappa_hi": np.nanpercentile(boots_k, 97.5),
               "present_rate_4b": (g.state == "present").mean(), "present_rate_ref": (g.state_ref == "present").mean(),
               "present_agree": percent_agreement(g.state == "present", g.state_ref == "present"),
               "kappa_presence": kappa_fast(pa, pb, 2), "kappa_presence_lo": np.nanpercentile(boots_p, 2.5),
               "kappa_presence_hi": np.nanpercentile(boots_p, 97.5)}
        for v in ["front", "back", "left", "right"]:
            gv = g[g.view == v]
            row[f"agree_{v}"] = percent_agreement(gv.state, gv.state_ref) if len(gv) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_fail_first"] = n_fail_first
    return out


def stability(full: pd.DataFrame) -> pd.DataFrame:
    pilot = states(W / "audit_pilot_qwen35_4b_v2.parquet")
    m = pilot.merge(full, on=["image_id", "item"], suffixes=("_run1", "_run2"))
    rows = [{"item": "ALL", "n": len(m), "agree": percent_agreement(m.state_run1, m.state_run2),
             "kappa": cohen_kappa(m.state_run1, m.state_run2, labels=STATES)}]
    for item, g in m.groupby("item"):
        rows.append({"item": item, "n": len(g), "agree": percent_agreement(g.state_run1, g.state_run2),
                     "kappa": cohen_kappa(g.state_run1, g.state_run2, labels=STATES)})
    return pd.DataFrame(rows)


def front_back(full: pd.DataFrame) -> pd.DataFrame:
    f = full.assign(pano=full.image_id.str.rsplit("_", n=1).str[0], view=full.image_id.str.rsplit("_", n=1).str[1])
    w = f[f.view.isin(["front", "back"])].pivot_table(index=["pano", "item"], columns="view", values="state", aggfunc="first").dropna()
    rows = []
    for item, g in w.reset_index().groupby("item"):
        rows.append({"item": item, "n_panoramas": len(g), "agree": percent_agreement(g.front, g.back),
                     "kappa": cohen_kappa(g.front, g.back, labels=STATES),
                     "present_front": (g.front == "present").mean(), "present_back": (g.back == "present").mean()})
    return pd.DataFrame(rows)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def segmentation_auc(full: pd.DataFrame) -> pd.DataFrame:
    seg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(W / "segmentation/*.parquet")))])
    rows = []
    for item, classes in SEG_PAIRS.items():
        g = full[full.item == item].merge(seg, on="image_id")
        g = g[g.state.isin(["present", "absent"])]
        x = g[[f"{c}_pixel_fraction" for c in classes]].sum(axis=1).to_numpy()
        pos, neg = x[(g.state == "present").to_numpy()], x[(g.state == "absent").to_numpy()]
        rows.append({"item": item, "class": "+".join(classes), "n_present": len(pos), "n_absent": len(neg),
                     "median_present": np.median(pos) if len(pos) else np.nan,
                     "median_absent": np.median(neg) if len(neg) else np.nan, "auc": auc(pos, neg)})
    return pd.DataFrame(rows)


def tiers(ref: pd.DataFrame | None, stab: pd.DataFrame) -> pd.DataFrame:
    s = stab.set_index("item").agree
    rows = []
    items = sorted(stab.item[stab.item != "ALL"])
    r = ref.set_index("item") if ref is not None else None
    for item in items:
        row = {"item": item, "stability_agree": s.get(item, np.nan), "stability_flag": bool(s.get(item, 1) < 0.9)}
        if r is not None and item in r.index:
            x = r.loc[item]
            row.update({"kappa_presence": x.kappa_presence, "kappa_presence_lo": x.kappa_presence_lo,
                        "kappa_presence_hi": x.kappa_presence_hi, "present_agree": x.present_agree,
                        "kappa_states": x.kappa, "present_rate_4b": x.present_rate_4b, "present_rate_ref": x.present_rate_ref})
            p = x.present_rate_ref
            if p < 0.03:
                row["tier"] = "rare"
            elif p > 0.97:
                row["tier"] = "near-universal"
            elif x.kappa_presence >= 0.6:
                row["tier"] = "A"
            elif x.kappa_presence >= 0.4:
                row["tier"] = "B"
            else:
                row["tier"] = "C"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.concat([states(f) for f in sorted(glob.glob(str(W / "audit_full_qwen35_4b_v2_shard*.parquet")))])
    ref = reference_agreement(full)
    stab = stability(full)
    fb = front_back(full)
    sa = segmentation_auc(full)
    t = tiers(ref, stab)
    for name, df in [("reference_agreement", ref), ("stability", stab), ("front_back", fb), ("segmentation_auc", sa), ("item_tiers", t)]:
        if df is not None:
            df.round(4).to_csv(OUT / f"{name}.csv", index=False)
            print(f"\n== {name}\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
