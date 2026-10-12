# -*- coding: utf-8 -*-
"""Step 2.1: compare the 4B model with v1 and v2 definitions against the 27B reference (v2).

Same 200 front views. Per item: percent agreement and Cohen's kappa over the four states,
and agreement on present vs not. Also 27B-v1 against 27B-v2, to see whether the v1
wording misleads the larger model too. Writes outputs/wp3/pilot_comparison.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from geoai_vlm import audit_table, cohen_kappa, percent_agreement

PROJ = Path(__file__).resolve().parents[1]
W = PROJ / "outputs" / "wp3"
STATES = ["present", "absent", "not_visible", "uncertain"]


def load(name: str) -> pd.DataFrame:
    t = audit_table(pd.read_parquet(W / f"{name}.parquet"))
    return t[["image_id", "item", "state"]]


def main():
    ref = load("reference_pilot_front200_qwen38_27b_fp8_v2")
    ids = set(ref["image_id"])
    runs = {
        "4B v1": load("audit_pilot_qwen35_4b"),
        "4B v2": load("audit_pilot_qwen35_4b_v2"),
        "27B v1": load("reference_pilot_front200_qwen38_27b_fp8_v1"),
    }
    rows = []
    for name, run in runs.items():
        m = run[run.image_id.isin(ids)].merge(ref, on=["image_id", "item"], suffixes=("", "_ref"))
        m = m[m.state.isin(STATES) & m.state_ref.isin(STATES)]
        rows.append({"run": name, "item": "ALL", "n": len(m), "agree": percent_agreement(m.state, m.state_ref),
                     "kappa": cohen_kappa(m.state, m.state_ref, labels=STATES),
                     "present_rate": (m.state == "present").mean(), "present_rate_ref": (m.state_ref == "present").mean()})
        for item, g in m.groupby("item"):
            rows.append({"run": name, "item": item, "n": len(g), "agree": percent_agreement(g.state, g.state_ref),
                         "kappa": cohen_kappa(g.state, g.state_ref, labels=STATES),
                         "present_agree": percent_agreement(g.state == "present", g.state_ref == "present"),
                         "present_rate": (g.state == "present").mean(), "present_rate_ref": (g.state_ref == "present").mean()})
    out = pd.DataFrame(rows)
    out.to_csv(W / "pilot_comparison.csv", index=False)
    show = out[out.item.isin(["ALL", "cycle_infrastructure", "sidewalk_parking", "sidewalk_obstruction", "sidewalk",
                              "pedestrian_crossing", "street_trees", "seating", "traffic_calming"])]
    print(show.pivot_table(index="item", columns="run", values=["agree", "kappa"]).round(3).to_string())
    print(show.pivot_table(index="item", columns="run", values=["present_rate", "present_rate_ref"]).round(3).to_string())


if __name__ == "__main__":
    main()
