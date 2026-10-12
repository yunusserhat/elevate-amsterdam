# -*- coding: utf-8 -*-
"""Steps 5.1-5.2: a HYPOTHETICAL stakeholder requirement matrix and a draft indicator long list.

The protocol's WP1 brings planners, health professionals, municipal representatives and
residents together (later also landscape architects and designers), maps stakeholders across
the public, private and civic domains, and turns workshop priorities into a ranked indicator
list. No workshop has taken place. The rows below are written by role-playing each group, to
show how its questions would map onto data, indicators, tool features and what a workshop
would have to validate. Status says what this demonstration already provides.

Usage: python 60_wp1_matrix.py
Writes outputs/wp1/stakeholder_matrix.csv, outputs/wp1/indicator_longlist.csv and docs/wp1_stakeholders.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
LABEL = "HYPOTHETICAL: written by role-play, to be validated in WP1 co-creation workshops"

MATRIX = [
    ("Residents", "civic", "Is my walk to the shops and the park comfortable and safe?",
     "Sidewalk, crossings, shade, seating (audit, tier A); green view; perceived safety (stand-in model); WP3 street segments along the route",
     "Address search with nearby street views and segment indicators; image-grounded answers",
     "Which features residents notice; whether model observations match lived experience (WP4 walk-alongs)",
     "partial: address and segment lookup; no route planner"),
    ("Residents (older adults, via neighbourhood ambassadors)", "civic", "Where can I sit and rest on the way?",
     "Seating and shade (tier A); coverage of footways and park paths (8% of their length in the 2025 archive)",
     "Map and image search for seating; coverage warning on paths",
     "Whether seating seen from the street is usable; places the car-mounted camera never sees",
     "partial: street seating only"),
    ("Residents of low-income buurten (via neighbourhood ambassadors)", "civic", "Are our streets as good as those elsewhere?",
     "Buurt indicators against CBS low-income share; data coverage by buurt",
     "Equity view: buurt comparison with uncertainty and coverage",
     "Which comparisons feel fair and meaningful; how to show uncertainty without discouraging",
     "available (descriptive)"),
    ("Urban planners (municipality)", "public", "Which intersections lack marked crossings?",
     "Intersection and pedestrian crossing (tier A), per view and street segment",
     "Gap layer: candidate locations, ranked by equity priority (WP6, step 7.1)",
     "Thresholds for calling something a gap; known crossings missed by the images",
     "available with a warning: WP6 gap map; the larger model confirms 31% of missing crossings"),
    ("Urban planners (municipality)", "public", "How complete and reliable is this evidence?",
     "Network coverage by road group and buurt; agreement with a larger model per item; tiers A/B/C",
     "Coverage map; reliability label next to every indicator",
     "What level of uncertainty is acceptable for which decision",
     "available"),
    ("Urban planners (municipality)", "public", "Where is cycling infrastructure missing on main roads?",
     "Cycle infrastructure (tier B: over-reported, 28% against 16% in the reference)",
     "Shown with a caution label only",
     "Cross-check against the city's own cycle network data before any use",
     "partial (low reliability)"),
    ("Health professionals (public health)", "public", "Which street environments go with more daily walking?",
     "WP2 steps and GPS with buffered street exposure; WP5 mixed negative binomial model",
     "Model page (currently on synthetic data, labelled as a pipeline test)",
     "Outcome definitions, confounders for the DAG, subgroups of interest",
     "pipeline only (synthetic)"),
    ("Health professionals (public health)", "public", "Do low-income neighbourhoods have less walkable streets?",
     "Equity analysis of WP3 indicators against CBS low-income share; spatial clusters",
     "Equity view with maps and tables",
     "Which indicators matter for health equity; which comparisons need adjustment for centrality",
     "available (descriptive)"),
    ("Health professionals (public health)", "public", "How much green do people pass while walking?",
     "Green view along walking and cycling paths, half-hour time-weighted (WP2 exposure)",
     "Exposure profile per (synthetic) person",
     "Exposure window and buffer; indoor time; time spent in parks versus on streets",
     "pipeline only (synthetic)"),
    ("Municipal representatives (district and policy)", "public", "Where should we invest first?",
     "Equity-priority rule: low green-view cluster in buurten with many low-income households (21 buurten)",
     "Priority map with the rule spelled out",
     "Whether the rule is acceptable; political and budget constraints; other priorities",
     "available (draft rule)"),
    ("Municipal representatives (district and policy)", "public", "Can this be updated every year, and at what cost?",
     "Reproducible pipeline on the city's yearly panoramas; GPU time (audit of 99,100 views: 2.5 h on two GPUs)",
     "Provenance page: model revisions, processing ids, scripts",
     "Data ownership, hosting, who maintains it",
     "available (documented)"),
    ("Municipal representatives (privacy)", "public", "Is this lawful and privacy-safe?",
     "City blurs faces and plates before release; no images hosted; no GPS tracks published (protocol: geomasked)",
     "Data card and licences page",
     "Review with the privacy officer; consent wording for WP2 participants",
     "available (documented)"),
    ("Landscape architects and designers", "private / public", "What would this street look like with trees or a crossing?",
     "Detected gaps; before/after visualisations (Qwen-Image-2.1, research licence)",
     "Gallery of AI visualisations, each re-audited and labelled 'not a design proposal'",
     "Realism and social fit (protocol's design-thinking sessions); risk of over-trusting images",
     "available: 47 AI visualisations, each checked (WP6)"),
    ("Landscape architects and designers", "private / public", "Which streets are similar to this one?",
     "Image+text index of 99,100 views (EmbeddingGemma 2)",
     "'Find similar scenes' by photo or words",
     "Usefulness for co-creating streetscape types (protocol objective 2)",
     "available"),
    ("Landscape architects and designers", "private / public", "What can the images not tell us?",
     "Limits: surface quality in detail, maintenance, lighting at night, noise, air, social safety",
     "Limitations section on each page",
     "Which other data sources should be added",
     "available (text)"),
    ("Civic organisations (neighbourhood, walking and cycling groups)", "civic", "Can residents' reports be linked to what the images show?",
     "City public-space reports (Signalen), if open; WP3 observations at the same places",
     "Triangulation layer (WP4, step 6.1)",
     "Which report categories matter for walking; whether to show reports publicly",
     "available per buurt (WP4)"),
]

LONGLIST = [  # draft order for workshop ranking: protocol relevance, reliability, coverage
    ("pedestrian crossing at intersections", "audit (tier A)", "0.83", "protocol Fig. 2 example; detects marked crossings only"),
    ("green view (vegetation share)", "segmentation", "AUC 0.96 vs street trees", "seasonal; leaf-on images only (June-October)"),
    ("shade on walkway", "audit (tier A)", "0.73", "depends on time of capture and sun"),
    ("seating", "audit (tier A)", "0.84", "street seating only; park benches rarely seen"),
    ("sidewalk present", "audit (tier A)", "0.81", "present on almost all observed streets; little contrast between buurten"),
    ("street lighting (pole present)", "audit (tier A)", "0.74", "daytime images: presence, not light levels"),
    ("intersection density (from images)", "audit (tier A)", "0.75", "better from the street network"),
    ("cycle infrastructure", "audit (tier B)", "0.59", "over-reported; check against the city's cycle network"),
    ("sidewalk obstruction", "audit (tier B)", "0.54", "transient (bins, bikes); single capture date"),
    ("active frontage", "audit (tier B)", "0.53", "over-reported in side views"),
    ("perceived beauty / safety / liveliness", "Place Pulse model (stand-in)", "two models agree, Spearman 0.15-0.74", "global crowd, not Amsterdam residents; to be replaced by ELEVATE's own model"),
    ("sidewalk parking", "audit (tier C)", "0.15", "not used: over-reported four-fold"),
    ("accessibility barrier", "audit (tier C)", "0.04", "not used: almost never reported by the small model"),
    ("litter", "audit (tier C)", "0.18", "not used: too few cases"),
    ("surface damage, traffic calming, sidewalk gap", "audit (rare)", "-", "not assessable from 500 views; too rare"),
]


def main():
    m = pd.DataFrame(MATRIX, columns=["stakeholder_group", "domain", "question", "indicators_and_data", "tool_feature",
                                      "to_validate_in_workshop", "status_in_demo"])
    m.insert(0, "label", LABEL)
    l = pd.DataFrame(LONGLIST, columns=["indicator", "source", "presence_kappa_or_check", "what_it_cannot_show"])
    l.insert(0, "draft_rank", range(1, len(l) + 1))
    l.insert(0, "label", "DRAFT order for workshop ranking, not stakeholder priorities")
    out = PROJ / "outputs/wp1"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "stakeholder_matrix.csv", index=False)
    l.to_csv(out / "indicator_longlist.csv", index=False)

    md = ["# WP1 stakeholder requirements (steps 5.1–5.2): HYPOTHETICAL", "",
          f"> **{LABEL}.** No workshop has taken place. The protocol's WP1 holds 3–4 co-creation workshops "
          "with planners, health professionals, municipal representatives and residents (later also landscape architects and "
          "designers). It maps stakeholders across the public, private and civic domains and turns workshop priorities into a "
          "ranked indicator list. This page shows how each group's questions would map onto data, indicators, tool features and "
          "workshop checks.", "",
          "Script: `scripts/60_wp1_matrix.py`. Tables: `outputs/wp1/stakeholder_matrix.csv`, `outputs/wp1/indicator_longlist.csv`.", "",
          "## Requirement matrix", "",
          "| Group | Question | Indicators and data | Tool feature | To validate in a workshop | Status |", "|---|---|---|---|---|---|"]
    for r in m.itertuples():
        md.append(f"| {r.stakeholder_group} | {r.question} | {r.indicators_and_data} | {r.tool_feature} | {r.to_validate_in_workshop} | {r.status_in_demo} |")
    md += ["", "## Draft indicator long list (for ranking in a workshop)", "",
           "- **Order rule:** relevance to walking in the protocol, then reliability (step 2.7), then coverage.",
           "- The order is a starting point for the workshop to re-rank. It is not a stakeholder priority.",
           "- The reliability column gives presence κ against the 27B reference, or the check used.", "",
           "| # | Indicator | Source | Reliability | What it cannot show |", "|---|---|---|---|---|"]
    for r in l.itertuples():
        md.append(f"| {r.draft_rank} | {r.indicator} | {r.source} | {r.presence_kappa_or_check} | {r.what_it_cannot_show} |")
    (PROJ / "docs/wp1_stakeholders.md").write_text("\n".join(md) + "\n")
    print(m.groupby("stakeholder_group").size().to_string())
    print(m.status_in_demo.value_counts().to_string())


if __name__ == "__main__":
    main()
