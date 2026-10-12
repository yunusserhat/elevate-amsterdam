# WP5 equity and spatial clustering (steps 4.1–4.2), 2026-10-11

Script: `scripts/50_wp5_equity.py`. Outputs in `outputs/wp5/`. Street-level values are model observations of 2025 street views, not measurements.

## Data

**Income.** CBS StatLine table 85618NED, "Kerncijfers wijken en buurten 2023" (CC BY 4.0, source Statistics Netherlands). It was fetched from the OData API on 2026-10-11 and is stored in `data/raw/cbs_85618NED_amsterdam_buurten.parquet`. Three measures are used:
- **low_income_households** (primary): share of households below the CBS low-income threshold, `HuishoudensMetEenLaagInkomen_87`. Student households and households without a full year of income are left out.
- **low40_households**: share of households in the national lowest 40% of disposable household income, `k_40HuishoudensMetLaagsteInkomen_85`.
  - This is the variable behind PDOK's `percentageHuishoudensMetLaagInkomen`, which we first took for a low-income share. On PDOK it is rounded and slightly older: BU0363AC02 is 42 on PDOK against 40.7 on StatLine.
  - It is inflated in the historic centre by small and student households. Begijnhofbuurt has 60% of its households in the lowest 40%, but only 7.4% of its households have a low income.
- **mean income per inhabitant** (× €1,000). StatLine fills it for 363 buurten, PDOK for only 147.

**Indicators.** WP3 buurt values (step 2.6, weighted by observed street length):
- green view (the vegetation share of the views);
- audit items of tier A or B only (step 2.7);
- perceived qualities from the Place Pulse ViT stand-in;
- data coverage: the network share observed in the sample, and the share covered by the 2025 archive (step 1.3).

**Unit.** Buurten with ≥ 10 sampled panoramas and a low-income value: 360 of 519.

## 4.1 Gradients by income (Spearman ρ, 95% bootstrap interval, 360 buurten)

| Indicator | Low-income households | Mean income per inhabitant |
|---|---|---|
| green view | 0.01 [−0.10, 0.12] | −0.33 [−0.42, −0.24] |
| pedestrian crossing (A) | 0.04 [−0.07, 0.15] | 0.32 [0.23, 0.41] |
| seating (A) | −0.04 [−0.14, 0.07] | 0.35 [0.25, 0.44] |
| perceived lively (stand-in) | 0.13 [0.02, 0.24] | 0.43 [0.35, 0.52] |
| network share observed (sample) | −0.01 [−0.12, 0.09] | 0.09 [−0.01, 0.20] |
| network share in 2025 archive | 0.05 [−0.05, 0.16] | 0.21 [0.12, 0.31] |

All indicators against all three measures are in `equity_spearman.csv`, and quintile means in `equity_by_low_income_quintile.csv`.

**Reading.**
- Street-level features show almost no gradient with the low-income share (|ρ| ≤ 0.13).
- The clearer gradients are with mean income. Richer buurten have less green in view but more crossings, seating and "lively" scenes. In Amsterdam, mean income tracks centrality, so these gradients mostly separate the dense centre from greener outer districts.
- The city's 2025 archive covers low-income buurten as well as others, so the evidence base is not thinner where deprivation is higher.
- Street trees and sidewalk are reported in nearly every view at buurt level (means 0.98–1.00). They do not discriminate between buurten and are kept for street-level use only.

## 4.2 Spatial clustering (k = 6 nearest buurt centroids, row-standardised, 999 permutations)

- Global Moran's I is positive and significant for every indicator except sidewalk (I ≈ 0, since it is near-constant).
- Highest: perceived lively 0.56, pedestrian crossing 0.38, green view 0.38, perceived beautiful 0.37. Low-income share: 0.14.
- The k-nearest-neighbour graph has two disconnected parts, Zuidoost and the rest, which is expected for Amsterdam.
- LISA (p < 0.05, no correction for multiple tests, exploratory):
  - green view forms low-low clusters in Centrum (28 buurten), West (15, around Westerpark and the Staatsliedenbuurt) and along the IJ waterfront: Eastern Docklands in Oost (15) and NDSM, Buiksloterham and Overhoeks in Noord (6);
  - it forms high-high clusters in Noord (21), Zuidoost (10) and Nieuw-West (9).
  - Counts by stadsdeel come from the buurt code (letter after 0363).

**Equity-priority flag for WP6** (`priority_low_green_low_income` in `buurt_equity.parquet`): a low-low green-view cluster and the two highest low-income quintiles (9.7–33.5% low-income households).
- 21 buurten qualify: 9 in Centrum, 8 in West, 3 in Noord and 1 in Oost. Among them are Westelijke eilanden, Haarlemmerbuurt-West, Zeeheldenbuurt, Staatsliedenbuurt-Noordoost, Borgerbuurt, Kop Zeedijk, IJplein e.o. and Buiksloterham-Zuid.
- This is a rule for choosing where to look first in WP6. It is not a finding about need, and it would be revisited with the WP1 stakeholders.
