# WP4 triangulation with residents' reports (step 6.1), 2026-10-11

Script: `scripts/65_wp4_signalen.py`. Outputs in `outputs/wp4/`.

**Why.** WP4 is qualitative: walk-along and on-site interviews, thematic analysis and triangulation with the quantitative results. No interviews exist yet. A second, open source of residents' own observations does exist: the city's public-space reports (Signalen). This step shows what triangulation looks like and where it breaks.

**Data.**
- "Meldingen over de Openbare Ruimte in Amsterdam" (SIA open dataset, `api.data.amsterdam.nl/v1/meldingen`), licence "Creative Commons, Naamsvermelding". It is a subset of all reports since mid-2018, in ten main categories.
- Window: June–October 2025, the capture months of the sampled panoramas: 223,592 reports, 99% of them closed.
- Report locations are "coordinates optimised for visualisation", so this works at buurt level only. Single reports are never matched to single images.
- The city's buurt codes match CBS buurtcode as "BU0363" + code; 99.4% of buurt names agree.

**Result.** Reports per km of walk/cycle network against the buurt share of views where the 4B model reports the related item. Spearman across 360 buurten, with a 95% bootstrap interval; the partial correlation holds address density fixed.

| Reports (count) | Model item (tier) | ρ [95% CI] | Partial, given density | ρ with low-income share |
|---|---|---|---|---|
| obstacles, bicycle and scooter wrecks (12,375) | sidewalk obstruction (B) | 0.79 [0.75, 0.83] | 0.53 | 0.16 |
| parking nuisance (18,762) | sidewalk parking (C) | 0.43 [0.34, 0.52] | 0.27 | 0.18 |
| street furniture (1,218) | seating (A) | 0.31 [0.21, 0.41] | 0.16 | 0.03 |
| litter and dumping (72,788) | litter (C) | 0.13 [0.03, 0.24] | 0.23 | 0.38 |
| trees: missing, dead, other (2,500) | green view | 0.10 [−0.01, 0.19] | 0.19 | −0.07 |
| street lighting (9,010) | street light pole (A) | −0.02 [−0.13, 0.09] | 0.00 | 0.14 |
| pavement maintenance (10,154) | surface damage (rare) | −0.38 [−0.46, −0.28] | −0.17 | 0.17 |

**Reading.**
- **Where both sources point the same way.**
  - Residents report obstacles most where the model sees obstructed sidewalks. This holds after accounting for density (0.53).
  - Parking reports follow the model's sidewalk parking, even though the model over-reports that item (tier C).
- **Where they cannot agree, by construction.**
  - A report of a broken street light says nothing about whether a pole is visible.
  - Surface damage is almost never reported by the model, so its buurt share is close to zero and the negative correlation is not interpretable.
- **Reports are behaviour, not only conditions.** Litter reports rise with the low-income share (0.38) more than with what the images show (0.13). Dumping and reporting habits differ between neighbourhoods, which is why the protocol pairs data with interviews.
- **What WP4 would add.** Walk-along interviews on the streets where the two sources disagree. This sample is the natural selection list for that.
