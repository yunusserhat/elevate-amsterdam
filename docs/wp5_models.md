# WP5 model chain on SYNTHETIC data (step 4.3), revised 2026-10-11

> **SYNTHETIC.** Every number on this page comes from the synthetic WP2 data (`docs/wp2_synthetic_design.md`). The aim is to show that the protocol's model chain runs end to end, and to test it against a known answer. Nothing here is a finding about Amsterdam or about walking.

**Revision.** The synthetic data were regenerated on 2026-10-11 after three generator fixes:
- lunch walks were reaching parks 3–17 km away;
- walks longer than 3 km now become bicycle trips;
- the City's ferries were added to the network.

A first version of this page reported that adding park-stay minutes removed a bias. On the regenerated data that no longer holds, and this page describes the current run.

Scripts: `scripts/51_wp5_models.py` (envs/stats), `scripts/52_wp5_nbmm.R` (R 4.5, glmmTMB 1.1.15). Outputs in `outputs/wp5/`.

## Data for the models

- 4,496 valid half-hours of 200 persons:
  - valid day, ≥ 4 valid days, phone worn the whole half-hour;
  - ≥ 2 walking or cycling fixes with exposure.
- **Exposure:** green view of the observed WP3 segments within 25 m of each walking or cycling fix, averaged over the half-hour. It is standardised over these half-hours (mean 0.259, SD 0.108).
- **Outcome:** steps per half-hour:
  - **A:** generated without any environmental effect;
  - **B:** generated with a planted effect of β = log(1.15) = 0.140 per SD of the true green view.

## Mixed negative binomial model: a specification curve

Model: `steps ~ z_green + hour + weekend [+ activity minutes] + (1 | person)`, glmmTMB `nbinom2`. Green-view coefficient with its 95% Wald interval:

| Adjusted for | A (truth 0) | B (truth 0.140) |
|---|---|---|
| hour, weekend | 0.012 [−0.015, 0.039] | 0.140 [0.113, 0.166] |
| + walking and cycling minutes | 0.019 [−0.004, 0.043] | 0.147 [0.124, 0.170] |
| + park-stay minutes, linear | −0.025 [−0.049, −0.002] | 0.101 [0.077, 0.124] |
| + all minutes, log(1 + x) | −0.019 [−0.042, 0.005] | 0.107 [0.084, 0.131] |
| + all minutes, binned | −0.021 [−0.045, 0.002] | 0.105 [0.081, 0.128] |

With the true exposure, B gives 0.163 when adjusted for walking and cycling minutes and 0.110 when park-stay minutes are added.

**Reading.**
- **Why no adjustment is exactly right.** The generator builds each half-hour's step rate additively: minutes walking, cycling and staying, with a park stay stepping fast, then multiplied in B by exp(β z). A log-linear model with activity minutes as covariates is therefore misspecified, whatever form the minutes take.
- **What the curve shows:**
  - Without park-stay minutes, the null comes out near 0 and the planted effect near 0.14.
  - With them, the null turns slightly negative and the planted effect falls to about 0.10.
  - Park visits raise both green view and steps, so a DAG lists them as a confounder. Here, adjusting for them overcorrects.
- **The lesson for the protocol.** A DAG settles which variables to adjust for, not the functional form. The protocol's plan for sensitivity analyses and residential self-selection matters as much as the DAG. In the first version of these data the same adjustment removed a bias instead; the direction depends on the data.

**Precision by sample size** (B, walking and cycling minutes, a specification fixed before this run; 20 random person subsets each):

| Persons | Mean estimate | SD | Mean CI width | CI excludes 0 | CI covers 0.140 |
|---|---|---|---|---|---|
| 50 | 0.151 | 0.033 | 0.096 | 100% | 85% |
| 100 | 0.148 | 0.016 | 0.067 | 100% | 95% |
| 150 | 0.147 | 0.011 | 0.054 | 100% | 95% |

The synthetic data have less noise than real data, so this says nothing about real power. The protocol's own simulation gives 96.6% power with 180 persons.

**Moderation by income stratum** (B, walking and cycling minutes; no moderation was generated):
- The green-view × stratum terms are 0.072 (middle, p = 0.015) and 0.092 (low, p = 0.002) against the high stratum. Both look "significant" with no true difference.
- Moderation estimates inherit any misspecification of the main model, and differences in activity patterns between strata are enough to produce them.
- Moderation tests (protocol objective 4) need a pre-specified hypothesis, a correction for multiple tests and the same sensitivity analyses.

## ST-DBSCAN

Per person, on 165,757 cleaned observed fixes; eps 50 m, 10 min, minimum 10 fixes. 881 clusters.
- Against the synthetic truth: 98.2% of fixes classified correctly as stationary or moving (precision 0.94, recall 0.98). It agrees with the step 3.4 stay rule on 96.4% of fixes.
- Steps per 30 s: 30.5 in clusters and 10.2 on paths.
- **Caveat:** the protocol's 5 m error rule removes almost all indoor fixes. The remaining clusters are outdoor places (parks, stops), so the most and least active deciles are both mostly parks.

## XGBoost + SHAP

- **Model:** Poisson objective, person-grouped 5-fold cross-validation. It explains 47% (A) and 50% (B) of deviance.
- **Green view, mean SHAP by decile** (`xgb_shap_green_view_synthetic.png`):
  - **A is not flat:** it falls from +0.06 to −0.10 towards the greenest half-hours, an association the trees learn from activity composition, not an effect.
  - **B rises:** from −0.08 to +0.12.
- **Lesson:** SHAP shows what the model learned, not what causes steps; only the comparison with A isolates the planted part.
- **Mean |SHAP| ranking:** walking, cycling and park minutes rank first in both; green view is 6th of 11 in both.
