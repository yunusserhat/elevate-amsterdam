# WP3 evaluation of the 4B audit (step 2.7), 2026-10-11

Scripts: `scripts/30_eval_sample.py`, `32_reference_eval.sh`, `31_eval.py`. Outputs in `outputs/wp3/eval/`.

**What this is and is not.**
- The reference is a larger open model, Qwen3.8-27B-FP8, run with the same template v2. Agreement with it shows how much the small model's answers depend on model size.
- It is **not accuracy**: neither model is ground truth.
- The protocol asks domain experts to compare model outputs with manually annotated ground truth. `outputs/wp3/manual_label_sample.csv` is ready for that (see below).

## Samples

- **eval_500:** 500 views, one per panorama, 125 per view direction.
  - Panoramas are drawn in proportion to road group: local 253, main 119, cycleway 87, footway/path 19, service 11, pedestrian 8, bus 3. Seed 20261011.
  - The 500 pilot panoramas are excluded, because the v2 definitions were written after looking at them. The pilot also came from a few recordings, not from the whole city.
- **Reference run** (vLLM, both GPUs, 0.96 s per view):
  - 31 of 500 answers (6.2%) were unparseable: JSON-schema decoding ran into a whitespace loop until the token limit.
  - They were re-run once with `repetition_penalty 1.05`; 27 recovered. 496 views are compared.

## (a) Agreement with the reference (496 views × 19 items)

All items together:
- Same state in 75.6% of item judgements [95% bootstrap interval 74.7–76.5]. Cohen's κ over states is 0.63 [0.62–0.65].
- Presence (present vs not present): 90.3% agreement, κ 0.77 [0.76–0.78].
- By direction: front 73.6%, back 74.9%, left 73.9%, right 79.9%.

**Why tiers use presence.** The indicators use presence only: present / (present + absent).
- A κ over all states also rewards agreeing on "absent" vs "not visible". It rates near-absent items highly; for example sidewalk_gap gets 0.73 while neither model reports the item present.
- The first tier rule used that κ. It was replaced by presence κ after the first look at the results, and both are reported.

**Item tiers** (A: presence κ ≥ 0.6; B: 0.4–0.6; C: < 0.4; "rare": the reference reports the item in < 3% of views, so κ is not informative):

| Item | Presence κ [95% CI] | Present: 4B / reference | State κ | Tier |
|---|---|---|---|---|
| seating | 0.84 [0.72–0.94] | 5.0% / 5.4% | 0.39 | A |
| pedestrian_crossing | 0.83 [0.76–0.90] | 13.5% / 12.9% | 0.41 | A |
| sidewalk | 0.81 [0.74–0.86] | 67.9% / 71.0% | 0.69 | A |
| street_trees | 0.75 [0.65–0.84] | 89.5% / 84.7% | 0.69 | A |
| intersection | 0.75 [0.67–0.81] | 27.4% / 20.6% | 0.49 | A |
| street_light_pole | 0.74 [0.68–0.80] | 57.3% / 67.9% | 0.54 | A |
| shade_on_walkway | 0.73 [0.67–0.79] | 47.8% / 46.0% | 0.73 | A |
| view_obstruction | 0.62 [0.55–0.70] | 31.8% / 27.4% | 0.46 | A |
| curb_ramp | 0.60 [0.44–0.73] | 5.4% / 7.5% | 0.24 | B |
| cycle_infrastructure | 0.59 [0.51–0.67] | 28.0% / 15.7% | 0.46 | B |
| sidewalk_obstruction | 0.54 [0.46–0.61] | 38.5% / 48.2% | 0.60 | B |
| active_frontage | 0.53 [0.46–0.60] | 49.6% / 29.0% | 0.46 | B |
| other_greenery | 0.40 [0.23–0.55] | 96.2% / 91.7% | 0.39 | B |
| litter | 0.18 [−0.01–0.39] | 0.8% / 3.4% | 0.52 | C |
| sidewalk_parking | 0.15 [0.04–0.26] | 13.7% / 3.2% | 0.56 | C |
| accessibility_barrier | 0.04 [0.00–0.10] | 0.8% / 20.6% | 0.44 | C |
| surface_damage | — | 0.6% / 0.2% | 0.64 | rare |
| traffic_calming | — | 3.2% / 2.8% | 0.53 | rare |
| sidewalk_gap | — | 0.2% / 0.0% | 0.73 | rare |

**Reading.**
- **Tier A items** can be used as street indicators, with the agreement shown next to them.
- **Tier B items** need the caveat on the page:
  - cycle infrastructure is still over-reported (28% against 16%);
  - active frontage is over-reported (50% against 29%).
- **Tier C items** are not used as indicators:
  - the 4B model over-reports sidewalk parking four-fold;
  - it almost never reports accessibility barriers, which the reference reports in a fifth of views;
  - litter rests on too few cases.
- **Rare items** cannot be judged from 500 views.
- **Correction to the pilot reading** (`docs/wp3_pilot.md`): the pilot called the 4B model weak on crossings and seating. That was mostly disagreement on "absent" vs "not visible". On presence, both are tier A here.

## (b) Stability

The same 2,000 pilot views were audited twice with identical settings: the v2 pilot run and the full run, in different batches.
- 96.2% of item states are the same, κ 0.94.
- The lowest item is view_obstruction at 92.1%; no item is below 90%.
- Temperature 0 is not fully deterministic under vLLM batching.

## (c) Front against back of the same panorama (24,775 panoramas)

- Agreement is 45–97% (presence rates are close in both directions).
- κ is 0.17–0.50, lowest for intersection and pedestrian crossing, which are often ahead in one direction only.
- The two views look at different stretches of street, so this measures how local the observations are, not error.

## (d) Segmentation cross-check (AUC of a Cityscapes class fraction, present vs absent)

| Item | Class | AUC |
|---|---|---|
| other_greenery | vegetation + terrain | 0.98 |
| street_trees | vegetation | 0.96 |
| shade_on_walkway | vegetation | 0.88 |
| street_light_pole | pole | 0.81 |
| sidewalk_parking | car | 0.80 |
| active_frontage | building | 0.71 |
| cycle_infrastructure | bicycle | 0.50 |

- The bicycle pixel fraction does not separate views with and without reported cycle infrastructure (AUC 0.50). So the v2 model's cycle infrastructure is not driven by parked bicycles, the v1 error.
- Sidewalk has only one "absent" view, so its AUC is not meaningful.

## Manual labelling sample (ready, not labelled)

`outputs/wp3/manual_label_sample.csv`:
- 380 usable views; pilot and eval_500 excluded.
- For each item, 10 views the 4B model reports present and 10 it does not, so rare items have positives. Seed 20261011.
- Columns: view path, direction and heading, location, buurt, road group, the City's panorama API URL, the model's 19 states, and 19 empty `label_*` columns.
- The sample over-represents positives by design, so agreement on it must be weighted back to the full set.
