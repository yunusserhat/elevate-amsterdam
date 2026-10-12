# WP3 pilot (step 2.1), 2026-10-10

**Setup.** 500 sampled panoramas, four views each (2,000 views). Qwen3.5-4B (revision 851bf6e8) through vLLM 0.31 with JSON-schema constrained decoding on one RTX 5090. Template `active_mobility_audit_v1`.

**Throughput.** 0.29 s per view, model loading included. No parse failures and no invalid states.

**Which view sees what.** Share of views reporting the item present, and in brackets the share marked not visible or uncertain:

| Item | Front | Right | Back | Left |
|---|---|---|---|---|
| sidewalk | 98 (2) | 68 (32) | 99 (1) | 83 (17) |
| street trees | 85 (1) | 53 (26) | 84 (1) | 75 (6) |
| cycle infrastructure | 63 (13) | 26 (64) | 64 (15) | 39 (46) |
| pedestrian crossing | 20 (56) | 10 (84) | 19 (56) | 12 (78) |
| active frontage | 89 (4) | 77 (15) | 90 (2) | 70 (19) |
| view obstruction | 22 | 54 | 20 | 70 |

Front and back views look along the street and see most items. Right and left views are close façade views: useful for frontage, blind for most ground-level items.

**Error found by visual check.** The 4B model reported parked bicycles as `cycle_infrastructure` ("bicycles parked on the sidewalk") and as `sidewalk_parking`. In one front view it missed a clear red, separated cycle track and cited the parked bikes instead. The v1 definitions are ambiguous in a Dutch context: "Infrastructure for cycling is visible" and "A vehicle is parked on the sidewalk".

**Decision.** We added `active_mobility_audit_v2` to GeoAI-VLM.
- Items, states, attributes and JSON schema are unchanged; the v1 text is unchanged, verified by SHA-256.
- Four definitions are sharpened: sidewalk, sidewalk_obstruction, cycle_infrastructure and sidewalk_parking. Parked bicycles are explicitly excluded from cycle infrastructure and from sidewalk parking.
- One rule is added about parked bicycles.

**Re-pilot.** The same 2,000 views are audited with v2. A reference run of Qwen3.8-27B with v2 covers 200 front views. The comparison of v1 and v2 against the reference decides which template the full run uses.

## Result of the comparison (200 front views, reference Qwen3.8-27B-FP8 with v2; `outputs/wp3/pilot_comparison.csv`)

Same state as the reference, with Cohen's kappa in brackets:

| Item | 4B v1 | 4B v2 | 27B v1 |
|---|---|---|---|
| all items | 69.4% (0.51) | **74.8% (0.59)** | 92.9% (0.88) |
| cycle_infrastructure | 33.5% (0.08) | **51.5% (0.19)** | 67.9% (0.36) |
| sidewalk_parking | 27.8% (0.01) | **74.7% (0.03)** | 95.2% (0.29) |
| sidewalk_obstruction | 76.8% (0.28) | 83.5% (0.31) | 91.4% (0.64) |
| street_trees | 98.5% (0.91) | 99.0% (0.94) | 100% (1.00) |
| pedestrian_crossing | 44.3% (0.22) | 45.4% (0.22) | 89.8% (0.79) |
| seating | 55.7% (0.21) | 60.3% (0.21) | 89.3% (0.65) |

Share reported present, against the reference (v2):
- cycle_infrastructure: 4B v1 68%, 4B v2 40%, 27B v1 35%; reference 13%.
- sidewalk_parking: 72%, 24%, 4%; reference 1%.

**Reading.**
- The v1 wording misleads even the 27B model: it reported cycle infrastructure in 35% of views under v1 against 14% under v2.
- v2 removes most of the parked-bicycle errors.
- The 4B model still over-reports cycle infrastructure and is weak on crossings, seating and traffic calming. Report this on the site. Items with low agreement need the reference model or manual labels before they are used as indicators.
- The reference runs had 6 (v2) and 8 (v1) unparseable answers out of 200.

**Decision.** The full run uses v2 (started 2026-10-10 22:06).

> **Correction (2026-10-11, step 2.7):** on 500 citywide views, with agreement measured on presence (present vs not present), crossings and seating are among the best items (presence κ 0.83 and 0.84). The weakness reported above was mostly disagreement on "absent" vs "not visible". See `docs/wp3_evaluation.md`.
