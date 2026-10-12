# WP2 synthetic mobility and physical activity: design (step 3.1), 2026-10-11

> **Everything produced from this design is SYNTHETIC.** No person was tracked. The data exist
> to show that the ELEVATE WP2 pipeline (cleaning → trips and stops → map matching → 25 m
> buffered segment exposure → half-hour time-weighted exposure → models) runs end to end on
> Amsterdam inputs. They carry no behavioural finding. Every file, figure and table made from
> them is labelled "synthetic".

## What the protocol fixes (taken over unchanged)

| Parameter | Protocol | Here |
|---|---|---|
| Participants | about 200 adults (180 give 96.6% power) | 200 synthetic persons |
| Monitoring | 7 days per participant | 7 consecutive days, Monday–Sunday |
| Daily window | 12 h/day in the power analysis | 07:00–19:00 local time |
| GPS | every ~30 s | one fix every 30 s while the phone records |
| Valid day / person | ≥ 8 h wear; ≥ 4 valid days | wear simulated; the same rules applied in cleaning |
| Cleaning | drop implausible speeds and horizontal error > 5 m; interpolate short gaps | same rules in step 3.4 |
| Steps | smartphone accelerometer, aggregated to epochs | steps per 30 s epoch, summed to half-hours |
| Step level | 9,448 steps/day (SD 2,218) | each person's mean daily steps drawn from N(9,448, 2,218), floored at 2,000 |
| Exposure | street-segment characteristics with optional 25 m buffer; time-weighted average per half-hour along paths | same, with the WP3 segment indicators |
| Outcome model | mixed negative binomial, random intercepts; effect log(1.15) per SD of greenness in the power analysis | step 4.3 |

## What we assume (not in the protocol; chosen to be plausible, not calibrated)

**Population and homes.**
- Buurten are split into tertiles by the CBS 2023 share of low-income households (mean income per inhabitant is filled for only 147 buurten). 68, 66 and 66 persons are drawn from the low, middle and high tertiles, a stand-in for the protocol's stratified sampling.
- Within a tertile, the buurt is drawn in proportion to its population. Buurten without income data are not used.
- The home is a random point on a local street, pedestrian street or footway inside the buurt, weighted by length.
- Attributes: gender (1:1); age group (18–34, 35–54, 55–75; 1:1:1); employed (65%, 40% for the 55–75 group); car available with probability equal to the buurt's cars per household, capped at 0.9.

**Destinations.**
- OpenStreetMap points of interest (step 3.2): supermarket and food shops, schools, parks, sports, healthcare, public-transport stops, offices and other workplaces, cafés and libraries.
- A destination of the needed type is chosen with probability ∝ exp(−d / s), up to 4 s away (the 5 nearest always qualify). The scale s is 1 km for daily shopping, 2 km for leisure and 3 km for other errands.
- A workplace is drawn once per employed person. Commute distance follows a log-normal distribution with a median of 4 km, and the workplace is the office or commercial POI nearest to the drawn distance and a random bearing.

**Daily programme.**
- Employed persons on weekdays: home → work (departure 07:30–09:00), work → home (16:30–18:00). With probability 0.3 they add a lunch-time walk, and with probability 0.4 an evening errand or leisure tour.
- Others and weekends: 1–3 tours, each to 1–2 destinations: shopping, park, sport, healthcare or social.
- Activity durations: shopping 10–40 min, park 20–90 min, sport 45–90 min, healthcare 20–60 min, social 60–180 min.

**Mode by trip distance** (logistic, then car access):
- walk: likely below 1 km, rare above 2.5 km;
- bicycle: most trips of 1–7.5 km;
- public transport or car: above 7.5 km. Car only if available, otherwise public transport.

**Routes.** Shortest paths on the WP3 network with each edge's length multiplied by U(1.0, 1.3), drawn per trip, so repeated trips vary a little.
- Walkers do not use cycle-only ways.
- Cyclists do not use footways.
- Cars and buses use main and local roads.
- Public-transport trips are routed along main roads at bus speed. The stop is reached on foot.

**Speeds.** Walk N(1.3, 0.15) m/s, bicycle N(4.3, 0.6) m/s, car or bus 7 m/s average with stops.

**GPS error.**
- Outdoors: Gaussian error with σ = 4 m. 5% of fixes have σ = 15 m. Each fix reports its own accuracy.
- Indoors at destinations: no fix with probability 0.7, otherwise an error of 20–60 m.
- Signal gaps of 1–10 min about twice a day. About 0.2% of fixes jump far enough to imply an impossible speed, so that the cleaning rules have something to remove.

**Wear.** Each day the phone is off or left behind for 0–3 h. About 10% of person-days fall below 8 h of wear and become invalid by the protocol's rule.

**Steps.**
- Walking epochs: cadence N(105, 8) steps/min.
- Cycling, car and bus epochs: Poisson(1) per 30 s.
- At destinations: Poisson with a rate by activity (work lowest, shopping and park highest).
- Each person's non-trip rates are scaled so that their mean valid-day total matches their drawn daily mean.

## Two step variants: no effect and a planted effect (decided 2026-10-11: both)

The persons, programmes, routes and GPS fixes are generated once. Steps are generated twice from the same movement, with the same random seed for everything except the effect. The pipeline test in WP5 (step 4.3) fits the protocol's model to both:

- **A. No effect.** Steps do not depend on the environment. The model should find an effect near zero, and the test shows that the pipeline produces no false effect.
- **B. Planted effect.** Walking steps in each epoch are multiplied by exp(β · z). Here z is the standardised green-view exposure of the street being walked (WP3 vegetation fraction), and β = log(1.15) per SD, the effect size of the protocol's power analysis. The model should recover β within its interval. This is how a simulation-based power analysis works (the protocol used R `simr`), and it shows the full chain from images to model estimate.

  The planted value is written in the data card and on every result. A recovered β is a check on the code, not evidence about Amsterdam.

## Outputs (step 3.3)

- `outputs/wp2/synthetic_persons.parquet`: person id, stratum, attributes and home buurt. The home point is not published; it is reported only as the buurt.
- `outputs/wp2/synthetic_gps.parquet`: one row per fix, with person, time, lon/lat, reported accuracy, true mode and true activity (the hidden truth, kept for checking the pipeline).
- `outputs/wp2/synthetic_steps_A_null.parquet` and `synthetic_steps_B_planted.parquet`: steps per person per 30 s epoch, plus a wear flag. B also stores the planted multiplier.
- `outputs/wp2/synthetic_truth.parquet`: the true trips and stays, for scoring trip and stop detection in step 3.4.

## Known simplifications

- No household structure, no children, no evening or night activity outside the 12-hour window.
- Mode choice ignores weather, parking and transit timetables.
- Indoor walking is approximated by Poisson steps.
- The WP3 indicators cover the sampled 526 km of street, not the whole network. Exposure is missing where a path runs along unobserved streets, and the share of time with exposure is reported.

## Revision 2026-10-11 (2)

A check of the first run found implausible walks. 128 of 2,441 walking legs were over 3 km, and they made up 42% of all walking distance. They came from lunch walks to parks 3–17 km away (the five nearest parks always qualified) and from detours over water. The generator now:

- sends a lunch walk only to a park or shop within 1 km;
- turns any walk longer than 3 km on the network, and any walk to or from a stop longer than 2 km, into a bicycle trip;
- includes the City's ferries (OSM `route=ferry`, Pampus excluded) as walk and bicycle edges, with 150 m added for boarding.

Result:

- No walking leg is over 3 km (the longest is 2.9 km).
- Walking totals 1,336 km and cycling 12,081 km.
- The 95th percentile of cycling trips fell from 12.3 to 6.6 km.
- Person means of daily steps average 9,587 with an SD of 2,248 across persons; the protocol gives 9,448 and 2,218.

Downstream steps 3.4, 4.3 and the site were rerun.
