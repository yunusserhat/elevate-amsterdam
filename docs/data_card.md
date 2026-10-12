# Data card: ELEVATE Amsterdam demonstration

## Street-level imagery

- **Source:** City of Amsterdam, *Kernregistratie Panoramabeelden*, via
  `https://api.data.amsterdam.nl/panorama/panoramas/`. Licence CC BY 4.0. Attribution:
  "Panoramas: Gemeente Amsterdam, CC BY 4.0".
- **What it is:** 360° equirectangular panoramas recorded from a car-mounted camera.
  The 2025 recordings are spaced 5 m apart, all on land (`surface_type` L), mission
  type `bi`. North-aligned: the centre column faces north, which we tested on
  consecutive frames.
- **2025 inventory:** 571,954 panoramas.
  - By month: January 29,519; June 112,680; July 136,129; August 4,660; September 258,280; October 21,406; December 9,280.
  - Local capture time about 05:00–18:00.
- **Network:** OpenStreetMap ways people can walk or cycle along, clipped to the CBS
  2023 boundary of the municipality. The rules are in `scripts/12_study_area.py`.
  - 115,569 segments, 4,813 km.
- **Coverage of that network in 2025:** 53% of street length lies within 12.5 m of a panorama.
  - By road group: main roads 94%, local streets 93%, cycleways 53%, pedestrian streets 30%, bus corridors 29%, service roads 19%, **footways and paths 8%**.
  - A car-based archive sees streets, not paths.
- **Audit sample (seed 2026):** 24,775 panoramas.
  - Eligible panoramas: matched to a segment within 20 m without ambiguity, June–October (leaf-on), daylight by solar elevation, known driving bearing. 388,533 qualify.
  - Spread: at most one per 50 m piece of a segment (60,608 pieces); buurten sampled in proportion with a floor of 5. The sample covers 19,410 segments and 493 of 519 buurten.
- **Views:** four per panorama (front, right, back, left of the driving direction). Each is a 90° horizontal field of view, 768 × 576 px, cropped below about −27° where the car roof begins. 99,100 views in total.
  - Only views are kept; panoramas are discarded after rendering.
  - Every view has a SHA-256 in `data/derived/views_manifest.parquet`.

## Neighbourhoods and socio-economic data

CBS Wijk- en Buurtkaart 2023 (PDOK WFS, CC0): 519 buurten.
- 2023 is the latest edition with income data. 420 buurten have the share of low-income households; CBS suppresses small buurten, which stay missing.
- Fields kept: population, density, households, address density, low-income shares, income per inhabitant, cars per household.

## Known limitations

- Imagery is from one year (2025), on days the city's car drove, in summer daylight.
  It shows what was visible then, from the carriageway.
- Footways, park paths and many cycleways are rarely recorded. Indicators for them rest on few images, and the map says so.
- The view headings follow the driving direction estimated from consecutive frames. On sharp turns it can be off by some degrees.
- People and licence plates are blurred by the city before publication. We do not store the panoramas themselves.
