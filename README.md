# ELEVATE in Amsterdam

An independent demonstration of the ELEVATE protocol (Helbich et al. 2026, *PLOS ONE* 21(5): e0348889)
in Amsterdam, built by Yunus Serhat Bıçakçı. It is not produced by or affiliated with the ELEVATE
project or Utrecht University.

- **Website:** https://www.yunusserhat.com/elevate/
- **Search API (Hugging Face Space):** https://huggingface.co/spaces/yunusserhat/elevate-amsterdam
- **Street-view audit and search:** [GeoAI-VLM 0.5.0](https://github.com/yunusserhat/GeoAI-VLM)

Model outputs are observations, not measurements of health, safety or walkability. Synthetic data is
labelled as such wherever it appears.

## What each work package shows

| WP | Status | What was done |
|---|---|---|
| WP1 Co-creation | hypothetical | 16 stakeholder questions linked to indicators, data and tool features; a draft list of 15 indicators for workshops to rank. Role-play, not workshop results. |
| WP2 Mobility and activity | synthetic | 200 synthetic people, 7 days of GPS every 30 s and steps, built to the protocol's parameters, run through its cleaning, mode detection and half-hourly exposure. No one was tracked. |
| WP3 Streetscape analysis | real data | 24,775 of the City's 2025 panoramas, 99,100 views, audited for 19 observable features with GeoAI-VLM, Cityscapes segmentation, the ELEVATE team's own Amsterdam perception model (svi_percept) with two Place Pulse models as cross-checks, an image and text search index, and aggregation to 19,410 street segments and 493 buurten with coverage. |
| WP4 Qualitative and mixed methods | real data | 223,592 public-space reports to the city (June–October 2025) compared with the image observations by buurt. |
| WP5 Spatial modelling | real data + synthetic | Equity analysis against CBS income, LISA clusters and 21 priority buurten (real data); the protocol's mixed negative binomial, ST-DBSCAN and XGBoost + SHAP run on the synthetic data with a planted effect (a pipeline test). |
| WP6 Recommender | idea | A gap layer, second-model confirmation, 47 AI visualisations re-audited after editing, and cited question answering. Every edited image is an AI visualisation, not a design. |
| WP7 Dissemination and open science | documentation | Story-map website, data card, licences, provenance and this code. |

## Layout

- `scripts/`: the pipeline, numbered by step.
  - `11`–`15`: panorama inventory, study area, coverage, sample and view extraction.
  - `21`–`32`: audit, reference model, pilot, segmentation, perception, search index, aggregation and evaluation.
  - `40`–`42`: WP2 synthetic population, generation and the protocol's pipeline.
  - `50`–`52`: WP5 equity analysis and models (`52` is R, glmmTMB).
  - `60`: WP1; `65`: WP4; `70`–`76`: WP6.
  - `80`–`86`: site data, packaging and publishing.
  - `87`: the 3DBAG buildings the home page serves itself: every LoD2.2 tile inside the A10 ring
    (`--crop 4.835,52.333,4.955,52.402 --refine-all --no-normals`), attribute tables and normals removed (a plain
    per-building id stays, for the drop-in), geometry re-encoded with Draco through `tools/gltf/` (Node 24,
    gltf-transform; `npm install` there first); the empty inner tiles always refine, so every building is drawn.
- `docs/`: protocol map, data card, licences, evaluation and design notes, one file per topic.
- `survey/schema.sql`: the pilot survey's PostgreSQL schema (Supabase): insert-only row-level security, constraints, a per-session cap and aggregate-only results.
- `space/`: the FastAPI back end (`app/`), the website source (`static/`) and the Space's Dockerfile.

`env.sh` sets the paths and Python environments used on the GPU server (vLLM, Transformers 5 with
Sentence Transformers, diffusers, and a statistics environment). Paths point to the original server
and need adjusting elsewhere.

## Not in this repository

- Street images, model outputs and intermediate tables, because of their size. The tables behind the
  website are published with the site; the panoramas come from the City of Amsterdam's API.
- Keys or secrets. The survey server reads `SUPABASE_URL` and `SUPABASE_KEY` from its environment, and
  the browser talks only to the API, never to the database.

## Data and model licences

See `docs/licences.md`. The main sources are the City of Amsterdam's 2025 panoramas (CC BY 4.0), PDOK
aerial imagery 2025 (CC BY 4.0), 3DBAG (CC BY 4.0), OpenStreetMap (ODbL) and CBS (CC BY 4.0 / CC0).
Some models carry non-commercial licences (SegFormer-B5 Cityscapes, Qwen-Image); they were used for
research only.

## Licence

The code is released under the GNU General Public License v3.0 (see `LICENSE`), the same licence as
GeoAI-VLM. Data and model licences are listed in `docs/licences.md`.
