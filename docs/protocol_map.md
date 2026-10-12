# ELEVATE protocol map (backbone of the demonstration)

Source: Helbich M, Labib SM, Liedtke I, et al. (2026) "Healthy living environments create
healthy physical activity-related habits (ELEVATE): A protocol for a smartphone-based
mixed-method study". PLOS ONE 21(5): e0348889. CC BY. Read in full on 2026-10-10.
WP numbers follow the project website (elevate.sites.uu.nl/work-packages); protocol
section names are given for each.

## Objectives (protocol, "Objectives")

1. How people perceive and experience the streetscape during walking-related physical activity (WPA).
2. Identify and co-create streetscape types and settings where adults are most and least active.
3. How streetscape environments met during daily mobility are (non-linearly) associated with WPA.
4. Whether associations are moderated by behaviour and socio-demographics (income, education, gender).
5. Co-create streetscape design principles.
6. A recommender system for urban (re)design suggestions.

Study site: Amsterdam, chosen for documented socio-economic and health inequalities and for
"rich diversity and exclusive street view data". About 200 adults, recruited by multi-stage
stratified sampling, with neighbourhood ambassadors for underserved groups. 2026–2030.

## WP1 Co-creation and stakeholder engagement (protocol: "Co-creating physically active daily life environments")

**Protocol:**
- Stakeholders: planners, health professionals, municipal representatives and residents. Later also urban planners, landscape architects and other design professionals.
- Stakeholder mapping across the public, private and civic domains.
- 3–4 co-creation workshops, including focus groups.
- Workshop priorities become a **ranked indicator list** that guides the quantitative indicators. Participants also help frame survey items.
- Interactive feedback sessions on preliminary models.
- Design-thinking sessions in year 4 to test the recommender's usability, realism and social fit.

**Our demonstration:**
- A stakeholder requirement matrix, built by role-playing each group: their questions → the indicator or data that answers each → the tool feature → what to validate in a workshop.
- A draft ranked indicator list, linked to what the images can and cannot show.
- Labelled **hypothetical, to be validated in WP1 workshops**.

## WP2 Smartphone-based mobility and physical activity (protocol: "Measuring PA under free-living conditions", "Power analysis")

**Protocol:**
- Smartphone accelerometer step counts (low-pass filtered) and GPS every ~30 s.
- Cleaning: drop implausible speeds and horizontal error > 5 m, interpolate short gaps, then derive transport modes.
- Steps aggregated into epochs along GPS paths. Experience sampling is possible.
- 7 days per participant. A valid day has ≥ 8 h of wear; participants need ≥ 4 valid days.
- Exposure is assigned from **street-segment characteristics with optional 25 m buffering**, then aggregated to the outcome resolution (**half-hour**) as **time-weighted averages along mobility paths**.
- Power analysis:
  - steps per half-hour over 12 h/day × 7 days, negative binomial outcome;
  - baseline 9,448 steps/day (SD 2,218);
  - greenness exposure NDVI mean 0.425 (SD 0.107), effect log(1.15);
  - mixed NB model with random intercepts, 500 simulations, R `simr`;
  - 180 participants give 96.6% power; target 200.

**Our demonstration:**
- Synthetic population of 200 placed by CBS buurt population.
- 7 days, activity chains to OSM destinations.
- GPS fixes every 30 s with realistic error and gaps.
- Steps calibrated to the protocol's distribution.
- The protocol's cleaning → map-matching → 25 m buffered segment exposure → half-hour time-weighted averages, on the WP3 indicators.
- Everything labelled **synthetic**; the aim is a working pipeline, not behavioural findings.

## WP3 AI-based streetscape and environmental analysis (protocol: "Micro-scale assessment of the built and natural environment")

**Protocol:**
- GIS metrics: walking infrastructure, traffic speed, street connectivity, access to greenery, land-use mix, amenities; from orthophotos and OSM.
- Eye-level indicators: computer vision on public street view images (Mapillary named), with preprocessing to filter unsuitable images.
- **Cityscapes-trained** semantic segmentation.
- **Perceptions** (safety, pleasantness, walkability) from models trained on their 20,000+ crowdsourced stated preferences.
- Domain experts compare model outputs with **manually annotated ground truth**.
- Indicators disaggregated at the **street segment** level.

**Our demonstration:**
- The City of Amsterdam's 2025 panoramas: 25k panoramas × 4 cube faces.
- GeoAI-VLM audit with 19 observable features, plus Cityscapes segmentation.
- The team's own Amsterdam perception model (svi_percept, public on Hugging Face; from the Percept survey, Danish et al. 2025) on all views, with two Place Pulse models as cross-checks. (Until 2026-10-12 the demonstration used the Place Pulse models as a stand-in, wrongly believing the team's model was not public.)
- EmbeddingGemma 2 index.
- Segment, buurt and city aggregation with coverage.
- Validation tooling: agreement checks and a seeded stratified sample ready for expert labelling.

## WP4 Qualitative and mixed methods (protocol: "In-situ qualitative inquiries", thematic analysis in "Data analysis plan")

**Protocol:**
- Semi-structured on-site and walk-along interviews, with photos taken to document places.
- Topics: activity environments, place meanings, routines, walking practices, facilitators and barriers.
- Thematic analysis in NVivo (deductive and inductive) with a codebook.
- Triangulation with quantitative findings.
- About 20 participants (saturation), 25% from marginalised groups.
- On-site interviews early (to inform the survey), walk-along interviews later (to ground and explain statistics).

**Our demonstration:**
- How image observations can spatially ground interview material: a route, its places and what the images show there.
- If Amsterdam's public-space reports (Signalen) are open, triangulation with them.
- Clearly marked illustrative where no real qualitative data exist.

## WP5 Spatial modelling and data integration (protocol: "Data analysis plan")

**Protocol:**
- Descriptives; χ² and Wilcoxon tests across socio-economic groups.
- **ST-DBSCAN** to find stationary clusters and linear paths, i.e. where people are most and least active.
- **Mixed negative binomial** models of half-hourly steps on environment, with **socio-economic moderation**.
- DAG-based adjustment and control for residential self-selection.
- Exploratory **XGBoost + SHAP** for non-linearity.

**Our demonstration:**
- Equity analysis on real data: segment and buurt indicators and coverage against CBS income and deprivation.
- Spatial autocorrelation (Moran's I / LISA).
- The protocol's model specification run end to end on the synthetic WP2 data: ST-DBSCAN, mixed NB, XGBoost/SHAP. Labelled a **pipeline test**.

## WP6 AI recommender system for urban design (protocol: "Recommender system for urban (re)design")

**Protocol:**
- Evidence-based, **equity-sensitive** suggestions, especially in low-activity and underserved neighbourhoods.
- Built on pre-trained or fine-tuned models (CLIP, "learning transferable visual models", world models), with training balanced across neighbourhood typologies.
- Fuses spatial, morphological, visual and textual data.
- **Fig 2 example:** "intersections without marked crossings are associated with low WPA" → traffic calming or wayfinding.
- Validation and co-creation workshops.
- A **publicly accessible chat-based web interface**: upload a street image or give an address → explainable recommendations grounded in data and citizen input, with feedback loops.
- Pipeline hosted openly (e.g. **Hugging Face**). Pilot in Amsterdam, prioritising areas where low WPA coincides with disadvantage.
- Fig 2's street images are illustrative, made with ChatGPT; the protocol does not plan generated imagery for participants.

**Our demonstration:**
- Evidence layer of segment "gaps", prioritised with CBS disadvantage.
- Grounded Q&A that cites images.
- An image or address flow in the Space.
- **~50 before/after visualisations** made with Qwen-Image-2.1, from detected gaps, **including marked crossings at intersections without them** (the protocol's own example).
- Each edit re-audited, with change outside the edit measured.
- Presented as an idea I would bring, not as part of the protocol.

## WP7 Dissemination, open science and impact (protocol: "Dissemination strategies", "Data management")

**Protocol:**
- Open-access papers, FAIR, environmental data made public, GPS tracks never shared and geomasked.
- Demonstration sessions with the recommender as a software-as-a-service platform.
- A website with a **story map** for residents.
- Infographics and social media.

**Our demonstration:**
- The Hugging Face Space works as a story map and demo.
- Data card, licences, provenance (model revisions, processing ids), code.
- Synthetic GPS only; no personal data.
