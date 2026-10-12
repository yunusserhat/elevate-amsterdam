# Hugging Face Space: outline (step 8.1), DRAFT for approval, 2026-10-11

## Principles
- Plain surface, depth on request. Each page has a short text, one main figure and a closed "Details" section.
- No decorative gradients, icons, mascots or "AI" styling. System fonts, one accent colour, light and dark themes.
- Every page states its data status in one word under the title:
  - **real data**: City panoramas, CBS, OSM, Signalen;
  - **synthetic**: WP2 and the WP5 model chain;
  - **hypothetical**: WP1;
  - **idea**: WP6 visualisations.
- Model outputs are called observations. Nothing is called a measure of health, safety or walkability.
- No street images are hosted, except the curated WP6 before/after pairs.
  - Every view links to the City's own image server: `api.data.amsterdam.nl/panorama/thumbnail/<pano_id>/?heading=<deg>` and the equirectangular original.
  - Attribution: "Panoramas: Gemeente Amsterdam, CC BY 4.0".

## Pages
**Home.**
- One sentence: *A worked example of the seven work packages of the ELEVATE protocol (Helbich et al., PLOS ONE 2026) in Amsterdam, built from the City's 2025 street panoramas, open models and synthetic mobility data.*
- Seven cards, one line each, with the status word.
- Footer: data card, licences, code, contact.

**WP1 Co-creation (hypothetical).**
- Main figure: the requirement matrix, filterable by stakeholder group.
- Details: the draft indicator long list with reliability and limits, and how a workshop would re-rank it.

**WP2 Mobility and physical activity (synthetic).**
- Main figure: one synthetic person-day on a map: true path, GPS fixes, detected stays and modes.
- Details: protocol parameters against the assumptions; pipeline scores against the synthetic truth.

**WP3 Streetscape analysis (real data).** The core page.
- Main figure: a map of observed street segments, coloured by one indicator chosen from a short list. Each indicator carries its reliability tier.
- Click a segment: its panoramas, the four views' observations and links to the City images.
- A search box ("benches under trees") ranks the 99,100 views. The query is embedded on the server's CPU.
- Details: data and sample, the 4B model and template v2, evaluation against the 27B model (tier table), segmentation, perception stand-in, coverage.

**WP4 Qualitative and mixed methods (real data, light).**
- Main figure: the triangulation table, residents' reports against model observations.
- Details: where the two disagree, as a selection list for walk-along interviews.

**WP5 Spatial modelling.**
- Main figure (real data): the equity map with LISA clusters, and the table of gradients against income.
- Second block (synthetic): the model-chain results, with the planted effect recovered and the confounding lesson.
- Details: methods, all coefficients, ST-DBSCAN, XGBoost + SHAP.

**WP6 Recommender (idea).**
- Main figure: the evidence layer of gaps, with the equity priority.
- Before/after gallery (~50), each pair with its re-audit result and the share of the image changed outside the target. Failures are shown too.
- A question box with precomputed grounded answers and their cited views. New questions list matching views.
- "Upload a street photo": similar Amsterdam views by image embedding on CPU. It is not a new audit, and the page says so.

**WP7 Dissemination.**
- This site as the story map.
- Data card, licences, provenance (model revisions, processing ids), code, and what is deliberately not shared.

## Technical shape
- **Space type:** Docker Space, so the page is plain hand-written HTML and not a Gradio look.
  - Back end: FastAPI (Python), CPU only.
  - Front end: static HTML, CSS and vanilla JS with Leaflet and the OSM basemap.
- **API:** `/api/search?q=`, `/api/near?lat=&lon=`, `/api/segment/<id>`, `/api/layer/<indicator>`, `/api/qa`, `/api/similar` (image upload).
- **Data shipped:**
  - index vectors as float16, ~51 MB;
  - view records ~17 MB;
  - simplified segment and buurt layers;
  - WP6 pairs;
  - small JSON and CSV tables.
  - EmbeddingGemma 2 (Apache-2.0, not gated) is downloaded at start.
- **Name and visibility:** to be decided by the user; private until approved.
