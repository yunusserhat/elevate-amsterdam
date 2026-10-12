# WP3 perceived qualities (step 2.4), 2026-10-11

**Why a stand-in (superseded 2026-10-12, see step 2.4b).** ELEVATE derives perceptions (safety, pleasantness, walkability) from models trained on its own 20,000+ crowdsourced stated preferences (protocol, WP3). We believed those models were not public; they are. We use open models trained on Place Pulse 2.0 for a similar purpose and label them as a stand-in, not as ELEVATE's model.

**Models.** Both come from the NUS Urban Analytics Lab and both are used by ZenSVI (MIT, `zensvi.cv.ClassifierPerceptionViT` and `ClassifierPerception`).

| Prefix | Model | Role | Architecture and score |
|---|---|---|---|
| `vit_` | `Jiani11/human-perception-place-pulse` @ 402164b5, Apache-2.0 | primary | ViT-B/16 at 384 px, one binary head per question; score = P(class 1) × 10, range 0–10 (authors' `eval.py`). Model card accuracy: safe 76.7%, lively 77.1%, wealthy 72.9%, beautiful 76.9%, boring 61.6%, depressing 67.2%. Used for Global Streetscapes (Hou et al. 2024, ISPRS J. 215:216–238). |
| `r50_` | `seshing/placepulse` @ 1d7f85c4, no licence file or model card | second opinion | Architecture taken from ZenSVI's code: ResNet-50 at 224 px, five classes; score = Σ p_k × (1, 3, 5, 7, 9), range 1–9. Used internally (decision 2026-10-11). |

Questions: safety, lively, wealthy, beautiful (high = positive); boring, depressing (high = negative).

Place Pulse 2.0: 110,988 Google Street View images from 56 cities in 28 countries, captured 2007–2012; 1.17 million pairwise comparisons by 81,630 online volunteers (Dubey et al. 2016, arXiv:1608.01769). We did not check whether Amsterdam is among the 56 cities.

**Loading.** No code from the checkpoints runs. The `seshing` files are plain state dicts and load with `torch.load(weights_only=True)`. The `Jiani11` files are pickled whole models. Their pickle globals were listed without unpickling: torch and torchvision ViT classes and `Model_01.Net` only. They are read with the weights-only unpickler and an explicit allow-list, and only their state dicts are kept. Both load with `strict=True` into architectures built in `scripts/26_perception.py`.

**Run.** 99,100 views × 6 questions × 2 models on two GPUs, 15 min. Input is the authors' transform: a plain resize of the 768 × 576 view to 384 × 384 (ViT) or 224 × 224 (ResNet), with ImageNet normalisation. Output: `outputs/wp3/perception/placepulse_shard*_part*.parquet`.

## Checks (`scripts/28_perception_qc.py`, `outputs/wp3/perception_qc/`)

Spearman correlations:

| Question | ViT vs ResNet, per view | ViT vs ResNet, per panorama (mean of 4 views) | ViT, front vs back | ViT, left vs right |
|---|---|---|---|---|
| safety | 0.28 | 0.35 | 0.54 | 0.28 |
| lively | 0.56 | 0.74 | 0.68 | 0.44 |
| wealthy | 0.37 | 0.50 | 0.54 | 0.29 |
| beautiful | 0.45 | 0.55 | 0.49 | 0.21 |
| boring | 0.20 | 0.34 | 0.39 | 0.20 |
| depressing | 0.15 | 0.18 | 0.41 | 0.24 |

Mean ViT score by view direction:

| Question | front | back | left | right |
|---|---|---|---|---|
| safety | 5.45 | 5.53 | 4.76 | 4.73 |
| lively | 5.85 | 6.13 | 4.60 | 4.43 |
| wealthy | 6.42 | 6.37 | 5.77 | 5.74 |
| beautiful | 6.92 | 6.87 | 5.73 | 5.61 |
| boring | 4.59 | 4.82 | 5.16 | 5.24 |
| depressing | 4.60 | 4.55 | 4.59 | 4.63 |

**Reading.**
- **The two models agree only moderately.** Per view, Spearman is 0.15 (depressing) to 0.56 (lively); per panorama, 0.18 to 0.74. A perception score depends on the model that produced it. Lively is the most stable question; depressing and boring are the least. Report this on the site. Use the scores as one indicator among several, never as a measure.
- **View direction matters.** Side views (left and right, mostly close façades) score lower on lively, beautiful and safety than views along the street. Aggregation must keep the mix of view directions fixed (step 2.6).
- **Unusable views distort the tails.** Views the audit flags `usable = False` make up 15–35% of the lowest 1% on safety, lively, wealthy and beautiful, and 30% of the highest 1% on boring. One of the eight least "wealthy" views is a close-up of an object blocking the camera. Exclude `usable = False` before aggregation.
- **Segmentation.** "Beautiful" rises with the vegetation pixel fraction in both models (0.49 and 0.47). ViT "safety" also rises with the car pixel fraction (0.30). These are associations learned from a global crowd, not safety.
- **Visual check** of the 8 highest and 8 lowest views per question (`sheet_vit_*.jpg`):
  - The results look plausible. High safety, beautiful and wealthy: tree-lined residential streets, 19th-century façades and park roads. Low: building sites, industrial yards, underpasses and open wasteland.
  - High depressing: open, sky-dominated urban edges and blank warehouse walls. Low depressing: close brick façades.
  - High lively: dense residential streets with parked bicycles. Low lively: fenced fields.

**Use (until 2026-10-12).** ViT scores were the perception indicator; ResNet scores were kept as a sensitivity check. Since step 2.4b both are cross-checks for the team's own model. Site wording: "perceived <quality>, predicted by an open model trained on a global crowdsourced dataset (Place Pulse 2.0); a stand-in for ELEVATE's own model; not a measure of safety or health".

## Step 2.4b: the ELEVATE team's own perception model (2026-10-12)

**Correction.** Step 2.4 assumed that ELEVATE's perception model was not public. It is. The Spatial Data Science and Geo-Intelligence Lab (Dr S.M. Labib, Utrecht University) publishes the `svi_percept` Python package (GitHub, GPL-3.0) and its Amsterdam model on Hugging Face (`Spatial-Data-Science-and-GEO-AI-Lab/svi_percept`, GPL-3.0). The model comes from the Percept citizen-science survey in Amsterdam (Danish, Labib, Ricker and Helbich 2025, *Computers, Environment and Urban Systems* 116), which collected 22,637 ratings from 331 people on walkability, bikeability, pleasantness, greenness and safety (1–5). The protocol's "20,000+ crowdsourced stated preferences" on safety, pleasantness and walkability match it. Found while reviewing the team's repositories.

**Method** (`scripts/26b_svi_percept.py`), as in the package:
- OpenCLIP ViT-H-14-378-quickgelu, pretrained `dfn5b` (`apple/DFN5B-CLIP-ViT-H-14-378` @ 01b771ed; Apple Machine Learning Research Model licence, research only) image features, L2-normalised;
- for each quality, the k = 40 most similar survey images (4,340–4,437 rated images per quality), weights softmax(10^similarity), score = weighted mean of their ratings; model data `svi_percept` @ d0c666d6;
- re-implemented for GPU batches. On 64 views it matches the package exactly (largest difference 0.00000; the package's own GPU path fails, so the comparison ran on the CPU);
- input is the open_clip transform: resize to 378 px and centre crop, i.e. the central square of each 768 × 576 view.

**Run.** 99,100 views on two GPUs in fp32, 20.5 min. Output `outputs/wp3/perception/svipercept_shard*_part*.parquet` (`svp_*` columns). Aggregated as the Place Pulse scores: mean of the usable front and back views per panorama, then segments and buurten (`29_aggregate.py`). Environment `envs/percept` (`PY_PERCEPT`).

**Checks** (`scripts/28b_svp_qc.py`; usable front and back views, n = 49,398; 24,743 panoramas):

| Quality | Compared with | Per view | Per panorama |
|---|---|---|---|
| safety | Place Pulse safety (cross-check) | ρ 0.52 | ρ 0.59 |
| pleasantness | Place Pulse beautiful (cross-check) | ρ 0.51 | ρ 0.56 |
| walkability | Place Pulse lively (cross-check) | ρ 0.46 | ρ 0.51 |
| greenness | green view (segmentation, vegetation share) | ρ 0.59 | ρ 0.60 |
| walkability | audit: sidewalk present vs absent | AUC 0.94 | – |
| bikeability | audit: cycle infrastructure present vs absent | AUC 0.59 | – |
| greenness | audit: street trees present vs absent | AUC 0.81 | – |
| safety | audit: street light pole present vs absent | AUC 0.42 | – |

- Distribution (views): means from 3.17 (pleasantness) to 3.46 (bikeability). Safety has the narrowest spread (SD 0.29), greenness the widest (SD 0.60).
- Front vs back of the same panorama: Spearman 0.69–0.74.
- Side views score lower than along-street views on greenness, pleasantness and bikeability (−0.15 to −0.17), not on walkability (+0.02).

**Reading.**
- The model moves with independent measures where it should. Walkability separates views with and without a sidewalk almost perfectly (AUC 0.94). Greenness follows street trees (AUC 0.81) and the segmentation's green view (ρ 0.60).
- Bikeability barely follows the audit's cycle infrastructure (AUC 0.59): the rating is broader than infrastructure, and the audit sees only what is in view.
- Safety does not follow street light poles (AUC 0.42). In daytime images a pole is a poor proxy, so this says nothing about safety either way.
- Agreement with the Place Pulse models (ρ 0.51–0.59 per panorama) is within the range of the two Place Pulse models' agreement with each other (0.18–0.74).
- The survey rated Mapillary photos; these are the City's panoramas, cropped to a square. No ratings of these views were collected. The scores are model observations, not validated measures.

**Equity (WP5, 360 buurten).** With the low-income share, all five qualities stay weak (|ρ| ≤ 0.10). With mean income per inhabitant: walkability 0.30 [0.20, 0.39], pleasantness 0.30, safety 0.30, bikeability −0.39 and greenness −0.46. The new indicators were appended last, so every earlier WP5 result, including the 21 priority buurten, is unchanged (checked against `outputs/_backup_2026-10-12_pre_svp/`).

**Use.** The team's model is the perception indicator on the site: the WP3 map toggle "Walkable", the WP5 layers and the WP5 income chart. The Place Pulse ViT and ResNet scores remain as cross-checks (WP3 table). Wording: "walkability (safety, pleasantness …) predicted by the ELEVATE team's open model, trained on ratings of Amsterdam street images; not a measure of safety or health".
