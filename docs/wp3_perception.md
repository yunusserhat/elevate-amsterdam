# WP3 perceived qualities (step 2.4), 2026-10-11

**Why a stand-in.** ELEVATE derives perceptions (safety, pleasantness, walkability) from models trained on its own 20,000+ crowdsourced stated preferences (protocol, WP3). Those models are not public. We use open models trained on Place Pulse 2.0 for a similar purpose and label them as a stand-in, not as ELEVATE's model.

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

**Use.** ViT scores are the perception indicator; ResNet scores are kept as a sensitivity check. Site wording: "perceived <quality>, predicted by an open model trained on a global crowdsourced dataset (Place Pulse 2.0); a stand-in for ELEVATE's own model; not a measure of safety or health".
