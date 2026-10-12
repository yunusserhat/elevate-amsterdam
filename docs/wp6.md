# WP6 evidence layer and before/after visualisations (steps 7.1–7.3), 2026-10-11

> **An idea I would bring, not part of the protocol.** The protocol's recommender is evidence-based and equity-sensitive, with a chat interface; its Fig. 2 street images are illustrative, made with ChatGPT. This step turns detected gaps into images that a workshop could react to. Every edited image is an **AI-generated visualisation, not a design proposal**. Nothing here checks whether an intervention is physically possible (cables and pipes, traffic, space, ownership).

Scripts: `scripts/70_wp6_gaps.py`, `72_wp6_confirm.py`, `71_wp6_edit.py`, `74_wp6_verify.py`, chains `73_`/`75_`. Outputs in `outputs/wp6/`.

## 7.1 Evidence layer
A gap is read from one usable front or back view: the view an edit would start from. The audit items are tier A unless stated.

| Gap | Rule | Views (of 49,398) |
|---|---|---|
| crossing | intersection present, pedestrian crossing absent (protocol Fig. 2 example) | 7,487 (15.2%) |
| seating | sidewalk present, active frontage present (B), seating absent | 18,613 (37.7%) |
| greenery | sidewalk present, vegetation < 5% of the view (segmentation) | 5,308 (10.7%) |
| cycle | main road, cycle infrastructure absent (B; absences are the safer side of an over-reported item) | 2,123 (4.3%) |
| trees | sidewalk present, street trees absent | 1,989 (4.0%) |

- 14,186 street segments have at least one gap view (`gap_segments.parquet`).
- By low-income quintile the gap rates show no clear gradient: crossing 13–18%, seating 35–50%, greenery 8–12%, cycle 4–5%, trees 3–5% (`gap_by_quintile.csv`).
- **Priority:**
  - 2 = buurt in the WP5 equity-priority set (low green-view cluster and many low-income households);
  - 1 = other buurten in the two most low-income quintiles;
  - 0 = the rest.

## 7.2 Candidates and edits
- **Pool:** 124 views from 20 of the 21 priority buurten: 2.5 × the quota per gap type, at most three per buurt and type, seeded.
- **Confirmation by the larger model.** The same views were audited with Qwen3.8-27B-FP8 and template v2. Shares where the reference also sees the gap:
  - crossing 31%;
  - cycle 80%;
  - trees 94%;
  - seating 96%;
  - greenery 100%.
  - The 4B model's "junction without a crossing" is mostly not confirmed: the reference often sees either no junction or a crossing already there.
- **Selection:** 47 confirmed candidates in 17 buurten: crossing 11 (quota 14, not enough confirmed), trees 13, seating 9, greenery 8, cycle 6.
- **Edits:**
  - Model: Qwen/Qwen-Image-2.1-Turbo @ d65dbc9a (Qwen Research Licence, non-commercial). It uses the 8-step schedule saved with the checkpoint, CFG 1 and the KV cache.
  - Hardware: the ~32 GB of weights are split over both GPUs (text encoder on one, transformer on the other).
  - Size: 1024 × 768. About 3 s per edit after a warm-up. 2400 × 1792 ran out of GPU memory.
  - Prompt: one fixed sentence per gap type, plus the same instruction to keep everything else unchanged. Prompts and seeds are in `edits_log.csv`.

## 7.3 Verification (all 47 kept, failures included)
Edits were resized to 768 × 576, then:
- **Target present after the edit:** 47/47 by the 4B model and 47/47 by the 27B reference. The models audit the image only; they are not told what was edited.
- **Other items changing state:**
  - 2.55 of 18 per edit on average, against 0.68 expected from the 4B model's run-to-run instability.
  - By type: cycle 4.0, trees 3.5, greenery 1.9, crossing 1.8, seating 1.8.
  - Some are expected side effects (trees bring shade, a cycle track changes the sidewalk). The rest is drift that a workshop image should disclose.
- **Pixels whose Cityscapes class changed:**
  - All classes: median 7% (crossing 3%, cycle 4%, seating 5%, greenery 11%, trees 22%).
  - Between classes the edit should not touch (for example buildings, cars or sky for a crossing): median 1.6%, 90th percentile 9.9%, maximum 16.7% (trees on the IJ waterfront, IJplein e.o.).
- **Visual check** of all pairs (`edits_gallery_qc.jpg`):
  - The edits add the requested element in plausible places and keep the scene.
  - A few put crossings or trees where they would not fit in reality, which is why each image is labelled a visualisation.

**Reading.**
- Generating a convincing before/after image is easy.
- Choosing where to place it reliably is the hard part. Without the reference check, two thirds of the "missing crossing" candidates would have been wrong.
- In ELEVATE, such images would only enter design-thinking sessions with the residents and designers of the place.
