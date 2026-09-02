# Fire360 Photo-Inclusive Indoor Baseline - 2026-07-04

## Scope

This run extends the previous Fire360 indoor video-frame baseline by adding a still-photo dataset. The mixed set contains real Fire360 indoor video-derived frames plus real indoor fire/smoke photo images. C-THRU/reference imagery is excluded from this run.

## Data

| Source | Type | Count | Notes |
| --- | ---: | ---: | --- |
| Fire360 `indoor_videos` extracted frames | video frames | 59 | Same multiclip Fire360 frame slice used in the previous baseline |
| Zenodo Indoor Fire Smoke Dataset | still photos | 36 | Sampled from train/valid/test images in the 5,000-image indoor fire/smoke dataset |
| Total mixed input | images | 95 | Stored under `data/fire360_photo_inclusive/all_images` |

Photo source: https://zenodo.org/records/15826133, DOI `10.5281/zenodo.15826133`, CC-BY-4.0, downloaded file `Indoor Fire Smoke.zip`, MD5 `086fbc3b874139276097f4057bc45a3c`.

Fire360 source reference: https://arxiv.org/html/2506.02167v1 and dataset entry https://uofi.box.com/v/fire360dataset.

## Device

This run was local CPU inference, not GPU inference.

| Field | Value |
| --- | --- |
| PyTorch | `2.12.1+cpu` |
| CUDA available | `false` |
| Device used | `cpu` |

Device artifact: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/device_info.json`

## Baselines

| Baseline | Scope | Result |
| --- | ---: | --- |
| HUD edge proxy | 95 / 95 images | Completed |
| YOLO11n COCO pretrained | 95 / 95 images | Completed |
| OWL-ViT open vocabulary | 16 representative mixed images | Completed |

## HUD Edge Proxy

HUD edge proxy covered all 95 mixed images: 59 Fire360 video frames and 36 still photos.

| Metric | Value |
| --- | ---: |
| Min edge ratio | 0.0049 |
| Median edge ratio | 0.0413 |
| Mean edge ratio | 0.0505 |
| Max edge ratio | 0.1928 |

Interpretation: HUD edge remains the most relevant baseline for a firefighter HUD/contour objective. It is not object detection, but it is useful for visualizing scene boundaries, silhouettes, and high-contrast structure under smoke or low visibility.

PNG proof: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/photo_inclusive_hud_edge_contact_sheet.png`

Thickness follow-up: the HUD renderer now supports adjustable contour width via `--edge-width`.
On the same 95-image mixed set, `edge_width=3` increased mean edge coverage by 2.54x
while remaining less visually saturated than `edge_width=5`, which increased mean edge
coverage by 3.67x. For current indoor photo plus Fire360 frame samples, `edge_width=3`
is the better default candidate and `edge_width=5` is useful as a strong-visibility debug view.

Comparison PNGs:

- `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/hud_edge_width1_vs_width3_contact_sheet.png`
- `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/hud_edge_width1_vs_width5_contact_sheet.png`

## YOLO11n COCO

YOLO11n ran on all 95 mixed images on CPU. It produced detections in 37 images.

| Source type | Frames | Frames with detection | Detections |
| --- | ---: | ---: | ---: |
| Fire360 video frames | 59 | 32 | 48 |
| Indoor fire/smoke photos | 36 | 5 | 6 |

Top detections included `person` 34, but also COCO-domain false-positive classes such as `giraffe`, `refrigerator`, `teddy bear`, `toothbrush`, and `motorcycle`.

Interpretation: YOLO11n COCO is a weak baseline for this domain. It can sometimes catch human silhouettes, but it does not understand firefighter/fire/smoke-specific indoor hazards without domain labels or fine-tuning.

PNG proof: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/photo_inclusive_yolo11n_contact_sheet.png`

## OWL-ViT Open Vocabulary

OWL-ViT was run on 16 representative mixed images: 8 Fire360 video frames and 8 indoor fire/smoke photos.

Prompts:

`firefighter`, `person`, `helmet`, `gas mask`, `fire`, `smoke`, `door`, `window`, `hose`, `air tank`

At threshold `0.10`, OWL-ViT returned only two `person` detections.

Interpretation: the open-vocabulary route executed successfully, but this model/prompt setup is not reliable enough for the desired firefighter HUD perception target. It should remain a failure baseline unless prompts/model choice are changed or Fire360/photo-domain labels are added.

PNG proof: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/photo_inclusive_owlvit_contact_sheet.png`

## Recommendation

1. Keep HUD edge proxy as the immediate visualization baseline.
2. Treat YOLO11n COCO and OWL-ViT as weak baselines only.
3. Build a small labeled eval subset next: 50-100 mixed images with labels for `firefighter/person`, `helmet`, `SCBA/air tank`, `fire/hotspot`, `smoke`, `door/window/exit`, `hose/tool`, and `obstacle`.
4. After the eval subset exists, fine-tune a detector or test a stronger open-vocabulary detector against quantitative precision/recall, not just visual inspection.

## Evidence

- Dataset summary: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/photo_inclusive_dataset_summary.json`
- Combined baseline summary: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/photo_inclusive_baseline_summary.json`
- Device info: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/device_info.json`
- Input PNG: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/photo_inclusive_input_contact_sheet.png`
- HUD summary: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/hud_edge_stats.json`
- YOLO summary: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/yolo11n_summary.json`
- OWL-ViT summary: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/open_vocab_owlvit/summary.json`
- RED proof: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/red-missing-photo-inclusive-summary.txt`
- GREEN data proof: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/green-photo-inclusive-dataset-check.txt`
- Code/slop review: `.omo/evidence/firesight-smoke-vision/fire360-photo-inclusive/fire360_photo_inclusive_code_slop_review.md`
