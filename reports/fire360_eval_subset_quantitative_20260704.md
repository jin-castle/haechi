# Fire360 Indoor Image-Level Eval Subset v0

Date: 2026-07-04

## Scope

This v0 evaluation moves beyond qualitative baseline screenshots into a small
image-level, multi-label quantitative gate. It uses local indoor fire/smoke
photos plus Fire360 frames only.

- Manifest: `data/eval/fire360_indoor_image_level_v0.json`
- Image root: `data/fire360_photo_inclusive/all_images`
- Items: 18 total, 7 photo images and 11 Fire360 video frames
- Classes: `fire`, `smoke`, `door`, `person`, `exit`, `obstacle`
- Metric level: image-level presence/absence
- Bbox mAP: deferred because this subset does not have complete bbox ground
  truth for the six-class ontology.

## Support

| Class | Positives |
| --- | ---: |
| fire | 9 |
| smoke | 13 |
| door | 5 |
| person | 11 |
| exit | 4 |
| obstacle | 11 |

## Baseline Mapping

YOLO11n COCO maps only `person -> person`. Non-task COCO detections are kept as
ignored labels and are not counted as `obstacle`, because that would inflate a
task label from unrelated COCO false positives.

OWL-ViT uses the prior bounded open-vocabulary run. Its mapped labels are
`fire`, `smoke`, `door`, `person`, and `firefighter -> person`. Images outside
the prior bounded run are treated as no prediction for this v0 comparison.

## Results

| Model | Macro Precision | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: |
| YOLO11n COCO | 0.1667 | 0.1061 | 0.1296 |
| OWL-ViT bounded run | 0.1667 | 0.0152 | 0.0278 |

### Per-Class Notes

YOLO11n COCO:
- `person`: precision 1.0000, recall 0.6364, TP 7, FP 0, FN 4
- `fire`, `smoke`, `door`, `exit`, `obstacle`: recall 0.0000 in this ontology

OWL-ViT bounded run:
- `person`: precision 1.0000, recall 0.0909, TP 1, FP 0, FN 10
- `fire`, `smoke`, `door`, `exit`, `obstacle`: recall 0.0000 in this ontology

## Interpretation

The existing YOLO/OWL-ViT runs are useful weak baselines, but not sufficient
for the intended indoor smoke/fire HUD-style task. Quantitatively, the baselines
mostly behave as person detectors in this subset and do not cover the operational
classes that matter most for navigation under limited visibility.

The next useful step is not more screenshot inspection. It is expanding this v0
subset into a reviewed image-level benchmark, then adding bbox labels for the
classes that require localization: `door`, `person`, `exit`, and `obstacle`.

The earlier planner artifact proposed a 60-image source-balanced target. This
run intentionally ships an 18-image v0 sanity gate; the reduction is recorded
in `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/scope-deviation-record.md`.

## Evidence

- Metrics JSON: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/image_level_metrics.json`
- Manifest summary: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/manifest_summary.json`
- Contact sheet PNG: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/image_level_eval_contact_sheet.png`
- Row-level label QA: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/row_level_label_qa.json`
- Scope/provenance: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/scope-deviation-record.md`, `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/provenance_hashes.json`
- YOLO predictions: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/yolo11n_coco_image_level_predictions.json`
- OWL-ViT predictions: `.omo/evidence/firesight-smoke-vision/fire360-eval-subset/owlvit_open_vocab_image_level_predictions.json`
