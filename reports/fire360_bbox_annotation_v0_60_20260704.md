# Fire360 Indoor BBox Eval Subset v0 60

Date: 2026-07-04

## Result

- Built a 60-image draft bbox eval subset from `data/fire360_photo_inclusive/all_images`.
- Composition: 30 indoor photo images and 30 Fire360/thermal video frames.
- Classes: `door`, `person`, `exit`, `obstacle`.
- COCO annotations: 64 boxes across 33 annotated images.
- Negative/empty images: 27 images, retained for false-positive evaluation.
- Category counts: door 8, person 17, exit 8, obstacle 31.
- Human review status: `required_before_training_or_benchmark`.
- Obstacle boxes were tightened after visual QA: maximum obstacle width ratio is now 0.70, preventing full-width lower-scene bands.

## Artifacts

- COCO bbox JSON: `data/eval/fire360_indoor_bbox_v0_60_coco.json`
- Manifest: `data/eval/fire360_indoor_bbox_v0_60_manifest.json`
- YOLO labels: `outputs/fire360_bbox_60/yolo_labels`
- YOLO dataset: `outputs/fire360_bbox_60/yolo_dataset`
- YOLO validation: `outputs/fire360_bbox_60/yolo_dataset_validation.json`
- Person provenance: `outputs/fire360_bbox_60/yolo_person_provenance.json`
- Overlay PNG: `outputs/fire360_bbox_60/bbox_overlay_contact_sheet.png`
- Simple viewer: `outputs/fire360_bbox_60/index.html`
- Convenient PNG copy: `fire360_bbox_overlay_contact_sheet.png`

## Annotation Policy

This subset is marked `draft_single_agent` and
`required_before_training_or_benchmark`. It is suitable for pipeline wiring,
format validation, and first quantitative baseline scaffolding. It is not yet a
training-ready, benchmark-ready, or safety-readiness dataset.

Person boxes are YOLO11n COCO-assisted where available. Door, exit, and obstacle
boxes are draft manual/proxy annotations from visible indoor structures and scene
regions. Human review is required before model-quality claims.

## Verification

- `uv run pytest tests/test_bbox_annotations.py -q`: passed.
- Targeted `ruff check` on bbox modules, script, and test: passed.
- Targeted `basedpyright` on bbox modules, script, and test: passed.
- COCO validator on generated JSON: passed with 60 images and 64 annotations.
- YOLO dataset validator: passed with 60 images, 60 label files, and 64 rows.
- Obstacle width QA: passed with 0 boxes above 0.75 image-width ratio.
