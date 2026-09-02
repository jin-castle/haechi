# Baseline Model Test - 2026-07-04

> Superseded for dataset evidence: this report used the C-THRU-style reference image as a visual sanity check. The actual dataset-based indoor Fire360 baseline is in `reports/fire360_indoor_baseline_20260704.md`.

## Goal

Test no-custom-data baselines against the indoor firefighter HUD target shown in the reference image.

## Inputs

- Reference image: `data/input/baseline/c-thru-reference.png`
- Existing fixture: `tests/fixtures/images/sample_room.jpg`
- Synthetic smoke variants: `.omo/evidence/firesight-smoke-vision/baseline/smoke-aug/`

## Baselines Run

| Baseline | Surface | Output |
| --- | --- | --- |
| Synthetic smoke robustness input | Existing `generate-smoke-aug` command | 4 variants: clean, mild, medium, dense |
| HUD edge proxy | New `run-hud-edge-baseline` command | Green edge overlay plus binary edge mask |
| YOLO11n COCO detector | Ultralytics `yolo predict` | Annotated images and YOLO label files |

## Results

### HUD Edge Proxy

The HUD edge proxy produced visible green line overlays from a classical edge pipeline.

| Image | Edge pixels | Total pixels | Edge ratio |
| --- | ---: | ---: | ---: |
| `c-thru-reference.png` | 29,062 | 199,699 | 0.1455 |
| `sample_room.jpg` | 467 | 1,536 | 0.3040 |

Artifacts:

- `.omo/evidence/firesight-smoke-vision/baseline/hud-edge/c-thru-reference_hud_edges.png`
- `.omo/evidence/firesight-smoke-vision/baseline/hud-edge/c-thru-reference_edge_mask.png`
- `.omo/evidence/firesight-smoke-vision/baseline/hud-edge/hud_edge_summary.json`

Interpretation: this is useful as a first visual baseline and regression surface, but it is not yet a learned structure model. It highlights many visual contours, including UI/product edges, so it does not distinguish walls, doors, people, masks, or egress geometry.

### YOLO11n COCO Detector

YOLO11n was run on the synthetic smoke variants of the reference image with `conf=0.25`, `imgsz=640`, CPU inference.

| Variant | Detection summary |
| --- | --- |
| clean | 1 `cell phone`, confidence 0.6932 |
| mild | 1 `cell phone`, confidence 0.5071 |
| medium | 1 `cell phone`, confidence 0.4050 |
| dense | no detections |

Artifacts:

- `.omo/evidence/firesight-smoke-vision/baseline/yolo/yolo11n-c-thru-smoke/`
- `.omo/evidence/firesight-smoke-vision/baseline/yolo/yolo11n-c-thru-smoke/labels/`
- `.omo/evidence/firesight-smoke-vision/baseline/weights/yolo11n.pt`

Interpretation: a generic COCO detector is not aligned with the target task. It misses firefighter/person/helmet/door/wall structure and only recognizes the visible handheld device. Dense smoke augmentation removes even that detection. This supports shifting the next baseline toward indoor scene segmentation, thermal/person detection, and custom label definitions rather than fire/smoke-only detection.

## Recommended Next Baseline

1. Add an indoor structure baseline using SUN RGB-D / NYU Depth-style classes: wall, floor, door, stair, obstacle.
2. Add a person/PPE baseline: person, helmet, mask, oxygen tank, hose.
3. Use Fire360 frames as the closest public firefighter-domain validation source.
4. Keep the current HUD edge overlay as a cheap visual regression check for every later model.

## Commands

```bash
uv run --frozen run-hud-edge-baseline \
  --input data/input/baseline/c-thru-reference.png tests/fixtures/images/sample_room.jpg \
  --out .omo/evidence/firesight-smoke-vision/baseline/hud-edge

uv run --frozen generate-smoke-aug \
  --input data/input/baseline/c-thru-reference.png \
  --out .omo/evidence/firesight-smoke-vision/baseline/smoke-aug \
  --severity clean mild medium dense \
  --seed 42

uv run --with ultralytics yolo predict \
  model=yolo11n.pt \
  source=C:/Users/bgs43/Desktop/haechi/.omo/evidence/firesight-smoke-vision/baseline/smoke-aug \
  project=C:/Users/bgs43/Desktop/haechi/.omo/evidence/firesight-smoke-vision/baseline/yolo \
  name=yolo11n-c-thru-smoke \
  save=True save_txt=True save_conf=True conf=0.25 imgsz=640 exist_ok=True
```
