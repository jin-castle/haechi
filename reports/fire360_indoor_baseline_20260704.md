# Fire360 Indoor Baseline - 2026-07-04

## Scope

This run uses real Fire360 indoor video frames as the baseline input. The C-THRU image is not used as experimental evidence in this report; it remains reference-only visual context for the desired HUD-style output.

## Dataset

- Dataset: Fire360
- Source folder: `indoor_videos`
- Fire360 Box link: https://uofi.box.com/v/fire360dataset
- Paper: https://arxiv.org/html/2506.02167v1
- Selected video: `IFSI Video 4 (2).mp4`
- Box file id: `1864227378988`
- Downloaded size: 25,933,721 bytes
- Extracted frames: 16
- Frame directory: `data/fire360/frames/ifsi_video_4`

The Fire360 paper describes the dataset as 228 real 360-degree firefighter videos, including indoor search/rescue scenarios, object localization labels, and low-visibility degradation metadata. For this baseline I used the public Box folder's `indoor_videos` branch rather than the outdoor/daytime folders or the large `indoor_videos.zip`.

## Baselines Run

| Baseline | Input | Output |
| --- | --- | --- |
| HUD edge proxy | 16 Fire360 indoor frames | green edge overlays and binary masks |
| YOLO11n COCO pretrained | same 16 Fire360 indoor frames | annotated object detections and YOLO labels |

## Results

### HUD Edge Proxy

The edge baseline is usable as a visibility/contour proxy on the Fire360 indoor thermal-style frames. It emphasizes door/wall boundaries, hot regions, firefighter silhouettes, and existing HUD scale marks.

| Metric | Value |
| --- | ---: |
| Frames | 16 |
| Min edge ratio | 0.0368 |
| Median edge ratio | 0.0462 |
| Mean edge ratio | 0.0481 |
| Max edge ratio | 0.0611 |

Artifacts:

- Input contact sheet: `.omo/evidence/firesight-smoke-vision/fire360/ifsi_video_4_contact_sheet.jpg`
- Edge contact sheet: `.omo/evidence/firesight-smoke-vision/fire360/ifsi_video_4_edge_contact_sheet.jpg`
- Edge output directory: `.omo/evidence/firesight-smoke-vision/fire360/ifsi_video_4_edges`
- Summary JSON: `.omo/evidence/firesight-smoke-vision/fire360/ifsi_video_4_baseline_summary.json`

### Object Detection

YOLO11n COCO is not a strong baseline for this domain. It detected objects in 7 of 16 frames, but the class set is mismatched for firefighter/thermal imagery.

| Detected COCO class | Count |
| --- | ---: |
| `person` | 3 |
| `dog` | 2 |
| `donut` | 2 |
| `refrigerator` | 1 |
| `suitcase` | 1 |

Interpretation: `person` detections are partially useful, but the `dog`, `donut`, `refrigerator`, and `suitcase` outputs are false-positive symptoms of domain mismatch. For this project, Fire360-style labels such as responder, helmet, gas mask, civilian, fire, and smoke are more relevant than COCO classes.

Artifacts:

- YOLO contact sheet: `.omo/evidence/firesight-smoke-vision/fire360/ifsi_video_4_yolo_contact_sheet.jpg`
- YOLO output directory: `.omo/evidence/firesight-smoke-vision/fire360/yolo11n-ifsi-video-4`
- YOLO summary JSON: `.omo/evidence/firesight-smoke-vision/fire360/yolo11n-ifsi-video-4_summary.json`

## Takeaways

1. Fire360 is the better baseline dataset direction than fire/smoke alarm datasets because it contains firefighter-perspective indoor imagery and object-localization-oriented tasks.
2. Classical edge overlays are a reasonable first proxy for the desired HUD contour visualization, especially before custom annotations exist.
3. COCO object detection should be treated only as a sanity check. It is not reliable enough for firefighter HUD object understanding without Fire360-domain fine-tuning or a model already trained for thermal/firefighter scenes.
4. Next baseline should use more indoor Fire360 clips, preferably selected by smoke/low-visibility metadata or manually stratified into clean, smoky, thermal, and high-heat frames.
