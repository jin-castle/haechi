# Fire360 Indoor Multiclip Baseline - 2026-07-04

## Scope

This expands the Fire360 indoor baseline from one clip to a small multiclip benchmark. It uses only real Fire360 `indoor_videos` frames. The C-THRU reference image is not included in this experiment.

## Dataset Slice

- Dataset: Fire360
- Source: `indoor_videos` folder from https://uofi.box.com/v/fire360dataset
- Clips used: 6
- Frames extracted: 59
- Combined frame directory: `data/fire360/frames_multiclip/all_frames`

Selected clips:

| Clip | Box file id | Downloaded size |
| --- | ---: | ---: |
| `02815 (2).MTS` | `1864227318988` | 9,830,400 |
| `02824 (2).MTS` | `1864227297388` | 16,601,088 |
| `02814 (2).MTS` | `1864227412588` | 20,367,360 |
| `IFSI Video 4 (2).mp4` | `1864227378988` | 25,933,721 |
| `IFSI Video 8 (2).mp4` | `1864227390988` | 27,465,405 |
| `GOPR8356 (2).MP4` | `1864227347788` | 32,067,190 |

Visual QA confirms this slice includes indoor fire glow, heavy smoke/low visibility, firefighter silhouettes, RGB low-light frames, and thermal/HUD-style frames.

## Baselines

| Baseline | Frames | Result |
| --- | ---: | --- |
| HUD edge proxy | 59 | Completed on every frame |
| YOLO11n COCO pretrained | 59 | Completed on every frame |
| OWL-ViT open vocabulary | 12 representative frames | Completed, sparse detections |

## HUD Edge Proxy

The edge proxy remains the best fit for the HUD-contour objective among the three baselines. It highlights fire borders, room/door structure, firefighter silhouettes, and thermal/HUD frame edges without needing labels.

| Metric | Value |
| --- | ---: |
| Frames | 59 |
| Min edge ratio | 0.0049 |
| Median edge ratio | 0.0249 |
| Mean edge ratio | 0.0271 |
| Max edge ratio | 0.0583 |

Artifact: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/multiclip_hud_edge_contact_sheet.jpg`

## YOLO11n COCO

YOLO11n found detections in 32 of 59 frames. `person` was the dominant class, but the remaining classes show clear domain mismatch.

| COCO class | Count |
| --- | ---: |
| `person` | 33 |
| `teddy bear` | 3 |
| `bottle` | 2 |
| `motorcycle` | 2 |
| `refrigerator` | 2 |
| `baseball bat` | 1 |
| `bed` | 1 |
| `bench` | 1 |
| `cell phone` | 1 |
| `clock` | 1 |
| `toothbrush` | 1 |

Interpretation: COCO-pretrained YOLO is useful as a failure baseline. It can often localize human silhouettes, but it mislabels firefighter gear, thermal regions, or building features as unrelated household objects.

Artifact: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/multiclip_yolo11n_contact_sheet.jpg`

## OWL-ViT Open Vocabulary

OWL-ViT was run on 12 representative frames with prompts:

`firefighter`, `person`, `helmet`, `gas mask`, `fire`, `smoke`, `door`, `window`, `hose`, `air tank`

At threshold `0.10`, it returned only four `person` detections. It did not reliably detect firefighter-specific or scene-specific prompts in this Fire360 slice.

Artifact: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/multiclip_owlvit_contact_sheet.jpg`

## Recommendation

1. Keep HUD edge proxy as the immediate visualization baseline.
2. Treat YOLO11n COCO and OWL-ViT as weak baselines only; they are not enough for firefighter HUD object understanding.
3. For object detection, the next meaningful step is a Fire360-domain label subset and fine-tuning or prompt-tuning around `firefighter`, `helmet`, `SCBA/air tank`, `door/window`, `fire/hotspot`, and `civilian`.
4. For edge/HUD work, add a lightweight human QA rubric: contour usefulness, noise level, silhouette clarity, and whether edges help navigation under smoke.

## Evidence

- Dataset summary: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/multiclip_dataset_summary.json`
- Combined summary: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/multiclip_baseline_summary.json`
- Input contact sheet: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/multiclip_input_contact_sheet.jpg`
- HUD edge summary: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/hud_edge_stats.json`
- YOLO summary: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/yolo11n_summary.json`
- OWL-ViT summary: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/open_vocab_owlvit/summary.json`
- Code review: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/fire360_step1_step3_code_review.md`
- Manual QA matrix: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/fire360_step1_step3_manual_qa.md`
- Provenance manifest: `.omo/evidence/firesight-smoke-vision/fire360-multiclip/fire360_step1_step3_provenance_manifest.json`
