# FireSight Smoke Vision

Initial reproducible Python research workspace for FireSight static smoke vision.

원형 Jetson Nano 2GB/4GB용 TEED 카메라·영상 실행 프로그램과 설치 방법은
[`deploy/jetson_nano/README.md`](deploy/jetson_nano/README.md)에 있다.

This scaffold intentionally avoids heavyweight model or data frameworks until a
baseline proves they are needed. Todo 1 establishes the package, strict Python
tooling, directory contract, smoke tests, and doctor evidence surface.

## Quick Checks

```bash
python -m pytest
python scripts/doctor.py --check scaffold --evidence .omo/evidence/firesight-smoke-vision/task-1-doctor.json
```

The project runtime default is Python 3.11. Raw datasets, model weights, and
large generated artifacts are not part of this scaffold.

## Step 1 Software-Only Edge Replay MVP

The first hardware-independent runtime replays a directory or single image
through the existing Pillow HUD edge baseline. It paces processing at the
requested FPS, writes one overlay and mask per frame, and records latency,
effective FPS, edge ratio, and pacing overruns.

```bash
uv run --frozen python scripts/run_edge_replay.py \
  --input data/fire360/frames_multiclip/ifsi_video_4 \
  --out .omo/evidence/firesight-smoke-vision/replay-step1 \
  --fps 15 \
  --profile dense-smoke
```

The output directory contains `frames/`, `replay_metrics.jsonl`, and
`replay_summary.json`. Use `--no-realtime` for a throughput-only benchmark.
This step intentionally consumes recorded image frames; it does not claim to
be a camera driver, FLIR Boson integration, XREAL display stream, or video
decoder.

## Step 2 Pretrained TEED Edge Replay

Step 2 adds [TEED (Tiny and Efficient Edge Detector)](https://github.com/xavysp/TEED)
as a second, learning-based edge candidate. It uses the official BIPED
checkpoint at `models/teed/5_model.pth`; no Fire360 images are used for training
or fine-tuning in this step. The model emits an edge-probability map, which the
replay adapter thresholds into a mask and draws as a green HUD overlay.

Install the optional runtime once:

```bash
uv sync --extra teed
```

Run a recorded replay and write TEED PNG artifacts plus JSONL metrics:

```bash
uv run --extra teed python scripts/run_teed_replay.py \
  --input data/fire360/frames_multiclip/ifsi_video_4 \
  --out .omo/evidence/firesight-smoke-vision/teed-fire360 \
  --checkpoint models/teed/5_model.pth \
  --threshold 0.75 \
  --edge-width 1 \
  --no-realtime \
  --device cpu
```

Each frame produces `*_teed_edges.png`, `*_teed_mask.png`, and
`*_teed_probability.png`. The threshold is an explicit visualization control,
not a Fire360 accuracy claim. Compare the two approaches with:

```bash
uv run --frozen python scripts/build_teed_comparison_contact_sheet.py \
  --input data/fire360/frames_multiclip/ifsi_video_4 \
  --baseline .omo/evidence/firesight-smoke-vision/replay-step1/frames \
  --teed .omo/evidence/firesight-smoke-vision/teed-fire360/frames \
  --out .omo/evidence/firesight-smoke-vision/teed-vs-classical-contact-sheet.png
```

The current CPU result is a development-PC reference only. Jetson Orin Nano
feasibility requires a separate CUDA/TensorRT benchmark after the camera and
device are available; the BIPED-pretrained output still needs smoke-domain
threshold and false-edge evaluation before it becomes the MVP default.

## Step 2.5 TEED + 거리 추정 오프라인 PoC

TEED 선에 단안 카메라 기반 미터 단위 거리 추정을 결합한다. 현재 PoC는
[Depth Anything V2 Metric Indoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
모델을 사용하며, 같은 프레임의 각 TEED 선을 예상 거리에 따라 다음처럼
표시한다.

- 0.2~0.5 m: 빨강, 굵은 선
- 0.5~1.0 m: 노랑, 중간 굵기
- 1.0~1.5 m: 청록, 가는 선
- 위 범위 밖 또는 유효하지 않은 거리: 회색

필요한 모델 실행 환경을 설치하고 Fire360 녹화 프레임을 처리한다.

```bash
uv sync --extra depth

uv run --extra depth run-teed-depth-replay \
  --input data/fire360/frames_multiclip/ifsi_video_8 \
  --out .omo/evidence/firesight-smoke-vision/teed-depth-poc-ifsi8 \
  --checkpoint models/teed/5_model.pth \
  --max-frames 3 \
  --device cpu \
  --ignore-fire360-osd
```

각 프레임에는 원본 비교표(`*_teed_depth_contact.png`), 거리 반영 선
(`*_teed_depth_overlay.png`), 미터 단위 원본 배열(`*_depth_m.npy`)과 거리
미리보기·유효 영역·거리 구간 mask가 저장된다. `replay_metrics.jsonl`에는
프레임별 거리 통계와 처리시간, `replay_summary.json`에는 전체 요약이
저장된다.

### ROS bag 열화상·RGB·정답 depth PNG 추출

Kaggle `thermal-to-depth-image-dataset`의 ROS1 bag에는 FLIR Boson `mono16`,
RealSense RGB, RGB 좌표계에 정렬된 `16UC1` depth가 함께 들어 있다. 다음
명령은 약 10초 간격으로 세 센서의 가까운 프레임을 골라, 손실 없는 16-bit
PNG와 사람이 확인할 수 있는 preview/contact PNG 및 `manifest.json`을 만든다.

```bash
uv sync --extra rosbag
uv run --extra rosbag extract-rosbag-depth-pairs \
  --bag data/external/thermal_to_depth_kaggle/hot_obj_circle.bag \
  --out .omo/evidence/firesight-smoke-vision/rosbag-hot-obj-circle \
  --max-pairs 3 \
  --interval-seconds 10
```

`*_depth_mm16.png`의 픽셀값은 millimetre 단위 RealSense depth이며 0은 무효
픽셀이다. 이 depth는 RGB에는 공간 정렬되어 있지만 raw thermal에는 공간
정렬되어 있지 않다. 따라서 thermal 예측과의 픽셀별 오차 평가는 카메라 보정
정보나 데이터셋의 processed aligned pair를 확보한 뒤 수행해야 한다.

이 값은 한 장의 영상으로 추정한 거리이므로 **연기 속 실제 거리 측정값이나
소방 안전 판정값이 아니다.** 우선 화면 표시 가능성을 확인하는 오프라인
PoC이며, 다음 단계에서 실측 거리 라벨과 mmWave 같은 거리 센서로 오차를
검증해야 한다. CPU 실행 속도 역시 Jetson 실시간 성능을 의미하지 않는다.

### Fire360 고정 OSD 무시

Fire360 영상의 좌측 기관 마크와 우측 온도 colorbar/숫자 overlay는 장면
구조가 아니므로 선택적으로 edge mask에서 제외할 수 있다. 두 replay CLI에
`--ignore-fire360-osd`를 붙이면 된다.

```bash
uv run --frozen python scripts/run_edge_replay.py \
  --input data/fire360/frames_multiclip/ifsi_video_8 \
  --out .omo/evidence/firesight-smoke-vision/osd-ignored/classical/ifsi_video_8 \
  --profile dense-smoke --no-realtime --ignore-fire360-osd

uv run --extra teed python scripts/run_teed_replay.py \
  --input data/fire360/frames_multiclip/ifsi_video_8 \
  --out .omo/evidence/firesight-smoke-vision/osd-ignored/teed/ifsi_video_8 \
  --checkpoint models/teed/5_model.pth \
  --threshold 0.75 --edge-width 1 --no-realtime --device cpu \
  --ignore-fire360-osd
```

현재 고정 영역은 정규화 좌표로 좌측 `(0.13, 0.72)-(0.27, 1.00)`, 우측
`(0.75, 0.00)-(0.90, 0.90)`이다. 원본/어두운 배경에는 OSD가 남지만, edge
mask와 초록 overlay에는 선이 그려지지 않는다.

같은 설정으로 아래 추가 시퀀스를 처리했다.

- `clip_02814` (10 frames)
- `clip_02815` (9 frames)
- `clip_02824` (10 frames)
- `gopr8356` (10 frames)
- `ifsi_video_4` (10 frames)
- `ifsi_video_8` (10 frames)

비교 contact sheet는
`.omo/evidence/firesight-smoke-vision/osd-ignored/comparisons/`, TEED preview
영상은 `.omo/evidence/firesight-smoke-vision/osd-ignored/videos/`에 저장했다.

### 원본 전체 길이 긴 영상 replay

앞의 `frames_multiclip` 샘플 영상은 9~10개 프레임만 묶은 약 2초짜리
시각화 preview다. 원본 MP4의 전체 구간을 보고 싶을 때는 아래 명령을
사용한다.

```bash
uv run --extra teed --with opencv-python python scripts/run_profiled_teed_videos.py \\
  --input-root data/fire360/raw \\
  --profiles data/fire360/video_profiles.json \\
  --out .omo/evidence/firesight-smoke-vision/long-videos-source-aware \\
  --checkpoint models/teed/5_model.pth \\
  --inference-fps 5 --output-width 1280 \\
  --threshold 0.75 --background-scale 0.8 --edge-width 1 --device cpu
```

이 명령은 원본 영상을 처음부터 끝까지 decode하면서 약 5 FPS 간격으로
TEED를 실행하고, 결과를 1280×720 MP4로 저장한다. 따라서 **영상의 시간
길이는 유지되지만 모든 원본 30/60 FPS 프레임에 TEED를 실행하는 실시간
처리 결과는 아니다.** 현재 출력은 CPU 개발 PC에서 만든 full-duration sampled
preview이며, Jetson Orin Nano의 실시간 FPS/지연시간을 의미하지 않는다.
어두운 열화상 원본의 가시성을 위해 이 preview에서는 `background-scale 0.8`을
사용한다. 이 값은 모델 품질이나 Jetson latency 측정값이 아니라 표시용 밝기
설정이다. 특히 IFSI 원본은 시작·끝 구간에 실제 검은 프레임이 있으므로, 첫
프레임만 보면 영상 전체가 검은 것으로 오해할 수 있다.

#### 입력 영상별 OSD 프로파일

왼쪽 기관 마크와 오른쪽 온도 colorbar는 영상마다 존재 여부가 다르다.
따라서 `data/fire360/video_profiles.json`을 기준으로만 선택적으로 무시한다.

| 입력 | OSD | `ignore_fire360_osd` | 처리 |
| --- | --- | --- | --- |
| `02814 (2).MTS` | 고정 OSD 없음 | `false` | 우측 장면을 그대로 edge 입력으로 유지 |
| `02815 (2).MTS` | 고정 OSD 없음 | `false` | 우측 장면을 그대로 edge 입력으로 유지 |
| `02824 (2).MTS` | 고정 OSD 없음 | `false` | 우측 장면을 그대로 edge 입력으로 유지 |
| `GOPR8356 (2).MP4` | 왼쪽 마크 + 오른쪽 colorbar | `true` | 고정 OSD 영역만 edge mask에서 제외 |
| `IFSI Video 4 (2).mp4` | 왼쪽 마크 + 오른쪽 colorbar | `true` | 고정 OSD 영역만 edge mask에서 제외 |
| `IFSI Video 8 (2).mp4` | 왼쪽 마크 + 오른쪽 colorbar | `true` | 고정 OSD 영역만 edge mask에서 제외 |
| `sample_3.MP4` | 고정 OSD 없음 | `false` | 우측 장면을 그대로 edge 입력으로 유지 |

생성된 source-aware 결과:

- `02814 (2)_teed_long.mp4`: 약 10.54초, 53 frames, OSD 무시 안 함
- `02815 (2)_teed_long.mp4`: 약 5.04초, 26 frames, OSD 무시 안 함
- `02824 (2)_teed_long.mp4`: 약 8.51초, 43 frames, OSD 무시 안 함
- `GOPR8356 (2)_teed_long_osd_ignored.mp4`: 약 8.41초, 42 frames
- `IFSI Video 4 (2)_teed_long_osd_ignored.mp4`: 약 21.02초, 105 frames
- `IFSI Video 8 (2)_teed_long_osd_ignored.mp4`: 약 22.22초, 111 frames
- `sample_3_teed_long.mp4`: 약 8.21초, 41 frames, OSD 무시 안 함

영상과 같은 이름의 `.json` 파일에는 source/output FPS, 원본/출력 길이,
sampling stride, frame별 latency와 edge ratio가 함께 기록된다. 파일은
`.omo/evidence/firesight-smoke-vision/long-videos-source-aware/`에서 확인한다.
이전에 생성된 `long-videos/sample_3_teed_long_osd_ignored.mp4`는 OSD 없는
영상에 고정 영역을 적용한 legacy 진단 산출물이므로 MVP 판단에 사용하지
않는다.

분류를 반영한 readiness 결과는
`.omo/evidence/firesight-smoke-vision/readiness-source-aware/`에 영상별로
저장되고, 통합 manifest는 `source_aware_edge_readiness_summary.json`이다.
전체 7개 원본은 decode/sampling이 PASS다. OSD가 있는 세 영상은 내부
마스킹이 `PASS`지만 경계 guard는 `REVIEW`로 남겨 두었다. 나머지 네 영상은
OSD 관련 항목이 `NOT_APPLICABLE`이다.

#### 장시간 Canny·TEED side-by-side 비교

같은 7개 원본과 source-aware OSD 프로파일을 사용해 Canny classical 결과와
TEED learned 결과를 좌우로 합성할 수 있다.

```bash
uv run --with opencv-python-headless python scripts/build_long_video_model_comparison.py \\
  --profiles data/fire360/video_profiles.json \\
  --teed-root .omo/evidence/firesight-smoke-vision/long-videos-source-aware \\
  --canny-root .omo/evidence/firesight-smoke-vision/long-videos-canny \\
  --out .omo/evidence/firesight-smoke-vision/long-videos-comparison
```

결과는 `.omo/evidence/firesight-smoke-vision/long-videos-comparison/`에
저장된다. 입력별 `*_canny_vs_teed.mp4`, 대표 중간 프레임 PNG, 통합
`long_video_comparison_summary.json`을 포함한다. 이 비교는 1280×720,
5 FPS, 개발 PC CPU에서 만든 full-duration sampled 시각화이며, Jetson
실측값이나 모델 품질 라벨이 아니다.

소방관용 열화상 화재 데이터셋 후보와 Fire360 적합성은
[`docs/thermal_fire_datasets.md`](docs/thermal_fire_datasets.md)에 정리했다.

CPU 기준선은 다음 명령으로 재현한다.

```bash
uv run --extra teed --with opencv-python python scripts/benchmark_edge_replay.py \\
  --input "data/fire360/raw/IFSI Video 4 (2).mp4" \\
  --out .omo/evidence/firesight-smoke-vision/benchmark/edge_replay_benchmark.json \\
  --checkpoint models/teed/5_model.pth \\
  --widths 480,640,1280 --inference-fps 2,5,10 --max-frames 8 \\
  --device cpu
```

이 benchmark는 Jetson 결과가 아니다. 개발 PC CPU에서 640×360 기준 TEED는
평균 약 72 ms, 전처리·고전 방식 포함 총 처리시간은 약 95 ms였고,
1280×720·5 FPS에서는 평균 총 처리시간이 약 405 ms로 5 FPS pacing을
충족하지 못했다. 실제 Jetson Orin Nano 가능성은 장치 확보 후 CUDA/TensorRT
측정으로 다시 판단한다.

### TEED 후보 30-frame 고정 평가표

다음 명령은 7개 영상에서 대표 프레임 30개를 결정론적으로 고정하고,
원본·Classical·TEED 결과를 동일한 640×360 입력으로 저장한다. 기존
source-aware 후보 프레임을 사용하므로 원본 영상을 다시 decode하지 않는다.

```bash
uv run --frozen python scripts/build_fire360_model_evaluation.py \\
  --out .omo/evidence/firesight-smoke-vision/model-evaluation-30 \\
  --checkpoint models/teed/5_model.pth \\
  --analysis-width 640 --classical-threshold 144 \\
  --classical-edge-width 3 --teed-threshold 0.75 \\
  --teed-edge-width 1 --background-scale 0.24 --device cpu
```

프레임 배정은 영상당 4장씩, 가장 긴 `IFSI Video 4`와 `IFSI Video 8`에
1장씩 추가하여 총 30장이다. 결과는 다음 위치에 있다.

- 평가 입력표: `.omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv`
- 고정 선택·파라미터 manifest: `.omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation_manifest.json`
- 영상별/통합 비교표: `.omo/evidence/firesight-smoke-vision/model-evaluation-30/contact_sheets/`
- 점수 기준: `.omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation_guide.md`

`evaluation.csv`의 `classical_score_1_to_5`, `teed_score_1_to_5`,
`preferred_method`, noise/breaks flag, `frame_flag`, `notes`를 사람이 채운
뒤에야 평균점수·TEED 승률·flagged frame을 계산한다. 현재 manifest의
`model_quality`는 의도적으로 `NOT_ESTABLISHED_WITHOUT_LABELS`이며, 이
artifact만으로 최종 모델을 확정하지 않는다.

### PiDiNet 추가 비교

TEED를 현재 기준선으로 유지하면서, 공식 [PiDiNet](https://github.com/hellozhuo/pidinet)
`table5_pidinet` 체크포인트를 같은 30개 프레임에 추가한다. `carv4`, SA,
dilation 구성을 사용하고, 추론 시 PDC를 일반 convolution으로 변환한다.
체크포인트는 `models/pidinet/table5_pidinet.pth`에 둔다.

```bash
uv run --extra teed python scripts/build_fire360_pidinet_comparison.py \
  --input-csv .omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv \
  --out .omo/evidence/firesight-smoke-vision/model-comparison-pidinet \
  --checkpoint models/pidinet/table5_pidinet.pth \
  --pidinet-threshold 0.30 --pidinet-edge-width 1 \
  --background-scale 0.24 --device cpu
```

결과는 `model-comparison-pidinet/comparison.csv`,
`model-comparison-pidinet/contact_sheets/`, `comparison_manifest.json`에
저장된다. contact sheet에서는 원본·Classical·TEED·PiDiNet을 한 행에서
비교하고, `comparison.csv`의 TEED/PiDiNet 점수와 `preferred_method`를 채워
추가 판단한다. PiDiNet threshold는 시각화 제어값이며 smoke-domain 정확도
검증 결과가 아니다.

두 모델의 CPU 기준 latency는 다음 명령으로 같은 30개 프레임에서 측정한다.

```bash
uv run --extra teed python scripts/benchmark_fire360_models.py \
  --input-csv .omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv \
  --out .omo/evidence/firesight-smoke-vision/model-comparison-pidinet/learned_model_benchmark.json \
  --sizes 640x360,320x240 --teed-threshold 0.75 \
  --pidinet-threshold 0.30 --warmup 2 --device cpu
```

현재 개발 PC CPU 측정은 640×360에서 TEED 평균 58.1 ms/p95 64.9 ms,
PiDiNet 평균 320.7 ms/p95 332.4 ms이고, 320×240에서는 TEED 평균 19.8
ms/p95 22.8 ms, PiDiNet 평균 104.9 ms/p95 119.8 ms이다. 이 값은 Jetson
Orin Nano 결과가 아니며, PiDiNet은 시각 비교 후보로는 남기되 현재 CPU
기준 실시간 후보로는 TEED보다 불리하다.

### Jetson Orin Nano latency projection

320×240 TEED p95 22.8 ms는 개발 PC CPU 실측값이다. 실제 Jetson Orin Nano
보드가 없으므로 TOPS만으로 변환하지 않고 실행 경로별 계획 범위를 둔다.

- Jetson CPU-only: p95 35–70 ms
- Jetson CUDA/PyTorch: p95 10–25 ms
- Jetson TensorRT FP16: p95 5–15 ms
- 현재 선택할 계획값: p95 8–25 ms, 대표값 15 ms, 보수적 검토값 30 ms

15 FPS의 frame period는 66.7 ms이고 MVP 모델 단독 목표는 p95 80 ms이다.
따라서 보수적 검토값 30 ms도 모델 단독 예산 안에 있지만, capture/decode,
resize, preprocessing, post-processing, overlay, encoding과 스케줄링은 아직
포함하지 않는다. 이는 실측이 아닌 계획용 추정치다.

추정치를 명시한 30-frame annotated video와 plot은 다음 명령으로 재생성한다.

```bash
uv run --with opencv-python-headless python scripts/build_teed_jetson_latency_projection.py
```

생성 위치는 `.omo/evidence/firesight-smoke-vision/jetson-latency-projection/`이며,
`teed_jetson_latency_projection.mp4`, `teed_jetson_latency_projection.png`,
`teed_jetson_latency_projection.json`, `teed_jetson_latency_projection.md`를
포함한다. JSON과 영상에는 `ESTIMATE_NOT_MEASURED`를 표시해 Jetson 실측과
혼동하지 않도록 했다. 최종 판단은 보드 확보 후 같은 30-frame protocol로
CUDA/TensorRT와 end-to-end latency를 측정해 갱신한다.

### 배포 runtime v1

배포 경로는 `firesight_vision.deployment.FireSightRuntime`으로 통일했다.
입력 프레임을 320×240으로 resize하고, 영상별 OSD profile을 선택적으로
적용한 뒤, TEED 또는 Canny backend의 overlay·mask·latency를 반환한다.
현재 backend는 `teed`와 `canny`이다. Canny는 checkpoint를 사용하는 학습
모델이 아니라 OpenCV의 Gaussian smoothing, gradient, non-maximum suppression,
hysteresis를 묶은 classical edge operator이며, 공식 `cv.Canny()` 설명은
[OpenCV Canny tutorial](https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)에
있다. 따라서 Canny를 “Canny neural model”로 해석하거나 TEED와 같은
학습 모델로 비교하지 않고, 매우 빠른 fallback/baseline으로 평가한다.

PC에서 실제 deployment CLI를 실행하는 예:

```bash
uv run --extra teed --with opencv-python-headless \\
  python scripts/run_firesight_deployment.py \\
  --backend teed --checkpoint models/teed/5_model.pth \\
  --input data/fire360/frames_multiclip/ifsi_video_4 \\
  --out .omo/evidence/firesight-smoke-vision/deployment/teed \\
  --profiles data/fire360/video_profiles.json \\
  --video-key "IFSI Video 4 (2).mp4" \\
  --width 320 --height 240 --fps 15 --threshold 0.75 \\
  --edge-width 1 --device cpu --no-realtime --write-video
```

Canny backend은 checkpoint 없이 같은 입력 계약을 사용한다.

```bash
uv run --with opencv-python-headless \\
  python scripts/run_firesight_deployment.py \\
  --backend canny \\
  --input data/fire360/frames_multiclip/ifsi_video_4 \\
  --out .omo/evidence/firesight-smoke-vision/deployment/canny \\
  --profiles data/fire360/video_profiles.json \\
  --video-key "IFSI Video 4 (2).mp4" \\
  --width 320 --height 240 --fps 15 \\
  --canny-low 50 --canny-high 150 \\
  --no-realtime --write-video
```

CLI는 `frames/`, `masks/`, `deployment_metrics.jsonl`,
`deployment_summary.json`과 선택적 `overlay.mp4`를 저장한다. 영상 파일을
직접 넣으면 OpenCV로 decode하고 목표 FPS에 맞춰 샘플링한다. Jetson에서는
보드에 맞는 JetPack CUDA/PyTorch와 시스템 OpenCV를 설치한 뒤 같은 CLI를
`--device cuda`로 실행하여 실제 수치를 기록한다. 현재 코드에는 TensorRT
변환을 넣지 않았으므로, TensorRT는 Jetson에서 TEED ONNX/export 경로를
검증한 뒤 별도 backend로 추가한다.

TEED와 Canny를 같은 고정 30-frame, 320×240 runtime에서 비교하는 명령:

```bash
uv run --extra teed --with opencv-python-headless \\
  python scripts/benchmark_firesight_backends.py \\
  --input-csv .omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv \\
  --profiles data/fire360/video_profiles.json \\
  --teed-checkpoint models/teed/5_model.pth \\
  --out .omo/evidence/firesight-smoke-vision/model-comparison-canny/deployment_backend_benchmark.json \\
  --backends teed,canny --width 320 --height 240 --device cpu
```

이번 PC CPU 실행 결과는 Canny 평균 5.43 ms/p95 6.25 ms, TEED 평균
23.82 ms/p95 30.37 ms였다. 이는 deployment runtime 내부의 resize와
post-processing을 포함한 개발 PC 기준이며 Jetson 측정값이 아니다. Canny의
edge ratio가 낮거나 높다는 사실만으로 smoke outline 품질을 판정할 수
없으므로, 품질 판단은 기존 30-frame 사람 평가표와 함께 진행한다.

같은 30개 프레임의 시각 결과와 진단 지표를 한 번에 다시 생성하려면 다음
명령을 사용한다.

```bash
uv run --extra teed --with opencv-python-headless \\
  python scripts/build_firesight_quality_comparison.py \\
  --input-csv .omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv \\
  --out .omo/evidence/firesight-smoke-vision/model-comparison-canny/quality \\
  --teed-checkpoint models/teed/5_model.pth \\
  --profiles data/fire360/video_profiles.json \\
  --width 320 --height 240 --threshold 0.75 \\
  --canny-low 50 --canny-high 150 --device cpu --warmup 2
```

현재 실행 결과는 Canny 평균 5.63 ms/p95 6.25 ms, TEED 평균 31.29
ms/p95 39.64 ms였다. 두 backend 모두 OSD 누출은 0 frame/0 pixel이었다.
Canny의 평균 edge ratio는 0.0154, TEED는 0.0642였지만, edge density는
smoke outline 정확도나 연속성을 나타내는 정답 지표가 아니다. 따라서
요약의 `quality_status`는 `NOT_ESTABLISHED_WITHOUT_LABELS`, `winner`는
`null`로 유지한다. contact sheet의 결과를 사람이 1~5점으로 평가한 뒤에야
품질 우위를 주장할 수 있다.

결과 산출물은 `model-comparison-canny/quality/quality_summary.json`,
`quality_comparison.csv`, `quality_contact_sheet.png`에 저장된다.

## ROS 2 mmWave Bring-Up

The hardware bring-up workspace lives in `ros2_ws`. It targets the LTS ROS 2
release for the Ubuntu host, with a first-pass mapper from mmWave `PointCloud2`
samples to front obstacle occupancy grid and RViz markers.

```bash
bash scripts/install_ros2_ubuntu.sh
source /opt/ros/${ROS_DISTRO:-lyrical}/setup.bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch firesight_ros2 mmwave_sensor_mapping.launch.py
```

See `docs/mmwave_ros2_calibration.md` for the 0.5 m, 1.0 m, and 1.5 m
calibration gate.

## Thermal + mmWave HUD Fusion MVP

The current v1 fusion path is a real-time-ish HUD pipeline check, not model
fine-tuning. Thermal frames produce green structure edges and red hotspot
outlines; mmWave adds front obstacle distance cues through
`/firesight/mmwave/front_obstacles`; the final HUD image publishes on
`/firesight/hud/fused_overlay`.

Offline mock artifact:

```bash
uv run --frozen python scripts/build_thermal_mmwave_hud_fusion_mock.py
```

ROS2 mock/replay path:

```bash
cd ros2_ws
source install/setup.bash
ros2 launch firesight_ros2 thermal_mmwave_fusion_demo.launch.py
```

See `docs/thermal_mmwave_hud_fusion.md` for the topic contract and acceptance
gates.
