# Jetson Nano TEED 실행 프로그램

이 폴더는 원형 NVIDIA Jetson Nano 2GB/4GB에서 `models/teed/5_model.pth`를
카메라, 이미지 또는 영상에 적용하는 독립 실행 프로그램이다. 프로젝트 본체는
Python 3.11을 요구하지만, 이 폴더의 Python 파일은 JetPack 4.x의 Python 3.6에서도
실행되도록 별도로 작성했다.

Jetson Orin Nano는 다른 보드이며 이 문서의 JetPack/Python 버전을 사용하지 않는다.

## 1. 지원 환경

- Jetson Nano 2GB/4GB
- JetPack 4.6.x, Python 3.6, CUDA 10.2
- JetPack 버전에 맞는 NVIDIA 제공 PyTorch wheel
- JetPack 시스템 OpenCV

JetPack 4.6.6은 원형 Jetson Nano를 지원하는 JetPack 4의 마지막 릴리스이며 현재
EOL 상태다. 새 PyPI `torch`를 설치하지 말고, 반드시 JetPack 버전에 맞는 NVIDIA
wheel을 사용한다.

- [JetPack 4.6.6](https://developer.nvidia.com/jetpack-sdk-466)
- [NVIDIA PyTorch for Jetson 설치 문서](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [Jetson용 PyTorch wheel 목록](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

JetPack 4.6 계열에서는 Python 3.6용 `torch-1.10.0-...-cp36-...aarch64.whl`
계열을 선택한다. 정확한 파일은 설치한 JetPack 세부 버전에 맞춰 위 목록에서 받는다.

## 2. 설치

저장소 전체를 Jetson Nano로 복사하고, 받은 NVIDIA wheel 경로를 설치 스크립트에
전달한다.

```bash
cd ~/haechi/deploy/jetson_nano
chmod +x install.sh
./install.sh ~/Downloads/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

설치가 끝나면 `CUDA available: True`가 출력되어야 한다.

## 3. 실행

USB 카메라 `/dev/video0`:

```bash
cd ~/haechi
python3 deploy/jetson_nano/run_teed.py \
  --source 0 \
  --checkpoint models/teed/5_model.pth \
  --device cuda \
  --width 320 --height 240 \
  --threshold 0.75
```

CSI 카메라 센서 0:

```bash
python3 deploy/jetson_nano/run_teed.py \
  --source csi://0 \
  --device cuda \
  --capture-width 640 --capture-height 480 --capture-fps 30 \
  --width 320 --height 240
```

화면이 없는 SSH 환경에서 영상 파일을 처리하고 결과와 성능 JSON을 저장:

```bash
python3 deploy/jetson_nano/run_teed.py \
  --source input.mp4 \
  --device cuda \
  --profiles data/fire360/video_profiles.json \
  --headless \
  --output teed_overlay.mp4 \
  --metrics jetson_teed_metrics.json
```

단일 이미지 확인:

```bash
python3 deploy/jetson_nano/run_teed.py \
  --source tests/fixtures/images/sample_room.jpg \
  --device cuda --headless \
  --output sample_room_teed.png \
  --metrics sample_room_teed.json
```

Fire360에서 추출한 단일 이미지나 카메라 입력처럼 파일명만으로 원본 영상을
알 수 없을 때는 `--video-key`를 함께 준다.

```bash
python3 deploy/jetson_nano/run_teed.py \
  --source ifsi_frame.jpg --device cuda --headless \
  --profiles data/fire360/video_profiles.json \
  --video-key "IFSI Video 4 (2).mp4" \
  --output ifsi_frame_teed.png
```

프로파일에서 `ignore_fire360_osd`가 참인 영상은 좌측 기관 로고와 우측 온도
colorbar가 edge 결과에서 제거된다.

화면 실행 중에는 `q` 또는 `Esc`로 종료한다. 기본값은 320x240, threshold 0.75,
3픽셀 초록색 edge overlay다. `--max-frames 100`을 주면 지정 프레임 수 뒤 자동
종료한다.

농연과 저조도에서 사람·구조물 경계가 약해지는 문제를 줄이기 위해 모델 입력에만
CLAHE 대비 보강을 적용한다. 기본 강도는 `--contrast-clip-limit 2.0`이며 출력
배경의 원래 색은 바뀌지 않는다. 일반 조명에서 보강이 불필요하면
`--contrast-clip-limit 0`으로 끌 수 있다.

## 4. Nano에서 확인할 항목

전원 모드와 클럭을 고정하고 측정하려면 실행 전 다음을 적용한다.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

`jetson_teed_metrics.json`에서 다음 값을 확인한다.

- `device`가 `cuda`인지
- `processed_frames`가 0보다 큰지
- `contrast_clip_limit`이 의도한 값인지
- `model_latency_ms.p95`
- `total_latency_ms.p95`
- `effective_fps`

현재 프로그램은 검증 우선의 PyTorch CUDA 경로다. 목표 FPS가 나오지 않을 때만
TEED 마지막 출력 하나를 ONNX로 export하고, Nano에서 TensorRT engine을 직접
빌드하는 다음 최적화 단계를 진행한다. TensorRT engine은 TensorRT 버전과 GPU에
종속되므로 다른 PC에서 만든 engine을 그대로 복사하지 않는다.

## 5. 파일 구성

```text
deploy/jetson_nano/
  install.sh       JetPack 시스템 의존성과 NVIDIA PyTorch wheel 설치
  run_teed.py      카메라·이미지·영상 실시간 추론 및 성능 기록
  teed_model.py    Python 3.6 호환 TEED 네트워크
models/teed/
  5_model.pth      기존 BIPED 사전학습 체크포인트
```
