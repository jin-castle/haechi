# FireSight 외부 열화상 화재 데이터셋 조사

조사 목적은 Fire360처럼 화재·농연 현장에서 열화상카메라로 촬영된 자료를
찾아 FireSight의 외곽선 검출 robustness 평가에 사용할 후보를 정리하는
것이다. 2026-08-19 기준으로 확인한 공개 자료를 기준으로 작성했다.

## 우선순위 요약

| 우선순위 | 후보 | 장면·센서 | 공개 형태와 라벨 | FireSight 적합성 |
| --- | --- | --- | --- | --- |
| 1 | [FirebotSLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC10490787/) | 소방관 훈련시설 주차장 화재. FLIR Boson 640 두 대, 640×512, 30 FPS, 16-bit radiometric thermal | 논문의 Data Availability Statement는 “available on request”이며, 공개 repository·직접 다운로드는 확인되지 않음 | Fire360과 가장 가까운 장면. 저자·기관에 데이터 접근 문의가 필요함 |
| 2 | [NRC full-scale room-fire / CFMRD 보고서](https://nrc-publications.canada.ca/eng/view/object/?id=84910e7f-4657-4529-8ecd-f74a9fad522d) | 실물 크기 방 화재 7회. thermal IR, RGB, 화재 센서 동시 수집 | 공개 보고서와 실험 설명. 원시 영상의 직접 다운로드·사용 조건은 별도 확인 필요 | 실내 화재·flashover 도메인에는 매우 적합하지만 firefighter POV는 아님 |
| 3 | [FLAME-T](https://data.mendeley.com/datasets/x6gty88k4f/1) | Gran Canaria prescribed burn, 지상 열화상카메라. FLIR Tau 2, A35, Lepton | 약 2,640 thermal images, 160×120~336×256, JPEG와 YOLO TXT 라벨, CC BY-NC 4.0 | 320×240에 가장 가까운 공개 지상 자료. 연속 영상이 아니라 프레임 묶음이므로 정적 robustness 평가에 우선 사용 |
| 4 | [FLAME](https://experts.nau.edu/en/datasets/the-flame-dataset-aerial-imagery-pile-burn-detection-using-drones/) | 드론의 visible 및 infrared thermal 영상, prescribed pile burn | 39,375 train frames, 8,617 test frames, 2,003 segmentation frames/masks, CC BY 4.0 | 공개 thermal video와 segmentation mask를 바로 활용할 수 있으나 항공 시점·야외 화재라 Fire360과 domain gap이 큼 |
| 5 | [FLAME 2](https://experts.nau.edu/en/datasets/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-d/) | 드론의 visible·infrared paired video, open-canopy prescribed fire | fire/smoke frame label, CC BY 4.0 / IEEE DataPort | smoke와 열화상 동시 비교에 유용하지만 firefighter-mounted 카메라는 아님 |
| 6 | [WIT-UAS](https://github.com/castacks/WIT-UAS-Dataset) | wildland fire LWIR. Seek S304SP 320×240, FLIR Boson 640×512, RGB, UAS ROS bags | 6,951 images, 그중 2,062개 수동 라벨 이미지 | 320×240·640×512 센서 조건은 좋지만 핵심 과제가 화재 주변 crew/vehicle 검출이며 실내 농연 외곽선 자료는 아님 |

## FirebotSLAM 상세 조사

[FirebotSLAM 논문](https://www.mdpi.com/1424-8220/23/17/7611)의 정확한 의미는
“소방관이 착용한 TIC 영상 데이터셋”이 아니라, 소방관 훈련시설의 농연 환경에서
UGV에 장착한 stereo 열화상카메라로 SLAM을 수행한 실험 데이터다. 따라서 Fire360과
장면·센서는 매우 가깝지만, firefighter POV 영상이나 농연 외곽선 segmentation
mask를 제공하는 데이터셋으로 보면 안 된다.

### 촬영 조건

| 구분 | 논문에 기술된 내용 |
| --- | --- |
| 장소 | 소방관 훈련시설 내부 주차장. 차량, 목재 팔레트, shipping container가 장면에 포함됨 |
| 화재 | 연기 발생원으로 목재 화재 두 개를 사용했고, 촬영 사이에 연료를 보충함 |
| 시점 | dataset 1은 점화 약 5분 후, dataset 2는 약 30분 후, dataset 3은 약 1시간 후. 순서대로 가장 차갑고 가장 뜨거운 상태 |
| 환경 | 12월 초 아침, 외기온은 결빙점보다 약간 높은 조건 |
| 카메라 | FLIR Boson 640 두 대, stereo baseline 80 mm, 640×512, 30 FPS |
| 데이터 | 16-bit radiometric raw thermal. stereo pair는 가장 가까운 timestamp로 맞췄고, hardware trigger가 아닌 점이 한계로 제시됨 |

카메라는 가열한 plywood calibration board와 원형 패턴으로 보정했으며, ROS
`image_pipeline`/OpenCV calibration을 사용했다. 일반적인 8-bit 변환은 radiometric
정보를 잃기 때문에 논문은 누적 histogram 기반 dynamic-range preprocessing을
사용했고, 뜨거운 화재가 전체 대비를 지배하지 않도록 `Imax=40,000`을 사용했다.
논문에서 중요한 환경 정보가 주로 20,000~40,000 영역에 있다고 설명한 점은
FireSight에서 8-bit preview만 바로 TEED/Canny에 넣을 때 주의할 이유다.

### SLAM 결과와 FireSight에 주는 의미

- ORB-SLAM3를 framework로 사용하고, 열화상에서 비교적 안정적인 SURF-BRIEF
  feature 조합으로 교체했다. 전체 FirebotSLAM 처리시간은 약 120~130 ms/frame으로
  기록되어, 현재 FireSight의 15 FPS 목표와 직접 비교할 때는 별도 pipeline이다.
- 얇은 smoke/haze에서는 feature detection과 matching이 크게 무너지지 않았다.
  반면 dataset 3에서는 주차장 진입 시 tracking을 잃고 첫 약 25초가 소실되었으며,
  출구에서 loop closure가 되지 않아 map이 합쳐지지 않았다.
- 주 화재 실험에는 정답 trajectory가 없었다. 저자들은 reprojection error와
  point-cloud/trajectory의 시각적 일관성을 근거로 path reversal·obstacle avoidance에
  충분한 수준이라고 평가했으므로, 이 데이터로 TEED가 smoke outline을 더 잘
  검출한다고 자동 판정할 수 없다.

### 데이터 접근성

논문의 공식 Data Availability Statement는 “Datasets used in the experiments are
available on request.”이다. 이번 조사에서 논문, [ResearchGate의 논문 사본](https://www.researchgate.net/publication/373666328_FirebotSLAM_Thermal_SLAM_to_Increase_Situational_Awareness_in_Smoke-Filled_Environments),
저자 페이지와 연결 자료를 확인했지만, raw stereo video를 내려받을 수 있는 공개
repository나 direct download URL은 찾지 못했다. 논문에 표기된 문의 대상은 다음과
같다.

- corresponding author: `b.r.vanmanen@saxion.nl`
- `v.i.sluiter@saxion.nl`
- `a.y.mersha@saxion.nl`

### 제보된 Zenodo 링크 교차 확인

사용자가 제보한 URL의 record ID `13983719`를 Zenodo 공개 API로 확인한 결과,
FirebotSLAM 자료가 아니었다. 해당 레코드의 제목은
`SERVADEI_DATASET_TEST_2-22-07-2024`이고, 생성자는 University of Udine이다.
설명도 소방·열화상·농연이 아니라 포도밭에서 자율주행 로봇으로 수집한 IMU,
GNSS, wheel odometry, point cloud 및 multispectral image 자료라고 되어 있다.

파일도 `0029SET.zip`, `2024-07-22-11-03-15.bag`, `gndvi.bag`, `ndvi.bag`,
`ndre.bag`로 구성되어 있어 FirebotSLAM의 640×512 stereo radiometric thermal
recording과 일치하지 않는다. [Zenodo record 13983719](https://zenodo.org/records/13983719)
및 [공개 API metadata](https://zenodo.org/api/records/13983719)를 근거로, 이 링크는
FireSight의 열화상 농연 검증 데이터로 사용하지 않는다. Zenodo API에서
`FirebotSLAM`을 직접 검색한 결과도 현재 0건이었다.

접근을 요청할 때는 단순히 “FirebotSLAM dataset”이라고 쓰기보다, `(1) left/right
16-bit radiometric frames, (2) timestamps, (3) intrinsic/extrinsic calibration,
(4) dataset 1/2/3 구분, (5) 사용·재배포 조건`을 함께 요청하는 것이 좋다. FireSight
검증에는 8-bit export보다 raw radiometric 원본과 calibration이 더 유용하다.

[논문에 연결된 실험 영상](https://youtu.be/9pRakFnuySc)은 결과 확인용 참고자료로
볼 수 있지만, raw dataset download 링크는 아니다.

## 가장 가까운 후보

FirebotSLAM은 공개 즉시 사용 자료가 아니라 접근 문의형 후보다. 따라서 당장은
장면·센서가 가까운 외부 robustness 후보로 기록하고, raw data 승인을 받은 뒤
FireSight의 동일한 30-frame protocol과 long-video replay를 반복하는 순서가 맞다.

[NRC 보고서](https://nrc-publications.canada.ca/eng/view/object/?id=84910e7f-4657-4529-8ecd-f74a9fad522d)는
실물 크기 방 화재 7회에 thermal IR·RGB·센서를 함께 사용하고 약 1.8 TB를
수집한 실험을 설명한다. 실제 실내 화재와 flashover 분석에는 강하지만, 보고서
페이지에서 원시 데이터의 즉시 다운로드 경로와 라이선스는 확인되지 않았다.

## 바로 사용할 공개 자료

[FLAME-T](https://data.mendeley.com/datasets/x6gty88k4f/1)는 FireSight의 현재
320×240 목표와 해상도 범위가 맞고, 지상 열화상카메라 및 YOLO 라벨을 제공한다.
약 1초 간격으로 20프레임씩 묶인 이미지 배치이므로, 첫 외부 검증은 다음처럼
진행하는 것이 현실적이다.

1. FLAME-T의 160×120~336×256 자료로 입력 크기·노이즈·flame boundary에 대한
   Canny/TEED 시각 비교를 수행한다.
2. [FLAME](https://experts.nau.edu/en/datasets/the-flame-dataset-aerial-imagery-pile-burn-detection-using-drones/)와
   [FLAME 2](https://experts.nau.edu/en/datasets/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-d/)의
   thermal video와 mask/label로 시간적 끊김과 smoke 오검출을 확인한다.
3. FirebotSLAM 또는 NRC raw data 접근이 확보되면 indoor firefighter-like
   domain에서 동일한 30-frame protocol을 반복한다.

## FLAME-T 실제 다운로드 및 FireSight replay 확인

2026-08-19에 [공식 Mendeley Data 페이지](https://data.mendeley.com/datasets/x6gty88k4f/1)의
공개 파일을 직접 내려받아 압축 무결성과 SHA-256을 확인했다. 로컬 archive의 크기는
267,078,842 bytes이고 SHA-256은 다음과 같다.

```text
c3bbd9e6960ce72756400136a643c69c00a76c0af1f5e197af65f2a96acbc19e
```

압축을 풀었을 때 실제 파일 수는 공식 페이지의 “약 2,640장” 설명보다 정확한
2,634장이다. Point A~F별로 thermal JPEG, TIFF, YOLO TXT가 각각 1:1로 존재한다.
JPEG/TIFF는 모두 단일 채널이며 카메라별 해상도는 A35 320×256, Lepton 160×120,
Tau 336×256이다. 비어 있지 않은 label은 2,494개이고 label class id는 `0` 20개,
`1` 2,474개다. 압축 파일 안에 class-name map이 따로 없으므로 `0`과 `1`을
임의로 flame/smoke라고 해석하지 않는다.

동일 입력을 비교하기 위해 각 Point와 카메라별로 시간축 시작·중간·끝을 고르게
8장씩 뽑아 144장 샘플을 만들고 Canny dense-smoke와 TEED pretrained를 실행했다.
두 replay 모두 144/144 프레임을 처리했다. 비교용 시각화에서는 원본 thermal
대비가 보이도록 `background_scale=0.8`을 사용했고, Canny는 `dense-smoke`, TEED는
checkpoint `models/teed/5_model.pth`, threshold `0.75`, CPU로 실행했다.

| 방법 | 평균 latency | p95 latency | edge pixel 비율 평균 | 비고 |
| --- | ---: | ---: | ---: | --- |
| Canny dense-smoke | 54.1 ms | 99.9 ms | 33.4% | 배경 texture까지 매우 조밀하게 반응 |
| TEED threshold 0.75 | 123.0 ms | 305.1 ms | 5.1% | 더 sparse하고 연속적인 contour 후보 |

입력 해상도가 섞였으므로 고정 해상도 Fire360 benchmark와 직접 비교하면 안 된다.
TEED의 해상도별 평균/p95는 160×120에서 56.5/91.5 ms, 320×256에서
131.7/232.3 ms, 336×256에서 180.8/362.0 ms였다. 따라서 이 결과는 Jetson
Orin Nano 예측치가 아니라, FLAME-T를 실제로 통과시킨 PC CPU replay 결과다.

YOLO box를 segmentation 정답으로 간주할 수 없기 때문에 별도의 위치 proxy만
계산했다. label box 내부에 edge pixel이 하나라도 들어가는 비율은 Canny가
평균 92.3%, TEED가 52.7%였지만, 전체 edge pixel 중 label box 안에 있는 비율은
Canny 0.58%, TEED 2.88%였다. 즉 Canny는 box를 거의 덮는 대신 배경 edge가 훨씬
많고, TEED는 box coverage를 일부 포기하는 대신 반응이 더 집중되어 있었다. 이는
농연 외곽선 품질의 정답 점수가 아니라 모델 선택을 위한 보조 지표다.

재현 산출물은 다음에 저장했다.

- 압축 원본: `data/external/flame_t/FLAME-T.zip`
- 추출 데이터: `data/external/flame_t/FLAME-T/`
- 샘플 manifest: `.omo/evidence/firesight-smoke-vision/flame-t/sample_144/sample_manifest.json`
- 비교 contact sheet: `.omo/evidence/firesight-smoke-vision/flame-t/flame_t_canny_vs_teed_contact_sheet.png`
- replay summary: `.omo/evidence/firesight-smoke-vision/flame-t/flame_t_verification_summary.json`
- 해상도·카메라별 latency: `.omo/evidence/firesight-smoke-vision/flame-t/flame_t_resolution_stratified_latency.json`

결론적으로 FLAME-T는 공개되어 있고 현재 FireSight에서 실제 실행 가능한 외부
열화상 robustness 자료로 확인됐다. 다만 prescribed-burn 지상 영상과 YOLO
bounding box 자료이므로, 실내 소방관 시점 농연 및 smoke-outline segmentation을
대체하지 않는다. 최종 모델 선택은 FLAME-T의 시각·proxy 결과를 참고하되,
FirebotSLAM/NRC 같은 indoor raw thermal 접근이 확보되면 별도 검증해야 한다.

## 중요한 한계

이번 조사에서 “소방관이 착용한 TIC로 촬영한 실내 농연 영상”과 “smoke
outline segmentation mask”를 동시에 제공하는 즉시 다운로드 가능한 공개
데이터셋은 확인하지 못했다. 공개 자료의 다수는 flame/fire detection,
wildland fire, drone view 또는 정적 thermal image 중심이다. 따라서 FLAME-T,
FLAME, FLAME 2, WIT-UAS는 우선 외부 robustness·시각 비교용으로 사용하고,
FirebotSLAM/NRC 자료는 접근 권한을 얻은 뒤 도메인 검증용으로 추가하는 것이
안전하다. 라벨 정의가 FireSight의 “농연 외곽선”과 다르므로, 이 자료만으로
모델 품질을 자동 확정하거나 곧바로 supervised fine-tuning을 시작하지 않는다.
