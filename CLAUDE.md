# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FireSight smoke vision (Python package `firesight_vision`): a research workspace for an AR firefighter HUD. Thermal/RGB frames go through an edge detector (classical Pillow/Canny, or learned TEED / PiDiNet), optionally fused with monocular depth (Depth Anything V2) or mmWave radar obstacles, and are drawn as HUD overlays. Target hardware is Jetson (original Nano and Orin Nano); most numbers in the repo are dev-PC CPU measurements. The README and much of the docs are written in Korean.

Evidence discipline matters here: outputs are labelled `NOT_ESTABLISHED_WITHOUT_LABELS`, `ESTIMATE_NOT_MEASURED`, etc. Do not claim model quality, a "winner", or Jetson latency from edge density or CPU runs; keep those status fields intact unless real labels/measurements exist.

## Commands

Python is pinned to 3.11 (`>=3.11,<3.12`), managed with `uv`. Heavy deps are optional extras: `teed` (torch, numpy), `depth` (+ transformers), `rosbag` (rosbags). OpenCV is never a declared dependency; scripts that need it are run with `--with opencv-python-headless`.

```bash
# Full test suite (without the extras, depth/rosbag tests fail at collection on missing numpy)
uv run --frozen --extra teed --extra depth --extra rosbag --with opencv-python-headless pytest

# Single test
uv run --frozen --extra teed pytest tests/test_deployment.py::test_name -q

# Lint / format / type check (pyright runs in strict "all" mode over src, tests, scripts)
uv run --frozen ruff check .
uv run --frozen ruff format .
uv run --frozen basedpyright
```

- pytest runs with `filterwarnings = ["error"]`, `--strict-markers`, and `pythonpath = ["src"]`. Tests that need torch/cv2 `pytest.skip` when the module is missing.
- `tests/test_doctor.py::test_doctor_writes_ok_evidence_when_scaffold_exists` fails on a fresh clone because it requires `.omo/evidence/firesight-smoke-vision/` (gitignored) to exist; `mkdir -p` it before running.
- Ruff uses `select = ["ALL"]` with Google docstrings. New modules that need relaxations get explicit `per-file-ignores` in `pyproject.toml`; torch-heavy modules are listed in basedpyright's `exclude`. Follow that pattern rather than adding inline `noqa`/`type: ignore` broadly (`reportUnnecessaryTypeIgnoreComment` is an error).
- Run-commands for each pipeline step (edge replay, TEED replay, depth replay, rosbag extraction, model evaluation/benchmarks, deployment CLI) are documented in `README.md` with exact flags; reuse them.

## Architecture

- `src/firesight_vision/` — library code. Key layers:
  - Edge backends: `hud_edges.py` (Pillow classical baseline + Fire360 OSD ignore regions), `canny.py`, `teed.py`/`teed_model.py`, `pidinet.py`/`pidinet_model.py`, `thermal_edges.py`.
  - Replay runners (`replay.py`, `teed_replay.py`, `teed_depth_replay.py`) consume image directories at a paced FPS and write `frames/`, `*_metrics.jsonl`, `*_summary.json`. `run_*.py` modules are the CLI entry points registered in `[project.scripts]`.
  - `deployment.py` — `FireSightRuntime`, the unified deployment path: resize to 320×240, apply per-video OSD profile, run a `teed` or `canny` backend behind the `_EdgeBackend` protocol, return overlay/mask/latency. New backends (e.g. TensorRT) plug in here via `_build_backend`.
  - Depth/fusion: `depth_anything.py`, `depth_edge_fusion.py` (distance-banded edge colouring), `mmwave_sim.py`, `mmwave_hud_fusion.py`.
  - Dataset/eval utilities: `bbox_*.py`, `image_level_eval*.py`, `yolo_dataset_validation.py`, `smoke.py` (deterministic smoke augmentation).
- `scripts/` — one-off experiment/evaluation/benchmark CLIs (packaged in the wheel too). Several use PEP 723 inline-script headers and are standalone (`doctor.py`, `validate_registry.py`, `validate_schema.py`).
- `data/fire360/video_profiles.json` decides per source video whether the fixed Fire360 OSD (left logo, right colorbar) is masked out; only apply OSD masking to videos flagged there.
- `models/teed/5_model.pth` (BIPED-pretrained, not fine-tuned) and `models/pidinet/table5_pidinet.pth` are committed checkpoints.
- `deploy/jetson_nano/` — standalone runtime for the original Jetson Nano (JetPack 4.6, **Python 3.6**). It is excluded from ruff and must keep Python 3.6 syntax; `tests/test_jetson_nano_runtime.py` checks that and that its `teed_model.py` stays numerically identical to `src/firesight_vision/teed_model.py`. Change both together.
- `ros2_ws/` — ROS 2 colcon workspace. `firesight_ros2` (Python) holds the mmWave obstacle mapper, thermal edge node, HUD fusion node, mocks, and launch files; the other packages (`haechi`, `fusion_imu_radar`, `imu`, `thermal_camera`, `ti_mmwave_rospkg*`, `serial-ros2-master`) are vendored hardware packages, excluded from ruff. `tests/test_ros2_scaffold.py` asserts these files exist and that mmWave topic names stay consistent across code, config, `docs/mmwave_ros2_calibration.md`, `README.md`, and `ros2_ws/README.md`, so rename topics everywhere at once. Build with `colcon build --symlink-install` inside `ros2_ws` (see README "ROS 2 mmWave Bring-Up").
- Generated artifacts go under `.omo/evidence/firesight-smoke-vision/` (the `evidence_root` in `[tool.firesight_vision]`), which is gitignored, as are raw videos (`data/fire360/raw/`) and `data/external/`.
- `docs/` holds topic contracts and calibration gates; `reports/` holds dated experiment write-ups.
