import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

JETSON_DIR = Path("deploy/jetson_nano")


@pytest.mark.parametrize("filename", ["teed_model.py", "run_teed.py"])
def test_jetson_runtime_keeps_python36_syntax(filename: str) -> None:
    source = (JETSON_DIR / filename).read_text(encoding="utf-8")
    _ = ast.parse(source, filename=filename, feature_version=(3, 6))


def test_jetson_model_matches_project_model() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    script = (
        "import torch\n"
        "from deploy.jetson_nano.teed_model import TED as JetsonTED\n"
        "from firesight_vision.teed_model import TED as ProjectTED\n"
        "torch.manual_seed(17)\n"
        "project_model = ProjectTED().eval()\n"
        "jetson_model = JetsonTED().eval()\n"
        "jetson_model.load_state_dict(project_model.state_dict(), strict=True)\n"
        "input_tensor = torch.randn(1, 3, 64, 64)\n"
        "with torch.no_grad():\n"
        "    project_output = project_model(input_tensor, single_test=True)[-1]\n"
        "    jetson_output = jetson_model(input_tensor, single_test=True)[-1]\n"
        "torch.testing.assert_close(\n"
        "    jetson_output, project_output, rtol=0.0, atol=0.0\n"
        ")\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_jetson_cli_processes_a_real_image(tmp_path: Path) -> None:
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("cv2 is not installed")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    output_path = tmp_path / "overlay.png"
    metrics_path = tmp_path / "metrics.json"
    command = [
        sys.executable,
        str(JETSON_DIR / "run_teed.py"),
        "--source",
        "tests/fixtures/images/sample_room.jpg",
        "--checkpoint",
        "models/teed/5_model.pth",
        "--device",
        "cpu",
        "--width",
        "64",
        "--height",
        "64",
        "--warmup",
        "0",
        "--headless",
        "--output",
        str(output_path),
        "--metrics",
        str(metrics_path),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    payload = cast(
        "dict[str, object]",
        json.loads(metrics_path.read_text(encoding="utf-8")),
    )
    assert payload["protocol"] == "teed_jetson_nano_runtime_v1"
    assert payload["source_kind"] == "image"
    assert payload["processed_frames"] == 1
    assert payload["resolution"] == [64, 64]
    assert payload["contrast_clip_limit"] == 2.0


def test_jetson_cli_applies_fire360_osd_profile(tmp_path: Path) -> None:
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("cv2 is not installed")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    output_path = tmp_path / "ifsi_overlay.png"
    metrics_path = tmp_path / "ifsi_metrics.json"
    command = [
        sys.executable,
        str(JETSON_DIR / "run_teed.py"),
        "--source",
        "data/fire360/frames_multiclip/ifsi_video_4/ifsi_video_4_frame_000284.jpg",
        "--checkpoint",
        "models/teed/5_model.pth",
        "--device",
        "cpu",
        "--width",
        "320",
        "--height",
        "240",
        "--warmup",
        "0",
        "--headless",
        "--profiles",
        "data/fire360/video_profiles.json",
        "--video-key",
        "IFSI Video 4 (2).mp4",
        "--output",
        str(output_path),
        "--metrics",
        str(metrics_path),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    payload = cast(
        "dict[str, object]",
        json.loads(metrics_path.read_text(encoding="utf-8")),
    )
    assert payload["ignored_region_count"] == 2
    with Image.open(output_path) as overlay:
        rgb_overlay = overlay.convert("RGB")
        green_pixel_count = sum(
            rgb_overlay.getpixel((x, y)) == (0, 255, 0)
            for y in range(217)
            for x in range(240, 289)
        )
    assert green_pixel_count == 0


def test_jetson_cli_recovers_more_02815_edges_with_contrast(tmp_path: Path) -> None:
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("cv2 is not installed")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    enhanced_path = tmp_path / "enhanced.png"
    baseline_path = tmp_path / "baseline.png"
    command = [
        sys.executable,
        str(JETSON_DIR / "run_teed.py"),
        "--source",
        "data/fire360/frames_multiclip/clip_02815/clip_02815_frame_000068.jpg",
        "--checkpoint",
        "models/teed/5_model.pth",
        "--device",
        "cpu",
        "--width",
        "320",
        "--height",
        "240",
        "--warmup",
        "0",
        "--headless",
        "--metrics",
        "",
    ]

    enhanced = subprocess.run(
        [*command, "--output", str(enhanced_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    baseline = subprocess.run(
        [
            *command,
            "--contrast-clip-limit",
            "0",
            "--output",
            str(baseline_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert enhanced.returncode == 0, enhanced.stderr
    assert baseline.returncode == 0, baseline.stderr
    green_counts: list[int] = []
    for path in (enhanced_path, baseline_path):
        with Image.open(path) as overlay:
            rgb_overlay = overlay.convert("RGB")
            green_counts.append(
                sum(
                    rgb_overlay.getpixel((x, y)) == (0, 255, 0)
                    for y in range(50, 235)
                    for x in range(55, 210)
                ),
            )
    assert green_counts[0] > green_counts[1] * 3
