import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw

from firesight_vision.mmwave_hud_fusion import (
    ROS2_FRONT_OBSTACLE_TOPIC,
    FusionError,
    FusionRenderConfig,
    MmWaveStreamConfig,
    build_mock_mmwave_stream,
    render_hud_mmwave_fusion,
    render_thermal_hud_mmwave_fusion,
    write_fusion_artifacts,
    write_thermal_fusion_artifacts,
)
from firesight_vision.mmwave_sim import (
    MmWaveScenario,
    MmWaveSimulationConfig,
    SimulatedObstacle,
)


def test_mock_stream_payload_uses_ros2_topic_contract() -> None:
    messages = build_mock_mmwave_stream(
        (
            MmWaveScenario(name="clear_path", obstacles=()),
            MmWaveScenario(
                name="center_near",
                obstacles=(
                    SimulatedObstacle(
                        label="box",
                        range_m=0.42,
                        angle_deg=0.0,
                        intensity=0.95,
                    ),
                ),
            ),
        ),
        MmWaveSimulationConfig(),
        stream_config=MmWaveStreamConfig(start_stamp_ns=1_000, period_ns=50),
    )

    assert [message.seq for message in messages] == [0, 1]
    assert [message.stamp_ns for message in messages] == [1_000, 1_050]
    assert {message.topic for message in messages} == {ROS2_FRONT_OBSTACLE_TOPIC}
    assert messages[0].result.detections == ()
    assert messages[1].result.detections[0].sector.value == "center"


def test_hud_mmwave_fusion_contains_fire_and_near_warning_pixels() -> None:
    source = _fire_fixture()
    messages = build_mock_mmwave_stream(
        (
            MmWaveScenario(
                name="center_near",
                obstacles=(
                    SimulatedObstacle(
                        label="box",
                        range_m=0.42,
                        angle_deg=0.0,
                        intensity=0.95,
                    ),
                ),
            ),
        ),
        MmWaveSimulationConfig(),
    )

    frame = render_hud_mmwave_fusion(source, messages[0])

    assert frame.image.mode == "RGB"
    assert frame.source_image.tobytes() == source.tobytes()
    assert _red_pixel_count(frame.image) > 100
    assert _near_warning_pixel_count(frame.image) > 20
    assert frame.hud_result.edge_pixels > 0


def test_thermal_hud_mmwave_fusion_contains_hotspot_outline_and_near_panel() -> None:
    source = _thermal_fixture()
    messages = build_mock_mmwave_stream(
        (
            MmWaveScenario(
                name="center_near",
                obstacles=(
                    SimulatedObstacle(
                        label="box",
                        range_m=0.42,
                        angle_deg=0.0,
                        intensity=0.95,
                    ),
                ),
            ),
        ),
        MmWaveSimulationConfig(),
    )

    frame = render_thermal_hud_mmwave_fusion(source, messages[0])

    assert frame.image.mode == "RGB"
    assert _red_pixel_count(frame.image) > 100
    assert _near_warning_pixel_count(frame.image) > 20
    assert frame.thermal_result.edge_pixels > 0
    assert frame.thermal_result.hotspot_pixels > 0


def test_write_fusion_artifacts_writes_png_jsonl_and_summary(tmp_path: Path) -> None:
    source = _fire_fixture()
    messages = build_mock_mmwave_stream(
        (
            MmWaveScenario(name="clear_path", obstacles=()),
            MmWaveScenario(
                name="right_mid",
                obstacles=(
                    SimulatedObstacle(
                        label="chair",
                        range_m=0.86,
                        angle_deg=26.0,
                        intensity=0.82,
                    ),
                ),
            ),
        ),
        MmWaveSimulationConfig(),
    )

    payload = write_fusion_artifacts((source, source), messages, tmp_path)

    stream_path = tmp_path / "mmwave_front_obstacle_mock_stream.jsonl"
    summary_path = tmp_path / "mmwave_hud_fusion_summary.json"
    sheet_path = tmp_path / "mmwave_hud_fusion_contact_sheet.png"
    stream_lines = stream_path.read_text("utf-8").splitlines()
    with Image.open(sheet_path) as sheet:
        assert sheet.width == 948
        assert sheet.height > 450
    assert summary_path.exists()
    assert payload["topic"] == ROS2_FRONT_OBSTACLE_TOPIC
    assert len(stream_lines) == 2
    assert f'"topic": "{ROS2_FRONT_OBSTACLE_TOPIC}"' in stream_lines[1]
    assert '"sector": "right"' in stream_lines[1]
    assert payload["frames"][1]["message"]["closest_range_m"] == 0.86


def test_write_fusion_artifacts_rejects_mismatched_per_frame_configs(
    tmp_path: Path,
) -> None:
    source = _fire_fixture()
    messages = build_mock_mmwave_stream(
        (MmWaveScenario(name="clear_path", obstacles=()),),
        MmWaveSimulationConfig(),
    )

    with pytest.raises(FusionError) as error_info:
        _ = write_fusion_artifacts(
            (source,),
            messages,
            tmp_path,
            render_configs=(
                FusionRenderConfig(threshold=180),
                FusionRenderConfig(threshold=220),
            ),
        )
    assert (
        str(error_info.value)
        == "source_images and render_configs must have the same length"
    )


def test_write_thermal_fusion_artifacts_writes_ros2_topic_summary(
    tmp_path: Path,
) -> None:
    source = _thermal_fixture()
    messages = build_mock_mmwave_stream(
        (
            MmWaveScenario(name="clear_path", obstacles=()),
            MmWaveScenario(
                name="right_mid",
                obstacles=(
                    SimulatedObstacle(
                        label="chair",
                        range_m=0.86,
                        angle_deg=26.0,
                        intensity=0.82,
                    ),
                ),
            ),
        ),
        MmWaveSimulationConfig(),
    )

    payload = write_thermal_fusion_artifacts((source, source), messages, tmp_path)

    assert (tmp_path / "mmwave_front_obstacles_mock_stream.jsonl").exists()
    assert (tmp_path / "thermal_mmwave_hud_fusion_summary.json").exists()
    with Image.open(tmp_path / "thermal_mmwave_hud_fusion_contact_sheet.png") as sheet:
        assert sheet.width == 948
        assert sheet.height > 450
    assert payload["thermal_input_topic"] == "/thermal_camera/image_raw"
    assert payload["thermal_edge_mask_topic"] == "/firesight/thermal/edge_mask"
    assert payload["thermal_edge_overlay_topic"] == "/firesight/thermal/edge_overlay"
    assert payload["mmwave_obstacle_topic"] == ROS2_FRONT_OBSTACLE_TOPIC
    assert payload["fused_overlay_topic"] == "/firesight/hud/fused_overlay"
    assert payload["frames"][1]["hotspot_pixels"] > 0


def test_build_mmwave_hud_fusion_mock_script_writes_easy_view_png(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "fusion"
    root_copy = tmp_path / "root_copy.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_mmwave_hud_fusion_mock.py",
            "--out",
            str(out_dir),
            "--root-copy",
            str(root_copy),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert root_copy.exists()
    assert (out_dir / "mmwave_front_obstacle_mock_stream.jsonl").exists()
    assert (out_dir / "mmwave_hud_fusion_summary.json").exists()
    with Image.open(root_copy) as image:
        assert image.width > 0
        assert image.height > 0


def test_build_thermal_mmwave_hud_fusion_mock_script_writes_easy_view_png(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "thermal_fusion"
    root_copy = tmp_path / "root_copy.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_thermal_mmwave_hud_fusion_mock.py",
            "--out",
            str(out_dir),
            "--root-copy",
            str(root_copy),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert root_copy.exists()
    assert (out_dir / "mmwave_front_obstacles_mock_stream.jsonl").exists()
    assert (out_dir / "thermal_mmwave_hud_fusion_summary.json").exists()
    with Image.open(root_copy) as image:
        assert image.width > 0
        assert image.height > 0


def test_build_fire360_thermal_display_replay_script_writes_proxy_summary(
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    _thermal_fixture().save(frame_dir / "frame_001.jpg")
    _thermal_fixture().save(frame_dir / "frame_002.jpg")
    out_dir = tmp_path / "thermal_replay"
    root_copy = tmp_path / "root_copy.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_fire360_thermal_display_replay.py",
            "--frame-dir",
            str(frame_dir),
            "--out",
            str(out_dir),
            "--root-copy",
            str(root_copy),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    summary = cast("dict[str, object]", json.loads(result.stdout))
    metrics = cast("dict[str, object]", summary["metrics"])
    assert result.returncode == 0
    assert root_copy.exists()
    assert summary["input_is_radiometric_lwir"] is False
    assert metrics["frames"] == 2
    assert summary["mmwave_input"] == "simulated clear-scene ROS2-style mock stream"


def test_build_fire360_thermal_display_multisequence_script_writes_summary(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    _thermal_fixture().save(first_dir / "frame_001.jpg")
    _thermal_fixture().save(second_dir / "frame_001.jpg")
    out_dir = tmp_path / "thermal_multisequence"
    root_copy = tmp_path / "root_copy.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_fire360_thermal_display_multisequence_eval.py",
            "--sequence",
            f"first={first_dir}",
            "--sequence",
            f"second={second_dir}",
            "--out",
            str(out_dir),
            "--root-copy",
            str(root_copy),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    summary = cast("dict[str, object]", json.loads(result.stdout))
    sequences = cast("dict[str, object]", summary["sequences"])
    assert result.returncode == 0
    assert root_copy.exists()
    assert summary["frame_count"] == 2
    assert set(sequences) == {"first", "second"}


def _fire_fixture() -> Image.Image:
    image = Image.new("RGB", (220, 140), (96, 98, 96))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 20, 196, 120), outline=(132, 134, 132), width=3)
    draw.rectangle((84, 58, 126, 120), outline=(140, 142, 140), width=3)
    draw.ellipse((150, 42, 190, 102), fill=(216, 122, 42))
    draw.polygon(((168, 36), (186, 82), (160, 106), (154, 68)), fill=(255, 184, 64))
    return image


def _thermal_fixture() -> Image.Image:
    image = Image.new("L", (220, 140), 88)
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 20, 196, 120), outline=118, width=3)
    draw.rectangle((84, 58, 126, 120), outline=126, width=3)
    draw.ellipse((150, 42, 190, 102), fill=226)
    draw.polygon(((168, 36), (186, 82), (160, 106), (154, 68)), fill=248)
    return image


def _red_pixel_count(image: Image.Image) -> int:
    values = image.convert("RGB").tobytes()
    return sum(
        1
        for red, green, blue in _rgb_triplets(values)
        if red > 180 and green < 90 and blue < 90
    )


def _near_warning_pixel_count(image: Image.Image) -> int:
    width, height = image.size
    panel_crop = image.crop((0, round(height * 0.68), width, height)).convert("RGB")
    values = panel_crop.tobytes()
    return sum(
        1
        for red, green, blue in _rgb_triplets(values)
        if red > 210 and green < 90 and blue < 90
    )


def _rgb_triplets(values: bytes) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (values[index], values[index + 1], values[index + 2])
        for index in range(0, len(values), 3)
    )
