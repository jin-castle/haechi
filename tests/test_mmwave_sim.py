from pathlib import Path

import pytest
from PIL import Image

from firesight_vision.mmwave_sim import (
    DistanceBand,
    MmWaveScenario,
    MmWaveSimulationConfig,
    ObstacleSector,
    SimulatedObstacle,
    simulate_scenario,
    write_simulation_artifacts,
)


def test_mmwave_sim_detects_center_close_obstacle() -> None:
    config = MmWaveSimulationConfig()
    result = simulate_scenario(
        MmWaveScenario(
            name="center_close_obstacle",
            obstacles=(
                SimulatedObstacle(
                    label="box",
                    range_m=0.42,
                    angle_deg=0.0,
                    intensity=0.95,
                ),
            ),
        ),
        config,
    )

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.sector is ObstacleSector.CENTER
    assert detection.distance_band is DistanceBand.NEAR
    assert detection.range_m == pytest.approx(0.42, abs=0.03)
    assert abs(detection.angle_deg) < 1.0
    assert detection.confidence > 0.90


def test_mmwave_sim_clear_path_has_no_detection() -> None:
    config = MmWaveSimulationConfig()
    result = simulate_scenario(
        MmWaveScenario(name="clear_path", obstacles=()),
        config,
    )

    assert result.detections == ()


def test_mmwave_sim_writes_heatmap_artifacts(tmp_path: Path) -> None:
    config = MmWaveSimulationConfig()
    out_dir = tmp_path / "mmwave-sim"

    payload = write_simulation_artifacts(
        (
            MmWaveScenario(name="clear_path", obstacles=()),
            MmWaveScenario(
                name="right_far_obstacle",
                obstacles=(
                    SimulatedObstacle(
                        label="table",
                        range_m=1.24,
                        angle_deg=32.0,
                        intensity=0.85,
                    ),
                ),
            ),
        ),
        config,
        out_dir,
    )

    summary_path = out_dir / "mmwave_front_obstacle_summary.json"
    image_path = out_dir / "mmwave_front_obstacle_heatmaps.png"
    summary_text = summary_path.read_text("utf-8")
    with Image.open(image_path) as image:
        assert image.width > 0
        assert image.height > 0
    assert payload["protocol"] == "mmwave_front_obstacle_range_angle_sim_v1"
    assert '"scenario": "clear_path"' in summary_text
    assert payload["scenarios"][0]["detections"] == []
    assert payload["scenarios"][1]["detections"][0]["sector"] == "right"
    assert payload["scenarios"][1]["detections"][0]["distance_band"] == "far"
