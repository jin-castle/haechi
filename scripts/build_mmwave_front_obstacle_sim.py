from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from firesight_vision.mmwave_sim import (
    MmWaveScenario,
    MmWaveSimulationConfig,
    SimulatedObstacle,
    write_simulation_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class BuildMmWaveSimNamespace(argparse.Namespace):
    out_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/mmwave-front-obstacle-sim",
    )
    root_copy: Path | None = Path("mmwave_front_obstacle_heatmaps.png")


def main(argv: Sequence[str] | None = None) -> int:
    namespace = BuildMmWaveSimNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--out",
        dest="out_dir",
        default=namespace.out_dir,
        type=Path,
    )
    _ = parser.add_argument(
        "--root-copy",
        dest="root_copy",
        default=namespace.root_copy,
        type=Path,
        help="Optional extra PNG copy for easy viewing; pass an empty string to skip.",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    root_copy = namespace.root_copy
    if root_copy is not None and root_copy.as_posix() == ".":
        root_copy = None

    config = MmWaveSimulationConfig()
    payload = write_simulation_artifacts(
        _default_scenarios(),
        config,
        namespace.out_dir,
    )
    image_path = namespace.out_dir / "mmwave_front_obstacle_heatmaps.png"
    if root_copy is not None:
        _ = shutil.copyfile(image_path, root_copy)
    _ = sys.stdout.write(
        json.dumps(
            {
                "image": image_path.as_posix(),
                "root_copy": None if root_copy is None else root_copy.as_posix(),
                "scenarios": len(payload["scenarios"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return 0


def _default_scenarios() -> tuple[MmWaveScenario, ...]:
    return (
        MmWaveScenario(name="clear_path", obstacles=()),
        MmWaveScenario(
            name="center_close_obstacle",
            obstacles=(
                SimulatedObstacle(
                    label="low_box",
                    range_m=0.42,
                    angle_deg=1.0,
                    intensity=0.95,
                ),
            ),
        ),
        MmWaveScenario(
            name="left_mid_doorframe",
            obstacles=(
                SimulatedObstacle(
                    label="doorframe",
                    range_m=0.82,
                    angle_deg=-34.0,
                    intensity=0.82,
                ),
            ),
        ),
        MmWaveScenario(
            name="right_far_table_edge",
            obstacles=(
                SimulatedObstacle(
                    label="table_edge",
                    range_m=1.26,
                    angle_deg=31.0,
                    intensity=0.78,
                ),
            ),
        ),
        MmWaveScenario(
            name="multi_obstacle_narrow_passage",
            obstacles=(
                SimulatedObstacle(
                    label="left_wall_return",
                    range_m=0.94,
                    angle_deg=-25.0,
                    intensity=0.72,
                ),
                SimulatedObstacle(
                    label="right_wall_return",
                    range_m=0.88,
                    angle_deg=27.0,
                    intensity=0.70,
                ),
                SimulatedObstacle(
                    label="front_debris",
                    range_m=1.18,
                    angle_deg=2.0,
                    intensity=0.58,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
