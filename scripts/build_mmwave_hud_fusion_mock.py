from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from firesight_vision.mmwave_hud_fusion import (
    build_mock_mmwave_stream,
    write_fusion_artifacts,
)
from firesight_vision.mmwave_sim import (
    MmWaveScenario,
    MmWaveSimulationConfig,
    SimulatedObstacle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class BuildFusionMockNamespace(argparse.Namespace):
    input_path: Path = Path("tests/fixtures/images/sample_room.jpg")
    out_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/mmwave-hud-fusion-mock",
    )
    root_copy: Path | None = Path("mmwave_hud_fusion_contact_sheet.png")


def main(argv: Sequence[str] | None = None) -> int:
    namespace = BuildFusionMockNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--input",
        dest="input_path",
        default=namespace.input_path,
        type=Path,
    )
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

    with Image.open(namespace.input_path) as source:
        base_image = source.convert("RGB")

    scenarios = _default_scenarios()
    messages = build_mock_mmwave_stream(scenarios, MmWaveSimulationConfig())
    frames = tuple(
        _demo_source_frame(base_image, index) for index in range(len(messages))
    )
    payload = write_fusion_artifacts(frames, messages, namespace.out_dir)
    image_path = namespace.out_dir / "mmwave_hud_fusion_contact_sheet.png"
    if root_copy is not None:
        _ = shutil.copyfile(image_path, root_copy)

    _ = sys.stdout.write(
        json.dumps(
            {
                "frames": len(payload["frames"]),
                "image": image_path.as_posix(),
                "root_copy": None if root_copy is None else root_copy.as_posix(),
                "stream": (
                    namespace.out_dir / "mmwave_front_obstacle_mock_stream.jsonl"
                ).as_posix(),
                "topic": payload["topic"],
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


def _demo_source_frame(base_image: Image.Image, index: int) -> Image.Image:
    resized = base_image.resize((640, 384), Image.Resampling.LANCZOS)
    smoke = Image.new("RGB", resized.size, (118, 122, 120))
    frame = Image.blend(resized, smoke, 0.38).filter(
        ImageFilter.GaussianBlur(radius=0.7)
    )
    frame = ImageEnhance.Contrast(frame).enhance(0.72)
    draw = ImageDraw.Draw(frame)
    x_offset = 8 * (index % 3)
    draw.ellipse((424 + x_offset, 150, 486 + x_offset, 238), fill=(210, 102, 34))
    draw.polygon(
        (
            (448 + x_offset, 140),
            (480 + x_offset, 198),
            (452 + x_offset, 236),
            (430 + x_offset, 190),
        ),
        fill=(255, 180, 58),
    )
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
