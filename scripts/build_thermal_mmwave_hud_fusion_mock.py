from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFilter

from firesight_vision.mmwave_hud_fusion import (
    build_mock_mmwave_stream,
    write_thermal_fusion_artifacts,
)
from firesight_vision.mmwave_sim import (
    MmWaveScenario,
    MmWaveSimulationConfig,
    SimulatedObstacle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class BuildThermalFusionMockNamespace(argparse.Namespace):
    input_path: Path = Path("tests/fixtures/images/sample_room.jpg")
    out_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/thermal-mmwave-hud-fusion-mock",
    )
    root_copy: Path | None = Path("thermal_mmwave_hud_fusion_contact_sheet.png")


def main(argv: Sequence[str] | None = None) -> int:
    namespace = BuildThermalFusionMockNamespace()
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
        _demo_thermal_frame(base_image, index) for index in range(len(messages))
    )
    payload = write_thermal_fusion_artifacts(frames, messages, namespace.out_dir)
    image_path = namespace.out_dir / "thermal_mmwave_hud_fusion_contact_sheet.png"
    if root_copy is not None:
        _ = shutil.copyfile(image_path, root_copy)

    _ = sys.stdout.write(
        json.dumps(
            {
                "frames": len(payload["frames"]),
                "fused_overlay_topic": payload["fused_overlay_topic"],
                "image": image_path.as_posix(),
                "root_copy": None if root_copy is None else root_copy.as_posix(),
                "stream": (
                    namespace.out_dir / "mmwave_front_obstacles_mock_stream.jsonl"
                ).as_posix(),
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
            name="near_center_debris",
            obstacles=(
                SimulatedObstacle(
                    label="low_debris",
                    range_m=0.46,
                    angle_deg=0.5,
                    intensity=0.96,
                ),
            ),
        ),
        MmWaveScenario(
            name="left_doorframe",
            obstacles=(
                SimulatedObstacle(
                    label="doorframe_return",
                    range_m=0.78,
                    angle_deg=-31.0,
                    intensity=0.84,
                ),
            ),
        ),
        MmWaveScenario(
            name="narrow_hot_passage",
            obstacles=(
                SimulatedObstacle(
                    label="left_wall_return",
                    range_m=0.92,
                    angle_deg=-24.0,
                    intensity=0.72,
                ),
                SimulatedObstacle(
                    label="right_wall_return",
                    range_m=0.91,
                    angle_deg=24.0,
                    intensity=0.72,
                ),
                SimulatedObstacle(
                    label="front_debris",
                    range_m=1.22,
                    angle_deg=2.0,
                    intensity=0.62,
                ),
            ),
        ),
    )


def _demo_thermal_frame(base_image: Image.Image, index: int) -> Image.Image:
    resized = base_image.convert("L").resize((640, 384), Image.Resampling.LANCZOS)
    low_contrast = Image.blend(resized, Image.new("L", resized.size, 92), 0.58)
    frame = low_contrast.filter(ImageFilter.GaussianBlur(radius=0.8))
    draw = ImageDraw.Draw(frame)

    x_offset = 10 * (index % 3)
    draw.rectangle((66, 60, 588, 318), outline=122, width=3)
    draw.rectangle((236, 122, 310, 318), outline=132, width=4)
    draw.line((80, 238, 230, 178, 384, 232, 560, 156), fill=124, width=4)
    draw.ellipse((430 + x_offset, 132, 488 + x_offset, 216), fill=228)
    draw.polygon(
        (
            (456 + x_offset, 108),
            (492 + x_offset, 176),
            (464 + x_offset, 236),
            (438 + x_offset, 174),
        ),
        fill=248,
    )
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
