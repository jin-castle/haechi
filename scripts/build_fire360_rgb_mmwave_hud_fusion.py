from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from firesight_vision.mmwave_hud_fusion import (
    FusionRenderConfig,
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


class BuildFire360FusionNamespace(argparse.Namespace):
    image_root: Path = Path("data/fire360_photo_inclusive/all_images")
    out_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/fire360-rgb-mmwave-hud-fusion",
    )
    root_copy: Path | None = Path("fire360_rgb_mmwave_hud_fusion_contact_sheet.png")


SOURCE_IMAGES = (
    "photo__test__000039_jpg.rf.29e227f738bbc4e32317c0d77b53bafd.jpg",
    "photo__valid__1014_jpg.rf.ee0255915e03654e8e164e38bbb8d49b.jpg",
    "video__clip_02814_frame_000028.jpg",
    "video__clip_02824_frame_000139.jpg",
)

CLAIM_TEXT = (
    "Actual Fire360 indoor RGB photo/video frames fused with simulated mmWave "
    "front-obstacle messages. This is an RGB HUD baseline, not LWIR validation "
    "or hardware sensor fusion."
)


def main(argv: Sequence[str] | None = None) -> int:
    namespace = BuildFire360FusionNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--image-root",
        default=namespace.image_root,
        type=Path,
    )
    _ = parser.add_argument(
        "--out", dest="out_dir", default=namespace.out_dir, type=Path
    )
    _ = parser.add_argument(
        "--root-copy",
        default=namespace.root_copy,
        type=Path,
        help="Optional extra PNG copy for easy viewing; pass an empty string to skip.",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    root_copy = namespace.root_copy
    if root_copy is not None and root_copy.as_posix() == ".":
        root_copy = None

    frames = _load_frames(namespace.image_root)
    messages = build_mock_mmwave_stream(_scenarios(), MmWaveSimulationConfig())
    render_configs = _render_configs()
    payload = write_fusion_artifacts(
        frames,
        messages,
        namespace.out_dir,
        render_configs=render_configs,
    )
    image_path = namespace.out_dir / "mmwave_hud_fusion_contact_sheet.png"
    named_image_path = (
        namespace.out_dir / "fire360_rgb_mmwave_hud_fusion_contact_sheet.png"
    )
    _ = shutil.copyfile(image_path, named_image_path)
    if root_copy is not None:
        _ = shutil.copyfile(named_image_path, root_copy)

    summary = {
        "claim": CLAIM_TEXT,
        "dataset": "fire360_photo_inclusive",
        "input_modality": "RGB",
        "mmwave_input": "simulated ROS2-style mock stream",
        "edge_thresholds": [config.threshold for config in render_configs],
        "edge_width": render_configs[0].edge_width,
        "source_images": list(SOURCE_IMAGES),
        "fused_overlay_topic": payload["topic"],
        "contact_sheet": named_image_path.as_posix(),
    }
    _ = (namespace.out_dir / "fire360_rgb_mmwave_hud_fusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _ = sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _load_frames(image_root: Path) -> tuple[Image.Image, ...]:
    frames: list[Image.Image] = []
    for name in SOURCE_IMAGES:
        input_path = image_root / name
        with Image.open(input_path) as image:
            frames.append(image.convert("RGB"))
    return tuple(frames)


def _scenarios() -> tuple[MmWaveScenario, ...]:
    return (
        MmWaveScenario(name="indoor_smoke_photo", obstacles=()),
        MmWaveScenario(
            name="indoor_fire_smoke_photo",
            obstacles=(
                SimulatedObstacle(
                    label="near_debris",
                    range_m=0.48,
                    angle_deg=1.0,
                    intensity=0.95,
                ),
            ),
        ),
        MmWaveScenario(
            name="dense_fire_smoke_video",
            obstacles=(
                SimulatedObstacle(
                    label="left_obstacle",
                    range_m=0.86,
                    angle_deg=-24.0,
                    intensity=0.78,
                ),
            ),
        ),
        MmWaveScenario(
            name="dense_smoke_video",
            obstacles=(
                SimulatedObstacle(
                    label="left_wall",
                    range_m=0.98,
                    angle_deg=-22.0,
                    intensity=0.73,
                ),
                SimulatedObstacle(
                    label="right_wall",
                    range_m=0.93,
                    angle_deg=23.0,
                    intensity=0.71,
                ),
            ),
        ),
    )


def _render_configs() -> tuple[FusionRenderConfig, ...]:
    return (
        FusionRenderConfig(threshold=220, edge_width=3),
        FusionRenderConfig(threshold=220, edge_width=3),
        FusionRenderConfig(threshold=128, edge_width=3),
        FusionRenderConfig(threshold=160, edge_width=3),
    )


if __name__ == "__main__":
    raise SystemExit(main())
