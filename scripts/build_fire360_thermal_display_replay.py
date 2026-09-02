from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from firesight_vision.mmwave_hud_fusion import (
    build_mock_mmwave_stream,
    write_thermal_fusion_artifacts,
)
from firesight_vision.mmwave_sim import MmWaveScenario, MmWaveSimulationConfig
from firesight_vision.thermal_edges import (
    ThermalHudConfig,
    ThermalMaskRegion,
    build_thermal_hud_overlay,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from firesight_vision.thermal_edges import ThermalHudResult


class BuildThermalReplayNamespace(argparse.Namespace):
    frame_dir: Path = Path("data/fire360/frames_multiclip/ifsi_video_4")
    out_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/fire360-thermal-display-replay",
    )
    root_copy: Path | None = Path("fire360_thermal_display_replay_contact_sheet.png")
    limit: int = 10


CLAIM_TEXT = (
    "Actual Fire360 indoor TIC-display video frames processed as a thermal-display "
    "proxy with a clear-scene simulated mmWave stream. The RGB-encoded display "
    "frames are non-radiometric and do not validate calibrated LWIR temperatures "
    "or real mmWave obstacle accuracy."
)
EDGE_SATURATION_RATIO = 0.15
THERMAL_CONFIG = ThermalHudConfig(
    edge_threshold=130,
    edge_width=3,
    hotspot_threshold=240,
    hotspot_outline_width=3,
    ignored_regions=(
        ThermalMaskRegion(left=0.74, top=0.0, right=1.0, bottom=1.0),
        ThermalMaskRegion(left=0.0, top=0.73, right=0.25, bottom=1.0),
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    namespace = BuildThermalReplayNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--frame-dir", default=namespace.frame_dir, type=Path)
    _ = parser.add_argument(
        "--out", dest="out_dir", default=namespace.out_dir, type=Path
    )
    _ = parser.add_argument("--limit", default=namespace.limit, type=int)
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

    frame_paths = _frame_paths(namespace.frame_dir, namespace.limit)
    frames = _load_frames(frame_paths)
    results = tuple(
        build_thermal_hud_overlay(frame, config=THERMAL_CONFIG) for frame in frames
    )
    messages = build_mock_mmwave_stream(
        tuple(MmWaveScenario(name="clear_scene", obstacles=()) for _ in frames),
        MmWaveSimulationConfig(),
    )
    payload = write_thermal_fusion_artifacts(
        frames,
        messages,
        namespace.out_dir,
        thermal_config=THERMAL_CONFIG,
    )
    source_sheet = namespace.out_dir / "thermal_mmwave_hud_fusion_contact_sheet.png"
    contact_sheet = (
        namespace.out_dir / "fire360_thermal_display_replay_contact_sheet.png"
    )
    _ = shutil.copyfile(source_sheet, contact_sheet)
    if root_copy is not None:
        _ = shutil.copyfile(contact_sheet, root_copy)

    summary = {
        "claim": CLAIM_TEXT,
        "dataset": "Fire360 indoor_videos / IFSI Video 4 (2).mp4",
        "input_modality": "RGB-encoded TIC display video (thermal-display proxy)",
        "input_is_radiometric_lwir": False,
        "mmwave_input": "simulated clear-scene ROS2-style mock stream",
        "source_frames": [path.name for path in frame_paths],
        "thermal_config": {
            "edge_threshold": THERMAL_CONFIG.edge_threshold,
            "edge_width": THERMAL_CONFIG.edge_width,
            "hotspot_threshold": THERMAL_CONFIG.hotspot_threshold,
            "hotspot_outline_width": THERMAL_CONFIG.hotspot_outline_width,
            "ignored_regions": [
                {
                    "left": region.left,
                    "top": region.top,
                    "right": region.right,
                    "bottom": region.bottom,
                }
                for region in THERMAL_CONFIG.ignored_regions
            ],
        },
        "metrics": _metrics(results),
        "fused_overlay_topic": payload["fused_overlay_topic"],
        "contact_sheet": contact_sheet.as_posix(),
    }
    _ = (namespace.out_dir / "fire360_thermal_display_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _ = sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _frame_paths(frame_dir: Path, limit: int) -> tuple[Path, ...]:
    if limit < 1:
        message = "limit must be at least 1"
        raise ValueError(message)
    paths = tuple(sorted(frame_dir.glob("*.jpg"))[:limit])
    if len(paths) == 0:
        message = f"no JPEG frames found in {frame_dir}"
        raise FileNotFoundError(message)
    return paths


def _load_frames(paths: tuple[Path, ...]) -> tuple[Image.Image, ...]:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    return tuple(frames)


def _metrics(results: tuple[ThermalHudResult, ...]) -> dict[str, float | int]:
    edge_ratios = tuple(result.edge_ratio for result in results)
    hotspot_ratios = tuple(result.hotspot_ratio for result in results)
    ratio_deltas = tuple(
        abs(current - previous) for previous, current in pairwise(edge_ratios)
    )
    edge_ious = tuple(
        _mask_iou(previous.edge_mask, current.edge_mask)
        for previous, current in pairwise(results)
    )
    return {
        "frames": len(results),
        "edge_ratio_mean": round(sum(edge_ratios) / len(edge_ratios), 4),
        "edge_ratio_min": round(min(edge_ratios), 4),
        "edge_ratio_max": round(max(edge_ratios), 4),
        "edge_ratio_saturation_limit": EDGE_SATURATION_RATIO,
        "edge_ratio_saturation_rate": round(
            sum(ratio >= EDGE_SATURATION_RATIO for ratio in edge_ratios)
            / len(edge_ratios),
            4,
        ),
        "hotspot_ratio_mean": round(sum(hotspot_ratios) / len(hotspot_ratios), 4),
        "hotspot_active_frame_rate": round(
            sum(result.hotspot_pixels > 0 for result in results) / len(results),
            4,
        ),
        "edge_ratio_abs_delta_mean": round(_mean(ratio_deltas), 4),
        "edge_ratio_abs_delta_p95": round(_percentile(ratio_deltas, 0.95), 4),
        "consecutive_edge_mask_iou_mean": round(_mean(edge_ious), 4),
        "consecutive_edge_mask_iou_min": round(min(edge_ious, default=1.0), 4),
    }


def _mask_iou(first: Image.Image, second: Image.Image) -> float:
    first_values = first.convert("L").tobytes()
    second_values = second.convert("L").tobytes()
    intersection = sum(
        first_value > 0 and second_value > 0
        for first_value, second_value in zip(first_values, second_values, strict=True)
    )
    union = sum(
        first_value > 0 or second_value > 0
        for first_value, second_value in zip(first_values, second_values, strict=True)
    )
    return 1.0 if union == 0 else intersection / union


def _mean(values: tuple[float, ...]) -> float:
    return 0.0 if len(values) == 0 else sum(values) / len(values)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if len(values) == 0:
        return 0.0
    ordered = tuple(sorted(values))
    return ordered[math.ceil(percentile * len(ordered)) - 1]


if __name__ == "__main__":
    raise SystemExit(main())
