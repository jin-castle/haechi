from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from PIL import Image, ImageDraw, ImageOps

from firesight_vision.mmwave_hud_fusion import (
    MmWaveMockMessage,
    build_mock_mmwave_stream,
    render_thermal_hud_mmwave_fusion,
    write_thermal_fusion_artifacts,
)
from firesight_vision.mmwave_sim import MmWaveScenario, MmWaveSimulationConfig
from firesight_vision.thermal_edges import (
    ThermalHudConfig,
    ThermalHudResult,
    ThermalMaskRegion,
    build_thermal_hud_overlay,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class BuildMultisequenceNamespace(argparse.Namespace):
    out_dir: Path = Path(
        ".omo/evidence/firesight-smoke-vision/fire360-thermal-display-multisequence",
    )
    root_copy: Path | None = Path(
        "fire360_thermal_display_multisequence_contact_sheet.png",
    )
    limit: int = 10
    sequences: list[str] | None = None


DEFAULT_SEQUENCES = (
    ("ifsi_video_4", Path("data/fire360/frames_multiclip/ifsi_video_4")),
    ("ifsi_video_8", Path("data/fire360/frames_multiclip/ifsi_video_8")),
)
CLAIM_TEXT = (
    "Two real Fire360 indoor TIC-display video sequences processed as thermal-display "
    "proxies with clear-scene simulated mmWave messages. The source frames are "
    "RGB-encoded displays, not radiometric LWIR or real mmWave captures."
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


@dataclass(frozen=True, slots=True)
class ReplaySample:
    sequence_name: str
    frame_name: str
    source: Image.Image
    thermal_result: ThermalHudResult
    fused: Image.Image


class _TextDrawer(Protocol):
    def text(
        self,
        xy: tuple[int, int],
        text: str,
        *,
        fill: tuple[int, int, int],
    ) -> None: ...


def main(argv: Sequence[str] | None = None) -> int:
    namespace = BuildMultisequenceNamespace()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--sequence",
        dest="sequences",
        action="append",
        help="Repeat name=frame-directory to replace the default Fire360 sequences.",
    )
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

    sequences = _parse_sequences(namespace.sequences)
    sequence_summaries: dict[str, object] = {}
    samples: list[ReplaySample] = []
    for sequence_name, frame_dir in sequences:
        frame_paths = _frame_paths(frame_dir, namespace.limit)
        frames = _load_frames(frame_paths)
        results = tuple(
            build_thermal_hud_overlay(frame, config=THERMAL_CONFIG) for frame in frames
        )
        messages = build_mock_mmwave_stream(
            tuple(MmWaveScenario(name="clear_scene", obstacles=()) for _ in frames),
            MmWaveSimulationConfig(),
        )
        sequence_out_dir = namespace.out_dir / sequence_name
        payload = write_thermal_fusion_artifacts(
            frames,
            messages,
            sequence_out_dir,
            thermal_config=THERMAL_CONFIG,
        )
        sequence_summaries[sequence_name] = {
            "frame_dir": frame_dir.as_posix(),
            "source_frames": [path.name for path in frame_paths],
            "metrics": _metrics(results),
            "fused_overlay_topic": payload["fused_overlay_topic"],
        }
        samples.extend(_samples(sequence_name, frame_paths, frames, results, messages))

    contact_sheet = _render_contact_sheet(tuple(samples))
    contact_path = (
        namespace.out_dir / "fire360_thermal_display_multisequence_contact_sheet.png"
    )
    namespace.out_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(contact_path)
    if root_copy is not None:
        _ = shutil.copyfile(contact_path, root_copy)

    summary = {
        "claim": CLAIM_TEXT,
        "input_modality": "RGB-encoded TIC display video (thermal-display proxy)",
        "input_is_radiometric_lwir": False,
        "mmwave_input": "simulated clear-scene ROS2-style mock stream",
        "thermal_config": _thermal_config_payload(),
        "sequence_count": len(sequences),
        "frame_count": sum(
            cast("dict[str, int]", value["metrics"])["frames"]
            for value in sequence_summaries.values()
            if isinstance(value, dict)
        ),
        "sequences": sequence_summaries,
        "contact_sheet": contact_path.as_posix(),
    }
    _ = (
        namespace.out_dir / "fire360_thermal_display_multisequence_summary.json"
    ).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _ = sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_sequences(values: list[str] | None) -> tuple[tuple[str, Path], ...]:
    if values is None:
        return DEFAULT_SEQUENCES
    sequences: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if separator == "" or name == "" or path == "":
            message = "sequence must use name=frame-directory"
            raise ValueError(message)
        sequences.append((name, Path(path)))
    return tuple(sequences)


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


def _samples(
    sequence_name: str,
    paths: tuple[Path, ...],
    frames: tuple[Image.Image, ...],
    results: tuple[ThermalHudResult, ...],
    messages: tuple[MmWaveMockMessage, ...],
) -> tuple[ReplaySample, ...]:
    selected_indices = (0,) if len(paths) == 1 else (0, len(paths) - 1)
    samples: list[ReplaySample] = []
    for index in selected_indices:
        fusion_frame = render_thermal_hud_mmwave_fusion(
            frames[index],
            messages[index],
            thermal_config=THERMAL_CONFIG,
        )
        samples.append(
            ReplaySample(
                sequence_name=sequence_name,
                frame_name=paths[index].name,
                source=frames[index],
                thermal_result=results[index],
                fused=fusion_frame.image,
            )
        )
    return tuple(samples)


def _render_contact_sheet(samples: tuple[ReplaySample, ...]) -> Image.Image:
    tile_width = 300
    tile_height = 169
    header_height = 28
    caption_height = 34
    pad = 12
    sheet = Image.new(
        "RGB",
        (
            (3 * tile_width) + (4 * pad),
            header_height + (len(samples) * (tile_height + caption_height + pad)) + pad,
        ),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(sheet)
    text_drawer = cast("_TextDrawer", draw)
    for index, label in enumerate(
        ("RAW TIC DISPLAY", "EDGE + HOTSPOT", "FUSED HUD + MMWAVE"),
    ):
        text_drawer.text(
            (pad + (index * (tile_width + pad)), 7),
            label,
            fill=(230, 230, 230),
        )
    for row, sample in enumerate(samples):
        y = header_height + (row * (tile_height + caption_height + pad))
        for column, image in enumerate(
            (sample.source, sample.thermal_result.overlay, sample.fused),
        ):
            x = pad + (column * (tile_width + pad))
            _paste_contained(sheet, image, (x, y, x + tile_width, y + tile_height))
        text_drawer.text(
            (pad, y + tile_height + 5),
            (
                f"{sample.sequence_name} {sample.frame_name} | "
                f"edge {sample.thermal_result.edge_ratio:.2%} "
                f"hot {sample.thermal_result.hotspot_ratio:.2%}"
            ),
            fill=(230, 230, 230),
        )
    return sheet


def _paste_contained(
    target: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    contained = ImageOps.contain(
        image.convert("RGB"),
        (right - left, bottom - top),
        Image.Resampling.LANCZOS,
    )
    x = left + ((right - left - contained.width) // 2)
    y = top + ((bottom - top - contained.height) // 2)
    target.paste(contained, (x, y))


def _thermal_config_payload() -> dict[str, object]:
    return {
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
    }


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
