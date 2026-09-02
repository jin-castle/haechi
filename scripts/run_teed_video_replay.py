from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import cv2
import numpy as np
from PIL import Image

from firesight_vision.hud_edges import FIRE360_OSD_IGNORED_REGIONS
from firesight_vision.teed import TeedError, TeedPredictor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from firesight_vision.hud_edges import HudIgnoreRegion


class TeedVideoArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    output_dir: Path = Path()
    checkpoint_path: Path = Path()
    inference_fps: float = 5.0
    output_width: int = 1280
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    ignore_fire360_osd: bool = False


class TeedVideoFramePayload(TypedDict):
    index: int
    source_frame: int
    latency_ms: float
    edge_ratio: float


class TeedVideoSummaryPayload(TypedDict):
    protocol: str
    claim: str
    input_file: str
    output_file: str
    checkpoint_path: str
    device: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_fps: float
    output_fps: float
    inference_fps_requested: float
    sampling_stride: int
    source_frames: int
    processed_frames: int
    source_duration_seconds: float
    output_duration_seconds: float
    ignored_regions: list[dict[str, float]]
    latency_ms: dict[str, float]
    edge_ratio: dict[str, float]
    frame_outputs: list[TeedVideoFramePayload]


@dataclass(frozen=True, slots=True)
class TeedVideoReplayConfig:
    input_path: Path
    output_dir: Path
    checkpoint_path: Path
    inference_fps: float = 5.0
    output_width: int = 1280
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    ignored_regions: tuple[HudIgnoreRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class TeedVideoReplayError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


PROTOCOL_NAME: Final = "teed_biped_video_replay_v1"
CLAIM_TEXT: Final = (
    "Full-duration video preview using sampled pretrained TEED BIPED inference; "
    "the model is not fine-tuned on Fire360 smoke footage."
)
SUPPORTED_VIDEO_SUFFIXES: Final = frozenset({".avi", ".mkv", ".mov", ".mp4", ".mts"})
MIN_OUTPUT_WIDTH: Final = 64


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_teed_video_replay(config)
    except (TeedError, TeedVideoReplayError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> TeedVideoReplayConfig:
    namespace = TeedVideoArgNamespace()
    parser = argparse.ArgumentParser(
        description="Replay full-duration videos through sampled TEED inference.",
    )
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument(
        "--inference-fps",
        default=5.0,
        type=float,
        help="run TEED on approximately this many source frames per second",
    )
    _ = parser.add_argument(
        "--output-width",
        default=1280,
        type=int,
        help="resize output frames to this width before inference",
    )
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=1, type=int)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.add_argument(
        "--ignore-fire360-osd",
        action="store_true",
        help="ignore the fixed left badge and right colorbar regions",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    return TeedVideoReplayConfig(
        input_path=namespace.input_path,
        output_dir=namespace.output_dir,
        checkpoint_path=namespace.checkpoint_path,
        inference_fps=namespace.inference_fps,
        output_width=namespace.output_width,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
        ignored_regions=(
            FIRE360_OSD_IGNORED_REGIONS if namespace.ignore_fire360_osd else ()
        ),
    )


def run_teed_video_replay(
    config: TeedVideoReplayConfig,
) -> list[TeedVideoSummaryPayload]:
    _validate_config(config)
    paths = _collect_video_paths(config.input_path)
    predictor = TeedPredictor(config.checkpoint_path, device=config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _process_video(path, config, predictor)
        for path in paths
    ]


def _process_video(
    input_path: Path,
    config: TeedVideoReplayConfig,
    predictor: TeedPredictor,
) -> TeedVideoSummaryPayload:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise TeedVideoReplayError(message=f"unable to open video: {input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0.0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise TeedVideoReplayError(message=f"invalid video metadata: {input_path}")

    sampling_stride = max(1, round(source_fps / config.inference_fps))
    output_fps = source_fps / sampling_stride
    output_width = min(config.output_width, source_width)
    output_height = round(source_height * output_width / source_width)
    if output_height % 2:
        output_height += 1
    output_suffix = (
        "_teed_long_osd_ignored"
        if len(config.ignored_regions) > 0
        else "_teed_long"
    )
    output_path = config.output_dir / f"{input_path.stem}{output_suffix}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise TeedVideoReplayError(message=f"unable to create video: {output_path}")

    frame_outputs: list[TeedVideoFramePayload] = []
    latencies: list[float] = []
    edge_ratios: list[float] = []
    source_frame_index = 0
    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break
            if source_frame_index % sampling_stride != 0:
                source_frame_index += 1
                continue
            resized_bgr = cv2.resize(
                bgr_frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            rgb_frame = Image.fromarray(
                cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB),
                mode="RGB",
            )
            started_at = cv2.getTickCount()
            result = predictor.process(
                rgb_frame,
                threshold=config.threshold,
                background_scale=config.background_scale,
                edge_width=config.edge_width,
                ignored_regions=config.ignored_regions,
            )
            elapsed_ms = (
                (cv2.getTickCount() - started_at)
                / cv2.getTickFrequency()
                * 1000.0
            )
            output_bgr = cv2.cvtColor(np.asarray(result.overlay), cv2.COLOR_RGB2BGR)
            writer.write(output_bgr)
            latencies.append(elapsed_ms)
            edge_ratios.append(result.edge_ratio)
            frame_outputs.append(
                TeedVideoFramePayload(
                    index=len(frame_outputs),
                    source_frame=source_frame_index,
                    latency_ms=round(elapsed_ms, 3),
                    edge_ratio=round(result.edge_ratio, 6),
                ),
            )
            source_frame_index += 1
    finally:
        capture.release()
        writer.release()

    if len(frame_outputs) == 0:
        raise TeedVideoReplayError(
            message=f"video produced no sampled frames: {input_path}",
        )
    summary = TeedVideoSummaryPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        input_file=input_path.as_posix(),
        output_file=output_path.as_posix(),
        checkpoint_path=config.checkpoint_path.as_posix(),
        device=predictor.device,
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
        source_fps=round(source_fps, 6),
        output_fps=round(output_fps, 6),
        inference_fps_requested=config.inference_fps,
        sampling_stride=sampling_stride,
        source_frames=source_frame_index,
        processed_frames=len(frame_outputs),
        source_duration_seconds=round(source_frame_index / source_fps, 6),
        output_duration_seconds=round(len(frame_outputs) / output_fps, 6),
        ignored_regions=_serialize_ignored_regions(config.ignored_regions),
        latency_ms=_summarize(latencies),
        edge_ratio=_summarize(edge_ratios),
        frame_outputs=frame_outputs,
    )
    _ = output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _collect_video_paths(input_path: Path) -> tuple[Path, ...]:
    if not input_path.exists():
        raise TeedVideoReplayError(message=f"missing video input: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise TeedVideoReplayError(message=f"unsupported video: {input_path}")
        return (input_path,)
    paths = tuple(
        sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        ),
    )
    if len(paths) == 0:
        raise TeedVideoReplayError(message=f"no supported videos found: {input_path}")
    return paths


def _validate_config(config: TeedVideoReplayConfig) -> None:
    if not math.isfinite(config.inference_fps) or config.inference_fps <= 0.0:
        raise TeedVideoReplayError(message="inference fps must be greater than zero")
    if config.output_width < MIN_OUTPUT_WIDTH:
        raise TeedVideoReplayError(
            message=f"output width must be at least {MIN_OUTPUT_WIDTH}",
        )


def _serialize_ignored_regions(
    regions: tuple[HudIgnoreRegion, ...],
) -> list[dict[str, float]]:
    return [
        {
            "left": region.left,
            "top": region.top,
            "right": region.right,
            "bottom": region.bottom,
        }
        for region in regions
    ]


def _summarize(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p50_index = min(len(ordered) - 1, max(0, math.ceil(0.50 * len(ordered)) - 1))
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": round(sum(values) / len(values), 6),
        "p50": round(ordered[p50_index], 6),
        "p95": round(ordered[p95_index], 6),
        "max": round(max(values), 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
