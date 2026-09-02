from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import cv2
from PIL import Image

from firesight_vision.hud_edges import HudEdgeProfile, build_hud_edge_overlay
from firesight_vision.teed import TeedError, TeedPredictor

if TYPE_CHECKING:
    from collections.abc import Sequence


class BenchmarkArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    output_path: Path = Path()
    checkpoint_path: Path = Path()
    widths: str = "480,640,1280"
    inference_fps: str = "2,5,10"
    threshold: float = 0.75
    background_scale: float = 0.24
    classical_threshold: int = 144
    classical_edge_width: int = 3
    teed_edge_width: int = 1
    max_frames: int = 8
    device: str = "cpu"


class BenchmarkResult(TypedDict):
    output_width: int
    output_height: int
    requested_fps: float
    actual_sample_fps: float
    sampling_stride: int
    sampled_frames: int
    source_frames_read: int
    decode_errors: int
    preprocess_ms: dict[str, float]
    classical_ms: dict[str, float]
    teed_ms: dict[str, float]
    total_ms: dict[str, float]
    teed_edge_ratio: dict[str, float]
    realtime_budget_ms: float
    pacing_overrun_rate: float
    classical_fps_equivalent: float
    teed_fps_equivalent: float


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    input_path: Path
    output_path: Path
    checkpoint_path: Path
    widths: tuple[int, ...] = (480, 640, 1280)
    inference_fps: tuple[float, ...] = (2.0, 5.0, 10.0)
    threshold: float = 0.75
    background_scale: float = 0.24
    classical_threshold: int = 144
    classical_edge_width: int = 3
    teed_edge_width: int = 1
    max_frames: int = 8
    device: str = "cpu"


class BenchmarkError(Exception):
    pass


PROTOCOL_NAME: Final = "edge_replay_benchmark_v1"
CLAIM_TEXT: Final = (
    "CPU-only software timing baseline for TEED and classical edge replay; "
    "not a Jetson performance result."
)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_benchmark(config)
    except (BenchmarkError, TeedError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> BenchmarkConfig:
    namespace = BenchmarkArgNamespace()
    parser = argparse.ArgumentParser(
        description="Benchmark TEED and classical edge replay on a video.",
    )
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="output_path", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument(
        "--widths",
        default=namespace.widths,
        help="comma-separated output widths",
    )
    _ = parser.add_argument(
        "--inference-fps",
        dest="inference_fps",
        default=namespace.inference_fps,
        help="comma-separated requested inference FPS values",
    )
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--classical-threshold", default=144, type=int)
    _ = parser.add_argument("--classical-edge-width", default=3, type=int)
    _ = parser.add_argument("--teed-edge-width", default=1, type=int)
    _ = parser.add_argument("--max-frames", default=8, type=int)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.parse_args(argv, namespace=namespace)
    return BenchmarkConfig(
        input_path=namespace.input_path,
        output_path=namespace.output_path,
        checkpoint_path=namespace.checkpoint_path,
        widths=_parse_ints(namespace.widths, "widths"),
        inference_fps=_parse_floats(namespace.inference_fps, "inference fps"),
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        classical_threshold=namespace.classical_threshold,
        classical_edge_width=namespace.classical_edge_width,
        teed_edge_width=namespace.teed_edge_width,
        max_frames=namespace.max_frames,
        device=namespace.device,
    )


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    _validate_config(config)
    if not config.input_path.is_file():
        raise BenchmarkError(f"missing benchmark input: {config.input_path}")
    capture = cv2.VideoCapture(str(config.input_path))
    if not capture.isOpened():
        raise BenchmarkError(f"unable to open benchmark input: {config.input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if source_fps <= 0.0 or source_width <= 0 or source_height <= 0:
        raise BenchmarkError(f"invalid benchmark metadata: {config.input_path}")
    predictor = TeedPredictor(config.checkpoint_path, device=config.device)
    results: list[BenchmarkResult] = []
    for width in config.widths:
        for requested_fps in config.inference_fps:
            results.extend(
                [
                    _benchmark_condition(
                        config,
                        predictor,
                        source_fps,
                        source_width,
                        source_height,
                        width,
                        requested_fps,
                    ),
                ],
            )
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": CLAIM_TEXT,
        "input_file": config.input_path.as_posix(),
        "checkpoint_path": config.checkpoint_path.as_posix(),
        "device": predictor.device,
        "source": {
            "width": source_width,
            "height": source_height,
            "fps": round(source_fps, 6),
            "frames": source_frames,
            "duration_seconds": round(source_frames / source_fps, 6),
        },
        "config": {
            "widths": list(config.widths),
            "inference_fps": list(config.inference_fps),
            "threshold": config.threshold,
            "background_scale": config.background_scale,
            "classical_threshold": config.classical_threshold,
            "classical_edge_width": config.classical_edge_width,
            "teed_edge_width": config.teed_edge_width,
            "max_frames": config.max_frames,
        },
        "results": results,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = config.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _benchmark_condition(
    config: BenchmarkConfig,
    predictor: TeedPredictor,
    source_fps: float,
    source_width: int,
    source_height: int,
    output_width: int,
    requested_fps: float,
) -> BenchmarkResult:
    actual_width = min(output_width, source_width)
    output_height = _even_height(source_height, source_width, actual_width)
    sampling_stride = max(1, round(source_fps / requested_fps))
    preprocess_ms: list[float] = []
    classical_ms: list[float] = []
    teed_ms: list[float] = []
    total_ms: list[float] = []
    teed_edge_ratios: list[float] = []
    overrun_count = 0
    sampled_frames = 0
    source_frame_index = 0
    decode_errors = 0
    capture = cv2.VideoCapture(str(config.input_path))
    if not capture.isOpened():
        raise BenchmarkError(f"unable to open benchmark input: {config.input_path}")
    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break
            if source_frame_index % sampling_stride != 0:
                source_frame_index += 1
                continue
            if sampled_frames >= config.max_frames:
                break
            preprocess_started = time.perf_counter()
            resized_bgr = cv2.resize(
                bgr_frame,
                (actual_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            rgb_frame = Image.fromarray(
                cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB),
                mode="RGB",
            )
            preprocess_elapsed = (time.perf_counter() - preprocess_started) * 1000.0
            try:
                classical_started = time.perf_counter()
                _ = build_hud_edge_overlay(
                    rgb_frame,
                    threshold=config.classical_threshold,
                    background_scale=config.background_scale,
                    edge_width=config.classical_edge_width,
                    profile=HudEdgeProfile.DENSE_SMOKE,
                )
                classical_elapsed = (time.perf_counter() - classical_started) * 1000.0
                teed_started = time.perf_counter()
                teed_result = predictor.process(
                    rgb_frame,
                    threshold=config.threshold,
                    background_scale=config.background_scale,
                    edge_width=config.teed_edge_width,
                )
                teed_elapsed = (time.perf_counter() - teed_started) * 1000.0
            except (OSError, TeedError) as error:
                decode_errors += 1
                raise BenchmarkError(
                    f"failed to process frame {source_frame_index}: {error}",
                ) from error
            total_elapsed = preprocess_elapsed + classical_elapsed + teed_elapsed
            preprocess_ms.append(preprocess_elapsed)
            classical_ms.append(classical_elapsed)
            teed_ms.append(teed_elapsed)
            total_ms.append(total_elapsed)
            teed_edge_ratios.append(teed_result.edge_ratio)
            if total_elapsed > 1000.0 / requested_fps:
                overrun_count += 1
            sampled_frames += 1
            source_frame_index += 1
    finally:
        capture.release()
    if sampled_frames == 0:
        raise BenchmarkError(f"benchmark produced no samples: {config.input_path}")
    return BenchmarkResult(
        output_width=actual_width,
        output_height=output_height,
        requested_fps=requested_fps,
        actual_sample_fps=source_fps / sampling_stride,
        sampling_stride=sampling_stride,
        sampled_frames=sampled_frames,
        source_frames_read=source_frame_index,
        decode_errors=decode_errors,
        preprocess_ms=_summarize(preprocess_ms),
        classical_ms=_summarize(classical_ms),
        teed_ms=_summarize(teed_ms),
        total_ms=_summarize(total_ms),
        teed_edge_ratio=_summarize(teed_edge_ratios),
        realtime_budget_ms=round(1000.0 / requested_fps, 6),
        pacing_overrun_rate=round(overrun_count / sampled_frames, 6),
        classical_fps_equivalent=round(1000.0 / _mean(classical_ms), 6),
        teed_fps_equivalent=round(1000.0 / _mean(teed_ms), 6),
    )


def _parse_ints(raw: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise BenchmarkError(f"invalid {name}: {raw}") from error
    if len(values) == 0:
        raise BenchmarkError(f"{name} must not be empty")
    return values


def _parse_floats(raw: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise BenchmarkError(f"invalid {name}: {raw}") from error
    if len(values) == 0:
        raise BenchmarkError(f"{name} must not be empty")
    return values


def _validate_config(config: BenchmarkConfig) -> None:
    if any(width < 64 for width in config.widths):
        raise BenchmarkError("benchmark widths must be at least 64")
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in config.inference_fps):
        raise BenchmarkError("benchmark inference fps values must be positive")
    if config.max_frames < 1:
        raise BenchmarkError("max frames must be greater than zero")


def _even_height(source_height: int, source_width: int, output_width: int) -> int:
    output_height = round(source_height * output_width / source_width)
    return output_height + (output_height % 2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summarize(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p50_index = min(len(ordered) - 1, max(0, math.ceil(0.50 * len(ordered)) - 1))
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": round(_mean(values), 6),
        "p50": round(ordered[p50_index], 6),
        "p95": round(ordered[p95_index], 6),
        "max": round(max(values), 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
