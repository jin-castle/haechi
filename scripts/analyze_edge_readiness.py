from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import cv2
import numpy as np
from PIL import Image

from firesight_vision.hud_edges import (
    FIRE360_OSD_IGNORED_REGIONS,
    HudEdgeProfile,
    HudIgnoreRegion,
    build_hud_edge_overlay,
)
from firesight_vision.teed import TeedError, TeedPredictor

if TYPE_CHECKING:
    from collections.abc import Sequence


class ReadinessArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    output_dir: Path = Path()
    checkpoint_path: Path = Path()
    analysis_width: int = 640
    inference_fps: float = 5.0
    threshold: float = 0.75
    classical_threshold: int = 144
    classical_edge_width: int = 3
    teed_edge_width: int = 1
    background_scale: float = 0.24
    guard_fraction: float = 0.02
    max_frames: int | None = None
    device: str = "cpu"
    ignore_fire360_osd: bool = False


class ReadinessError(Exception):
    pass


class MethodSummary(TypedDict):
    frames: int
    edge_ratio: dict[str, float]
    temporal_iou: dict[str, float]
    temporal_xor_rate: dict[str, float]
    temporal_edge_ratio_delta: dict[str, float]
    latency_ms: dict[str, float]
    raw_osd_edge_pixels: int
    raw_guard_ring_edge_pixels: int
    guard_ring_area_pixels: int
    guard_ring_edge_ratio: dict[str, float]
    post_ignore_edge_pixels_in_osd: int


@dataclass(frozen=True, slots=True)
class ReadinessConfig:
    input_path: Path
    output_dir: Path
    checkpoint_path: Path
    analysis_width: int = 640
    inference_fps: float = 5.0
    threshold: float = 0.75
    classical_threshold: int = 144
    classical_edge_width: int = 3
    teed_edge_width: int = 1
    background_scale: float = 0.24
    guard_fraction: float = 0.02
    max_frames: int | None = None
    device: str = "cpu"
    ignore_fire360_osd: bool = False


@dataclass(slots=True)
class MethodAccumulator:
    edge_ratios: list[float] = field(default_factory=list)
    temporal_ious: list[float] = field(default_factory=list)
    temporal_xor_rates: list[float] = field(default_factory=list)
    temporal_edge_ratio_deltas: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    guard_ring_ratios: list[float] = field(default_factory=list)
    raw_osd_edge_pixels: int = 0
    raw_guard_ring_edge_pixels: int = 0
    guard_ring_area_pixels: int = 0
    post_ignore_edge_pixels_in_osd: int = 0
    previous_mask: np.ndarray | None = None
    previous_edge_ratio: float | None = None

    def record(
        self,
        raw_mask: np.ndarray,
        ignored_mask: np.ndarray,
        osd_region: np.ndarray,
        guard_ring: np.ndarray,
        latency_ms: float,
    ) -> None:
        edge_ratio = float(np.mean(ignored_mask))
        self.edge_ratios.append(edge_ratio)
        self.latencies_ms.append(latency_ms)
        self.raw_osd_edge_pixels += int(np.count_nonzero(raw_mask & osd_region))
        self.raw_guard_ring_edge_pixels += int(
            np.count_nonzero(raw_mask & guard_ring),
        )
        self.guard_ring_area_pixels += int(np.count_nonzero(guard_ring))
        self.guard_ring_ratios.append(
            _ratio(
                np.count_nonzero(raw_mask & guard_ring),
                np.count_nonzero(guard_ring),
            ),
        )
        self.post_ignore_edge_pixels_in_osd = max(
            self.post_ignore_edge_pixels_in_osd,
            int(np.count_nonzero(ignored_mask & osd_region)),
        )
        if self.previous_mask is not None and self.previous_edge_ratio is not None:
            intersection = np.count_nonzero(self.previous_mask & ignored_mask)
            union = np.count_nonzero(self.previous_mask | ignored_mask)
            self.temporal_ious.append(_ratio(intersection, union))
            self.temporal_xor_rates.append(
                float(np.mean(self.previous_mask != ignored_mask)),
            )
            self.temporal_edge_ratio_deltas.append(
                abs(edge_ratio - self.previous_edge_ratio),
            )
        self.previous_mask = ignored_mask
        self.previous_edge_ratio = edge_ratio

    def summary(self) -> MethodSummary:
        return MethodSummary(
            frames=len(self.edge_ratios),
            edge_ratio=_summarize(self.edge_ratios),
            temporal_iou=_summarize(self.temporal_ious),
            temporal_xor_rate=_summarize(self.temporal_xor_rates),
            temporal_edge_ratio_delta=_summarize(self.temporal_edge_ratio_deltas),
            latency_ms=_summarize(self.latencies_ms),
            raw_osd_edge_pixels=self.raw_osd_edge_pixels,
            raw_guard_ring_edge_pixels=self.raw_guard_ring_edge_pixels,
            guard_ring_area_pixels=self.guard_ring_area_pixels,
            guard_ring_edge_ratio=_summarize(self.guard_ring_ratios),
            post_ignore_edge_pixels_in_osd=self.post_ignore_edge_pixels_in_osd,
        )


PROTOCOL_NAME: Final = "edge_readiness_analysis_v1"
CLAIM_TEXT: Final = (
    "Software-only temporal, OSD-boundary, model-comparison, and input replay "
    "checks; no smoke-domain accuracy or Jetson real-time claim."
)
SUPPORTED_VIDEO_SUFFIXES: Final = frozenset({".avi", ".mkv", ".mov", ".mp4", ".mts"})
MIN_ANALYSIS_WIDTH: Final = 64


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        summary = run_edge_readiness_analysis(config)
    except (ReadinessError, TeedError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> ReadinessConfig:
    namespace = ReadinessArgNamespace()
    parser = argparse.ArgumentParser(
        description="Analyze TEED/classical edge replay readiness on Fire360 videos.",
    )
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
        type=Path,
    )
    _ = parser.add_argument("--analysis-width", default=640, type=int)
    _ = parser.add_argument("--inference-fps", default=5.0, type=float)
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--classical-threshold", default=144, type=int)
    _ = parser.add_argument("--classical-edge-width", default=3, type=int)
    _ = parser.add_argument("--teed-edge-width", default=1, type=int)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument(
        "--guard-fraction",
        default=0.02,
        type=float,
        help="normalized guard band around each fixed Fire360 OSD region",
    )
    _ = parser.add_argument("--max-frames", default=None, type=int)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.add_argument(
        "--ignore-fire360-osd",
        action="store_true",
        help="mask the fixed Fire360 badge and colorbar regions",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    return ReadinessConfig(
        input_path=namespace.input_path,
        output_dir=namespace.output_dir,
        checkpoint_path=namespace.checkpoint_path,
        analysis_width=namespace.analysis_width,
        inference_fps=namespace.inference_fps,
        threshold=namespace.threshold,
        classical_threshold=namespace.classical_threshold,
        classical_edge_width=namespace.classical_edge_width,
        teed_edge_width=namespace.teed_edge_width,
        background_scale=namespace.background_scale,
        guard_fraction=namespace.guard_fraction,
        max_frames=namespace.max_frames,
        device=namespace.device,
        ignore_fire360_osd=namespace.ignore_fire360_osd,
    )


def run_edge_readiness_analysis(config: ReadinessConfig) -> dict[str, object]:
    _validate_config(config)
    video_paths = _collect_video_paths(config.input_path)
    predictor = TeedPredictor(config.checkpoint_path, device=config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ignored_regions = (
        FIRE360_OSD_IGNORED_REGIONS if config.ignore_fire360_osd else ()
    )
    videos = [
        _analyze_video(path, config, predictor, ignored_regions)
        for path in video_paths
    ]
    summary: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": CLAIM_TEXT,
        "checkpoint_path": config.checkpoint_path.as_posix(),
        "device": predictor.device,
        "config": {
            "analysis_width": config.analysis_width,
            "inference_fps": config.inference_fps,
            "threshold": config.threshold,
            "classical_threshold": config.classical_threshold,
            "teed_edge_width": config.teed_edge_width,
            "background_scale": config.background_scale,
            "guard_fraction": config.guard_fraction,
            "max_frames": config.max_frames,
            "ignore_fire360_osd": config.ignore_fire360_osd,
            "ignored_regions": _serialize_regions(ignored_regions),
        },
        "acceptance": _build_acceptance(
            videos,
            osd_enabled=config.ignore_fire360_osd,
        ),
        "videos": videos,
        "aggregate": _aggregate_videos(videos),
    }
    _ = (config.output_dir / "edge_readiness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _analyze_video(
    input_path: Path,
    config: ReadinessConfig,
    predictor: TeedPredictor,
    ignored_regions: tuple[HudIgnoreRegion, ...],
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ReadinessError(f"unable to open video: {input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0.0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise ReadinessError(f"invalid video metadata: {input_path}")

    sampling_stride = max(1, round(source_fps / config.inference_fps))
    analysis_width = min(config.analysis_width, source_width)
    analysis_height = _even_height(source_height, source_width, analysis_width)
    osd_region, guard_ring = _build_osd_masks(
        analysis_width,
        analysis_height,
        ignored_regions,
        config.guard_fraction,
    )
    classical = MethodAccumulator()
    teed = MethodAccumulator()
    comparison_ious: list[float] = []
    comparison_xor_rates: list[float] = []
    teed_only_ratios: list[float] = []
    classical_only_ratios: list[float] = []
    source_frame_index = 0
    sampled_frames = 0
    preprocess_ms: list[float] = []
    decode_errors = 0
    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break
            if source_frame_index % sampling_stride != 0:
                source_frame_index += 1
                continue
            if config.max_frames is not None and sampled_frames >= config.max_frames:
                break
            preprocess_started = time.perf_counter()
            resized_bgr = cv2.resize(
                bgr_frame,
                (analysis_width, analysis_height),
                interpolation=cv2.INTER_AREA,
            )
            rgb_frame = Image.fromarray(
                cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB),
                mode="RGB",
            )
            preprocess_ms.append((time.perf_counter() - preprocess_started) * 1000.0)
            try:
                classical_started = time.perf_counter()
                classical_result = build_hud_edge_overlay(
                    rgb_frame,
                    threshold=config.classical_threshold,
                    background_scale=config.background_scale,
                    edge_width=config.classical_edge_width,
                    profile=HudEdgeProfile.DENSE_SMOKE,
                )
                classical_latency = (time.perf_counter() - classical_started) * 1000.0
                teed_started = time.perf_counter()
                teed_result = predictor.process(
                    rgb_frame,
                    threshold=config.threshold,
                    background_scale=config.background_scale,
                    edge_width=config.teed_edge_width,
                )
                teed_latency = (time.perf_counter() - teed_started) * 1000.0
            except (OSError, TeedError) as error:
                decode_errors += 1
                raise ReadinessError(
                    f"failed to process sampled frame {source_frame_index} "
                    f"from {input_path}: {error}",
                ) from error

            classical_raw = np.asarray(classical_result.mask, dtype=np.uint8) > 0
            teed_raw = np.asarray(teed_result.mask, dtype=np.uint8) > 0
            classical_mask = classical_raw & ~osd_region
            teed_mask = teed_raw & ~osd_region
            classical.record(
                classical_raw,
                classical_mask,
                osd_region,
                guard_ring,
                classical_latency,
            )
            teed.record(teed_raw, teed_mask, osd_region, guard_ring, teed_latency)
            comparison = _compare_masks(classical_mask, teed_mask)
            comparison_ious.append(comparison["iou"])
            comparison_xor_rates.append(comparison["xor_rate"])
            teed_only_ratios.append(comparison["teed_only_ratio"])
            classical_only_ratios.append(comparison["classical_only_ratio"])
            sampled_frames += 1
            source_frame_index += 1
    finally:
        capture.release()

    expected_samples = min(
        math.ceil(metadata_frames / sampling_stride),
        config.max_frames if config.max_frames is not None else math.inf,
    )
    return {
        "input_file": input_path.as_posix(),
        "source": {
            "width": source_width,
            "height": source_height,
            "fps": round(source_fps, 6),
            "frames_metadata": metadata_frames,
            "duration_seconds": round(metadata_frames / source_fps, 6),
        },
        "analysis": {
            "width": analysis_width,
            "height": analysis_height,
            "sampling_stride": sampling_stride,
            "requested_fps": config.inference_fps,
            "sampled_frames": sampled_frames,
            "expected_samples": int(expected_samples),
            "source_frames_read": source_frame_index,
            "decode_errors": decode_errors,
            "preprocess_ms": _summarize(preprocess_ms),
        },
        "methods": {
            "classical": classical.summary(),
            "teed": teed.summary(),
        },
        "comparison": {
            "frames": len(comparison_ious),
            "mask_iou": _summarize(comparison_ious),
            "mask_xor_rate": _summarize(comparison_xor_rates),
            "teed_only_ratio": _summarize(teed_only_ratios),
            "classical_only_ratio": _summarize(classical_only_ratios),
            "interpretation": (
                "Agreement/disagreement only; without smoke-domain labels this "
                "does not establish which model is more accurate."
            ),
        },
    }


def _build_acceptance(
    videos: list[dict[str, object]],
    *,
    osd_enabled: bool,
) -> dict[str, object]:
    all_decoded = all(
        video["analysis"]["sampled_frames"] == video["analysis"]["expected_samples"]
        and video["analysis"]["decode_errors"] == 0
        for video in videos
    )
    max_osd_pixels = max(
        int(video["methods"][method]["post_ignore_edge_pixels_in_osd"])
        for video in videos
        for method in ("classical", "teed")
    )
    guard_review = any(
        int(video["methods"][method]["raw_guard_ring_edge_pixels"]) > 0
        for video in videos
        for method in ("classical", "teed")
    )
    return {
        "input_decode_and_sampling": "PASS" if all_decoded else "FAIL",
        "osd_interior_masked": (
            ("PASS" if max_osd_pixels == 0 else "FAIL")
            if osd_enabled
            else "NOT_APPLICABLE"
        ),
        "osd_boundary_guard": (
            ("REVIEW" if guard_review else "CLEAR")
            if osd_enabled
            else "NOT_APPLICABLE"
        ),
        "temporal_stability": "REPORT_ONLY",
        "model_quality": "NOT_ESTABLISHED_WITHOUT_LABELS",
        "jetson_realtime": "NOT_TESTED_HARDWARE_UNAVAILABLE",
        "notes": [
            (
                "Boundary guard reports raw edge pixels immediately outside the "
                "fixed OSD boxes; inspect visually before expanding the default "
                "boxes."
            ),
            (
                "Temporal IoU/XOR and TEED/classical agreement are diagnostic "
                "metrics, not smoke-domain accuracy metrics."
            ),
        ],
    }


def _aggregate_videos(videos: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for method in ("classical", "teed"):
        aggregate[method] = {
            "frames": sum(
                int(video["methods"][method]["frames"]) for video in videos
            ),
            "edge_ratio": _merge_metric(videos, method, "edge_ratio"),
            "temporal_iou": _merge_metric(videos, method, "temporal_iou"),
            "temporal_xor_rate": _merge_metric(
                videos,
                method,
                "temporal_xor_rate",
            ),
            "temporal_edge_ratio_delta": _merge_metric(
                videos,
                method,
                "temporal_edge_ratio_delta",
            ),
            "latency_ms": _merge_metric(videos, method, "latency_ms"),
            "guard_ring_edge_ratio": _merge_metric(
                videos,
                method,
                "guard_ring_edge_ratio",
            ),
        }
    return {
        "videos": len(videos),
        "sampled_frames": sum(
            int(video["analysis"]["sampled_frames"]) for video in videos
        ),
        "methods": aggregate,
        "comparison_mask_iou": _merge_comparison_metric(videos, "mask_iou"),
        "comparison_mask_xor_rate": _merge_comparison_metric(
            videos,
            "mask_xor_rate",
        ),
    }


def _merge_metric(
    videos: list[dict[str, object]],
    method: str,
    metric: str,
) -> dict[str, float]:
    values = [
        float(video["methods"][method][metric]["mean"])
        for video in videos
    ]
    return _summarize(values)


def _merge_comparison_metric(
    videos: list[dict[str, object]],
    metric: str,
) -> dict[str, float]:
    values = [float(video["comparison"][metric]["mean"]) for video in videos]
    return _summarize(values)


def _compare_masks(
    classical_mask: np.ndarray,
    teed_mask: np.ndarray,
) -> dict[str, float]:
    intersection = int(np.count_nonzero(classical_mask & teed_mask))
    union = int(np.count_nonzero(classical_mask | teed_mask))
    total = classical_mask.size
    return {
        "iou": _ratio(intersection, union),
        "xor_rate": float(np.mean(classical_mask != teed_mask)),
        "teed_only_ratio": _ratio(
            np.count_nonzero(teed_mask & ~classical_mask),
            total,
        ),
        "classical_only_ratio": _ratio(
            np.count_nonzero(classical_mask & ~teed_mask),
            total,
        ),
    }


def _build_osd_masks(
    width: int,
    height: int,
    regions: tuple[HudIgnoreRegion, ...],
    guard_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    osd_region = np.zeros((height, width), dtype=bool)
    expanded_region = np.zeros((height, width), dtype=bool)
    for region in regions:
        _paint_region(osd_region, region)
        _paint_region(
            expanded_region,
            HudIgnoreRegion(
                left=max(0.0, region.left - guard_fraction),
                top=max(0.0, region.top - guard_fraction),
                right=min(1.0, region.right + guard_fraction),
                bottom=min(1.0, region.bottom + guard_fraction),
            ),
        )
    return osd_region, expanded_region & ~osd_region


def _paint_region(mask: np.ndarray, region: HudIgnoreRegion) -> None:
    height, width = mask.shape
    left = max(0, min(width - 1, round(region.left * width)))
    top = max(0, min(height - 1, round(region.top * height)))
    right = max(left, min(width - 1, round(region.right * width)))
    bottom = max(top, min(height - 1, round(region.bottom * height)))
    mask[top : bottom + 1, left : right + 1] = True


def _collect_video_paths(input_path: Path) -> tuple[Path, ...]:
    if not input_path.exists():
        raise ReadinessError(f"missing video input: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ReadinessError(f"unsupported video: {input_path}")
        return (input_path,)
    paths = tuple(
        sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        ),
    )
    if len(paths) == 0:
        raise ReadinessError(f"no supported videos found: {input_path}")
    return paths


def _validate_config(config: ReadinessConfig) -> None:
    if config.analysis_width < MIN_ANALYSIS_WIDTH:
        raise ReadinessError(f"analysis width must be at least {MIN_ANALYSIS_WIDTH}")
    if not math.isfinite(config.inference_fps) or config.inference_fps <= 0.0:
        raise ReadinessError("inference fps must be greater than zero")
    if not math.isfinite(config.guard_fraction) or config.guard_fraction < 0.0:
        raise ReadinessError("guard fraction must be non-negative")
    if config.max_frames is not None and config.max_frames < 1:
        raise ReadinessError("max frames must be greater than zero")


def _even_height(source_height: int, source_width: int, output_width: int) -> int:
    output_height = round(source_height * output_width / source_width)
    return output_height + (output_height % 2)


def _serialize_regions(
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


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


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
