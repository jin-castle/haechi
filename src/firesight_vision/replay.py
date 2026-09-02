from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from PIL import Image

from firesight_vision.hud_edges import (
    HudEdgeError,
    HudEdgeProfile,
    HudIgnoreRegion,
    build_hud_edge_overlay,
)


class ReplayFramePayload(TypedDict):
    index: int
    input_file: str
    overlay_file: str
    mask_file: str
    latency_ms: float
    edge_ratio: float


class ReplaySummaryPayload(TypedDict):
    protocol: str
    claim: str
    input_path: str
    output_dir: str
    input_kind: str
    realtime_requested: bool
    pacing_enabled: bool
    target_fps: float
    ignored_regions: list[dict[str, float]]
    target_interval_ms: float
    frames_available: int
    frames_requested: int
    frames_processed: int
    elapsed_seconds: float
    effective_fps: float
    overrun_frames: int
    latency_ms: dict[str, float]
    edge_ratio: dict[str, float]
    frame_outputs: list[ReplayFramePayload]


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    input_path: Path
    out_dir: Path
    fps: float = 15.0
    max_frames: int | None = None
    threshold: int | None = None
    background_scale: float = 0.24
    edge_width: int | None = None
    profile: HudEdgeProfile = HudEdgeProfile.DENSE_SMOKE
    realtime: bool = True
    ignored_regions: tuple[HudIgnoreRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


PROTOCOL_NAME: Final = "pillow_hud_edge_replay_v1"
CLAIM_TEXT: Final = (
    "Paced image-folder replay for software-only HUD edge validation; it is not "
    "a hardware camera stream or video decoder."
)
SUPPORTED_IMAGE_SUFFIXES: Final = frozenset({".bmp", ".jpeg", ".jpg", ".png"})


def collect_replay_frames(input_path: Path) -> tuple[Path, ...]:
    if not input_path.exists():
        raise ReplayError(message=f"missing replay input: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ReplayError(message=f"unsupported replay image: {input_path}")
        return (input_path,)

    frame_paths = tuple(
        sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        ),
    )
    if len(frame_paths) == 0:
        raise ReplayError(message=f"no supported replay images found: {input_path}")
    return frame_paths


def run_edge_replay(config: ReplayConfig) -> ReplaySummaryPayload:
    _validate_config(config)
    all_frame_paths = collect_replay_frames(config.input_path)
    frame_paths = all_frame_paths
    if config.max_frames is not None:
        frame_paths = frame_paths[: config.max_frames]

    config.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = config.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    threshold = (
        config.threshold
        if config.threshold is not None
        else _default_threshold(config.profile)
    )
    edge_width = (
        config.edge_width
        if config.edge_width is not None
        else _default_edge_width(config.profile)
    )
    interval_seconds = 1.0 / config.fps
    started_at = time.perf_counter()
    frame_outputs: list[ReplayFramePayload] = []
    latencies: list[float] = []
    edge_ratios: list[float] = []
    overrun_frames = 0

    for index, input_path in enumerate(frame_paths):
        scheduled_at = started_at + index * interval_seconds
        if config.realtime:
            remaining_seconds = scheduled_at - time.perf_counter()
            if remaining_seconds > 0.0:
                time.sleep(remaining_seconds)
        frame_started_at = time.perf_counter()
        if frame_started_at > scheduled_at + interval_seconds:
            overrun_frames += 1
        try:
            with Image.open(input_path) as source_image:
                result = build_hud_edge_overlay(
                    source_image,
                    threshold=threshold,
                    background_scale=config.background_scale,
                    edge_width=edge_width,
                    profile=config.profile,
                    ignored_regions=config.ignored_regions,
                )
        except OSError as error:
            raise ReplayError(
                message=f"unable to decode replay frame: {input_path}",
            ) from error
        except HudEdgeError as error:
            raise ReplayError(message=str(error)) from error

        stem = input_path.stem
        overlay_name = f"frame_{index:06d}_{stem}_hud_edges.png"
        mask_name = f"frame_{index:06d}_{stem}_edge_mask.png"
        overlay_path = frames_dir / overlay_name
        mask_path = frames_dir / mask_name
        result.overlay.save(overlay_path)
        result.mask.save(mask_path)
        latency_ms = (time.perf_counter() - frame_started_at) * 1000.0
        latencies.append(latency_ms)
        edge_ratios.append(result.edge_ratio)
        frame_outputs.append(
            ReplayFramePayload(
                index=index,
                input_file=input_path.as_posix(),
                overlay_file=(Path("frames") / overlay_name).as_posix(),
                mask_file=(Path("frames") / mask_name).as_posix(),
                latency_ms=round(latency_ms, 3),
                edge_ratio=round(result.edge_ratio, 6),
            ),
        )

    elapsed_seconds = time.perf_counter() - started_at
    frames_processed = len(frame_outputs)
    summary = ReplaySummaryPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        input_path=config.input_path.as_posix(),
        output_dir=config.out_dir.as_posix(),
        input_kind="image-file" if config.input_path.is_file() else "image-directory",
        realtime_requested=config.realtime,
        pacing_enabled=config.realtime,
        target_fps=config.fps,
        ignored_regions=_serialize_ignored_regions(config.ignored_regions),
        target_interval_ms=1000.0 / config.fps,
        frames_available=len(all_frame_paths),
        frames_requested=len(frame_paths),
        frames_processed=frames_processed,
        elapsed_seconds=round(elapsed_seconds, 6),
        effective_fps=(
            round(frames_processed / elapsed_seconds, 3)
            if elapsed_seconds > 0.0
            else 0.0
        ),
        overrun_frames=overrun_frames,
        latency_ms=_summarize(latencies),
        edge_ratio=_summarize(edge_ratios),
        frame_outputs=frame_outputs,
    )
    _write_jsonl(config.out_dir / "replay_metrics.jsonl", frame_outputs)
    _ = (config.out_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_config(config: ReplayConfig) -> None:
    if not math.isfinite(config.fps) or config.fps <= 0.0:
        raise ReplayError(message="fps must be a finite value greater than zero")
    if config.max_frames is not None and config.max_frames < 1:
        raise ReplayError(message="max frames must be greater than zero")


def _write_jsonl(path: Path, frame_outputs: list[ReplayFramePayload]) -> None:
    lines = [json.dumps(frame, sort_keys=True) for frame in frame_outputs]
    _ = path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _summarize(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 6),
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "max": round(max(values), 6),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _default_threshold(profile: HudEdgeProfile) -> int:
    match profile:
        case HudEdgeProfile.STANDARD:
            return 36
        case HudEdgeProfile.DENSE_SMOKE:
            return 144
        case HudEdgeProfile.FIRE_LINE:
            return 192


def _default_edge_width(profile: HudEdgeProfile) -> int:
    match profile:
        case HudEdgeProfile.STANDARD:
            return 1
        case HudEdgeProfile.DENSE_SMOKE:
            return 3
        case HudEdgeProfile.FIRE_LINE:
            return 5


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
