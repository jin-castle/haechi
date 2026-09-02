from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from PIL import Image

from firesight_vision.replay import ReplayError, collect_replay_frames
from firesight_vision.teed import TeedError, TeedPredictor

if TYPE_CHECKING:
    from firesight_vision.hud_edges import HudIgnoreRegion


class TeedReplayFramePayload(TypedDict):
    index: int
    input_file: str
    overlay_file: str
    mask_file: str
    probability_file: str
    latency_ms: float
    edge_ratio: float


class TeedReplaySummaryPayload(TypedDict):
    protocol: str
    claim: str
    input_path: str
    output_dir: str
    checkpoint_path: str
    device: str
    threshold: float
    edge_width: int
    realtime_requested: bool
    target_fps: float
    ignored_regions: list[dict[str, float]]
    frames_available: int
    frames_requested: int
    frames_processed: int
    elapsed_seconds: float
    effective_fps: float
    overrun_frames: int
    latency_ms: dict[str, float]
    edge_ratio: dict[str, float]
    frame_outputs: list[TeedReplayFramePayload]


@dataclass(frozen=True, slots=True)
class TeedReplayConfig:
    input_path: Path
    out_dir: Path
    checkpoint_path: Path
    fps: float = 15.0
    max_frames: int | None = None
    threshold: float = 0.35
    background_scale: float = 0.24
    edge_width: int = 3
    device: str = "auto"
    realtime: bool = True
    ignored_regions: tuple[HudIgnoreRegion, ...] = ()


PROTOCOL_NAME: Final = "teed_biped_replay_v1"
CLAIM_TEXT: Final = (
    "Pretrained TEED BIPED checkpoint replay for edge visibility comparison; "
    "the model is not fine-tuned on Fire360 smoke footage."
)


def run_teed_replay(config: TeedReplayConfig) -> TeedReplaySummaryPayload:
    _validate_config(config)
    try:
        all_frame_paths = collect_replay_frames(config.input_path)
    except ReplayError as error:
        raise TeedError(str(error)) from error
    frame_paths = all_frame_paths
    if config.max_frames is not None:
        frame_paths = frame_paths[: config.max_frames]

    predictor = TeedPredictor(config.checkpoint_path, device=config.device)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = config.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    interval_seconds = 1.0 / config.fps
    started_at = time.perf_counter()
    frame_outputs: list[TeedReplayFramePayload] = []
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
                result = predictor.process(
                    source_image,
                    threshold=config.threshold,
                    background_scale=config.background_scale,
                    edge_width=config.edge_width,
                    ignored_regions=config.ignored_regions,
                )
        except OSError as error:
            message = f"unable to decode replay frame: {input_path}"
            raise TeedError(message) from error

        stem = input_path.stem
        overlay_name = f"frame_{index:06d}_{stem}_teed_edges.png"
        mask_name = f"frame_{index:06d}_{stem}_teed_mask.png"
        probability_name = f"frame_{index:06d}_{stem}_teed_probability.png"
        result.overlay.save(frames_dir / overlay_name)
        result.mask.save(frames_dir / mask_name)
        result.probability.save(frames_dir / probability_name)
        latency_ms = (time.perf_counter() - frame_started_at) * 1000.0
        latencies.append(latency_ms)
        edge_ratios.append(result.edge_ratio)
        frame_outputs.append(
            TeedReplayFramePayload(
                index=index,
                input_file=input_path.as_posix(),
                overlay_file=(Path("frames") / overlay_name).as_posix(),
                mask_file=(Path("frames") / mask_name).as_posix(),
                probability_file=(Path("frames") / probability_name).as_posix(),
                latency_ms=round(latency_ms, 3),
                edge_ratio=round(result.edge_ratio, 6),
            ),
        )

    elapsed_seconds = time.perf_counter() - started_at
    frames_processed = len(frame_outputs)
    summary = TeedReplaySummaryPayload(
        protocol=PROTOCOL_NAME,
        claim=CLAIM_TEXT,
        input_path=config.input_path.as_posix(),
        output_dir=config.out_dir.as_posix(),
        checkpoint_path=config.checkpoint_path.as_posix(),
        device=predictor.device,
        threshold=config.threshold,
        edge_width=config.edge_width,
        realtime_requested=config.realtime,
        target_fps=config.fps,
        ignored_regions=_serialize_ignored_regions(config.ignored_regions),
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


def _validate_config(config: TeedReplayConfig) -> None:
    if not math.isfinite(config.fps) or config.fps <= 0.0:
        message = "fps must be a finite value greater than zero"
        raise TeedError(message)
    if config.max_frames is not None and config.max_frames < 1:
        message = "max frames must be greater than zero"
        raise TeedError(message)


def _write_jsonl(path: Path, frame_outputs: list[TeedReplayFramePayload]) -> None:
    lines = [json.dumps(frame, sort_keys=True) for frame in frame_outputs]
    _ = path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
