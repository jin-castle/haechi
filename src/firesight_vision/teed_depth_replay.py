from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image, ImageDraw

from firesight_vision.depth_anything import (
    DEFAULT_DEPTH_MODEL_ID,
    DEFAULT_DEPTH_MODEL_REVISION,
    DepthAnythingPredictor,
)
from firesight_vision.depth_edge_fusion import (
    DepthEdgeFusionConfig,
    build_depth_aware_edge_overlay,
)
from firesight_vision.replay import ReplayError, collect_replay_frames
from firesight_vision.teed import TeedPredictor

if TYPE_CHECKING:
    from pathlib import Path

    from firesight_vision.hud_edges import HudIgnoreRegion


PROTOCOL_NAME: Final = "teed_depth_anything_metric_indoor_replay_v1"
CLAIM_TEXT: Final = (
    "Offline TEED plus monocular metric-depth visualization proof of concept; "
    "distance is estimated, not mmWave-confirmed or validated for dense smoke."
)


class TeedDepthReplayError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class TeedDepthReplayConfig:
    input_path: Path
    out_dir: Path
    checkpoint_path: Path
    depth_model_id: str = DEFAULT_DEPTH_MODEL_ID
    depth_model_revision: str = DEFAULT_DEPTH_MODEL_REVISION
    fps: float = 5.0
    max_frames: int | None = None
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    realtime: bool = False
    preview_max_depth_m: float = 6.0
    min_depth_m: float = 0.20
    near_max_m: float = 0.50
    mid_max_m: float = 1.00
    far_max_m: float = 1.50
    ignored_regions: tuple[HudIgnoreRegion, ...] = ()


def run_teed_depth_replay(config: TeedDepthReplayConfig) -> dict[str, object]:
    _validate_config(config)
    try:
        all_frame_paths = collect_replay_frames(config.input_path)
    except ReplayError as error:
        raise TeedDepthReplayError(str(error)) from error
    frame_paths = all_frame_paths[: config.max_frames]
    teed_predictor = TeedPredictor(config.checkpoint_path, device=config.device)
    depth_predictor = DepthAnythingPredictor(
        model_id=config.depth_model_id,
        revision=config.depth_model_revision,
        device=config.device,
        preview_max_depth_m=config.preview_max_depth_m,
    )
    fusion_config = DepthEdgeFusionConfig(
        min_depth_m=config.min_depth_m,
        near_max_m=config.near_max_m,
        mid_max_m=config.mid_max_m,
        far_max_m=config.far_max_m,
        background_scale=config.background_scale,
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = config.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    interval_seconds = 1.0 / config.fps
    frame_outputs: list[dict[str, object]] = []
    teed_latencies: list[float] = []
    depth_latencies: list[float] = []
    fusion_latencies: list[float] = []
    total_latencies: list[float] = []
    model_valid_edge_ratios: list[float] = []
    banded_edge_ratios: list[float] = []
    overrun_frames = 0

    for index, input_path in enumerate(frame_paths):
        scheduled_at = started_at + (index * interval_seconds)
        if config.realtime:
            remaining_seconds = scheduled_at - time.perf_counter()
            if remaining_seconds > 0.0:
                time.sleep(remaining_seconds)
        frame_started_at = time.perf_counter()
        if frame_started_at > scheduled_at + interval_seconds:
            overrun_frames += 1
        try:
            with Image.open(input_path) as source_image:
                source = source_image.convert("RGB")
        except OSError as error:
            message = f"unable to decode replay frame: {input_path}"
            raise TeedDepthReplayError(message) from error

        teed_started_at = time.perf_counter()
        teed_result = teed_predictor.process(
            source,
            threshold=config.threshold,
            background_scale=config.background_scale,
            edge_width=config.edge_width,
            ignored_regions=config.ignored_regions,
        )
        teed_latency_ms = (time.perf_counter() - teed_started_at) * 1000.0

        depth_started_at = time.perf_counter()
        depth_result = depth_predictor.process(source)
        depth_latency_ms = (time.perf_counter() - depth_started_at) * 1000.0

        fusion_started_at = time.perf_counter()
        fusion_result = build_depth_aware_edge_overlay(
            source,
            teed_result.mask,
            depth_result.depth_m,
            config=fusion_config,
        )
        fusion_latency_ms = (time.perf_counter() - fusion_started_at) * 1000.0
        total_latency_ms = (time.perf_counter() - frame_started_at) * 1000.0

        stem = f"frame_{index:06d}_{input_path.stem}"
        files = _write_frame_artifacts(
            frames_dir,
            stem,
            source,
            teed_result.overlay,
            teed_result.probability,
            teed_result.mask,
            depth_result.depth_m,
            depth_result.preview,
            depth_result.valid_mask,
            fusion_result.overlay,
            fusion_result.band_mask,
        )
        teed_latencies.append(teed_latency_ms)
        depth_latencies.append(depth_latency_ms)
        fusion_latencies.append(fusion_latency_ms)
        total_latencies.append(total_latency_ms)
        model_valid_edge_ratios.append(fusion_result.edge_depth_model_valid_ratio)
        banded_edge_ratios.append(fusion_result.edge_depth_banded_ratio)
        frame_outputs.append(
            {
                "index": index,
                "input_file": input_path.as_posix(),
                **files,
                "teed_latency_ms": round(teed_latency_ms, 3),
                "depth_latency_ms": round(depth_latency_ms, 3),
                "fusion_latency_ms": round(fusion_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "teed_edge_ratio": round(teed_result.edge_ratio, 6),
                "depth_valid_ratio": round(depth_result.valid_ratio, 6),
                "depth_min_m": round(depth_result.min_depth_m, 4),
                "depth_median_m": round(depth_result.median_depth_m, 4),
                "depth_p95_m": round(depth_result.p95_depth_m, 4),
                "depth_max_m": round(depth_result.max_depth_m, 4),
                "edge_depth_model_valid_ratio": round(
                    fusion_result.edge_depth_model_valid_ratio,
                    6,
                ),
                "edge_depth_banded_ratio": round(
                    fusion_result.edge_depth_banded_ratio,
                    6,
                ),
                "near_edge_ratio": round(fusion_result.near_edge_ratio, 6),
                "mid_edge_ratio": round(fusion_result.mid_edge_ratio, 6),
                "far_edge_ratio": round(fusion_result.far_edge_ratio, 6),
                "out_of_band_edge_ratio": round(
                    fusion_result.out_of_band_edge_ratio,
                    6,
                ),
                "invalid_depth_edge_ratio": round(
                    fusion_result.invalid_depth_edge_ratio,
                    6,
                ),
                "unknown_edge_ratio": round(
                    fusion_result.unknown_edge_ratio,
                    6,
                ),
            },
        )

    elapsed_seconds = time.perf_counter() - started_at
    frames_processed = len(frame_outputs)
    summary: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": CLAIM_TEXT,
        "input_path": config.input_path.as_posix(),
        "output_dir": config.out_dir.as_posix(),
        "teed_checkpoint_path": config.checkpoint_path.as_posix(),
        "depth_model_id": depth_predictor.model_id,
        "depth_model_revision": depth_predictor.revision,
        "depth_model_max_m": depth_predictor.model_max_depth_m,
        "device": depth_predictor.device,
        "threshold": config.threshold,
        "edge_width": config.edge_width,
        "background_scale": config.background_scale,
        "distance_bands_m": {
            "minimum": config.min_depth_m,
            "near_max": config.near_max_m,
            "mid_max": config.mid_max_m,
            "far_max": config.far_max_m,
        },
        "realtime_requested": config.realtime,
        "target_fps": config.fps,
        "ignored_regions": _serialize_ignored_regions(config.ignored_regions),
        "frames_available": len(all_frame_paths),
        "frames_requested": len(frame_paths),
        "frames_processed": frames_processed,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "effective_fps": (
            round(frames_processed / elapsed_seconds, 3)
            if elapsed_seconds > 0.0
            else 0.0
        ),
        "overrun_frames": overrun_frames,
        "latency_ms": {
            "teed": _summarize(teed_latencies),
            "depth": _summarize(depth_latencies),
            "fusion": _summarize(fusion_latencies),
            "total": _summarize(total_latencies),
        },
        "edge_depth_model_valid_ratio": _summarize(model_valid_edge_ratios),
        "edge_depth_banded_ratio": _summarize(banded_edge_ratios),
        "frame_outputs": frame_outputs,
    }
    _write_jsonl(config.out_dir / "replay_metrics.jsonl", frame_outputs)
    _ = (config.out_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_frame_artifacts(
    frames_dir: Path,
    stem: str,
    source: Image.Image,
    teed_overlay: Image.Image,
    teed_probability: Image.Image,
    teed_mask: Image.Image,
    depth_m: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    depth_preview: Image.Image,
    depth_valid_mask: Image.Image,
    depth_overlay: Image.Image,
    depth_band_mask: Image.Image,
) -> dict[str, str]:
    paths = {
        "teed_overlay_file": frames_dir / f"{stem}_teed_edges.png",
        "teed_probability_file": frames_dir / f"{stem}_teed_probability.png",
        "teed_mask_file": frames_dir / f"{stem}_teed_mask.png",
        "depth_m_file": frames_dir / f"{stem}_depth_m.npy",
        "depth_preview_file": frames_dir / f"{stem}_depth_preview.png",
        "depth_valid_mask_file": frames_dir / f"{stem}_depth_valid_mask.png",
        "depth_overlay_file": frames_dir / f"{stem}_teed_depth_overlay.png",
        "depth_band_mask_file": frames_dir / f"{stem}_teed_depth_band_mask.png",
        "contact_sheet_file": frames_dir / f"{stem}_teed_depth_contact.png",
    }
    teed_overlay.save(paths["teed_overlay_file"])
    teed_probability.save(paths["teed_probability_file"])
    teed_mask.save(paths["teed_mask_file"])
    np.save(paths["depth_m_file"], depth_m, allow_pickle=False)
    depth_preview.save(paths["depth_preview_file"])
    depth_valid_mask.save(paths["depth_valid_mask_file"])
    depth_overlay.save(paths["depth_overlay_file"])
    depth_band_mask.save(paths["depth_band_mask_file"])
    _render_contact_sheet(
        source,
        teed_overlay,
        depth_preview,
        depth_overlay,
    ).save(paths["contact_sheet_file"])
    return {
        key: path.relative_to(frames_dir.parent).as_posix()
        for key, path in paths.items()
    }


def _render_contact_sheet(
    source: Image.Image,
    teed_overlay: Image.Image,
    depth_preview: Image.Image,
    depth_overlay: Image.Image,
) -> Image.Image:
    tile_width = 300
    tile_height = max(1, round(tile_width * source.height / source.width))
    pad = 12
    header_height = 32
    labels = (
        "RGB INPUT",
        "TEED",
        "DEPTH (RED NEAR / BLUE 6M)",
        "TEED+DEPTH (0.5 / 1.0 / 1.5M)",
    )
    images = (source, teed_overlay, depth_preview, depth_overlay)
    footer_height = 20
    sheet = Image.new(
        "RGB",
        (
            (tile_width * len(images)) + (pad * (len(images) + 1)),
            tile_height + header_height + footer_height + 12,
        ),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        x = pad + (index * (tile_width + pad))
        draw.text((x, 8), label, fill=(235, 235, 235))
        resized = image.convert("RGB").resize(
            (tile_width, tile_height),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(resized, (x, header_height))
    draw.text(
        (pad, header_height + tile_height + 6),
        "Estimated monocular depth: near=red/thick, mid=yellow, far=cyan/thin, "
        "gray=outside 0.2-1.5m",
        fill=(210, 210, 210),
    )
    return sheet


def _validate_config(config: TeedDepthReplayConfig) -> None:
    if not math.isfinite(config.fps) or config.fps <= 0.0:
        message = "fps must be greater than zero"
        raise TeedDepthReplayError(message)
    if config.max_frames is not None and config.max_frames < 1:
        message = "max frames must be greater than zero"
        raise TeedDepthReplayError(message)
    if not 0.0 <= config.threshold <= 1.0:
        message = "TEED threshold must be between 0 and 1"
        raise TeedDepthReplayError(message)
    maximum_edge_width = 15
    if config.edge_width < 1 or config.edge_width > maximum_edge_width:
        message = "TEED edge width must be between 1 and 15"
        raise TeedDepthReplayError(message)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
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
