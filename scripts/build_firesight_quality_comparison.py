from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, cast

from PIL import Image, ImageDraw, ImageFont

from firesight_vision.canny import CannyError
from firesight_vision.deployment import (
    FireSightDeploymentError,
    FireSightFrameResult,
    FireSightRuntime,
    FireSightRuntimeConfig,
)
from firesight_vision.teed import TeedError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from firesight_vision.hud_edges import HudIgnoreRegion


class QualityArgNamespace(argparse.Namespace):
    input_csv: Path = Path(
        ".omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv",
    )
    output_dir: Path = Path()
    checkpoint_path: Path = Path("models/teed/5_model.pth")
    profile_path: Path = Path("data/fire360/video_profiles.json")
    width: int = 320
    height: int = 240
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "cpu"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    warmup: int = 2


@dataclass(frozen=True, slots=True)
class QualityComparisonConfig:
    input_csv: Path
    output_dir: Path
    checkpoint_path: Path
    profile_path: Path
    width: int = 320
    height: int = 240
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "cpu"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    warmup: int = 2


@dataclass(frozen=True, slots=True)
class QualityComparisonError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class QualityArtifactPaths:
    original: Path
    canny: Path
    teed: Path
    canny_mask: Path
    teed_mask: Path


class QualityRecord(TypedDict):
    evaluation_id: str
    input_file: str
    ignore_fire360_osd: bool
    original_path: str
    canny_path: str
    teed_path: str
    canny_mask_path: str
    teed_mask_path: str
    canny_latency_ms: float
    teed_latency_ms: float
    canny_edge_ratio: float
    teed_edge_ratio: float
    canny_edge_pixels: int
    teed_edge_pixels: int
    canny_osd_leak_pixels: int
    teed_osd_leak_pixels: int


PROTOCOL_NAME: Final = "firesight_deployment_quality_comparison_v1"
EXPECTED_FRAME_COUNT: Final = 30
CONTACT_SHEET_COUNT: Final = 10
MIN_QUALITY_DIMENSION: Final = 64
MASK_EDGE_VALUE: Final = 255


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_quality_comparison(config)
    except (
        CannyError,
        FireSightDeploymentError,
        QualityComparisonError,
        TeedError,
    ) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> QualityComparisonConfig:
    namespace = QualityArgNamespace()
    parser = argparse.ArgumentParser(
        description="Compare TEED and Canny quality on fixed FireSight frames.",
    )
    _ = parser.add_argument("--input-csv", default=namespace.input_csv, type=Path)
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument(
        "--teed-checkpoint",
        default=namespace.checkpoint_path,
        type=Path,
    )
    _ = parser.add_argument(
        "--profiles",
        dest="profile_path",
        default=namespace.profile_path,
        type=Path,
    )
    _ = parser.add_argument("--width", default=320, type=int)
    _ = parser.add_argument("--height", default=240, type=int)
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=1, type=int)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.add_argument("--canny-low", default=50, type=int)
    _ = parser.add_argument("--canny-high", default=150, type=int)
    _ = parser.add_argument("--warmup", default=2, type=int)
    _ = parser.parse_args(argv, namespace=namespace)
    return QualityComparisonConfig(
        input_csv=namespace.input_csv,
        output_dir=namespace.output_dir,
        checkpoint_path=namespace.checkpoint_path,
        profile_path=namespace.profile_path,
        width=namespace.width,
        height=namespace.height,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
        canny_low_threshold=namespace.canny_low_threshold,
        canny_high_threshold=namespace.canny_high_threshold,
        warmup=namespace.warmup,
    )


def run_quality_comparison(config: QualityComparisonConfig) -> dict[str, object]:
    rows = _load_rows(config.input_csv)
    _validate_config(config, rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = config.output_dir / "original"
    canny_dir = config.output_dir / "canny"
    teed_dir = config.output_dir / "teed"
    canny_mask_dir = config.output_dir / "canny_masks"
    teed_mask_dir = config.output_dir / "teed_masks"
    for path in (
        original_dir,
        canny_dir,
        teed_dir,
        canny_mask_dir,
        teed_mask_dir,
    ):
        path.mkdir(exist_ok=True)

    teed = FireSightRuntime(
        FireSightRuntimeConfig(
            backend="teed",
            checkpoint_path=config.checkpoint_path,
            width=config.width,
            height=config.height,
            threshold=config.threshold,
            background_scale=config.background_scale,
            edge_width=config.edge_width,
            device=config.device,
            profile_path=config.profile_path,
        ),
    )
    canny = FireSightRuntime(
        FireSightRuntimeConfig(
            backend="canny",
            width=config.width,
            height=config.height,
            background_scale=config.background_scale,
            edge_width=config.edge_width,
            profile_path=config.profile_path,
            canny_low_threshold=config.canny_low_threshold,
            canny_high_threshold=config.canny_high_threshold,
        ),
    )
    _warmup_runtimes(rows, config, teed, canny)
    records: list[QualityRecord] = []
    for row in rows:
        source_path = _workspace_absolute(row["original_image"])
        with Image.open(source_path) as source_image:
            source = source_image.convert("RGB")
        original = source.resize(
            (config.width, config.height),
            Image.Resampling.LANCZOS,
        )
        teed_result = teed.process_frame(source, row["input_file"])
        canny_result = canny.process_frame(source, row["input_file"])
        identifier = row["evaluation_id"]
        original_path = original_dir / f"{identifier}_original.png"
        canny_path = canny_dir / f"{identifier}_canny.png"
        teed_path = teed_dir / f"{identifier}_teed.png"
        canny_mask_path = canny_mask_dir / f"{identifier}_canny_mask.png"
        teed_mask_path = teed_mask_dir / f"{identifier}_teed_mask.png"
        original.save(original_path)
        canny_result.overlay.save(canny_path)
        teed_result.overlay.save(teed_path)
        canny_result.mask.save(canny_mask_path)
        teed_result.mask.save(teed_mask_path)
        ignored_regions = canny.ignored_regions_for(row["input_file"])
        record = _comparison_record(
            row,
            teed_result,
            canny_result,
            ignored_regions,
            QualityArtifactPaths(
                original=original_path,
                canny=canny_path,
                teed=teed_path,
                canny_mask=canny_mask_path,
                teed_mask=teed_mask_path,
            ),
        )
        records.append(record)
        original.close()
        source.close()
        teed_result.overlay.close()
        teed_result.mask.close()
        canny_result.overlay.close()
        canny_result.mask.close()

    comparison_path = config.output_dir / "quality_comparison.csv"
    _write_records(comparison_path, records)
    contact_sheet_path = config.output_dir / "quality_contact_sheet.png"
    _build_contact_sheet(records, contact_sheet_path, config.width, config.height)
    summary = _build_summary(config, records, contact_sheet_path, comparison_path)
    summary_path = config.output_dir / "quality_summary.json"
    _ = summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _warmup_runtimes(
    rows: list[dict[str, str]],
    config: QualityComparisonConfig,
    teed: FireSightRuntime,
    canny: FireSightRuntime,
) -> None:
    for row in rows[: config.warmup]:
        path = _workspace_absolute(row["original_image"])
        with Image.open(path) as source_image:
            source = source_image.convert("RGB")
        teed_result = teed.process_frame(source, row["input_file"])
        canny_result = canny.process_frame(source, row["input_file"])
        teed_result.overlay.close()
        teed_result.mask.close()
        canny_result.overlay.close()
        canny_result.mask.close()
        source.close()


def _comparison_record(
    row: dict[str, str],
    teed_result: FireSightFrameResult,
    canny_result: FireSightFrameResult,
    ignored_regions: tuple[HudIgnoreRegion, ...],
    paths: QualityArtifactPaths,
) -> QualityRecord:
    return {
        "evaluation_id": row["evaluation_id"],
        "input_file": row["input_file"],
        "ignore_fire360_osd": row["ignore_fire360_osd"].lower() == "true",
        "original_path": paths.original.as_posix(),
        "canny_path": paths.canny.as_posix(),
        "teed_path": paths.teed.as_posix(),
        "canny_mask_path": paths.canny_mask.as_posix(),
        "teed_mask_path": paths.teed_mask.as_posix(),
        "canny_latency_ms": canny_result.latency_ms,
        "teed_latency_ms": teed_result.latency_ms,
        "canny_edge_ratio": canny_result.edge_ratio,
        "teed_edge_ratio": teed_result.edge_ratio,
        "canny_edge_pixels": canny_result.edge_pixels,
        "teed_edge_pixels": teed_result.edge_pixels,
        "canny_osd_leak_pixels": _ignored_mask_pixels(
            canny_result.mask,
            ignored_regions,
        ),
        "teed_osd_leak_pixels": _ignored_mask_pixels(
            teed_result.mask,
            ignored_regions,
        ),
    }


def _ignored_mask_pixels(
    mask: Image.Image,
    regions: tuple[HudIgnoreRegion, ...],
) -> int:
    width, height = mask.size
    return sum(
        value == MASK_EDGE_VALUE
        for region in regions
        for value in mask.crop(
            (
                round(region.left * width),
                round(region.top * height),
                round(region.right * width),
                round(region.bottom * height),
            ),
        ).tobytes()
    )


def _write_records(path: Path, records: list[QualityRecord]) -> None:
    if len(records) == 0:
        message = "quality comparison produced no records"
        raise QualityComparisonError(message)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _build_contact_sheet(
    records: list[QualityRecord],
    output_path: Path,
    width: int,
    height: int,
) -> None:
    step = max(1, math.ceil(len(records) / CONTACT_SHEET_COUNT))
    selected = records[::step][:CONTACT_SHEET_COUNT]
    label_height = 28
    sheet = Image.new(
        "RGB",
        (width * 3, (height + label_height) * len(selected)),
        (18, 18, 18),
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, record in enumerate(selected):
        top = row_index * (height + label_height)
        paths = (
            ("original", record["original_path"]),
            ("Canny classical", record["canny_path"]),
            ("TEED learned", record["teed_path"]),
        )
        for column, (label, raw_path) in enumerate(paths):
            left = column * width
            draw_text = cast("Callable[..., None]", draw.text)
            draw_text(
                (left + 6, top + 7),
                f"{record['evaluation_id']} | {label}",
                fill="white",
                font=font,
            )
            with Image.open(Path(str(raw_path))) as image:
                sheet.paste(image.convert("RGB"), (left, top + label_height))
    sheet.save(output_path)


def _build_summary(
    config: QualityComparisonConfig,
    records: list[QualityRecord],
    contact_sheet_path: Path,
    comparison_path: Path,
) -> dict[str, object]:
    canny_latencies = [float(record["canny_latency_ms"]) for record in records]
    teed_latencies = [float(record["teed_latency_ms"]) for record in records]
    canny_ratios = [float(record["canny_edge_ratio"]) for record in records]
    teed_ratios = [float(record["teed_edge_ratio"]) for record in records]
    canny_leaks = [int(record["canny_osd_leak_pixels"]) for record in records]
    teed_leaks = [int(record["teed_osd_leak_pixels"]) for record in records]
    return {
        "protocol": PROTOCOL_NAME,
        "quality_status": "NOT_ESTABLISHED_WITHOUT_LABELS",
        "claim": (
            "Same fixed frames and source-aware OSD profile at 320x240; "
            "edge density and latency are diagnostics, not smoke-domain accuracy."
        ),
        "frames": len(records),
        "target": {
            "width": config.width,
            "height": config.height,
            "threshold": config.threshold,
            "canny_low": config.canny_low_threshold,
            "canny_high": config.canny_high_threshold,
        },
        "results": {
            "canny": {
                "latency_ms": _summarize(canny_latencies),
                "edge_ratio": _summarize(canny_ratios),
                "nonempty_frames": sum(value > 0.0 for value in canny_ratios),
                "osd_leak_frames": sum(value > 0 for value in canny_leaks),
                "osd_leak_pixels": sum(canny_leaks),
            },
            "teed": {
                "latency_ms": _summarize(teed_latencies),
                "edge_ratio": _summarize(teed_ratios),
                "nonempty_frames": sum(value > 0.0 for value in teed_ratios),
                "osd_leak_frames": sum(value > 0 for value in teed_leaks),
                "osd_leak_pixels": sum(teed_leaks),
            },
        },
        "winner": None,
        "interpretation": (
            "Canny may win latency, but this artifact does not establish a quality "
            "winner. Human 1-5 scoring and smoke-domain labels remain required."
        ),
        "artifacts": {
            "comparison_csv": comparison_path.as_posix(),
            "contact_sheet": contact_sheet_path.as_posix(),
        },
    }


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p50_index = min(len(ordered) - 1, max(0, math.ceil(0.50 * len(ordered)) - 1))
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": statistics.mean(values),
        "p50": ordered[p50_index],
        "p95": ordered[p95_index],
        "max": max(values),
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        message = f"input evaluation CSV is missing: {path}"
        raise QualityComparisonError(message)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]
    if len(rows) != EXPECTED_FRAME_COUNT:
        message = f"expected {EXPECTED_FRAME_COUNT} input rows, got {len(rows)}"
        raise QualityComparisonError(message)
    return rows


def _validate_config(
    config: QualityComparisonConfig,
    rows: list[dict[str, str]],
) -> None:
    if not config.checkpoint_path.is_file():
        message = f"missing TEED checkpoint: {config.checkpoint_path}"
        raise QualityComparisonError(message)
    if not config.profile_path.is_file():
        message = f"missing Fire360 OSD profile: {config.profile_path}"
        raise QualityComparisonError(message)
    if config.width < MIN_QUALITY_DIMENSION or config.height < MIN_QUALITY_DIMENSION:
        message = "quality dimensions must be at least 64"
        raise QualityComparisonError(message)
    if not math.isfinite(config.threshold) or not 0.0 <= config.threshold <= 1.0:
        message = "threshold must be between 0 and 1"
        raise QualityComparisonError(message)
    if config.warmup < 0:
        message = "warmup must be non-negative"
        raise QualityComparisonError(message)
    for row in rows:
        path = _workspace_absolute(row["original_image"])
        if not path.is_file():
            message = f"input artifact is missing: {path}"
            raise QualityComparisonError(message)


def _workspace_absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    raise SystemExit(main())
