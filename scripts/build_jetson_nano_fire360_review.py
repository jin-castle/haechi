from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from PIL import Image, ImageDraw, ImageFont


class _LatencySummary(TypedDict):
    mean: float


class _Metrics(TypedDict):
    processed_frames: int
    ignored_region_count: int
    total_latency_ms: _LatencySummary


@dataclass(frozen=True, slots=True)
class _ReviewSource:
    label: str
    key: str
    source_path: Path


@dataclass(frozen=True, slots=True)
class _ReviewRow:
    label: str
    source: Image.Image
    overlay: Image.Image
    metrics: _Metrics
    passed: bool


OUTPUT_ROOT = Path("outputs/jetson_nano_fire360_qa")
OUTPUT_PATH = Path("jetson_nano_fire360_review.png")
FRAME_SIZE = (320, 240)
RIGHT_OSD_BOX = (240, 0, 289, 217)
LEFT_OSD_BOX = (42, 173, 87, 240)


def _sources() -> tuple[_ReviewSource, ...]:
    frame_root = Path("data/fire360/frames_multiclip")
    return (
        _ReviewSource(
            "02814",
            "clip_02814",
            frame_root / "clip_02814/clip_02814_frame_000143.jpg",
        ),
        _ReviewSource(
            "02815",
            "clip_02815",
            frame_root / "clip_02815/clip_02815_frame_000068.jpg",
        ),
        _ReviewSource(
            "02824",
            "clip_02824",
            frame_root / "clip_02824/clip_02824_frame_000115.jpg",
        ),
        _ReviewSource(
            "GOPR8356",
            "gopr8356",
            frame_root / "gopr8356/gopr8356_frame_000227.jpg",
        ),
        _ReviewSource(
            "IFSI Video 4",
            "ifsi_video_4",
            frame_root / "ifsi_video_4/ifsi_video_4_frame_000284.jpg",
        ),
        _ReviewSource(
            "IFSI Video 8",
            "ifsi_video_8",
            frame_root / "ifsi_video_8/ifsi_video_8_frame_000300.jpg",
        ),
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{filename}", size)


def _count_green(image: Image.Image, box: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = box
    return sum(
        image.getpixel((x, y)) == (0, 255, 0)
        for y in range(top, bottom)
        for x in range(left, right)
    )


def _load_row(source: _ReviewSource) -> _ReviewRow:
    with Image.open(source.source_path) as input_image:
        original = input_image.convert("RGB").resize(
            FRAME_SIZE,
            Image.Resampling.LANCZOS,
        )
    with Image.open(OUTPUT_ROOT / f"{source.key}_teed_profiled.png") as output_image:
        overlay = output_image.convert("RGB")
    metrics = cast(
        "_Metrics",
        cast(
            "object",
            json.loads(
                (OUTPUT_ROOT / f"{source.key}_profiled_metrics.json").read_text(
                    encoding="utf-8",
                ),
            ),
        ),
    )
    ignored = metrics["ignored_region_count"]
    osd_is_clean = ignored == 0 or (
        _count_green(overlay, RIGHT_OSD_BOX) == 0
        and _count_green(overlay, LEFT_OSD_BOX) == 0
    )
    return _ReviewRow(
        label=source.label,
        source=original,
        overlay=overlay,
        metrics=metrics,
        passed=metrics["processed_frames"] == 1 and osd_is_clean,
    )


def build_review_png() -> Path:
    rows = tuple(_load_row(source) for source in _sources())
    canvas = Image.new("RGB", (720, 1907), (20, 23, 27))
    draw = ImageDraw.Draw(canvas)
    body_font = _font(18)
    small_font = _font(14)
    draw.text(
        (20, 16),
        "Jetson Nano TEED - Fire360 QA",
        font=_font(28, bold=True),
        fill=(245, 245, 245),
    )
    draw.text((20, 56), "ORIGINAL FIRE360", font=body_font, fill=(170, 190, 210))
    draw.text((380, 56), "TEED EDGE OVERLAY", font=body_font, fill=(170, 190, 210))

    for index, row in enumerate(rows):
        y = 92 + index * 292
        status = "PASS" if row.passed else "REVIEW"
        status_color = (28, 120, 72) if row.passed else (160, 70, 40)
        detail_color = (135, 220, 175) if row.passed else (255, 150, 100)
        draw.text((20, y), row.label, font=body_font, fill=(255, 220, 90))
        draw.rounded_rectangle(
            (610, y - 2, 695, y + 22),
            radius=7,
            fill=status_color,
        )
        draw.text((628, y + 1), status, font=small_font, fill=(255, 255, 255))
        canvas.paste(row.source, (20, y + 30))
        canvas.paste(row.overlay, (380, y + 30))
        profile_text = (
            "OSD edge regions masked"
            if row.metrics["ignored_region_count"]
            else "No fixed OSD profile"
        )
        latency = row.metrics["total_latency_ms"]["mean"]
        draw.text(
            (380, y + 272),
            f"{profile_text} | total {latency:.1f} ms",
            font=small_font,
            fill=detail_color,
        )

    passed_count = sum(row.passed for row in rows)
    result_color = (90, 235, 150) if passed_count == len(rows) else (255, 150, 100)
    draw.text(
        (20, 1857),
        (
            f"RESULT: {passed_count}/{len(rows)} PASS | CPU functional QA; "
            "Jetson CUDA timing requires the board."
        ),
        font=body_font,
        fill=result_color,
    )
    canvas.save(OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> int:
    print(build_review_png().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
