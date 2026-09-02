from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / ".omo/evidence/firesight-smoke-vision/model-comparison-pidinet/"
    / "learned_model_benchmark.json"
)
DEFAULT_EVALUATION_CSV = (
    PROJECT_ROOT
    / ".omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / ".omo/evidence/firesight-smoke-vision/jetson-latency-projection"
)


class ProjectionError(Exception):
    pass


class _VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: object) -> None: ...

    def release(self) -> None: ...


class _OpenCv(Protocol):
    COLOR_RGB2BGR: int

    def VideoWriter(
        self, filename: str, fourcc: int, fps: float, size: tuple[int, int]
    ) -> _VideoWriter: ...

    def VideoWriter_fourcc(self, *codes: str) -> int: ...

    def cvtColor(self, frame: object, code: int) -> object: ...


class _Numpy(Protocol):
    def asarray(self, value: object) -> object: ...


@dataclass(frozen=True, slots=True)
class Projection:
    pc_cpu_p95_ms: float
    pc_cpu_mean_ms: float
    jetson_cpu_only_low_ms: float = 35.0
    jetson_cpu_only_high_ms: float = 70.0
    jetson_cuda_low_ms: float = 10.0
    jetson_cuda_high_ms: float = 25.0
    jetson_tensorrt_low_ms: float = 5.0
    jetson_tensorrt_high_ms: float = 15.0
    selected_low_ms: float = 8.0
    selected_high_ms: float = 25.0
    nominal_ms: float = 15.0
    conservative_review_ms: float = 30.0
    target_p95_ms: float = 80.0
    target_fps: float = 15.0


class ProjectionArgNamespace(argparse.Namespace):
    benchmark_path: Path = DEFAULT_BENCHMARK
    evaluation_csv: Path = DEFAULT_EVALUATION_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR
    fps: float = 15.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an explicitly labeled Jetson Orin Nano TEED latency projection "
            "plot and video."
        ),
    )
    _ = parser.add_argument(
        "--benchmark",
        dest="benchmark_path",
        default=DEFAULT_BENCHMARK,
        type=Path,
    )
    _ = parser.add_argument(
        "--evaluation-csv",
        default=DEFAULT_EVALUATION_CSV,
        type=Path,
    )
    _ = parser.add_argument(
        "--out-dir",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
    )
    _ = parser.add_argument("--fps", default=15.0, type=float)
    args = ProjectionArgNamespace()
    _ = parser.parse_args(argv, namespace=args)

    try:
        payload = build_projection(
            benchmark_path=_workspace_absolute(args.benchmark_path),
            evaluation_csv=_workspace_absolute(args.evaluation_csv),
            output_dir=_workspace_absolute(args.output_dir),
            fps=args.fps,
        )
    except ProjectionError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2

    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def build_projection(
    *,
    benchmark_path: Path,
    evaluation_csv: Path,
    output_dir: Path,
    fps: float = 15.0,
) -> dict[str, object]:
    if fps <= 0.0:
        raise ProjectionError("fps must be positive")

    benchmark = _load_json(benchmark_path)
    measured = _get_320x240_teed_result(benchmark)
    projection = Projection(
        pc_cpu_p95_ms=float(measured["p95_ms"]),
        pc_cpu_mean_ms=float(measured["mean_ms"]),
        target_fps=fps,
    )
    rows = _load_evaluation_rows(evaluation_csv)
    _validate_evaluation_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "teed_jetson_latency_projection.png"
    video_path = output_dir / "teed_jetson_latency_projection.mp4"
    report_path = output_dir / "teed_jetson_latency_projection.md"
    json_path = output_dir / "teed_jetson_latency_projection.json"

    plot = _render_projection_plot(
        projection, current_frame=None, frame_count=len(rows)
    )
    plot.save(plot_path)
    _build_video(
        rows,
        projection,
        output_path=video_path,
        fps=fps,
    )

    payload = _build_payload(
        benchmark_path=benchmark_path,
        evaluation_csv=evaluation_csv,
        output_dir=output_dir,
        projection=projection,
        rows=rows,
        plot_path=plot_path,
        video_path=video_path,
        report_path=report_path,
    )
    _ = json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _ = report_path.write_text(_build_report(payload), encoding="utf-8")
    return payload


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ProjectionError(f"benchmark JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectionError(f"unable to read benchmark JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ProjectionError("benchmark JSON root must be an object")
    return cast("dict[str, object]", payload)


def _get_320x240_teed_result(benchmark: dict[str, object]) -> dict[str, float]:
    raw_results = benchmark.get("results")
    if not isinstance(raw_results, list):
        raise ProjectionError("benchmark JSON has no results list")
    result_values = cast("list[object]", raw_results)
    for result_value in result_values:
        if not isinstance(result_value, dict):
            continue
        result = cast("dict[str, object]", result_value)
        if result.get("resolution") != [320, 240]:
            continue
        raw_teed_value = result.get("teed")
        if not isinstance(raw_teed_value, dict):
            break
        raw_teed = cast("dict[str, object]", raw_teed_value)
        mean_ms = raw_teed.get("mean_ms")
        p95_ms = raw_teed.get("p95_ms")
        if not isinstance(mean_ms, int | float) or not isinstance(p95_ms, int | float):
            break
        return {
            "mean_ms": float(mean_ms),
            "p95_ms": float(p95_ms),
        }
    raise ProjectionError("benchmark JSON has no numeric 320x240 TEED result")


def _load_evaluation_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ProjectionError(f"evaluation CSV is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]
    except OSError as error:
        raise ProjectionError(f"unable to read evaluation CSV: {path}") from error
    return rows


def _validate_evaluation_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ProjectionError("evaluation CSV has no frame rows")
    required = ("evaluation_id", "video_key", "original_image", "teed_image")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ProjectionError(
            f"evaluation CSV is missing columns: {', '.join(missing)}"
        )
    for row in rows:
        for key in ("original_image", "teed_image"):
            path = _workspace_absolute(Path(row[key]))
            if not path.is_file():
                raise ProjectionError(f"evaluation artifact is missing: {path}")


def _render_projection_plot(
    projection: Projection,
    *,
    current_frame: int | None,
    frame_count: int,
) -> Image.Image:
    width, height = 1400, 900
    background = (17, 22, 29)
    panel = (27, 34, 43)
    white = (239, 244, 248)
    muted = (166, 179, 190)
    grid = (64, 76, 89)
    green = (70, 190, 128)
    amber = (235, 176, 72)
    blue = (82, 157, 235)
    purple = (176, 117, 226)
    red = (232, 103, 103)
    canvas = Image.new("RGB", (width, height), background)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)
    font = _font(22)
    small = _font(17)
    title = _font(30)
    _draw_text(
        draw,
        (48, 30),
        "TEED latency projection for Jetson Orin Nano",
        fill=white,
        font=title,
    )
    _draw_text(
        draw,
        (48, 70),
        "320x240 | p95 planning view | ESTIMATE ONLY, NO JETSON MEASUREMENT",
        fill=amber,
        font=font,
    )

    cards = [
        ("PC CPU measured p95", f"{projection.pc_cpu_p95_ms:.1f} ms", blue),
        (
            "Jetson selected range",
            f"{projection.selected_low_ms:.0f}-{projection.selected_high_ms:.0f} ms",
            green,
        ),
        ("Nominal estimate", f"{projection.nominal_ms:.0f} ms", purple),
        ("15 FPS frame budget", f"{_frame_budget_ms(projection):.1f} ms", amber),
        ("Target p95", f"<= {projection.target_p95_ms:.0f} ms", red),
    ]
    card_width = 252
    for index, (label, value, color) in enumerate(cards):
        left = 48 + index * (card_width + 15)
        draw.rounded_rectangle(
            (left, 118, left + card_width, 192),
            radius=10,
            fill=panel,
            outline=color,
            width=2,
        )
        _draw_text(draw, (left + 12, 130), label, fill=muted, font=small)
        _draw_text(draw, (left + 12, 158), value, fill=white, font=font)

    x0, x1 = 290, 1325
    axis_max = 90.0
    y_rows = [255, 305, 355, 405, 455]
    scenarios = [
        ("PC CPU measured p95", 0.0, projection.pc_cpu_p95_ms, blue),
        (
            "Jetson CPU-only",
            projection.jetson_cpu_only_low_ms,
            projection.jetson_cpu_only_high_ms,
            amber,
        ),
        (
            "Jetson CUDA PyTorch",
            projection.jetson_cuda_low_ms,
            projection.jetson_cuda_high_ms,
            purple,
        ),
        (
            "Jetson TensorRT FP16",
            projection.jetson_tensorrt_low_ms,
            projection.jetson_tensorrt_high_ms,
            green,
        ),
        (
            "Selected planning range",
            projection.selected_low_ms,
            projection.selected_high_ms,
            green,
        ),
    ]
    _draw_text(draw, (48, 218), "Scenario p95 ranges (ms)", fill=white, font=font)
    for tick in range(0, 91, 10):
        x = _scale_x(float(tick), x0, x1, axis_max)
        draw.line((x, 245, x, 478), fill=grid, width=1)
        _draw_text(draw, (x - 7, 482), str(tick), fill=muted, font=small)
    for y, (label, low, high, color) in zip(y_rows, scenarios, strict=True):
        _draw_text(draw, (48, y - 12), label, fill=white, font=small)
        start = _scale_x(low, x0, x1, axis_max)
        end = _scale_x(high, x0, x1, axis_max)
        draw.rounded_rectangle(
            (start, y - 10, max(start + 4, end), y + 10), radius=7, fill=color
        )
        if low == 0.0:
            draw.line((end, y - 18, end, y + 18), fill=white, width=3)
        else:
            draw.line((start, y - 18, start, y + 18), fill=color, width=3)
            draw.line((end, y - 18, end, y + 18), fill=color, width=3)
        _draw_text(
            draw,
            (min(end + 10, 1260), y - 12),
            f"{low:.0f}-{high:.1f}",
            fill=white,
            font=small,
        )
    for value, label, color in (
        (_frame_budget_ms(projection), "15 FPS budget", amber),
        (projection.target_p95_ms, "target", red),
    ):
        x = _scale_x(value, x0, x1, axis_max)
        draw.line((x, 238, x, 475), fill=color, width=2)
        _draw_text(draw, (x - 28, 520), label, fill=color, font=small)

    chart_top, chart_bottom = 585, 805
    _draw_text(
        draw,
        (48, 545),
        "Per-frame planning envelope (not measured trace)",
        fill=white,
        font=font,
    )
    plot_x0, plot_x1 = 100, 1325
    plot_y0, plot_y1 = chart_bottom, chart_top
    for value in (0.0, 15.0, 25.0, 50.0, 80.0):
        y = _scale_y(value, plot_y0, plot_y1, 90.0)
        draw.line((plot_x0, y, plot_x1, y), fill=grid, width=1)
        _draw_text(draw, (48, y - 10), f"{value:.0f}", fill=muted, font=small)
    band_low_y = _scale_y(projection.selected_low_ms, plot_y0, plot_y1, 90.0)
    band_high_y = _scale_y(projection.selected_high_ms, plot_y0, plot_y1, 90.0)
    draw.rectangle((plot_x0, band_high_y, plot_x1, band_low_y), fill=(29, 83, 66))
    nominal_y = _scale_y(projection.nominal_ms, plot_y0, plot_y1, 90.0)
    draw.line((plot_x0, nominal_y, plot_x1, nominal_y), fill=purple, width=3)
    measured_y = _scale_y(projection.pc_cpu_p95_ms, plot_y0, plot_y1, 90.0)
    draw.line((plot_x0, measured_y, plot_x1, measured_y), fill=blue, width=2)
    target_y = _scale_y(projection.target_p95_ms, plot_y0, plot_y1, 90.0)
    draw.line((plot_x0, target_y, plot_x1, target_y), fill=red, width=2)
    for frame in range(frame_count):
        x = _scale_x_frame(frame, frame_count, plot_x0, plot_x1)
        if frame % 5 == 0 or frame == frame_count - 1:
            _draw_text(
                draw, (x - 8, chart_bottom + 12), str(frame + 1), fill=muted, font=small
            )
    if current_frame is not None:
        current_x = _scale_x_frame(current_frame, frame_count, plot_x0, plot_x1)
        draw.line((current_x, plot_y1, current_x, plot_y0), fill=white, width=3)
        draw.ellipse(
            (current_x - 7, nominal_y - 7, current_x + 7, nominal_y + 7), fill=white
        )
        _draw_text(
            draw,
            (max(plot_x0, current_x - 42), chart_top - 42),
            f"F{current_frame + 1:02d}",
            fill=white,
            font=small,
        )

    legend_y = 848
    _draw_text(
        draw,
        (48, legend_y),
        "green band: 8-25 ms selected planning range",
        fill=green,
        font=small,
    )
    _draw_text(draw, (440, legend_y), "purple: nominal 15 ms", fill=purple, font=small)
    _draw_text(draw, (700, legend_y), "blue: PC CPU p95 22.8 ms", fill=blue, font=small)
    _draw_text(draw, (1030, legend_y), "red: 80 ms target", fill=red, font=small)
    return canvas


def _build_video(
    rows: list[dict[str, str]],
    projection: Projection,
    *,
    output_path: Path,
    fps: float,
) -> None:
    try:
        cv2 = cast("_OpenCv", cast("object", importlib.import_module("cv2")))
        numpy = cast("_Numpy", cast("object", importlib.import_module("numpy")))
    except ImportError as error:
        raise ProjectionError(
            "video output needs OpenCV; run with `uv run --with opencv-python-headless ...`",
        ) from error

    width, height = 1600, 900
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: _VideoWriter = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ProjectionError(f"unable to open video writer: {output_path}")
    try:
        for index, row in enumerate(rows):
            frame = _render_video_frame(
                row,
                projection,
                frame_index=index,
                frame_count=len(rows),
                size=(width, height),
            )
            frame_array = numpy.asarray(frame)
            writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _render_video_frame(
    row: dict[str, str],
    projection: Projection,
    *,
    frame_index: int,
    frame_count: int,
    size: tuple[int, int],
) -> Image.Image:
    canvas = Image.new("RGB", size, (17, 22, 29))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    font = _font(21)
    small = _font(17)
    _draw_text(
        draw,
        (35, 20),
        f"TEED Fire360 evaluation frame {row['evaluation_id']} | {row['video_key']}",
        fill=(239, 244, 248),
        font=title_font,
    )
    _draw_text(
        draw,
        (35, 58),
        "PC CPU p95 measured: 22.8 ms | Jetson Orin Nano: ESTIMATED 8-25 ms | nominal 15 ms",
        fill=(235, 176, 72),
        font=font,
    )
    image_width, image_height = 760, 428
    original = _open_rgb(_workspace_absolute(Path(row["original_image"])))
    teed = _open_rgb(_workspace_absolute(Path(row["teed_image"])))
    try:
        original_preview = ImageOps.contain(original, (image_width, image_height))
        teed_preview = ImageOps.contain(teed, (image_width, image_height))
    finally:
        original.close()
        teed.close()
    canvas.paste(original_preview, (35, 105))
    canvas.paste(teed_preview, (805, 105))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (35, 105, 35 + image_width, 105 + image_height), outline=(82, 157, 235), width=3
    )
    draw.rectangle(
        (805, 105, 805 + image_width, 105 + image_height),
        outline=(70, 190, 128),
        width=3,
    )
    _draw_text(draw, (52, 122), "Original", fill=(239, 244, 248), font=font)
    _draw_text(draw, (822, 122), "TEED overlay", fill=(239, 244, 248), font=font)

    plot = _render_projection_plot(
        projection,
        current_frame=frame_index,
        frame_count=frame_count,
    )
    plot_preview = ImageOps.contain(plot, (760, 300))
    canvas.paste(plot_preview, (35, 570))
    draw = ImageDraw.Draw(canvas)
    _draw_text(
        draw,
        (825, 610),
        "STATUS: ESTIMATE ONLY / NOT HARDWARE DATA",
        fill=(232, 103, 103),
        font=font,
    )
    details = [
        f"frame {frame_index + 1}/{frame_count} at {projection.target_fps:.0f} FPS",
        f"frame period: {_frame_budget_ms(projection):.1f} ms",
        f"selected p95 range: {projection.selected_low_ms:.0f}-{projection.selected_high_ms:.0f} ms",
        f"conservative review: {projection.conservative_review_ms:.0f} ms",
        f"target p95: <= {projection.target_p95_ms:.0f} ms",
        "model-only estimate; capture/decode/overlay not included",
    ]
    for line_index, line in enumerate(details):
        _draw_text(
            draw,
            (825, 655 + line_index * 30),
            line,
            fill=(239, 244, 248),
            font=small,
        )
    return canvas


def _build_payload(
    *,
    benchmark_path: Path,
    evaluation_csv: Path,
    output_dir: Path,
    projection: Projection,
    rows: list[dict[str, str]],
    plot_path: Path,
    video_path: Path,
    report_path: Path,
) -> dict[str, object]:
    return {
        "protocol": "teed_jetson_orin_nano_latency_projection_v1",
        "status": "ESTIMATE_NOT_MEASURED",
        "claim": "Jetson Orin Nano values are planning estimates, not hardware measurements.",
        "measurement": {
            "device": "development_pc_cpu",
            "resolution": [320, 240],
            "frames": 30,
            "teed_mean_ms": projection.pc_cpu_mean_ms,
            "teed_p95_ms": projection.pc_cpu_p95_ms,
            "benchmark_json": _workspace_relative(benchmark_path),
        },
        "projection": {
            "device": "jetson_orin_nano",
            "resolution": [320, 240],
            "target_fps": projection.target_fps,
            "frame_budget_ms": _frame_budget_ms(projection),
            "scenarios": {
                "cpu_only_p95_ms": [
                    projection.jetson_cpu_only_low_ms,
                    projection.jetson_cpu_only_high_ms,
                ],
                "cuda_pytorch_p95_ms": [
                    projection.jetson_cuda_low_ms,
                    projection.jetson_cuda_high_ms,
                ],
                "tensorrt_fp16_p95_ms": [
                    projection.jetson_tensorrt_low_ms,
                    projection.jetson_tensorrt_high_ms,
                ],
            },
            "selected_planning_range_p95_ms": [
                projection.selected_low_ms,
                projection.selected_high_ms,
            ],
            "nominal_p95_ms": projection.nominal_ms,
            "conservative_review_p95_ms": projection.conservative_review_ms,
            "target_p95_ms": projection.target_p95_ms,
            "model_only_within_target_at_conservative_review": (
                projection.conservative_review_ms <= projection.target_p95_ms
            ),
        },
        "assumptions": [
            "320x240 input and warmed-up inference",
            "selected range assumes Jetson CUDA/TensorRT-style execution for a tiny TEED model",
            "nominal value is a planning point, not a predicted guarantee",
            "latency excludes camera capture, decode, resize, overlay, encoding, and scheduling",
            "TensorRT FP16 conversion and operator support are not yet verified",
        ],
        "evaluation_video": {
            "frames": len(rows),
            "fps": projection.target_fps,
            "evaluation_csv": _workspace_relative(evaluation_csv),
        },
        "artifacts": {
            "plot_png": _workspace_relative(plot_path),
            "video_mp4": _workspace_relative(video_path),
            "report_markdown": _workspace_relative(report_path),
            "output_dir": _workspace_relative(output_dir),
        },
        "official_hardware_context": {
            "jetson_orin_nano_user_guide": (
                "https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/index.html"
            ),
            "jetson_power_performance_guide": (
                "https://docs.nvidia.com/jetson/archives/r36.4.4/"
                "DeveloperGuide/SD/PlatformPowerAndPerformance/"
                "JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html"
            ),
        },
    }


def _build_report(payload: dict[str, object]) -> str:
    measurement = payload["measurement"]
    projection = payload["projection"]
    if not isinstance(measurement, dict) or not isinstance(projection, dict):
        raise ProjectionError("projection payload has invalid report sections")
    return (
        "# TEED Jetson Orin Nano latency projection\n\n"
        "Status: **ESTIMATE_NOT_MEASURED**\n\n"
        "This artifact projects the 320x240 TEED p95 result from the development PC "
        "onto a Jetson Orin Nano planning envelope. It is not a Jetson benchmark.\n\n"
        "## Reference measurement\n\n"
        f"- Device: `{measurement['device']}`\n"
        f"- Resolution: `{measurement['resolution'][0]}x{measurement['resolution'][1]}`\n"
        f"- TEED mean: `{measurement['teed_mean_ms']:.1f} ms`\n"
        f"- TEED p95: `{measurement['teed_p95_ms']:.1f} ms`\n\n"
        "## Planning estimate\n\n"
        "- Selected p95 range: **8-25 ms**\n"
        "- Nominal planning value: **15 ms**\n"
        "- Conservative review value: **30 ms**\n"
        f"- 15 FPS frame period: **{projection['frame_budget_ms']:.1f} ms**\n"
        f"- MVP model-only target: **<= {projection['target_p95_ms']:.0f} ms p95**\n\n"
        "The 30 ms conservative review value remains below the 80 ms model-only target, "
        "but it does not establish end-to-end readiness. Camera capture, decode, resize, "
        "preprocessing, post-processing, overlay, encoding, and runtime scheduling still "
        "need to be measured on the actual board.\n\n"
        "## Scenario ranges\n\n"
        "| Execution path | Planning p95 range |\n"
        "| --- | ---: |\n"
        "| Jetson CPU-only | 35-70 ms |\n"
        "| Jetson CUDA PyTorch | 10-25 ms |\n"
        "| Jetson TensorRT FP16 | 5-15 ms |\n"
        "| Selected planning range | **8-25 ms** |\n\n"
        "The range is an engineering inference from the PC CPU reference and the expected "
        "execution path. TOPS alone is not a valid latency conversion for this small model "
        "because framework, launch, memory-transfer, and preprocessing overhead can dominate.\n\n"
        "## Artifacts\n\n"
        "- Plot: `teed_jetson_latency_projection.png`\n"
        "- Annotated evaluation video: `teed_jetson_latency_projection.mp4`\n"
        "- Machine-readable payload: `teed_jetson_latency_projection.json`\n\n"
        "## Official hardware context\n\n"
        "- [Jetson Orin Nano Developer Kit User Guide]"
        "(https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/index.html)\n"
        "- [NVIDIA Jetson Orin power and performance guide]"
        "(https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/"
        "PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)\n"
    )


def _open_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as error:
        raise ProjectionError(f"unable to open image: {path}") from error


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    **kwargs: object,
) -> None:
    draw_text = cast("Callable[..., None]", draw.text)
    draw_text(xy, text, **kwargs)


def _scale_x(value: float, x0: int, x1: int, axis_max: float) -> int:
    return round(x0 + (value / axis_max) * (x1 - x0))


def _scale_y(value: float, y0: int, y1: int, axis_max: float) -> int:
    return round(y0 - (value / axis_max) * (y0 - y1))


def _scale_x_frame(index: int, frame_count: int, x0: int, x1: int) -> int:
    if frame_count <= 1:
        return x0
    return round(x0 + (index / (frame_count - 1)) * (x1 - x0))


def _workspace_absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _frame_budget_ms(projection: Projection) -> float:
    return 1000.0 / projection.target_fps


def _workspace_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
