from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from PIL import Image

from firesight_vision.canny import CannyError
from firesight_vision.deployment import (
    DeploymentBackend,
    FireSightDeploymentError,
    FireSightRuntime,
    FireSightRuntimeConfig,
)
from firesight_vision.teed import TeedError

if TYPE_CHECKING:
    from collections.abc import Sequence


class BenchmarkArgNamespace(argparse.Namespace):
    input_csv: Path = Path(
        ".omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv",
    )
    output_path: Path = Path()
    checkpoint_path: Path = Path("models/teed/5_model.pth")
    profile_path: Path = Path("data/fire360/video_profiles.json")
    backends: str = "teed,canny"
    width: int = 320
    height: int = 240
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "cpu"
    warmup: int = 2
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150


@dataclass(frozen=True, slots=True)
class BackendBenchmarkConfig:
    input_csv: Path
    output_path: Path
    checkpoint_path: Path
    profile_path: Path
    backends: tuple[str, ...] = ("teed", "canny")
    width: int = 320
    height: int = 240
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "cpu"
    warmup: int = 2
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150


class BackendBenchmarkError(Exception):
    pass


PROTOCOL_NAME: Final = "firesight_deployment_backend_benchmark_v1"
EXPECTED_FRAME_COUNT: Final = 30


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_benchmark(config)
    except (
        BackendBenchmarkError,
        CannyError,
        FireSightDeploymentError,
        TeedError,
    ) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> BackendBenchmarkConfig:
    namespace = BenchmarkArgNamespace()
    parser = argparse.ArgumentParser(
        description="Benchmark TEED and Canny through the deployment runtime.",
    )
    _ = parser.add_argument("--input-csv", default=namespace.input_csv, type=Path)
    _ = parser.add_argument("--out", dest="output_path", required=True, type=Path)
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
    _ = parser.add_argument("--backends", default=namespace.backends)
    _ = parser.add_argument("--width", default=320, type=int)
    _ = parser.add_argument("--height", default=240, type=int)
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=1, type=int)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.add_argument("--warmup", default=2, type=int)
    _ = parser.add_argument("--canny-low", default=50, type=int)
    _ = parser.add_argument("--canny-high", default=150, type=int)
    _ = parser.parse_args(argv, namespace=namespace)
    return BackendBenchmarkConfig(
        input_csv=namespace.input_csv,
        output_path=namespace.output_path,
        checkpoint_path=namespace.checkpoint_path,
        profile_path=namespace.profile_path,
        backends=_parse_backends(namespace.backends),
        width=namespace.width,
        height=namespace.height,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
        warmup=namespace.warmup,
        canny_low_threshold=namespace.canny_low_threshold,
        canny_high_threshold=namespace.canny_high_threshold,
    )


def run_benchmark(config: BackendBenchmarkConfig) -> dict[str, object]:
    rows = _load_rows(config.input_csv)
    _validate_config(config, rows)
    images = _load_images(rows)
    results: dict[str, object] = {}
    try:
        for backend in config.backends:
            runtime = FireSightRuntime(
                FireSightRuntimeConfig(
                    backend=cast("DeploymentBackend", backend),
                    checkpoint_path=(
                        config.checkpoint_path if backend == "teed" else None
                    ),
                    width=config.width,
                    height=config.height,
                    threshold=config.threshold,
                    background_scale=config.background_scale,
                    edge_width=config.edge_width,
                    device=config.device if backend == "teed" else "cpu",
                    canny_low_threshold=config.canny_low_threshold,
                    canny_high_threshold=config.canny_high_threshold,
                    profile_path=config.profile_path,
                ),
            )
            for image, row in images[: config.warmup]:
                _ = runtime.process_frame(image, row["input_file"])
            latencies: list[float] = []
            edge_ratios: list[float] = []
            for image, row in images:
                result = runtime.process_frame(image, row["input_file"])
                latencies.append(result.latency_ms)
                edge_ratios.append(result.edge_ratio)
            results[backend] = {
                "device": runtime.device,
                "frames": len(latencies),
                "latency_ms": _summarize(latencies),
                "edge_ratio": _summarize(edge_ratios),
                "canny_thresholds": {
                    "low": config.canny_low_threshold,
                    "high": config.canny_high_threshold,
                }
                if backend == "canny"
                else None,
            }
    finally:
        for image, _ in images:
            image.close()
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": (
            "Same fixed Fire360 frames and same 320x240 deployment path; "
            "latency is a development-machine reference, not a Jetson result."
        ),
        "input_csv": config.input_csv.as_posix(),
        "profile_path": config.profile_path.as_posix(),
        "frames": len(rows),
        "warmup_frames": config.warmup,
        "target": {
            "width": config.width,
            "height": config.height,
            "threshold": config.threshold,
            "background_scale": config.background_scale,
            "edge_width": config.edge_width,
        },
        "checkpoints": {"teed": config.checkpoint_path.as_posix()},
        "results": results,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = config.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BackendBenchmarkError(f"input evaluation CSV is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]
    except OSError as error:
        message = f"unable to read input evaluation CSV: {path}"
        raise BackendBenchmarkError(message) from error
    if len(rows) != EXPECTED_FRAME_COUNT:
        message = f"expected {EXPECTED_FRAME_COUNT} input rows, got {len(rows)}"
        raise BackendBenchmarkError(message)
    return rows


def _load_images(
    rows: list[dict[str, str]],
) -> list[tuple[Image.Image, dict[str, str]]]:
    images: list[tuple[Image.Image, dict[str, str]]] = []
    try:
        for row in rows:
            path = _workspace_absolute(row["original_image"])
            with Image.open(path) as source:
                images.append((source.convert("RGB"), row))
    except OSError as error:
        for image, _ in images:
            image.close()
        raise BackendBenchmarkError("unable to load fixed evaluation frames") from error
    return images


def _validate_config(
    config: BackendBenchmarkConfig,
    rows: list[dict[str, str]],
) -> None:
    if not config.checkpoint_path.is_file() and "teed" in config.backends:
        message = f"missing TEED checkpoint: {config.checkpoint_path}"
        raise BackendBenchmarkError(message)
    if not config.profile_path.is_file():
        message = f"missing Fire360 OSD profile: {config.profile_path}"
        raise BackendBenchmarkError(message)
    if config.width < 64 or config.height < 64:
        raise BackendBenchmarkError("benchmark dimensions must be at least 64")
    if not math.isfinite(config.threshold) or not 0.0 <= config.threshold <= 1.0:
        raise BackendBenchmarkError("threshold must be between 0 and 1")
    if config.warmup < 0:
        raise BackendBenchmarkError("warmup must be non-negative")
    for row in rows:
        path = _workspace_absolute(row["original_image"])
        if not path.is_file():
            raise BackendBenchmarkError(f"input artifact is missing: {path}")


def _parse_backends(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if len(values) == 0 or any(value not in ("teed", "canny") for value in values):
        message = "backends must be a comma-separated subset of teed,canny"
        raise BackendBenchmarkError(message)
    return values


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


def _workspace_absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    raise SystemExit(main())
