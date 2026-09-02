from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from PIL import Image

from firesight_vision.hud_edges import FIRE360_OSD_IGNORED_REGIONS
from firesight_vision.pidinet import PidinetError, PidinetPredictor
from firesight_vision.teed import TeedError, TeedPredictor

if TYPE_CHECKING:
    from collections.abc import Sequence


class BenchmarkArgNamespace(argparse.Namespace):
    input_csv: Path = Path(
        ".omo/evidence/firesight-smoke-vision/model-evaluation-30/evaluation.csv",
    )
    teed_checkpoint: Path = Path("models/teed/5_model.pth")
    pidinet_checkpoint: Path = Path("models/pidinet/table5_pidinet.pth")
    output_path: Path = Path()
    sizes: str = "640x360,320x240"
    teed_threshold: float = 0.75
    pidinet_threshold: float = 0.30
    warmup: int = 2
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class BenchmarkSize:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ModelBenchmarkConfig:
    input_csv: Path
    teed_checkpoint: Path
    pidinet_checkpoint: Path
    output_path: Path
    sizes: tuple[BenchmarkSize, ...]
    teed_threshold: float = 0.75
    pidinet_threshold: float = 0.30
    warmup: int = 2
    device: str = "cpu"


class ModelBenchmarkError(Exception):
    pass


PROTOCOL_NAME: Final = "fire360_learned_model_benchmark_v1"
EXPECTED_FRAME_COUNT: Final = 30


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_benchmark(config)
    except (ModelBenchmarkError, PidinetError, TeedError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> ModelBenchmarkConfig:
    namespace = BenchmarkArgNamespace()
    parser = argparse.ArgumentParser(
        description="Benchmark TEED and PiDiNet on fixed Fire360 frames.",
    )
    _ = parser.add_argument("--input-csv", default=namespace.input_csv, type=Path)
    _ = parser.add_argument(
        "--teed-checkpoint",
        default=namespace.teed_checkpoint,
        type=Path,
    )
    _ = parser.add_argument(
        "--pidinet-checkpoint",
        default=namespace.pidinet_checkpoint,
        type=Path,
    )
    _ = parser.add_argument("--out", dest="output_path", required=True, type=Path)
    _ = parser.add_argument("--sizes", default=namespace.sizes)
    _ = parser.add_argument("--teed-threshold", default=0.75, type=float)
    _ = parser.add_argument("--pidinet-threshold", default=0.30, type=float)
    _ = parser.add_argument("--warmup", default=2, type=int)
    _ = parser.add_argument("--device", default="cpu")
    _ = parser.parse_args(argv, namespace=namespace)
    return ModelBenchmarkConfig(
        input_csv=namespace.input_csv,
        teed_checkpoint=namespace.teed_checkpoint,
        pidinet_checkpoint=namespace.pidinet_checkpoint,
        output_path=namespace.output_path,
        sizes=_parse_sizes(namespace.sizes),
        teed_threshold=namespace.teed_threshold,
        pidinet_threshold=namespace.pidinet_threshold,
        warmup=namespace.warmup,
        device=namespace.device,
    )


def run_benchmark(config: ModelBenchmarkConfig) -> dict[str, object]:
    rows = _load_rows(config.input_csv)
    _validate_config(config, rows)
    teed = TeedPredictor(config.teed_checkpoint, device=config.device)
    pidinet = PidinetPredictor(config.pidinet_checkpoint, device=config.device)
    results: list[dict[str, object]] = []
    for size in config.sizes:
        images = _load_images(rows, size)
        try:
            results.append(
                {
                    "resolution": [size.width, size.height],
                    "frames": len(images),
                    "teed": _measure_model(
                        teed,
                        images,
                        config.teed_threshold,
                        config.warmup,
                    ),
                    "pidinet": _measure_model(
                        pidinet,
                        images,
                        config.pidinet_threshold,
                        config.warmup,
                    ),
                },
            )
        finally:
            for image, _ in images:
                image.close()
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": "Development-PC CPU reference only; not a Jetson measurement.",
        "input_csv": _workspace_relative(config.input_csv),
        "frames": len(rows),
        "device": config.device,
        "warmup_frames": config.warmup,
        "thresholds": {
            "teed": config.teed_threshold,
            "pidinet": config.pidinet_threshold,
        },
        "checkpoints": {
            "teed": _workspace_relative(config.teed_checkpoint),
            "pidinet": _workspace_relative(config.pidinet_checkpoint),
        },
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
        raise ModelBenchmarkError(f"input evaluation CSV is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]
    except OSError as error:
        message = f"unable to read input evaluation CSV: {path}"
        raise ModelBenchmarkError(message) from error
    if len(rows) != EXPECTED_FRAME_COUNT:
        raise ModelBenchmarkError(
            f"expected {EXPECTED_FRAME_COUNT} input rows, got {len(rows)}",
        )
    return rows


def _validate_config(
    config: ModelBenchmarkConfig,
    rows: list[dict[str, str]],
) -> None:
    if not config.sizes:
        raise ModelBenchmarkError("at least one benchmark size is required")
    if not 0.0 <= config.teed_threshold <= 1.0:
        raise ModelBenchmarkError("TEED threshold must be between 0 and 1")
    if not 0.0 <= config.pidinet_threshold <= 1.0:
        raise ModelBenchmarkError("PiDiNet threshold must be between 0 and 1")
    if config.warmup < 0:
        raise ModelBenchmarkError("warmup must be non-negative")
    for row in rows:
        path = _workspace_absolute(row["original_image"])
        if not path.is_file():
            raise ModelBenchmarkError(f"input artifact is missing: {path}")


def _parse_sizes(value: str) -> tuple[BenchmarkSize, ...]:
    sizes: list[BenchmarkSize] = []
    for raw_size in value.split(","):
        parts = raw_size.strip().lower().split("x")
        if len(parts) != 2:
            raise ModelBenchmarkError(f"invalid benchmark size: {raw_size}")
        try:
            width, height = (int(part) for part in parts)
        except ValueError as error:
            raise ModelBenchmarkError(f"invalid benchmark size: {raw_size}") from error
        if width < 64 or height < 64:
            raise ModelBenchmarkError(f"benchmark size is too small: {raw_size}")
        sizes.append(BenchmarkSize(width=width, height=height))
    return tuple(sizes)


def _load_images(
    rows: list[dict[str, str]],
    size: BenchmarkSize,
) -> list[tuple[Image.Image, bool]]:
    images: list[tuple[Image.Image, bool]] = []
    try:
        for row in rows:
            with Image.open(_workspace_absolute(row["original_image"])) as source:
                image = source.convert("RGB").resize(
                    (size.width, size.height),
                    Image.Resampling.LANCZOS,
                )
            images.append((image, row["ignore_fire360_osd"].lower() == "true"))
    except OSError as error:
        for image, _ in images:
            image.close()
        raise ModelBenchmarkError("unable to load benchmark images") from error
    return images


def _measure_model(
    predictor: TeedPredictor | PidinetPredictor,
    images: list[tuple[Image.Image, bool]],
    threshold: float,
    warmup: int,
) -> dict[str, float]:
    for image, ignored in images[:warmup]:
        _ = _process(predictor, image, ignored, threshold)
    timings: list[float] = []
    for image, ignored in images:
        start = time.perf_counter()
        _ = _process(predictor, image, ignored, threshold)
        timings.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.mean(timings),
        "p95_ms": ordered[p95_index],
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def _process(
    predictor: TeedPredictor | PidinetPredictor,
    image: Image.Image,
    ignored: bool,
    threshold: float,
) -> object:
    return predictor.process(
        image,
        threshold=threshold,
        background_scale=0.24,
        edge_width=1,
        ignored_regions=FIRE360_OSD_IGNORED_REGIONS if ignored else (),
    )


def _workspace_relative(path: Path) -> str:
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


def _workspace_absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    raise SystemExit(main())
