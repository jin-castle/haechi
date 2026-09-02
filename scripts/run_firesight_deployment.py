from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from PIL import Image

from firesight_vision.canny import CannyError
from firesight_vision.deployment import (
    DeploymentBackend,
    FireSightDeploymentError,
    FireSightFrameResult,
    FireSightRuntime,
    FireSightRuntimeConfig,
)
from firesight_vision.teed import TeedError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class DeploymentArgNamespace(argparse.Namespace):
    input_path: Path = Path()
    output_dir: Path = Path()
    checkpoint_path: Path | None = None
    backend: str = "teed"
    width: int = 320
    height: int = 240
    fps: float = 15.0
    max_frames: int | None = None
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    profile_path: Path | None = None
    video_key: str | None = None
    realtime: bool = True
    write_video: bool = False


@dataclass(frozen=True, slots=True)
class DeploymentRunConfig:
    input_path: Path
    output_dir: Path
    checkpoint_path: Path | None = None
    backend: str = "teed"
    width: int = 320
    height: int = 240
    fps: float = 15.0
    max_frames: int | None = None
    threshold: float = 0.75
    background_scale: float = 0.24
    edge_width: int = 1
    device: str = "auto"
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    profile_path: Path | None = None
    video_key: str | None = None
    realtime: bool = True
    write_video: bool = False


@dataclass(frozen=True, slots=True)
class _InputFrame:
    image: Image.Image
    index: int
    source_frame: int
    source_name: str
    decode_ms: float


class _ImageArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def tobytes(self) -> bytes: ...


class _VideoCapture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, _ImageArray]: ...

    def release(self) -> None: ...


class _VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: _ImageArray) -> None: ...

    def release(self) -> None: ...


class _Cv2Module(Protocol):
    CAP_PROP_FPS: int
    COLOR_BGR2RGB: int
    COLOR_RGB2BGR: int

    def VideoCapture(self, path: str) -> _VideoCapture: ...

    def VideoWriter(
        self,
        path: str,
        fourcc: int,
        fps: float,
        size: tuple[int, int],
    ) -> _VideoWriter: ...

    def VideoWriter_fourcc(self, *chars: str) -> int: ...

    def cvtColor(self, image: _ImageArray, code: int) -> _ImageArray: ...


class _NumpyModule(Protocol):
    def asarray(self, image: Image.Image) -> _ImageArray: ...


IMAGE_SUFFIXES: Final = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
PROTOCOL_NAME: Final = "firesight_deployment_runtime_v1"
CLAIM_TEXT: Final = (
    "Deployment runtime result for the selected backend and device; "
    "Jetson performance is established only by running this same command on Jetson."
)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        payload = run_deployment(config)
    except (CannyError, FireSightDeploymentError, TeedError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    _ = sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None) -> DeploymentRunConfig:
    namespace = DeploymentArgNamespace()
    parser = argparse.ArgumentParser(
        description="Run the FireSight deployment edge runtime.",
    )
    _ = parser.add_argument("--input", dest="input_path", required=True, type=Path)
    _ = parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    _ = parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path)
    _ = parser.add_argument("--backend", choices=("teed", "canny"), default="teed")
    _ = parser.add_argument("--width", default=320, type=int)
    _ = parser.add_argument("--height", default=240, type=int)
    _ = parser.add_argument("--fps", default=15.0, type=float)
    _ = parser.add_argument("--max-frames", type=int)
    _ = parser.add_argument("--threshold", default=0.75, type=float)
    _ = parser.add_argument("--background-scale", default=0.24, type=float)
    _ = parser.add_argument("--edge-width", default=1, type=int)
    _ = parser.add_argument("--device", default="auto")
    _ = parser.add_argument("--canny-low", default=50, type=int)
    _ = parser.add_argument("--canny-high", default=150, type=int)
    _ = parser.add_argument("--profiles", dest="profile_path", type=Path)
    _ = parser.add_argument("--video-key")
    _ = parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="run as fast as possible without pacing",
    )
    _ = parser.add_argument(
        "--write-video",
        action="store_true",
        help="write overlay.mp4 in addition to per-frame PNG files",
    )
    _ = parser.parse_args(argv, namespace=namespace)
    return DeploymentRunConfig(
        input_path=namespace.input_path,
        output_dir=namespace.output_dir,
        checkpoint_path=namespace.checkpoint_path,
        backend=namespace.backend,
        width=namespace.width,
        height=namespace.height,
        fps=namespace.fps,
        max_frames=namespace.max_frames,
        threshold=namespace.threshold,
        background_scale=namespace.background_scale,
        edge_width=namespace.edge_width,
        device=namespace.device,
        canny_low_threshold=namespace.canny_low_threshold,
        canny_high_threshold=namespace.canny_high_threshold,
        profile_path=namespace.profile_path,
        video_key=namespace.video_key,
        realtime=namespace.realtime,
        write_video=namespace.write_video,
    )


def run_deployment(config: DeploymentRunConfig) -> dict[str, object]:
    _validate_run_config(config)
    effective_video_key = config.video_key or config.input_path.name
    runtime = FireSightRuntime(
        FireSightRuntimeConfig(
            backend=cast("DeploymentBackend", config.backend),
            checkpoint_path=config.checkpoint_path,
            width=config.width,
            height=config.height,
            threshold=config.threshold,
            background_scale=config.background_scale,
            edge_width=config.edge_width,
            device=config.device,
            canny_low_threshold=config.canny_low_threshold,
            canny_high_threshold=config.canny_high_threshold,
            profile_path=config.profile_path,
            video_key=effective_video_key,
        ),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = config.output_dir / "frames"
    masks_dir = config.output_dir / "masks"
    frames_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    metrics_path = config.output_dir / "deployment_metrics.jsonl"
    output_video_path: Path | None = None
    video_writer: _VideoWriter | None = None
    video_cv2: _Cv2Module | None = None
    video_numpy: _NumpyModule | None = None
    if config.write_video:
        output_video_path, video_writer, video_cv2, video_numpy = _open_video_writer(
            config.output_dir,
            runtime.target_size,
            config.fps,
        )

    latencies: list[float] = []
    decode_times: list[float] = []
    overrun_count = 0
    processed_frames = 0
    started_at = time.perf_counter()
    next_deadline = started_at
    frame_iterator = _iter_input_frames(config.input_path, config.fps)
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for frame in frame_iterator:
                if (
                    config.max_frames is not None
                    and processed_frames >= config.max_frames
                ):
                    break
                try:
                    result = runtime.process_frame(frame.image, config.video_key)
                    _write_frame_outputs(
                        result,
                        frames_dir / f"frame_{processed_frames:06d}.png",
                        masks_dir / f"frame_{processed_frames:06d}.png",
                    )
                    if video_writer is not None:
                        if video_cv2 is None or video_numpy is None:
                            message = "video writer backend was not initialized"
                            raise FireSightDeploymentError(message)
                        _write_video_frame(result, video_writer, video_cv2, video_numpy)
                    latencies.append(result.latency_ms)
                    decode_times.append(frame.decode_ms)
                    if result.latency_ms > 1000.0 / config.fps:
                        overrun_count += 1
                    record = _frame_record(frame, result, processed_frames)
                    _ = metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                finally:
                    frame.image.close()
                processed_frames += 1
                if config.realtime:
                    next_deadline += 1.0 / config.fps
                    delay = next_deadline - time.perf_counter()
                    if delay > 0.0:
                        time.sleep(delay)
                else:
                    next_deadline = time.perf_counter()
    finally:
        if video_writer is not None:
            video_writer.release()

    if processed_frames == 0:
        message = f"deployment produced no frames: {config.input_path}"
        raise FireSightDeploymentError(message)
    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    payload: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "claim": CLAIM_TEXT,
        "input": config.input_path.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "output_video": output_video_path.as_posix() if output_video_path else None,
        "backend": runtime.backend,
        "device": runtime.device,
        "target": {
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "frame_budget_ms": 1000.0 / config.fps,
        },
        "processed_frames": processed_frames,
        "effective_fps": processed_frames / elapsed_seconds,
        "realtime": config.realtime,
        "pacing_overrun_rate": overrun_count / processed_frames,
        "latency_ms": _summarize(latencies),
        "decode_ms": _summarize(decode_times),
        "profile_path": config.profile_path.as_posix()
        if config.profile_path is not None
        else None,
        "video_key": effective_video_key,
        "artifacts": {
            "metrics": metrics_path.as_posix(),
            "frames": frames_dir.as_posix(),
            "masks": masks_dir.as_posix(),
        },
    }
    summary_path = config.output_dir / "deployment_summary.json"
    _ = summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _iter_input_frames(input_path: Path, fps: float) -> Iterator[_InputFrame]:
    if input_path.is_dir():
        paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for index, path in enumerate(paths):
            started_at = time.perf_counter()
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
            except OSError as error:
                message = f"unable to read input image: {path}"
                raise FireSightDeploymentError(message) from error
            yield _InputFrame(
                image=image,
                index=index,
                source_frame=index,
                source_name=path.name,
                decode_ms=(time.perf_counter() - started_at) * 1000.0,
            )
        return
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_SUFFIXES:
        started_at = time.perf_counter()
        try:
            with Image.open(input_path) as source:
                image = source.convert("RGB")
        except OSError as error:
            message = f"unable to read input image: {input_path}"
            raise FireSightDeploymentError(message) from error
        yield _InputFrame(
            image=image,
            index=0,
            source_frame=0,
            source_name=input_path.name,
            decode_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        return
    yield from _iter_video_frames(input_path, fps)


def _iter_video_frames(input_path: Path, fps: float) -> Iterator[_InputFrame]:
    if not input_path.is_file():
        message = f"missing deployment input: {input_path}"
        raise FireSightDeploymentError(message)
    cv2, _ = _load_opencv()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        message = f"unable to open deployment video: {input_path}"
        raise FireSightDeploymentError(message)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    stride = max(1, round(source_fps / fps)) if source_fps > 0.0 else 1
    source_frame = 0
    output_index = 0
    try:
        while True:
            started_at = time.perf_counter()
            success, bgr_frame = capture.read()
            decode_ms = (time.perf_counter() - started_at) * 1000.0
            if not success:
                break
            if source_frame % stride != 0:
                source_frame += 1
                continue
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            height, width = rgb_frame.shape[0], rgb_frame.shape[1]
            image = Image.frombytes("RGB", (width, height), rgb_frame.tobytes())
            yield _InputFrame(
                image=image,
                index=output_index,
                source_frame=source_frame,
                source_name=input_path.name,
                decode_ms=decode_ms,
            )
            output_index += 1
            source_frame += 1
    finally:
        capture.release()


def _open_video_writer(
    output_dir: Path,
    target_size: tuple[int, int],
    fps: float,
) -> tuple[Path, _VideoWriter, _Cv2Module, _NumpyModule]:
    cv2, numpy = _load_opencv()
    output_path = output_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        target_size,
    )
    if not writer.isOpened():
        message = f"unable to create deployment video: {output_path}"
        raise FireSightDeploymentError(message)
    return output_path, writer, cv2, numpy


def _write_video_frame(
    result: FireSightFrameResult,
    writer: _VideoWriter,
    cv2: _Cv2Module,
    numpy: _NumpyModule,
) -> None:
    bgr = cv2.cvtColor(numpy.asarray(result.overlay), cv2.COLOR_RGB2BGR)
    writer.write(bgr)


def _write_frame_outputs(
    result: FireSightFrameResult,
    frame_path: Path,
    mask_path: Path,
) -> None:
    result.overlay.save(frame_path)
    result.mask.save(mask_path)


def _frame_record(
    frame: _InputFrame,
    result: FireSightFrameResult,
    index: int,
) -> dict[str, object]:
    return {
        "index": index,
        "source_frame": frame.source_frame,
        "source_name": frame.source_name,
        "decode_ms": frame.decode_ms,
        "latency_ms": result.latency_ms,
        "edge_ratio": result.edge_ratio,
        "edge_pixels": result.edge_pixels,
        "input_size": list(result.input_size),
        "processed_size": list(result.processed_size),
        "backend": result.backend,
        "device": result.device,
        "ignored_region_count": result.ignored_region_count,
    }


def _load_opencv() -> tuple[_Cv2Module, _NumpyModule]:
    try:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError as error:
        message = (
            "video deployment input/output requires OpenCV and NumPy; install "
            "opencv-python-headless or use an image directory."
        )
        raise FireSightDeploymentError(message) from error
    return cast("_Cv2Module", cast("object", cv2)), cast(
        "_NumpyModule",
        cast("object", numpy),
    )


def _summarize(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p50_index = min(len(ordered) - 1, max(0, math.ceil(0.50 * len(ordered)) - 1))
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": statistics.mean(values),
        "p50": ordered[p50_index],
        "p95": ordered[p95_index],
        "max": max(values),
    }


def _validate_run_config(config: DeploymentRunConfig) -> None:
    if not config.input_path.exists():
        message = f"missing deployment input: {config.input_path}"
        raise FireSightDeploymentError(message)
    if not math.isfinite(config.fps) or config.fps <= 0.0:
        message = "deployment FPS must be a finite positive value"
        raise FireSightDeploymentError(message)
    if config.max_frames is not None and config.max_frames < 1:
        message = "max frames must be greater than zero"
        raise FireSightDeploymentError(message)


if __name__ == "__main__":
    raise SystemExit(main())
