from __future__ import annotations

import hashlib
import importlib
import json
from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from PIL import Image, ImageDraw, ImageOps

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


THERMAL_TOPIC: Final = "/flir_boson/image_raw"
RGB_TOPIC: Final = "/camera/color/image_raw"
DEPTH_TOPIC: Final = "/camera/aligned_depth_to_color/image_raw"
SUPPORTED_ENCODINGS: Final = frozenset({"mono16", "16UC1", "rgb8"})
PROTOCOL_NAME: Final = "thermal_rgb_aligned_depth_rosbag_extract_v1"
Y16_14_BIT_MAX: Final = 16383.0
Y16_MSB_DIVISIBLE_RATIO: Final = 0.995


class RosbagDepthExtractError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RosbagDepthExtractConfig:
    bag_path: Path
    out_dir: Path
    max_pairs: int = 3
    start_seconds: float = 0.0
    interval_seconds: float = 10.0
    max_sync_delta_ms: float = 50.0
    preview_max_depth_m: float = 6.0


@dataclass(frozen=True, slots=True)
class ImageStamp:
    topic: str
    topic_index: int
    bag_timestamp_ns: int
    header_timestamp_ns: int


@dataclass(frozen=True, slots=True)
class SynchronizedPair:
    pair_index: int
    thermal: ImageStamp
    rgb: ImageStamp
    depth: ImageStamp


@dataclass(frozen=True, slots=True)
class DecodedImage:
    encoding: str
    width: int
    height: int
    values: NDArray[np.uint8] | NDArray[np.uint16]


def extract_rosbag_depth_pairs(
    config: RosbagDepthExtractConfig,
) -> dict[str, object]:
    _validate_config(config)
    numpy, any_reader = _load_dependencies()
    bag_path = config.bag_path.resolve()
    stamps = _collect_image_stamps(bag_path, any_reader)
    pairs = synchronize_image_stamps(
        stamps,
        max_pairs=config.max_pairs,
        start_seconds=config.start_seconds,
        interval_seconds=config.interval_seconds,
        max_sync_delta_ms=config.max_sync_delta_ms,
    )
    decoded = _read_selected_images(bag_path, pairs, any_reader, numpy)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    pair_outputs = [
        _write_pair_artifacts(
            config.out_dir,
            pair,
            decoded,
            preview_max_depth_m=config.preview_max_depth_m,
            numpy_module=numpy,
        )
        for pair in pairs
    ]
    manifest: dict[str, object] = {
        "protocol": PROTOCOL_NAME,
        "bag_file": bag_path.as_posix(),
        "bag_size_bytes": bag_path.stat().st_size,
        "bag_sha256": _sha256_file(bag_path),
        "topics": {
            "thermal": THERMAL_TOPIC,
            "rgb": RGB_TOPIC,
            "depth": DEPTH_TOPIC,
        },
        "sync_clock": "rosbag_record_timestamp",
        "max_sync_delta_ms": config.max_sync_delta_ms,
        "pairs_requested": config.max_pairs,
        "pairs_written": len(pair_outputs),
        "rgb_depth_spatial_alignment": (
            "Depth topic is aligned to the RGB camera coordinate system."
        ),
        "thermal_depth_spatial_alignment": (
            "Not established by the raw bag. Thermal-to-depth pixel metrics require "
            "camera calibration or processed aligned pairs."
        ),
        "haechiar_thermal2depth_input": {
            "expected": "14-bit Y16 aligned to bits 0..13 or shifted into bits 2..15",
            "all_extracted_frames_compatible": all(
                bool(output["haechiar_thermal2depth_input_compatible"])
                for output in pair_outputs
            ),
            "action": (
                "Do not run the existing checkpoint on incompatible frames. Verify the "
                "Boson output mode and conversion, or retrain on this dataset format."
            ),
        },
        "distance_unit": "RealSense 16UC1 values interpreted as millimetres",
        "pair_outputs": pair_outputs,
    }
    manifest_path = config.out_dir / "manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def synchronize_image_stamps(
    stamps: dict[str, list[ImageStamp]],
    *,
    max_pairs: int,
    start_seconds: float,
    interval_seconds: float,
    max_sync_delta_ms: float,
) -> list[SynchronizedPair]:
    for topic in (THERMAL_TOPIC, RGB_TOPIC, DEPTH_TOPIC):
        if not stamps.get(topic):
            raise RosbagDepthExtractError(f"required image topic is empty: {topic}")

    depth_stamps = stamps[DEPTH_TOPIC]
    start_ns = depth_stamps[0].bag_timestamp_ns + round(start_seconds * 1e9)
    interval_ns = round(interval_seconds * 1e9)
    maximum_delta_ns = round(max_sync_delta_ms * 1e6)
    selected_depth: list[ImageStamp] = []
    next_target_ns = start_ns
    for depth_stamp in depth_stamps:
        if depth_stamp.bag_timestamp_ns < next_target_ns:
            continue
        selected_depth.append(depth_stamp)
        if len(selected_depth) == max_pairs:
            break
        next_target_ns += interval_ns

    if not selected_depth:
        raise RosbagDepthExtractError(
            "no depth frames matched the requested time range"
        )

    synchronized: list[SynchronizedPair] = []
    for depth_stamp in selected_depth:
        thermal = _nearest_stamp(stamps[THERMAL_TOPIC], depth_stamp.bag_timestamp_ns)
        rgb = _nearest_stamp(stamps[RGB_TOPIC], depth_stamp.bag_timestamp_ns)
        thermal_delta = abs(thermal.bag_timestamp_ns - depth_stamp.bag_timestamp_ns)
        rgb_delta = abs(rgb.bag_timestamp_ns - depth_stamp.bag_timestamp_ns)
        if thermal_delta > maximum_delta_ns or rgb_delta > maximum_delta_ns:
            continue
        synchronized.append(
            SynchronizedPair(
                pair_index=len(synchronized),
                thermal=thermal,
                rgb=rgb,
                depth=depth_stamp,
            ),
        )
    if not synchronized:
        message = (
            "no complete thermal/RGB/depth pairs met the sync tolerance of "
            f"{max_sync_delta_ms:.3f} ms"
        )
        raise RosbagDepthExtractError(message)
    return synchronized


def decode_ros_image(message: Any, numpy_module: Any) -> DecodedImage:
    encoding = str(message.encoding)
    if encoding not in SUPPORTED_ENCODINGS:
        raise RosbagDepthExtractError(f"unsupported ROS image encoding: {encoding}")
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    channels = 3 if encoding == "rgb8" else 1
    bytes_per_channel = 1 if encoding == "rgb8" else 2
    active_row_bytes = width * channels * bytes_per_channel
    if width < 1 or height < 1 or step < active_row_bytes:
        raise RosbagDepthExtractError(
            f"invalid ROS image dimensions: {width}x{height}, step={step}",
        )

    raw = numpy_module.asarray(message.data, dtype=numpy_module.uint8)
    expected_bytes = height * step
    if raw.size < expected_bytes:
        raise RosbagDepthExtractError(
            f"truncated ROS image: expected {expected_bytes} bytes, got {raw.size}",
        )
    active = raw[:expected_bytes].reshape(height, step)[:, :active_row_bytes].copy()
    if encoding == "rgb8":
        values = active.reshape(height, width, 3)
    else:
        byte_order = ">u2" if bool(message.is_bigendian) else "<u2"
        values = active.reshape(-1).view(numpy_module.dtype(byte_order))
        values = values.astype(numpy_module.uint16, copy=False).reshape(height, width)
    return DecodedImage(
        encoding=encoding,
        width=width,
        height=height,
        values=values,
    )


def classify_haechiar_y16_alignment(
    raw_mono16: NDArray[np.uint16],
    numpy_module: Any,
) -> tuple[str | None, float, float]:
    high_percentile = float(numpy_module.percentile(raw_mono16, 99.5))
    divisible_by_four = float(numpy_module.mean((raw_mono16 & 0x3) == 0))
    if high_percentile <= Y16_14_BIT_MAX:
        return "lsb", high_percentile, divisible_by_four
    if divisible_by_four >= Y16_MSB_DIVISIBLE_RATIO:
        return "msb", high_percentile, divisible_by_four
    return None, high_percentile, divisible_by_four


def _validate_config(config: RosbagDepthExtractConfig) -> None:
    if not config.bag_path.is_file():
        raise RosbagDepthExtractError(f"ROS bag not found: {config.bag_path}")
    if config.max_pairs < 1:
        raise RosbagDepthExtractError("max_pairs must be at least 1")
    if config.start_seconds < 0.0:
        raise RosbagDepthExtractError("start_seconds must not be negative")
    if config.interval_seconds <= 0.0:
        raise RosbagDepthExtractError("interval_seconds must be greater than zero")
    if config.max_sync_delta_ms <= 0.0:
        raise RosbagDepthExtractError("max_sync_delta_ms must be greater than zero")
    if config.preview_max_depth_m <= 0.0:
        raise RosbagDepthExtractError("preview_max_depth_m must be greater than zero")


def _load_dependencies() -> tuple[Any, Any]:
    try:
        numpy = importlib.import_module("numpy")
        highlevel = importlib.import_module("rosbags.highlevel")
    except ModuleNotFoundError as error:
        message = (
            "ROS bag extraction requires optional dependencies; "
            "run `uv sync --extra rosbag`."
        )
        raise RosbagDepthExtractError(message) from error
    return numpy, highlevel.AnyReader


def _collect_image_stamps(
    bag_path: Path, any_reader: Any
) -> dict[str, list[ImageStamp]]:
    topics = (THERMAL_TOPIC, RGB_TOPIC, DEPTH_TOPIC)
    stamps = {topic: [] for topic in topics}
    topic_indexes = dict.fromkeys(topics, 0)
    with any_reader([bag_path]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in topics
        ]
        available = {connection.topic for connection in connections}
        missing = sorted(set(topics) - available)
        if missing:
            raise RosbagDepthExtractError(
                f"required ROS image topics are missing: {', '.join(missing)}",
            )
        for connection, bag_timestamp_ns, rawdata in reader.messages(
            connections=connections,
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            topic = str(connection.topic)
            stamps[topic].append(
                ImageStamp(
                    topic=topic,
                    topic_index=topic_indexes[topic],
                    bag_timestamp_ns=int(bag_timestamp_ns),
                    header_timestamp_ns=_header_timestamp_ns(message),
                ),
            )
            topic_indexes[topic] += 1
    return stamps


def _header_timestamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)


def _nearest_stamp(stamps: list[ImageStamp], target_ns: int) -> ImageStamp:
    timestamps = [stamp.bag_timestamp_ns for stamp in stamps]
    insertion = bisect_left(timestamps, target_ns)
    candidates = stamps[max(0, insertion - 1) : min(len(stamps), insertion + 1)]
    return min(candidates, key=lambda stamp: abs(stamp.bag_timestamp_ns - target_ns))


def _read_selected_images(
    bag_path: Path,
    pairs: Iterable[SynchronizedPair],
    any_reader: Any,
    numpy_module: Any,
) -> dict[tuple[str, int], DecodedImage]:
    selected = {
        (stamp.topic, stamp.topic_index)
        for pair in pairs
        for stamp in (pair.thermal, pair.rgb, pair.depth)
    }
    decoded: dict[tuple[str, int], DecodedImage] = {}
    topic_indexes = dict.fromkeys((THERMAL_TOPIC, RGB_TOPIC, DEPTH_TOPIC), 0)
    with any_reader([bag_path]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in topic_indexes
        ]
        for connection, _, rawdata in reader.messages(connections=connections):
            topic = str(connection.topic)
            key = (topic, topic_indexes[topic])
            if key in selected:
                message = reader.deserialize(rawdata, connection.msgtype)
                decoded[key] = decode_ros_image(message, numpy_module)
                if len(decoded) == len(selected):
                    break
            topic_indexes[topic] += 1
    missing = sorted(selected - decoded.keys())
    if missing:
        raise RosbagDepthExtractError(
            f"unable to decode selected ROS images: {missing}"
        )
    return decoded


def _write_pair_artifacts(
    out_dir: Path,
    pair: SynchronizedPair,
    decoded: dict[tuple[str, int], DecodedImage],
    *,
    preview_max_depth_m: float,
    numpy_module: Any,
) -> dict[str, object]:
    thermal = decoded[(pair.thermal.topic, pair.thermal.topic_index)]
    rgb = decoded[(pair.rgb.topic, pair.rgb.topic_index)]
    depth = decoded[(pair.depth.topic, pair.depth.topic_index)]
    if (
        thermal.encoding != "mono16"
        or rgb.encoding != "rgb8"
        or depth.encoding != "16UC1"
    ):
        raise RosbagDepthExtractError(
            "unexpected encodings for synchronized thermal/RGB/depth pair",
        )
    if (rgb.width, rgb.height) != (depth.width, depth.height):
        raise RosbagDepthExtractError(
            "RealSense RGB and aligned depth dimensions do not match",
        )

    stem = f"pair_{pair.pair_index:03d}"
    thermal_raw_path = out_dir / f"{stem}_thermal_raw16.png"
    thermal_preview_path = out_dir / f"{stem}_thermal_preview.png"
    rgb_path = out_dir / f"{stem}_rgb.png"
    depth_raw_path = out_dir / f"{stem}_depth_mm16.png"
    depth_preview_path = out_dir / f"{stem}_depth_preview.png"
    contact_path = out_dir / f"{stem}_contact.png"

    thermal_values = thermal.values
    rgb_values = rgb.values
    depth_mm = depth.values
    thermal_raw = Image.fromarray(thermal_values, mode="I;16")
    thermal_preview = _render_thermal_preview(thermal_values, numpy_module)
    rgb_image = Image.fromarray(rgb_values, mode="RGB")
    depth_raw = Image.fromarray(depth_mm, mode="I;16")
    depth_preview = _render_depth_preview(
        depth_mm,
        preview_max_depth_m=preview_max_depth_m,
        numpy_module=numpy_module,
    )
    thermal_raw.save(thermal_raw_path)
    thermal_preview.save(thermal_preview_path)
    rgb_image.save(rgb_path)
    depth_raw.save(depth_raw_path)
    depth_preview.save(depth_preview_path)
    _render_contact_sheet(thermal_preview, rgb_image, depth_preview).save(contact_path)

    valid_depth = depth_mm > 0
    valid_values_m = depth_mm[valid_depth].astype(numpy_module.float32) / 1000.0
    depth_stats = _depth_statistics(valid_values_m, depth_mm.size, numpy_module)
    alignment, thermal_p99_5, thermal_divisible_by_four = (
        classify_haechiar_y16_alignment(thermal_values, numpy_module)
    )
    return {
        "pair_index": pair.pair_index,
        "thermal_raw_png": thermal_raw_path.name,
        "thermal_preview_png": thermal_preview_path.name,
        "rgb_png": rgb_path.name,
        "depth_mm16_png": depth_raw_path.name,
        "depth_preview_png": depth_preview_path.name,
        "contact_png": contact_path.name,
        "thermal_topic_index": pair.thermal.topic_index,
        "rgb_topic_index": pair.rgb.topic_index,
        "depth_topic_index": pair.depth.topic_index,
        "thermal_bag_timestamp_ns": pair.thermal.bag_timestamp_ns,
        "rgb_bag_timestamp_ns": pair.rgb.bag_timestamp_ns,
        "depth_bag_timestamp_ns": pair.depth.bag_timestamp_ns,
        "thermal_header_timestamp_ns": pair.thermal.header_timestamp_ns,
        "rgb_header_timestamp_ns": pair.rgb.header_timestamp_ns,
        "depth_header_timestamp_ns": pair.depth.header_timestamp_ns,
        "thermal_depth_bag_delta_ms": round(
            (pair.thermal.bag_timestamp_ns - pair.depth.bag_timestamp_ns) / 1e6,
            3,
        ),
        "rgb_depth_bag_delta_ms": round(
            (pair.rgb.bag_timestamp_ns - pair.depth.bag_timestamp_ns) / 1e6,
            3,
        ),
        "thermal_depth_header_delta_ms": round(
            (pair.thermal.header_timestamp_ns - pair.depth.header_timestamp_ns) / 1e6,
            3,
        ),
        "rgb_depth_header_delta_ms": round(
            (pair.rgb.header_timestamp_ns - pair.depth.header_timestamp_ns) / 1e6,
            3,
        ),
        "thermal_shape": [thermal.height, thermal.width],
        "rgb_shape": [rgb.height, rgb.width, 3],
        "depth_shape": [depth.height, depth.width],
        "thermal_min_raw16": int(thermal_values.min()),
        "thermal_median_raw16": round(float(numpy_module.median(thermal_values)), 3),
        "thermal_p99_5_raw16": round(thermal_p99_5, 3),
        "thermal_max_raw16": int(thermal_values.max()),
        "thermal_divisible_by_four_ratio": round(thermal_divisible_by_four, 6),
        "haechiar_thermal2depth_detected_alignment": alignment,
        "haechiar_thermal2depth_input_compatible": alignment is not None,
        **depth_stats,
    }


def _render_thermal_preview(values: Any, numpy_module: Any) -> Image.Image:
    low, high = numpy_module.percentile(values, [2.0, 98.0])
    normalized = numpy_module.clip(
        (values.astype(numpy_module.float32) - low) / (high - low + 1e-8),
        0.0,
        1.0,
    )
    return Image.fromarray((normalized * 255.0).astype(numpy_module.uint8), mode="L")


def _render_depth_preview(
    depth_mm: Any,
    *,
    preview_max_depth_m: float,
    numpy_module: Any,
) -> Image.Image:
    depth_m = depth_mm.astype(numpy_module.float32) / 1000.0
    valid = depth_mm > 0
    normalized = numpy_module.clip(depth_m / preview_max_depth_m, 0.0, 1.0)
    colors = numpy_module.zeros((*depth_m.shape, 3), dtype=numpy_module.uint8)
    colors[..., 0] = (255.0 * (1.0 - normalized)).astype(numpy_module.uint8)
    colors[..., 1] = (
        255.0 * (1.0 - numpy_module.abs((2.0 * normalized) - 1.0))
    ).astype(numpy_module.uint8)
    colors[..., 2] = (255.0 * normalized).astype(numpy_module.uint8)
    colors[~valid] = 0
    return Image.fromarray(colors, mode="RGB")


def _render_contact_sheet(
    thermal: Image.Image,
    rgb: Image.Image,
    depth: Image.Image,
) -> Image.Image:
    panel_size = (480, 300)
    labels = ("Thermal (not spatially aligned)", "RGB", "RealSense depth (RGB-aligned)")
    images = (thermal.convert("RGB"), rgb.convert("RGB"), depth.convert("RGB"))
    sheet = Image.new("RGB", (panel_size[0] * 3, panel_size[1] + 28), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        fitted = ImageOps.contain(image, panel_size, method=Image.Resampling.BILINEAR)
        x = (index * panel_size[0]) + ((panel_size[0] - fitted.width) // 2)
        y = 28 + ((panel_size[1] - fitted.height) // 2)
        sheet.paste(fitted, (x, y))
        draw.text((index * panel_size[0] + 8, 7), label, fill="white")
    return sheet


def _depth_statistics(
    values_m: Any, total_pixels: int, numpy_module: Any
) -> dict[str, float]:
    valid_pixels = int(values_m.size)
    if valid_pixels == 0:
        return {
            "depth_valid_ratio": 0.0,
            "depth_min_m": 0.0,
            "depth_median_m": 0.0,
            "depth_p95_m": 0.0,
            "depth_max_m": 0.0,
        }
    return {
        "depth_valid_ratio": round(valid_pixels / total_pixels, 6),
        "depth_min_m": round(float(values_m.min()), 4),
        "depth_median_m": round(float(numpy_module.median(values_m)), 4),
        "depth_p95_m": round(float(numpy_module.quantile(values_m, 0.95)), 4),
        "depth_max_m": round(float(values_m.max()), 4),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
