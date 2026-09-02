from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from firesight_vision.rosbag_depth_extract import (
    DEPTH_TOPIC,
    RGB_TOPIC,
    THERMAL_TOPIC,
    ImageStamp,
    RosbagDepthExtractError,
    classify_haechiar_y16_alignment,
    decode_ros_image,
    synchronize_image_stamps,
)


@dataclass
class FakeImageMessage:
    width: int
    height: int
    encoding: str
    is_bigendian: int
    step: int
    data: np.ndarray


def test_decode_ros_image_preserves_little_endian_16_bit_values_and_padding() -> None:
    row0 = np.array([1, 256, 999], dtype="<u2").tobytes() + b"xx"
    row1 = np.array([2, 512, 1000], dtype="<u2").tobytes() + b"yy"
    message = FakeImageMessage(
        width=3,
        height=2,
        encoding="mono16",
        is_bigendian=0,
        step=8,
        data=np.frombuffer(row0 + row1, dtype=np.uint8),
    )

    decoded = decode_ros_image(message, np)

    assert decoded.values.dtype == np.uint16
    assert decoded.values.tolist() == [[1, 256, 999], [2, 512, 1000]]


def test_decode_ros_image_preserves_rgb_channels() -> None:
    message = FakeImageMessage(
        width=2,
        height=1,
        encoding="rgb8",
        is_bigendian=0,
        step=6,
        data=np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8),
    )

    decoded = decode_ros_image(message, np)

    assert decoded.values.shape == (1, 2, 3)
    assert decoded.values.tolist() == [[[1, 2, 3], [4, 5, 6]]]


def test_synchronize_uses_rosbag_timestamp_and_enforces_tolerance() -> None:
    base = 1_000_000_000
    stamps = {
        THERMAL_TOPIC: [
            _stamp(THERMAL_TOPIC, 0, base + 20_000_000, base + 220_000_000)
        ],
        RGB_TOPIC: [_stamp(RGB_TOPIC, 0, base + 10_000_000, base)],
        DEPTH_TOPIC: [_stamp(DEPTH_TOPIC, 0, base, base)],
    }

    pairs = synchronize_image_stamps(
        stamps,
        max_pairs=1,
        start_seconds=0.0,
        interval_seconds=1.0,
        max_sync_delta_ms=50.0,
    )

    assert len(pairs) == 1
    assert (
        pairs[0].thermal.header_timestamp_ns - pairs[0].depth.header_timestamp_ns
        == 220_000_000
    )


def test_synchronize_rejects_pairs_outside_tolerance() -> None:
    base = 1_000_000_000
    stamps = {
        THERMAL_TOPIC: [_stamp(THERMAL_TOPIC, 0, base + 60_000_000, base)],
        RGB_TOPIC: [_stamp(RGB_TOPIC, 0, base, base)],
        DEPTH_TOPIC: [_stamp(DEPTH_TOPIC, 0, base, base)],
    }

    with pytest.raises(RosbagDepthExtractError, match="no complete"):
        _ = synchronize_image_stamps(
            stamps,
            max_pairs=1,
            start_seconds=0.0,
            interval_seconds=1.0,
            max_sync_delta_ms=50.0,
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([[0, 100, 16383]], dtype=np.uint16), "lsb"),
        (np.array([[0, 20000, 64000]], dtype=np.uint16), "msb"),
        (np.array([[23649, 24002, 25899]], dtype=np.uint16), None),
    ],
)
def test_classify_haechiar_y16_alignment(
    values: np.ndarray,
    expected: str | None,
) -> None:
    if expected == "msb":
        values = values & np.uint16(0xFFFC)

    alignment, _, _ = classify_haechiar_y16_alignment(values, np)

    assert alignment == expected


def _stamp(topic: str, index: int, bag_ns: int, header_ns: int) -> ImageStamp:
    return ImageStamp(
        topic=topic,
        topic_index=index,
        bag_timestamp_ns=bag_ns,
        header_timestamp_ns=header_ns,
    )
