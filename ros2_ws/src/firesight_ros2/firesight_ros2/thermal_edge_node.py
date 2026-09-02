from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import rclpy
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMessage
from std_msgs.msg import String

try:
    from firesight_vision.thermal_edges import (
        ThermalHudConfig,
        build_thermal_hud_overlay,
    )
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[4] / "src"
    if repo_src.exists():
        sys.path.insert(0, repo_src.as_posix())
    from firesight_vision.thermal_edges import (  # type: ignore[no-redef]
        ThermalHudConfig,
        build_thermal_hud_overlay,
    )


class ThermalEdgeNode(Node):
    def __init__(self) -> None:
        super().__init__("thermal_edge_node")
        self._declare_parameters()
        self._thermal_topic = self.get_parameter("thermal_topic").value
        self._edge_mask_topic = self.get_parameter("edge_mask_topic").value
        self._edge_overlay_topic = self.get_parameter("edge_overlay_topic").value
        self._debug_topic = self.get_parameter("debug_topic").value
        self._config = ThermalHudConfig(
            edge_threshold=int(self.get_parameter("edge_threshold").value),
            hotspot_threshold=int(self.get_parameter("hotspot_threshold").value),
            background_scale=float(self.get_parameter("background_scale").value),
            edge_width=int(self.get_parameter("edge_width").value),
            hotspot_outline_width=int(
                self.get_parameter("hotspot_outline_width").value,
            ),
        )
        self._mask_publisher = self.create_publisher(
            ImageMessage,
            self._edge_mask_topic,
            10,
        )
        self._overlay_publisher = self.create_publisher(
            ImageMessage,
            self._edge_overlay_topic,
            10,
        )
        self._debug_publisher = self.create_publisher(String, self._debug_topic, 10)
        self._subscription = self.create_subscription(
            ImageMessage,
            self._thermal_topic,
            self._on_image,
            10,
        )
        self.get_logger().info(
            f"thermal edge node: {self._thermal_topic} -> "
            f"{self._edge_mask_topic}, {self._edge_overlay_topic}",
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("thermal_topic", "/thermal_camera/image_raw")
        self.declare_parameter("edge_mask_topic", "/firesight/thermal/edge_mask")
        self.declare_parameter("edge_overlay_topic", "/firesight/thermal/edge_overlay")
        self.declare_parameter("debug_topic", "/firesight/thermal/debug")
        self.declare_parameter("edge_threshold", 46)
        self.declare_parameter("hotspot_threshold", 205)
        self.declare_parameter("background_scale", 0.30)
        self.declare_parameter("edge_width", 5)
        self.declare_parameter("hotspot_outline_width", 5)

    def _on_image(self, message: ImageMessage) -> None:
        try:
            source = _image_message_to_pil(message)
            result = build_thermal_hud_overlay(source, config=self._config)
        except ValueError as error:
            self.get_logger().warning(f"skipping thermal frame: {error}")
            return

        self._mask_publisher.publish(
            _pil_to_image_message(result.edge_mask, message, encoding="mono8"),
        )
        self._overlay_publisher.publish(
            _pil_to_image_message(result.overlay, message, encoding="rgb8"),
        )
        debug = String()
        debug.data = json.dumps(
            {
                "edge_ratio": round(result.edge_ratio, 4),
                "hotspot_ratio": round(result.hotspot_ratio, 4),
                "hotspot_pixels": result.hotspot_pixels,
                "config": asdict(self._config),
            },
            sort_keys=True,
        )
        self._debug_publisher.publish(debug)


def _image_message_to_pil(message: ImageMessage) -> Image.Image:
    encoding = message.encoding.lower()
    width = int(message.width)
    height = int(message.height)
    if encoding in {"mono8", "8uc1"}:
        data = _compact_rows(bytes(message.data), height, int(message.step), width)
        return Image.frombytes("L", (width, height), data)
    if encoding in {"rgb8"}:
        data = _compact_rows(bytes(message.data), height, int(message.step), width * 3)
        return Image.frombytes("RGB", (width, height), data)
    if encoding in {"bgr8"}:
        data = _compact_rows(bytes(message.data), height, int(message.step), width * 3)
        return Image.frombytes("RGB", (width, height), data, "raw", "BGR")
    if encoding in {"mono16", "16uc1"}:
        data = _compact_rows(bytes(message.data), height, int(message.step), width * 2)
        mode = "I;16B" if bool(message.is_bigendian) else "I;16L"
        return Image.frombytes(mode, (width, height), data)
    msg = f"unsupported thermal image encoding: {message.encoding}"
    raise ValueError(msg)


def _pil_to_image_message(
    image: Image.Image,
    source: ImageMessage,
    *,
    encoding: str,
) -> ImageMessage:
    converted = image.convert("L" if encoding == "mono8" else "RGB")
    message = ImageMessage()
    message.header = source.header
    message.height = converted.height
    message.width = converted.width
    message.encoding = encoding
    message.is_bigendian = False
    message.step = converted.width if encoding == "mono8" else converted.width * 3
    message.data = converted.tobytes()
    return message


def _compact_rows(data: bytes, height: int, step: int, row_bytes: int) -> bytes:
    if height <= 0:
        return b""
    if step == row_bytes and len(data) == height * row_bytes:
        return data
    return b"".join(
        data[row * step : (row * step) + row_bytes] for row in range(height)
    )


def main() -> None:
    rclpy.init()
    node = ThermalEdgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
