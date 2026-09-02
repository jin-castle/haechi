from __future__ import annotations

import itertools
from collections.abc import Iterator

import rclpy
from PIL import Image, ImageDraw, ImageFilter
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMessage


class ThermalMockPublisher(Node):
    def __init__(self) -> None:
        super().__init__("thermal_mock_publisher")
        self.declare_parameter("topic", "/thermal_camera/image_raw")
        self.declare_parameter("frame_id", "thermal_camera_link")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 384)
        self.declare_parameter("rate_hz", 10.0)
        self._topic = self.get_parameter("topic").value
        self._frame_id = self.get_parameter("frame_id").value
        self._width = int(self.get_parameter("width").value)
        self._height = int(self.get_parameter("height").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._frames = _frame_counter()
        self._publisher = self.create_publisher(ImageMessage, self._topic, 10)
        self._timer = self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info(f"publishing mock LWIR frames to {self._topic}")

    def _publish(self) -> None:
        frame_index = next(self._frames)
        image = _build_frame(self._width, self._height, frame_index)
        message = ImageMessage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.height = image.height
        message.width = image.width
        message.encoding = "mono8"
        message.is_bigendian = False
        message.step = image.width
        message.data = image.tobytes()
        self._publisher.publish(message)


def _build_frame(width: int, height: int, frame_index: int) -> Image.Image:
    image = Image.new("L", (width, height), 86)
    draw = ImageDraw.Draw(image)
    margin_x = round(width * 0.10)
    margin_y = round(height * 0.16)
    draw.rectangle(
        (margin_x, margin_y, width - margin_x, height - margin_y),
        outline=124,
        width=3,
    )
    draw.rectangle(
        (round(width * 0.36), round(height * 0.34), round(width * 0.49), height - margin_y),
        outline=136,
        width=4,
    )
    draw.line(
        (
            round(width * 0.12),
            round(height * 0.64),
            round(width * 0.34),
            round(height * 0.46),
            round(width * 0.57),
            round(height * 0.62),
            round(width * 0.86),
            round(height * 0.42),
        ),
        fill=126,
        width=4,
    )
    offset = round((frame_index % 18) - 9)
    draw.ellipse(
        (
            round(width * 0.66) + offset,
            round(height * 0.33),
            round(width * 0.75) + offset,
            round(height * 0.56),
        ),
        fill=228,
    )
    draw.polygon(
        (
            (round(width * 0.70) + offset, round(height * 0.28)),
            (round(width * 0.76) + offset, round(height * 0.46)),
            (round(width * 0.72) + offset, round(height * 0.62)),
            (round(width * 0.68) + offset, round(height * 0.45)),
        ),
        fill=248,
    )
    return image.filter(ImageFilter.GaussianBlur(radius=0.45))


def _frame_counter() -> Iterator[int]:
    return itertools.count()


def main() -> None:
    rclpy.init()
    node = ThermalMockPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
