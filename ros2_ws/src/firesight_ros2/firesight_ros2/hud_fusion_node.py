from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from PIL import Image, ImageDraw
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMessage
from visualization_msgs.msg import Marker, MarkerArray

PANEL_BACKGROUND = (0, 0, 0, 150)
PANEL_GRID = (140, 150, 155, 125)
NEAR_WARNING = (255, 50, 50, 230)
MID_WARNING = (255, 178, 36, 220)
FAR_WARNING = (66, 180, 255, 210)
CLEAR_STATUS = (42, 220, 120, 230)


@dataclass(frozen=True, slots=True)
class FrontObstacle:
    range_m: float
    angle_deg: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PanelGeometry:
    left: int
    right: int
    top: int
    bottom: int
    center_x: int


class HudFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("hud_fusion_node")
        self._declare_parameters()
        self._thermal_overlay_topic = self.get_parameter("thermal_overlay_topic").value
        self._mmwave_obstacle_topic = self.get_parameter("mmwave_obstacle_topic").value
        self._fused_overlay_topic = self.get_parameter("fused_overlay_topic").value
        self._max_sync_delay_ns = int(
            float(self.get_parameter("max_sync_delay_ms").value) * 1_000_000,
        )
        self._range_max_m = float(self.get_parameter("range_max_m").value)
        self._fov_deg = float(self.get_parameter("fov_deg").value)
        self._latest_obstacles: MarkerArray | None = None

        self._publisher = self.create_publisher(
            ImageMessage,
            self._fused_overlay_topic,
            10,
        )
        self._overlay_subscription = self.create_subscription(
            ImageMessage,
            self._thermal_overlay_topic,
            self._on_overlay,
            10,
        )
        self._obstacle_subscription = self.create_subscription(
            MarkerArray,
            self._mmwave_obstacle_topic,
            self._on_obstacles,
            10,
        )
        self.get_logger().info(
            f"HUD fusion node: {self._thermal_overlay_topic} + "
            f"{self._mmwave_obstacle_topic} -> {self._fused_overlay_topic}",
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "thermal_overlay_topic",
            "/firesight/thermal/edge_overlay",
        )
        self.declare_parameter(
            "mmwave_obstacle_topic",
            "/firesight/mmwave/front_obstacles",
        )
        self.declare_parameter("fused_overlay_topic", "/firesight/hud/fused_overlay")
        self.declare_parameter("max_sync_delay_ms", 100.0)
        self.declare_parameter("range_max_m", 1.5)
        self.declare_parameter("fov_deg", 120.0)

    def _on_obstacles(self, message: MarkerArray) -> None:
        self._latest_obstacles = message

    def _on_overlay(self, message: ImageMessage) -> None:
        try:
            overlay = _image_message_to_rgb(message)
        except ValueError as error:
            self.get_logger().warning(f"skipping overlay frame: {error}")
            return

        obstacles = self._synced_obstacles(message)
        fused = overlay.convert("RGBA")
        panel_layer = Image.new("RGBA", fused.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel_layer)
        _draw_mmwave_panel(
            draw,
            fused.size,
            obstacles,
            range_max_m=self._range_max_m,
            fov_deg=self._fov_deg,
        )
        fused_message = _pil_to_image_message(
            Image.alpha_composite(fused, panel_layer).convert("RGB"),
            message,
        )
        self._publisher.publish(fused_message)

    def _synced_obstacles(self, image_message: ImageMessage) -> tuple[FrontObstacle, ...]:
        latest = self._latest_obstacles
        if latest is None or len(latest.markers) == 0:
            return ()
        latest_stamp_ns = _marker_stamp_ns(latest)
        image_stamp_ns = _stamp_to_ns(image_message.header.stamp)
        if latest_stamp_ns is not None:
            delta_ns = abs(image_stamp_ns - latest_stamp_ns)
            if delta_ns > self._max_sync_delay_ns:
                return ()
        return tuple(_front_obstacle(marker) for marker in latest.markers if _is_add(marker))


def _draw_mmwave_panel(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    obstacles: tuple[FrontObstacle, ...],
    *,
    range_max_m: float,
    fov_deg: float,
) -> None:
    width, height = image_size
    panel_height = max(92, round(height * 0.28))
    geometry = PanelGeometry(
        left=round(width * 0.08),
        right=round(width * 0.92),
        top=height - panel_height - 10,
        bottom=height - 10,
        center_x=width // 2,
    )
    draw.rectangle(
        (geometry.left, geometry.top, geometry.right, geometry.bottom),
        fill=PANEL_BACKGROUND,
    )
    for angle_deg in (-20.0, 20.0):
        x = _angle_to_x(angle_deg, geometry.left, geometry.right, fov_deg)
        draw.line((x, geometry.top, x, geometry.bottom), fill=PANEL_GRID, width=1)
    for range_m in (0.5, 1.0):
        y = _range_to_y(range_m, geometry.top, geometry.bottom, range_max_m)
        draw.line((geometry.left, y, geometry.right, y), fill=PANEL_GRID, width=1)
    draw.line(
        (geometry.center_x, geometry.bottom, geometry.center_x, geometry.top),
        fill=(185, 185, 185, 150),
        width=1,
    )

    if len(obstacles) == 0:
        draw.text(
            (geometry.left + 8, geometry.top + 8),
            "mmWave: clear",
            fill=CLEAR_STATUS,
        )
        return

    closest = min(obstacles, key=lambda obstacle: obstacle.range_m)
    draw.text(
        (geometry.left + 8, geometry.top + 8),
        f"mmWave: {_distance_band(closest.range_m)} {closest.range_m:.2f}m",
        fill=_band_color(closest.range_m),
    )
    for obstacle in obstacles:
        _draw_detection_marker(
            draw,
            obstacle,
            geometry,
            range_max_m=range_max_m,
            fov_deg=fov_deg,
        )


def _draw_detection_marker(
    draw: ImageDraw.ImageDraw,
    obstacle: FrontObstacle,
    geometry: PanelGeometry,
    *,
    range_max_m: float,
    fov_deg: float,
) -> None:
    x = _angle_to_x(obstacle.angle_deg, geometry.left, geometry.right, fov_deg)
    y = _range_to_y(obstacle.range_m, geometry.top, geometry.bottom, range_max_m)
    color = _band_color(obstacle.range_m)
    radius = round(7 + (obstacle.confidence * 7))
    draw.line((geometry.center_x, geometry.bottom, x, y), fill=color, width=2)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=3,
    )
    inner_radius = max(3, radius // 2)
    draw.ellipse(
        (x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius),
        fill=color,
    )


def _image_message_to_rgb(message: ImageMessage) -> Image.Image:
    encoding = message.encoding.lower()
    width = int(message.width)
    height = int(message.height)
    if encoding == "rgb8":
        data = _compact_rows(bytes(message.data), height, int(message.step), width * 3)
        return Image.frombytes("RGB", (width, height), data)
    if encoding == "bgr8":
        data = _compact_rows(bytes(message.data), height, int(message.step), width * 3)
        return Image.frombytes("RGB", (width, height), data, "raw", "BGR")
    if encoding in {"mono8", "8uc1"}:
        data = _compact_rows(bytes(message.data), height, int(message.step), width)
        return Image.frombytes("L", (width, height), data).convert("RGB")
    msg = f"unsupported overlay image encoding: {message.encoding}"
    raise ValueError(msg)


def _pil_to_image_message(image: Image.Image, source: ImageMessage) -> ImageMessage:
    converted = image.convert("RGB")
    message = ImageMessage()
    message.header = source.header
    message.height = converted.height
    message.width = converted.width
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = converted.width * 3
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


def _is_add(marker: Marker) -> bool:
    return marker.action == Marker.ADD


def _front_obstacle(marker: Marker) -> FrontObstacle:
    x = float(marker.pose.position.x)
    y = float(marker.pose.position.y)
    scale = max(float(marker.scale.x), float(marker.scale.y), 0.12)
    return FrontObstacle(
        range_m=math.hypot(x, y),
        angle_deg=math.degrees(math.atan2(y, x)),
        confidence=min(1.0, max(0.25, scale / 0.35)),
    )


def _marker_stamp_ns(markers: MarkerArray) -> int | None:
    for marker in markers.markers:
        if _is_add(marker):
            return _stamp_to_ns(marker.header.stamp)
    return None


def _stamp_to_ns(stamp: object) -> int:
    return (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)


def _angle_to_x(angle_deg: float, left: int, right: int, fov_deg: float) -> int:
    half_fov = max(1.0, fov_deg / 2.0)
    normalized = (max(-half_fov, min(half_fov, angle_deg)) + half_fov) / fov_deg
    return left + round((right - left) * normalized)


def _range_to_y(range_m: float, top: int, bottom: int, range_max_m: float) -> int:
    normalized = (max(0.2, min(range_max_m, range_m)) - 0.2) / (range_max_m - 0.2)
    return bottom - round((bottom - top) * normalized)


def _distance_band(range_m: float) -> str:
    if range_m <= 0.5:
        return "near"
    if range_m <= 1.0:
        return "mid"
    return "far"


def _band_color(range_m: float) -> tuple[int, int, int, int]:
    if range_m <= 0.5:
        return NEAR_WARNING
    if range_m <= 1.0:
        return MID_WARNING
    return FAR_WARNING


def main() -> None:
    rclpy.init()
    node = HudFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
