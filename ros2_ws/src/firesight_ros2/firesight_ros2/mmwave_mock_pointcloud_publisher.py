from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class MmWaveMockPointCloudPublisher(Node):
    def __init__(self) -> None:
        super().__init__("mmwave_mock_pointcloud_publisher")
        self.declare_parameter("topic", "/ti_mmwave/radar_scan_pcl")
        self.declare_parameter("frame_id", "mmwave_front_link")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("range_m", 1.0)
        self.declare_parameter("angle_deg", 0.0)
        self.declare_parameter("spread_m", 0.08)
        self.declare_parameter("point_count", 36)
        self._topic = str(self.get_parameter("topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._publisher = self.create_publisher(PointCloud2, self._topic, 10)
        self._tick = 0
        self.create_timer(1.0 / max(rate_hz, 0.1), self._publish_points)
        self.get_logger().info(f"publishing mock PointCloud2 on {self._topic}")

    def _publish_points(self) -> None:
        range_m = float(self.get_parameter("range_m").value)
        angle_deg = float(self.get_parameter("angle_deg").value)
        spread_m = float(self.get_parameter("spread_m").value)
        point_count = int(self.get_parameter("point_count").value)
        phase = self._tick * 0.08
        center_angle = math.radians(angle_deg + (math.sin(phase) * 12.0))
        center_x = range_m * math.cos(center_angle)
        center_y = range_m * math.sin(center_angle)
        points: list[tuple[float, float, float]] = []
        for index in range(max(point_count, 1)):
            theta = (math.tau * index / max(point_count, 1)) + phase
            radius = spread_m * (0.35 + (0.65 * ((index % 5) / 4.0)))
            points.append(
                (
                    center_x + (math.cos(theta) * radius),
                    center_y + (math.sin(theta) * radius),
                    -0.05 + (0.1 * ((index % 3) / 2.0)),
                ),
            )

        message = point_cloud2.create_cloud_xyz32(
            header=self._header(),
            points=points,
        )
        self._publisher.publish(message)
        self._tick += 1

    def _header(self) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        return header


def main() -> None:
    rclpy.init()
    node = MmWaveMockPointCloudPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
