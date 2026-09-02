from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from firesight_ros2.mmwave_temporal_filter import (
    FrontPoint,
    GridSpec,
    OccupiedCell,
    ObstacleCluster,
    StabilizationConfig,
    TemporalObstacleTracker,
    cluster_occupied_cells,
    point_xyz,
    range_color,
)

@dataclass(frozen=True, slots=True)
class MapperConfig:
    input_topic: str
    grid_topic: str
    marker_topic: str
    obstacle_topic: str
    frame_id: str
    range_min_m: float
    range_max_m: float
    fov_deg: float
    z_min_m: float
    z_max_m: float
    grid_resolution_m: float
    occupancy_value: int
    marker_lifetime_sec: float
    publish_empty_grid: bool
    stabilization: StabilizationConfig


class MmWaveObstacleMapper(Node):
    def __init__(self) -> None:
        super().__init__("mmwave_obstacle_mapper")
        self._declare_parameters()
        self.config = self._read_config()
        self._side_extent_m = math.tan(math.radians(self.config.fov_deg / 2.0)) * (
            self.config.range_max_m
        )
        self._grid_width = max(
            1,
            math.ceil(self.config.range_max_m / self.config.grid_resolution_m),
        )
        self._grid_height = max(
            1,
            math.ceil((self._side_extent_m * 2.0) / self.config.grid_resolution_m),
        )
        grid = GridSpec(
            width=self._grid_width,
            height=self._grid_height,
            side_extent_m=self._side_extent_m,
            resolution_m=self.config.grid_resolution_m,
        )
        self._tracker = TemporalObstacleTracker(grid=grid, config=self.config.stabilization)
        self._grid_publisher = self.create_publisher(
            OccupancyGrid,
            self.config.grid_topic,
            10,
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            self.config.marker_topic,
            10,
        )
        self._obstacle_publisher = self.create_publisher(
            MarkerArray,
            self.config.obstacle_topic,
            10,
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            self.config.input_topic,
            self._on_points,
            10,
        )
        self.get_logger().info(
            f"mapping {self.config.input_topic} to "
            f"{self.config.grid_topic} and {self.config.marker_topic}",
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_topic", "/ti_mmwave/radar_scan_pcl")
        self.declare_parameter("grid_topic", "/firesight/mmwave/front_grid")
        self.declare_parameter("marker_topic", "/firesight/mmwave/front_markers")
        self.declare_parameter("obstacle_topic", "/firesight/mmwave/front_obstacles")
        self.declare_parameter("frame_id", "mmwave_front_link")
        self.declare_parameter("range_min_m", 0.2)
        self.declare_parameter("range_max_m", 1.5)
        self.declare_parameter("fov_deg", 120.0)
        self.declare_parameter("z_min_m", -0.6)
        self.declare_parameter("z_max_m", 0.8)
        self.declare_parameter("grid_resolution_m", 0.05)
        self.declare_parameter("occupancy_value", 100)
        self.declare_parameter("marker_lifetime_sec", 0.25)
        self.declare_parameter("publish_empty_grid", value=True)
        self.declare_parameter("stability_hit_increment", 0.6)
        self.declare_parameter("stability_decay_per_frame", 0.18)
        self.declare_parameter("stability_occupied_threshold", 1.0)
        self.declare_parameter("stability_max_score", 2.0)
        self.declare_parameter("min_cluster_cells", 1)

    def _read_config(self) -> MapperConfig:
        return MapperConfig(
            input_topic=self.get_parameter("input_topic").value,
            grid_topic=self.get_parameter("grid_topic").value,
            marker_topic=self.get_parameter("marker_topic").value,
            obstacle_topic=self.get_parameter("obstacle_topic").value,
            frame_id=self.get_parameter("frame_id").value,
            range_min_m=float(self.get_parameter("range_min_m").value),
            range_max_m=float(self.get_parameter("range_max_m").value),
            fov_deg=float(self.get_parameter("fov_deg").value),
            z_min_m=float(self.get_parameter("z_min_m").value),
            z_max_m=float(self.get_parameter("z_max_m").value),
            grid_resolution_m=float(self.get_parameter("grid_resolution_m").value),
            occupancy_value=int(self.get_parameter("occupancy_value").value),
            marker_lifetime_sec=float(
                self.get_parameter("marker_lifetime_sec").value,
            ),
            publish_empty_grid=bool(self.get_parameter("publish_empty_grid").value),
            stabilization=StabilizationConfig(
                hit_increment=float(self.get_parameter("stability_hit_increment").value),
                decay_per_frame=float(
                    self.get_parameter("stability_decay_per_frame").value,
                ),
                occupied_threshold=float(
                    self.get_parameter("stability_occupied_threshold").value,
                ),
                max_score=float(self.get_parameter("stability_max_score").value),
                min_cluster_cells=int(self.get_parameter("min_cluster_cells").value),
            ),
        )

    def _on_points(self, message: PointCloud2) -> None:
        points = tuple(self._front_points(message))
        stable_cells = self._tracker.update(points)
        if stable_cells or self.config.publish_empty_grid:
            self._grid_publisher.publish(self._build_grid(message, stable_cells))
        clusters = cluster_occupied_cells(
            stable_cells,
            min_cluster_cells=self.config.stabilization.min_cluster_cells,
        )
        markers = self._build_markers(message, clusters)
        self._marker_publisher.publish(markers)
        self._obstacle_publisher.publish(markers)

    def _front_points(self, message: PointCloud2) -> list[FrontPoint]:
        front_points: list[FrontPoint] = []
        for raw_point in point_cloud2.read_points(
            message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            x, y, z = point_xyz(raw_point)
            range_m = math.hypot(x, y)
            if range_m < self.config.range_min_m or range_m > self.config.range_max_m:
                continue
            if z < self.config.z_min_m or z > self.config.z_max_m:
                continue
            angle_deg = math.degrees(math.atan2(y, x))
            if abs(angle_deg) > self.config.fov_deg / 2.0:
                continue
            front_points.append(
                FrontPoint(
                    x=x,
                    y=y,
                    z=z,
                    range_m=range_m,
                    angle_deg=angle_deg,
                ),
            )
        return front_points

    def _build_grid(
        self,
        source: PointCloud2,
        cells: tuple[OccupiedCell, ...],
    ) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = source.header.stamp
        grid.header.frame_id = self.config.frame_id
        grid.info.resolution = self.config.grid_resolution_m
        grid.info.width = self._grid_width
        grid.info.height = self._grid_height
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -self._side_extent_m
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        data = [0] * (self._grid_width * self._grid_height)
        for occupied in cells:
            data[(occupied.cell.y * self._grid_width) + occupied.cell.x] = (
                self.config.occupancy_value
            )
        grid.data = data
        return grid

    def _build_markers(
        self,
        source: PointCloud2,
        clusters: tuple[ObstacleCluster, ...],
    ) -> MarkerArray:
        markers = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.stamp = source.header.stamp
        clear_marker.header.frame_id = self.config.frame_id
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        for index, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.stamp = source.header.stamp
            marker.header.frame_id = self.config.frame_id
            marker.ns = "stable_front_obstacle"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=cluster.center_x_m, y=cluster.center_y_m, z=0.0)
            marker.pose.orientation.w = 1.0
            diameter = min(
                0.35,
                max(0.12, math.sqrt(cluster.cell_count) * self.config.grid_resolution_m * 1.5),
            )
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = 0.12
            marker.color.a = 0.9
            marker.color.r, marker.color.g, marker.color.b = range_color(cluster.range_m)
            marker.lifetime.sec = int(self.config.marker_lifetime_sec)
            marker.lifetime.nanosec = int(
                (self.config.marker_lifetime_sec % 1.0) * 1_000_000_000,
            )
            markers.markers.append(marker)
        return markers

def main() -> None:
    rclpy.init()
    node = MmWaveObstacleMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
