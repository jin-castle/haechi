from firesight_ros2.mmwave_mapping_launch import (
    mapper_launch_arguments,
    mapper_nodes,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("firesight_ros2")
    default_params = PathJoinSubstitution(
        [package_share, "config", "mmwave_mapping.yaml"],
    )
    input_topic = LaunchConfiguration("input_topic")
    thermal_topic = LaunchConfiguration("thermal_topic")
    edge_overlay_topic = LaunchConfiguration("edge_overlay_topic")
    obstacle_topic = LaunchConfiguration("obstacle_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "input_topic",
                default_value="/ti_mmwave/radar_scan_pcl",
            ),
            DeclareLaunchArgument("thermal_topic", default_value="/thermal_camera/image_raw"),
            DeclareLaunchArgument(
                "edge_mask_topic",
                default_value="/firesight/thermal/edge_mask",
            ),
            DeclareLaunchArgument(
                "edge_overlay_topic",
                default_value="/firesight/thermal/edge_overlay",
            ),
            DeclareLaunchArgument(
                "obstacle_topic",
                default_value="/firesight/mmwave/front_obstacles",
            ),
            DeclareLaunchArgument(
                "fused_overlay_topic",
                default_value="/firesight/hud/fused_overlay",
            ),
            DeclareLaunchArgument("mock_range_m", default_value="0.85"),
            DeclareLaunchArgument("mock_angle_deg", default_value="0.0"),
            *mapper_launch_arguments(),
            Node(
                package="firesight_ros2",
                executable="thermal_mock_publisher",
                name="thermal_mock_publisher",
                output="screen",
                parameters=[{"topic": thermal_topic}],
            ),
            Node(
                package="firesight_ros2",
                executable="mmwave_mock_pointcloud_publisher",
                name="mmwave_mock_pointcloud_publisher",
                output="screen",
                parameters=[
                    {
                        "topic": input_topic,
                        "range_m": LaunchConfiguration("mock_range_m"),
                        "angle_deg": LaunchConfiguration("mock_angle_deg"),
                    },
                ],
            ),
            *mapper_nodes(input_topic),
            Node(
                package="firesight_ros2",
                executable="thermal_edge_node",
                name="thermal_edge_node",
                output="screen",
                parameters=[
                    {
                        "thermal_topic": thermal_topic,
                        "edge_mask_topic": LaunchConfiguration("edge_mask_topic"),
                        "edge_overlay_topic": edge_overlay_topic,
                    },
                ],
            ),
            Node(
                package="firesight_ros2",
                executable="hud_fusion_node",
                name="hud_fusion_node",
                output="screen",
                parameters=[
                    {
                        "thermal_overlay_topic": edge_overlay_topic,
                        "mmwave_obstacle_topic": obstacle_topic,
                        "fused_overlay_topic": LaunchConfiguration("fused_overlay_topic"),
                    },
                ],
            ),
        ],
    )
