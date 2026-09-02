from firesight_ros2.mmwave_mapping_launch import (
    mapper_launch_arguments,
    mapper_nodes,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("firesight_ros2")
    default_params = PathJoinSubstitution(
        [package_share, "config", "mmwave_mapping.yaml"],
    )
    default_rviz = PathJoinSubstitution(
        [package_share, "rviz", "mmwave_mapping.rviz"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "input_topic",
                default_value="/ti_mmwave/radar_scan_pcl",
            ),
            *mapper_launch_arguments(),
            *mapper_nodes(LaunchConfiguration("input_topic")),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                additional_env={
                    "LIBGL_ALWAYS_SOFTWARE": "1",
                    "QT_QPA_PLATFORM": "xcb",
                },
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ],
    )
