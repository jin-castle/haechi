from firesight_ros2.mmwave_mapping_launch import (
    mapper_launch_arguments,
    mapper_nodes,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    firesight_share = FindPackageShare("firesight_ros2")
    ti_share = FindPackageShare("ti_mmwave_rospkg")
    input_topic = LaunchConfiguration("input_topic")
    default_params = PathJoinSubstitution(
        [firesight_share, "config", "mmwave_mapping.yaml"],
    )
    default_rviz = PathJoinSubstitution(
        [firesight_share, "rviz", "mmwave_mapping.rviz"],
    )
    ti_launch = PathJoinSubstitution(
        [ti_share, "launch", "6843AOP_Standard.py"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("ti_rviz", default_value="false"),
            DeclareLaunchArgument("cfg_file", default_value="6843AOP_Standard.cfg"),
            DeclareLaunchArgument("command_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("data_port", default_value="/dev/ttyUSB1"),
            DeclareLaunchArgument(
                "input_topic",
                default_value="/ti_mmwave/radar_scan_pcl",
            ),
            *mapper_launch_arguments(),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(ti_launch),
                launch_arguments={
                    "rviz": LaunchConfiguration("ti_rviz"),
                    "cfg_file": LaunchConfiguration("cfg_file"),
                    "command_port": LaunchConfiguration("command_port"),
                    "data_port": LaunchConfiguration("data_port"),
                }.items(),
            ),
            *mapper_nodes(input_topic),
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
