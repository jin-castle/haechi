from firesight_ros2.mmwave_mapping_launch import mapper_launch_arguments
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    firesight_share = FindPackageShare("firesight_ros2")
    mmwave_launch = PathJoinSubstitution(
        [firesight_share, "launch", "mmwave_sensor_mapping.launch.py"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("ti_rviz", default_value="false"),
            DeclareLaunchArgument("cfg_file", default_value="6843AOP_Standard.cfg"),
            DeclareLaunchArgument("command_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("data_port", default_value="/dev/ttyUSB1"),
            DeclareLaunchArgument("input_topic", default_value="/ti_mmwave/radar_scan_pcl"),
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
            DeclareLaunchArgument("start_mmwave", default_value="true"),
            DeclareLaunchArgument("start_imu", default_value="true"),
            DeclareLaunchArgument("start_fusion", default_value="true"),
            DeclareLaunchArgument("imu_driver", default_value="lsm6dsv16x"),
            DeclareLaunchArgument("i2c_bus", default_value="7"),
            DeclareLaunchArgument("i2c_address", default_value="0"),
            DeclareLaunchArgument("imu_publish_rate", default_value="50.0"),
            *mapper_launch_arguments(),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(mmwave_launch),
                launch_arguments={
                    "rviz": LaunchConfiguration("rviz"),
                    "ti_rviz": LaunchConfiguration("ti_rviz"),
                    "cfg_file": LaunchConfiguration("cfg_file"),
                    "command_port": LaunchConfiguration("command_port"),
                    "data_port": LaunchConfiguration("data_port"),
                    "input_topic": LaunchConfiguration("input_topic"),
                    "extrinsic_x": LaunchConfiguration("extrinsic_x"),
                    "extrinsic_y": LaunchConfiguration("extrinsic_y"),
                    "extrinsic_z": LaunchConfiguration("extrinsic_z"),
                    "extrinsic_roll": LaunchConfiguration("extrinsic_roll"),
                    "extrinsic_pitch": LaunchConfiguration("extrinsic_pitch"),
                    "extrinsic_yaw": LaunchConfiguration("extrinsic_yaw"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("start_mmwave")),
            ),
            Node(
                package="imu",
                executable="imu_node",
                name="imu_node",
                output="screen",
                parameters=[
                    {"driver": LaunchConfiguration("imu_driver")},
                    {"i2c_bus": ParameterValue(LaunchConfiguration("i2c_bus"), value_type=int)},
                    {"i2c_address": ParameterValue(LaunchConfiguration("i2c_address"), value_type=int)},
                    {
                        "publish_rate": ParameterValue(
                            LaunchConfiguration("imu_publish_rate"),
                            value_type=float,
                        ),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_imu")),
            ),
            Node(
                package="fusion_imu_radar",
                executable="fusion_imu_radar_node",
                name="fusion_imu_radar_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_fusion")),
            ),
            Node(
                package="firesight_ros2",
                executable="thermal_edge_node",
                name="thermal_edge_node",
                output="screen",
                parameters=[
                    {
                        "thermal_topic": LaunchConfiguration("thermal_topic"),
                        "edge_mask_topic": LaunchConfiguration("edge_mask_topic"),
                        "edge_overlay_topic": LaunchConfiguration("edge_overlay_topic"),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_fusion")),
            ),
            Node(
                package="firesight_ros2",
                executable="hud_fusion_node",
                name="hud_fusion_node",
                output="screen",
                parameters=[
                    {
                        "thermal_overlay_topic": LaunchConfiguration(
                            "edge_overlay_topic",
                        ),
                        "mmwave_obstacle_topic": LaunchConfiguration("obstacle_topic"),
                        "fused_overlay_topic": LaunchConfiguration("fused_overlay_topic"),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_fusion")),
            ),
        ],
    )
