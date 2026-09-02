from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def mapper_launch_arguments() -> list[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument("extrinsic_x", default_value="0.035"),
        DeclareLaunchArgument("extrinsic_y", default_value="0.0"),
        DeclareLaunchArgument("extrinsic_z", default_value="-0.015"),
        DeclareLaunchArgument("extrinsic_roll", default_value="0.0"),
        DeclareLaunchArgument("extrinsic_pitch", default_value="0.0"),
        DeclareLaunchArgument("extrinsic_yaw", default_value="0.0"),
    ]


def mapper_nodes(input_topic: LaunchConfiguration) -> list[Node]:
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="glasses_to_mmwave_tf",
            arguments=[
                "--x",
                LaunchConfiguration("extrinsic_x"),
                "--y",
                LaunchConfiguration("extrinsic_y"),
                "--z",
                LaunchConfiguration("extrinsic_z"),
                "--roll",
                LaunchConfiguration("extrinsic_roll"),
                "--pitch",
                LaunchConfiguration("extrinsic_pitch"),
                "--yaw",
                LaunchConfiguration("extrinsic_yaw"),
                "--frame-id",
                "glasses_frame",
                "--child-frame-id",
                "mmwave_front_link",
            ],
        ),
        Node(
            package="firesight_ros2",
            executable="mmwave_obstacle_mapper",
            name="mmwave_obstacle_mapper",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"input_topic": input_topic},
            ],
        ),
    ]
