from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROS2_PACKAGE = ROOT / "ros2_ws" / "src" / "firesight_ros2"
ROS2_SRC = ROOT / "ros2_ws" / "src"


def test_ros2_mmwave_mapper_scaffold_exists() -> None:
    expected_paths = (
        ROS2_PACKAGE / "package.xml",
        ROS2_PACKAGE / "setup.py",
        ROS2_PACKAGE / "launch" / "mmwave_mapping.launch.py",
        ROS2_PACKAGE / "launch" / "mmwave_mapping_demo.launch.py",
        ROS2_PACKAGE / "launch" / "mmwave_sensor_mapping.launch.py",
        ROS2_PACKAGE / "launch" / "thermal_mmwave_fusion_demo.launch.py",
        ROS2_PACKAGE / "config" / "mmwave_mapping.yaml",
        ROS2_PACKAGE / "config" / "extrinsics.yaml",
        ROS2_PACKAGE / "rviz" / "mmwave_mapping.rviz",
        ROS2_PACKAGE / "firesight_ros2" / "hud_fusion_node.py",
        ROS2_PACKAGE / "firesight_ros2" / "mmwave_obstacle_mapper.py",
        ROS2_PACKAGE / "firesight_ros2" / "mmwave_mock_pointcloud_publisher.py",
        ROS2_PACKAGE / "firesight_ros2" / "thermal_edge_node.py",
        ROS2_PACKAGE / "firesight_ros2" / "thermal_mock_publisher.py",
    )

    for path in expected_paths:
        assert path.exists(), path


def test_haechi_embedded_ros2_packages_are_available() -> None:
    expected_packages = (
        "fusion_imu_radar",
        "haechi",
        "haechi_cmake_compat",
        "imu",
        "serial-ros2-master",
        "thermal_camera",
        "ti_mmwave_rospkg",
        "ti_mmwave_rospkg_msgs",
    )

    for package in expected_packages:
        assert (ROS2_SRC / package / "package.xml").exists(), package


def test_ros2_mmwave_topics_are_documented_consistently() -> None:
    mapper = (ROS2_PACKAGE / "firesight_ros2" / "mmwave_obstacle_mapper.py").read_text(
        encoding="utf-8"
    )
    config = (ROS2_PACKAGE / "config" / "mmwave_mapping.yaml").read_text(
        encoding="utf-8",
    )
    docs = (ROOT / "docs" / "mmwave_ros2_calibration.md").read_text(
        encoding="utf-8",
    )
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ros2_readme = (ROOT / "ros2_ws" / "README.md").read_text(
        encoding="utf-8",
    )
    launch = (ROS2_PACKAGE / "launch" / "mmwave_mapping.launch.py").read_text(
        encoding="utf-8",
    )
    sensor_launch = (
        ROS2_PACKAGE / "launch" / "mmwave_sensor_mapping.launch.py"
    ).read_text(
        encoding="utf-8",
    )
    demo_launch = (ROS2_PACKAGE / "launch" / "mmwave_mapping_demo.launch.py").read_text(
        encoding="utf-8",
    )
    thermal_demo_launch = (
        ROS2_PACKAGE / "launch" / "thermal_mmwave_fusion_demo.launch.py"
    ).read_text(
        encoding="utf-8",
    )
    real_fusion_launch = (
        ROS2_PACKAGE / "launch" / "real_sensor_fusion.launch.py"
    ).read_text(
        encoding="utf-8",
    )
    rviz = (ROS2_PACKAGE / "rviz" / "mmwave_mapping.rviz").read_text(
        encoding="utf-8",
    )

    for topic in (
        "/ti_mmwave/radar_scan_pcl",
        "/firesight/mmwave/front_grid",
        "/firesight/mmwave/front_markers",
        "/firesight/mmwave/front_obstacles",
    ):
        assert topic in (
            f"{mapper}\n{launch}\n{demo_launch}\n{sensor_launch}\n"
            f"{thermal_demo_launch}\n{real_fusion_launch}"
        )
        assert topic in f"{config}\n{rviz}"
        assert topic in f"{docs}\n{root_readme}\n{ros2_readme}"

    assert "input_topic:=/mmwave/points" not in root_readme
    assert "input_topic:=/mmwave/points" not in ros2_readme


def test_ros2_thermal_mmwave_fusion_topics_are_scaffolded() -> None:
    setup_py = (ROS2_PACKAGE / "setup.py").read_text(encoding="utf-8")
    package_xml = (ROS2_PACKAGE / "package.xml").read_text(encoding="utf-8")
    thermal_node = (ROS2_PACKAGE / "firesight_ros2" / "thermal_edge_node.py").read_text(
        encoding="utf-8",
    )
    hud_node = (ROS2_PACKAGE / "firesight_ros2" / "hud_fusion_node.py").read_text(
        encoding="utf-8",
    )
    demo_launch = (
        ROS2_PACKAGE / "launch" / "thermal_mmwave_fusion_demo.launch.py"
    ).read_text(
        encoding="utf-8",
    )
    real_launch = (ROS2_PACKAGE / "launch" / "real_sensor_fusion.launch.py").read_text(
        encoding="utf-8",
    )
    docs = (ROOT / "docs" / "thermal_mmwave_hud_fusion.md").read_text(
        encoding="utf-8",
    )

    for executable in (
        "thermal_edge_node",
        "thermal_mock_publisher",
        "hud_fusion_node",
    ):
        assert executable in setup_py
        assert executable in f"{demo_launch}\n{real_launch}"

    for topic in (
        "/thermal_camera/image_raw",
        "/firesight/thermal/edge_mask",
        "/firesight/thermal/edge_overlay",
        "/firesight/hud/fused_overlay",
    ):
        assert topic in f"{thermal_node}\n{hud_node}\n{demo_launch}\n{real_launch}"
        assert topic in docs

    assert "python3-pil" in package_xml
    assert "max_sync_delay_ms" in hud_node
    assert "100.0" in hud_node


def test_haechi_ti_launch_matches_mapper_frame_and_config() -> None:
    launch = (ROS2_SRC / "ti_mmwave_rospkg" / "launch" / "IWR6843.py").read_text(
        encoding="utf-8",
    )

    assert "'global_params.yaml'" in launch
    assert "'launch/*.rviz'" not in launch
    assert '{"frame_id": "mmwave_front_link"}' in launch


def test_real_mapping_launch_opens_rviz_by_default() -> None:
    launch = (ROS2_PACKAGE / "launch" / "mmwave_mapping.launch.py").read_text(
        encoding="utf-8",
    )

    assert 'DeclareLaunchArgument("rviz", default_value="true")' in launch
    assert 'package="rviz2"' in launch
    assert 'condition=IfCondition(LaunchConfiguration("rviz"))' in launch


def test_sensor_mapping_launch_starts_ti_driver_and_mapper() -> None:
    launch = (ROS2_PACKAGE / "launch" / "mmwave_sensor_mapping.launch.py").read_text(
        encoding="utf-8",
    )

    assert '"6843AOP_Standard.py"' in launch
    assert (
        'DeclareLaunchArgument("command_port", default_value="/dev/ttyUSB0")' in launch
    )
    assert 'DeclareLaunchArgument("data_port", default_value="/dev/ttyUSB1")' in launch
    assert '"rviz": LaunchConfiguration("ti_rviz")' in launch
    assert 'default_value="/ti_mmwave/radar_scan_pcl"' in launch
    assert "mapper_nodes(input_topic)" in launch


def test_default_rviz_view_is_simple_for_obstacle_checking() -> None:
    rviz = (ROS2_PACKAGE / "rviz" / "mmwave_mapping.rviz").read_text(
        encoding="utf-8",
    )

    assert "Enabled: true\n      Line Style:" in rviz
    assert "Name: Grid\n      Normal Cell Count: 0" in rviz
    assert "Plane Cell Count: 6" in rviz
    assert "All Enabled: false" in rviz
    assert "glasses_frame:\n          Value: false" in rviz
    assert "mmwave_front_link:\n          Value: true" in rviz
    assert "Name: TF\n      Show Arrows: false" in rviz
    assert "Show Axes: true" in rviz
    assert "Show Names: false" in rviz
    assert "Hide Left Dock: true" in rviz


def test_ros2_setup_script_targets_host_lts_pair() -> None:
    install_script = (ROOT / "scripts" / "install_ros2_ubuntu.sh").read_text(
        encoding="utf-8",
    )
    ros2_readme = (ROOT / "ros2_ws" / "README.md").read_text(encoding="utf-8")

    assert 'ROS_DISTRO_NAME="jazzy"' in install_script
    assert 'ROS_DISTRO_NAME="lyrical"' in install_script
    assert "jazzy:noble | lyrical:resolute" in install_script
    assert "Lyrical on Ubuntu 26.04 Resolute" in ros2_readme
