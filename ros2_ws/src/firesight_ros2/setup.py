from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "firesight_ros2"


def data_files() -> list[tuple[str, list[str]]]:
    share = Path("share") / package_name
    return [
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (str(share), ["package.xml"]),
        (str(share / "config"), glob("config/*.yaml")),
        (str(share / "launch"), glob("launch/*.launch.py")),
        (str(share / "rviz"), glob("rviz/*.rviz")),
    ]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FireSight",
    maintainer_email="dev@firesight.local",
    description="ROS 2 bring-up package for FireSight mmWave front obstacle mapping.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "hud_fusion_node = firesight_ros2.hud_fusion_node:main",
            "mmwave_obstacle_mapper = firesight_ros2.mmwave_obstacle_mapper:main",
            "mmwave_mock_pointcloud_publisher = firesight_ros2.mmwave_mock_pointcloud_publisher:main",
            "sensor_readiness_check = firesight_ros2.sensor_readiness_check:main",
            "thermal_edge_node = firesight_ros2.thermal_edge_node:main",
            "thermal_mock_publisher = firesight_ros2.thermal_mock_publisher:main",
        ],
    },
)
