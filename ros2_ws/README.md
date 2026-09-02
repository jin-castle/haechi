# FireSight ROS 2 Sensor Workspace

This workspace is for the first hardware gate: prove that an mmWave sensor can
map front obstacles in the 0.2-1.5 m range before fusing thermal camera data.

The setup script targets the LTS ROS 2 release matching the Ubuntu host:
Lyrical on Ubuntu 26.04 Resolute, or Jazzy on Ubuntu 24.04 Noble. The current
local WSL distro is Ubuntu 26.04, so it will select Lyrical unless
`ROS_DISTRO_NAME` is overridden.

## Install on WSL2 or Native Ubuntu 24.04

From the repo root:

```bash
bash scripts/install_ros2_ubuntu.sh
source /opt/ros/${ROS_DISTRO_NAME:-lyrical}/setup.bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Quick check:

```bash
bash ../scripts/ros2_sensor_doctor.sh
```

## Run mmWave Sensor Mapping

The combined launch starts the Haechi/TI mmWave driver, the FireSight mapper,
and RViz. It expects the sensor command port on `/dev/ttyUSB0` and data port on
`/dev/ttyUSB1`, matching the upstream Haechi launch defaults.

```bash
source /opt/ros/${ROS_DISTRO_NAME:-lyrical}/setup.bash
source install/setup.bash
ros2 launch firesight_ros2 mmwave_sensor_mapping.launch.py
```

This launch opens the FireSight RViz view by default and disables the upstream
TI RViz view. For terminal-only checks, add `rviz:=false`. If your WSL/Linux
ports differ, pass `command_port:=... data_port:=...`.

If the TI driver is already running separately, use the mapper-only launch:

```bash
ros2 launch firesight_ros2 mmwave_mapping.launch.py
```

Without hardware attached, run the demo publisher to verify RViz PointCloud2,
grid, and marker displays:

```bash
ros2 launch firesight_ros2 mmwave_mapping_demo.launch.py rviz:=true
```

Published topics:

- `/ti_mmwave/radar_scan_pcl`: Haechi/TI mmWave `PointCloud2` input.
- `/firesight/mmwave/front_grid`: 2D occupancy grid in front of the glasses.
- `/firesight/mmwave/front_markers`: RViz markers for obstacle points.
- `/firesight/mmwave/front_obstacles`: stable front obstacle markers for HUD fusion.

If the vendor driver publishes a different point cloud topic, pass it through
`input_topic:=...` or update `src/firesight_ros2/config/mmwave_mapping.yaml`.

## Run Thermal + mmWave HUD Fusion Demo

After `colcon build`, the mock demo starts thermal mock frames, mmWave mock
points, the mmWave mapper, the thermal edge node, and the fused HUD node:

```bash
source /opt/ros/${ROS_DISTRO_NAME:-lyrical}/setup.bash
source install/setup.bash
ros2 launch firesight_ros2 thermal_mmwave_fusion_demo.launch.py
```

Fusion topics:

- `/thermal_camera/image_raw`: LWIR or mock thermal image input.
- `/firesight/thermal/edge_mask`: mono thermal edge mask.
- `/firesight/thermal/edge_overlay`: green thermal edge and red hotspot outline.
- `/firesight/hud/fused_overlay`: thermal HUD plus mmWave 1.5 m range panel.

For real hardware, pass the mmWave-only calibration gate first, then launch:

```bash
ros2 launch firesight_ros2 real_sensor_fusion.launch.py \
  thermal_topic:=/thermal_camera/image_raw \
  input_topic:=/ti_mmwave/radar_scan_pcl
```

## Docker Fallback

Docker is useful for dependency checks, but physical sensors usually need extra
USB/video device forwarding. This Dockerfile stays on Jazzy because many vendor
drivers lag behind the newest LTS; prefer native WSL2/Ubuntu for first hardware
bring-up.

```bash
cd ros2_ws
docker build -t firesight-ros2-jazzy .
docker run --rm -it --net=host -v "$PWD":/workspace/ros2_ws firesight-ros2-jazzy
```
