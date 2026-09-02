# mmWave ROS 2 Bring-Up and Calibration

Goal: verify that a front mmWave sensor can produce stable 1-1.5 m obstacle
mapping for AR glasses before thermal fusion is added.

## Baseline Environment

- Host: Ubuntu 26.04 Resolute with ROS 2 Lyrical, or Ubuntu 24.04 Noble with
  ROS 2 Jazzy if a vendor driver has not caught up to Lyrical.
- Workspace: `ros2_ws`.
- Required ROS topics:
  - input: `/ti_mmwave/radar_scan_pcl` as `sensor_msgs/msg/PointCloud2`
  - output: `/firesight/mmwave/front_grid` as `nav_msgs/msg/OccupancyGrid`
  - output: `/firesight/mmwave/front_markers` as `visualization_msgs/msg/MarkerArray`
  - output: `/firesight/mmwave/front_obstacles` as the HUD fusion
    `visualization_msgs/msg/MarkerArray`
- Frame convention:
  - `glasses_frame`: AR glasses bridge, X forward, Y left, Z up.
  - `mmwave_front_link`: mmWave sensor origin, X forward, Y left, Z up.

## Hardware Bring-Up Order

1. Connect the mmWave sensor and confirm the OS sees it:

   ```bash
   bash scripts/ros2_sensor_doctor.sh
   ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
   ```

2. Start the Haechi/TI mmWave driver so it publishes a point cloud on
   `/ti_mmwave/radar_scan_pcl`.

3. Build and run the FireSight mapper:

   ```bash
   source /opt/ros/${ROS_DISTRO:-lyrical}/setup.bash
   cd ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   colcon build --symlink-install
   source install/setup.bash
   ros2 launch firesight_ros2 mmwave_sensor_mapping.launch.py
   ```

   RViz opens by default. Use `rviz:=false` only when you want a
   terminal-only launch for scripted checks. If the sensor appears under
   different serial ports, pass `command_port:=... data_port:=...`.

4. Inspect the mapping:

   ```bash
   ros2 topic hz /ti_mmwave/radar_scan_pcl
   ros2 topic hz /firesight/mmwave/front_grid
   ros2 topic hz /firesight/mmwave/front_obstacles
   ros2 topic echo /firesight/mmwave/front_grid --once
   ```

If hardware is not publishing yet, verify the display path with:

```bash
ros2 launch firesight_ros2 mmwave_mapping_demo.launch.py rviz:=true
```

In RViz, set `Fixed Frame` to `mmwave_front_link`, then add:

- `PointCloud2` on `/ti_mmwave/radar_scan_pcl`
- `MarkerArray` on `/firesight/mmwave/front_markers`
- `MarkerArray` on `/firesight/mmwave/front_obstacles` for the HUD consumer

`/firesight/mmwave/front_grid` is still published, but the default WSLg RViz
config leaves the OccupancyGrid display out because the Map shader can be noisy
under software rendering.

On WSLg, the demo launch starts RViz with software OpenGL fallback because
hardware GLX can fail to create an RViz render window.

## First Calibration Pass

Use a high-reflectivity target such as a corner reflector, metal plate, or flat
box with foil. Mark floor positions at 0.5 m, 1.0 m, and 1.5 m from the glasses
bridge frame, then repeat at -30, 0, and +30 degrees.

Record for each target pose:

- physical range from `glasses_frame`
- physical angle from the forward axis
- peak range reported by mmWave
- peak angle reported by mmWave
- whether the marker lands in the expected near, mid, or far band
- whether the grid cell is stable for at least 5 seconds

Acceptance criteria for this gate:

- 0.5 m, 1.0 m, and 1.5 m targets appear in the correct distance band.
- Center target angle error is within 5 degrees after yaw correction.
- Range bias is stable enough to correct with a single offset or scale.
- Empty scene produces no persistent front obstacle markers after static clutter
  is removed by the vendor driver or by threshold tuning.

## Calibration Adjustments

Update `ros2_ws/src/firesight_ros2/config/extrinsics.yaml` after measurement:

- `translation_m.x`: physical forward offset from glasses bridge to radar phase
  center.
- `translation_m.y`: lateral offset. Positive is left.
- `translation_m.z`: vertical offset. Positive is up.
- `rotation_rpy_deg.yaw`: corrects left/right angular bias.
- `rotation_rpy_deg.pitch`: corrects systematic floor/ceiling tilt.

Launch-time overrides for the first pass:

```bash
ros2 launch firesight_ros2 mmwave_mapping.launch.py \
  input_topic:=/ti_mmwave/radar_scan_pcl \
  extrinsic_x:=0.035 \
  extrinsic_y:=0.0 \
  extrinsic_z:=-0.015 \
  extrinsic_yaw:=0.0
```

If the sensor reports too many floor/body reflections, tune these first:

- `z_min_m` and `z_max_m`: reject points outside the useful AR-glasses band.
- `range_min_m`: reject self-reflections from the glasses frame or mount.
- `grid_resolution_m`: start at 0.05 m; lower only after the signal is stable.
- vendor-side CFAR/static-clutter settings: prefer fixing this before app-side
  filtering.

## Thermal Camera Fusion Readiness

Do not calibrate thermal fusion until mmWave alone passes the gate above. Once
it passes, capture synchronized samples with:

- a warm target at known image pixel coordinates
- a radar reflector at the same physical pose
- ROS timestamps from both streams
- static TF from `glasses_frame` to `thermal_camera_link`

Thermal fusion should then solve an extrinsic projection problem, not compensate
for unstable radar mapping.
