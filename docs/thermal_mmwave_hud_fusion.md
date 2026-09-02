# Thermal + mmWave HUD Fusion MVP

This v1 path is a sensor-fusion HUD pipeline check, not fine-tuning.

## Runtime Contract

Inputs:

- `/thermal_camera/image_raw`: LWIR or mock thermal frame, `sensor_msgs/msg/Image`.
- `/ti_mmwave/radar_scan_pcl`: TI mmWave point cloud, `sensor_msgs/msg/PointCloud2`.
- `/imu`: optional IMU input for the external IMU/radar fusion package.

Intermediate outputs:

- `/firesight/thermal/edge_mask`: mono thermal structure edge mask.
- `/firesight/thermal/edge_overlay`: RGB thermal HUD overlay with green edges and red hotspot outlines.
- `/firesight/mmwave/front_grid`: front occupancy grid from the existing mmWave mapper.
- `/firesight/mmwave/front_markers`: RViz marker view of front obstacles.
- `/firesight/mmwave/front_obstacles`: marker payload consumed by the HUD fusion node.

Final output:

- `/firesight/hud/fused_overlay`: RGB HUD image that combines the thermal overlay with the bottom 1.5 m mmWave range panel.

## Offline Mock

The local mock path creates a PNG contact sheet and ROS2-style JSONL stream:

```bash
uv run --frozen python scripts/build_thermal_mmwave_hud_fusion_mock.py
```

The easy-view copy is written to `thermal_mmwave_hud_fusion_contact_sheet.png`.

## Fire360 Thermal-Display Replay

The Fire360 `IFSI Video 4` frames are real indoor firefighter footage with a
thermal-camera display, but the extracted files are RGB-encoded display video,
not radiometric LWIR frames. Run the replay as a thermal-display proxy only:

```bash
uv run --frozen python scripts/build_fire360_thermal_display_replay.py
```

It writes `fire360_thermal_display_replay_contact_sheet.png` and a JSON summary
with edge saturation, hotspot activity, and consecutive-frame edge stability.
The mmWave panel remains an explicitly clear-scene mock until a recorded radar
stream is available.

For a 20-frame two-sequence stability check, run:

```bash
uv run --frozen python scripts/build_fire360_thermal_display_multisequence_eval.py
```

The per-sequence JSON metrics do not compare masks across sequence boundaries.

## ROS2 Mock Demo

Use this before real hardware. It starts a mock LWIR publisher, mock mmWave
point cloud publisher, mmWave mapper, thermal edge node, and fused HUD node:

```bash
source /opt/ros/${ROS_DISTRO_NAME:-lyrical}/setup.bash
cd ros2_ws
source install/setup.bash
ros2 launch firesight_ros2 thermal_mmwave_fusion_demo.launch.py
```

Expected topics:

```bash
ros2 topic hz /firesight/thermal/edge_overlay
ros2 topic hz /firesight/mmwave/front_obstacles
ros2 topic hz /firesight/hud/fused_overlay
```

## Real Sensor Bring-Up

First pass the mmWave-only gate from `docs/mmwave_ros2_calibration.md`. Then run:

```bash
ros2 launch firesight_ros2 real_sensor_fusion.launch.py \
  thermal_topic:=/thermal_camera/image_raw \
  input_topic:=/ti_mmwave/radar_scan_pcl
```

The first fused HUD version draws mmWave cues in a bottom range panel. Do not use
image-space projection markers until `glasses_frame`, `mmwave_front_link`, and
`thermal_camera_link` are calibrated in `ros2_ws/src/firesight_ros2/config/extrinsics.yaml`.

## Acceptance Gates

- Thermal edge node publishes nonzero `/firesight/thermal/edge_mask` and `/firesight/thermal/edge_overlay`.
- Hot thermal regions are red outlines, independent of RGB fire color thresholding.
- A 0.5 m, 1.0 m, and 1.5 m mmWave target lands in the expected near/mid/far panel band.
- Empty scene does not create persistent `/firesight/mmwave/front_obstacles` markers.
- `/firesight/hud/fused_overlay` publishes at nonzero rate with p95 sync delay under 100 ms for replay/mock bags.
