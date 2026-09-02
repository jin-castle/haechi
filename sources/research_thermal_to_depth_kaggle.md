# Thermal to Depth Image Dataset 확인 기록

- Kaggle ref: `hrkanahin/thermal-to-depth-image-dataset`
- Kaggle page: https://www.kaggle.com/datasets/bab5526b08d7023b20b28947c31b9e8f9f71d9ec6fc6aa55debcce221a08c305
- Paper: https://arxiv.org/abs/2603.14998
- Version checked: 2
- Updated: 2026-02-12
- Total size: 48,938,560,801 bytes (48.94 GB)
- File count shown by Kaggle: about 228,000
- License: Database Open Database License / Database Contents License

## Dataset contents

- Spatially aligned thermal and depth image pairs in a VIVID++-like layout.
- Some sequences include RGB images.
- `zoom_*` and `no_zoom_*` sequences include Vicon ground-truth poses.
- Liquid Recurrent Block model checkpoints and result artifacts are included.
- Two real-sensor ROS1 bag recordings are included:
  - `hot_obj_circle.bag`: 7.14 GB
  - `walking_far.bag`: 7.92 GB

## Downloaded recording

- Path: `data/external/thermal_to_depth_kaggle/hot_obj_circle.bag`
- Size: 7,139,417,275 bytes
- SHA-256: `6edbecee799c30eef60cd60c0ff1660ec0d8383e64f5b72c36a8c9bab88393ca`
- Format header: `#ROSBAG V2.0`
- Duration: 59.952933537 seconds
- Total messages: 11,355

Topics:

| Topic | Type | Messages | Approx. rate |
| --- | --- | ---: | ---: |
| `/flir_boson/image_raw` | `sensor_msgs/Image` | 1,799 | 30.0 Hz |
| `/camera/color/image_raw` | `sensor_msgs/Image` | 1,737 | 29.0 Hz |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 627 | 10.5 Hz |
| `/local_pose_vicon/pose` | `geometry_msgs/PoseStamped` | 7,192 | 120.0 Hz |

## Relevance to FireSight

This is useful for validating thermal-to-depth alignment, depth-unit handling,
temporal synchronization, and thermal monocular-depth training. It is not a
dense-smoke firefighter dataset, so it cannot by itself establish accuracy or
safety in fireground smoke. The downloaded bag is a ROS recording, not an MP4;
its image topics must be extracted before the current image replay CLI can use
them.
