# Future Sensor Frame Schema

This document defines the future multimodal frame contract for FireSight static-image research handoff. It is a data interchange shape only; it does not implement SLAM, AR UI, PS-LTE, or live mmWave fusion.

## Frame Contract

Each frame is one offline alignment unit with these top-level fields:

- `timestamp`: UTC capture time for the multimodal observation.
- `frame_id`: stable identifier shared across camera, thermal, mmWave, and IMU/AHRS records.
- `camera`: RGB camera payload with an image URI plus intrinsics and extrinsics placeholders.
- `thermal`: thermal frame payload with sensor ID, frame URI, encoding, dimensions, and temperature unit metadata.
- `mmwave`: mmWave radar payload with a point cloud. Each point records x/y/z position, range, velocity, bearing, and intensity.
- `imu_ahrs`: IMU/AHRS payload with orientation, angular velocity, linear acceleration, and pose placeholder.
- `calibration`: calibration metadata linking camera-IMU, mmWave, and thermal references with validity notes.

## Calibration Notes

Camera intrinsics and extrinsics are placeholders for later calibration products such as Kalibr camera-IMU output. mmWave fields follow the future point-cloud shape needed for offline comparison with TI-style radar output: x, y, z, range, velocity, bearing, and intensity. Calibration metadata may record temperature or process-drift compensation references, but no runtime correction is performed in Phase 1.

## Scope Boundary

The schema supports future dataset capture and replay. It intentionally stops at recorded fields and calibration references, so downstream work can align modalities later without treating this phase as a live fusion or navigation system.
