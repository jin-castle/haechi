# HaechiAR depth repository review

Reviewed on 2026-09-01 against private repository commit
`175ed092e46fc1b9e2e38d39e97b21a5a4bd23ee`.

## Repository

- Repository: `HaechiAR/firesight_edge_depth-jetson-` (private)
- Purpose: FLIR Boson Y16 thermal input to Thermal2Depth dense depth, followed by
  TEED edges rendered over the predicted depth view.
- Runtime target: Linux / Jetson Orin Nano 8 GB.
- Pinned README: https://github.com/HaechiAR/firesight_edge_depth-jetson-/blob/175ed092e46fc1b9e2e38d39e97b21a5a4bd23ee/README.md
- Pinned model card: https://github.com/HaechiAR/firesight_edge_depth-jetson-/blob/175ed092e46fc1b9e2e38d39e97b21a5a4bd23ee/MODEL_CARD.md
- Pinned runtime: https://github.com/HaechiAR/firesight_edge_depth-jetson-/blob/175ed092e46fc1b9e2e38d39e97b21a5a4bd23ee/depth_edge/thermal2depth_runtime.py
- Pinned config: https://github.com/HaechiAR/firesight_edge_depth-jetson-/blob/175ed092e46fc1b9e2e38d39e97b21a5a4bd23ee/config/orin_nano.yaml

## Confirmed model contract

- Model: `DispNetSeq` with a Liquid recurrent temporal aggregator and
  EfficientNet-B0 encoder.
- Input: five consecutive raw `uint16` thermal frames, resized to
  `B x T x 1 x 256 x 320` and normalized from verified 14-bit Y16 data.
- Output: one dense depth array for the last frame at `256 x 320`.
- Checkpoint: local Epoch 9 EMA, 4,564,393 parameters, 18,570,295 bytes.
- Reported RAT validation: AbsRel 0.1117, RMSE 0.5194, MAE 0.3135,
  delta-1 0.8971.
- TEED consumes either predicted depth (default) or normalized thermal data;
  its default input is 320 x 240 with threshold 0.75.

## Important limitation

The repository model card explicitly describes the output as a learned monocular
estimate, not a calibrated safety sensor. The code does not establish that its
numeric output is in metres for this Kaggle recording. Therefore the model's
depth values must not yet be labelled as physical distance bands.

The downloaded ROS bag has RealSense depth aligned to the RGB camera, not to the
thermal camera. RGB-to-depth pixel metrics are valid directly. Thermal-to-depth
pixel metrics require the dataset's thermal/RGB calibration or already aligned
processed pairs. Time synchronization alone is insufficient for pixelwise error.

The extracted `hot_obj_circle.bag` thermal samples also fail the current runtime's
Y16 input gate: p99.5 is about 24,535 to 24,708 (above the 14-bit maximum 16,383)
while only about 25% of values are divisible by four (far below the 99.5% MSB
alignment rule). The existing checkpoint therefore must not be run directly on
these raw frames. The Boson output mode/conversion needs verification, or the
model must be retrained for this dataset representation.

## Integration decision

1. Extract synchronized thermal, RGB, and RealSense depth frames from the bag.
2. Save lossless 16-bit thermal/depth PNGs plus human-readable preview PNGs and
   timestamp deltas.
3. Use RGB plus `aligned_depth_to_color` for the first valid metric-depth
   baseline.
4. Reuse the private Thermal2Depth runtime for thermal inference only after a
   thermal-to-RGB/depth calibration mapping or processed aligned pair is present.
5. Calibrate or fit the Thermal2Depth output to metres before applying near/mid/
   far line colours in a field-facing display.

## Related repositories

- `HaechiAR/firesight_edge`: TEED-only implementation.
- `HaechiAR/firesight_sw`: calibration design material, not a completed metric
  thermal-depth registration runtime at the reviewed revision.
- `HaechiAR/firesight_hw`: no implementation content at the reviewed time.

The upstream `RBs-thermal2depth` source included by the private repository has no
license file. The HaechiAR README therefore requires the derived repository to
remain private until rights are clarified.
