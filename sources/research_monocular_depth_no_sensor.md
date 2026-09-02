# Monocular depth without a depth sensor at inference

Reviewed 2026-09-01. `parallel-cli` was unavailable in this environment, so the
lookup used primary project repositories and CVF/arXiv papers as the fallback.

## Main finding

A trained monocular depth model can infer a dense depth map from one camera image
without a depth sensor at inference time. This does not create a physical
measurement: the model learned statistical visual cues during training. Relative
depth is usually more robust than absolute metric scale, and metric outputs can
degrade under camera, scene, or modality domain shift.

## Current RGB candidates

- Depth Anything V2: single-image relative models plus indoor/outdoor metric
  variants. The official metric example returns depth in metres.
  - https://github.com/DepthAnything/Depth-Anything-V2
  - https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md
- UniDepthV2: camera-agnostic monocular metric depth with confidence output.
  - https://github.com/lpiccinelli-eth/UniDepth
  - https://arxiv.org/abs/2502.20110
- Metric3Dv2: zero-shot metric depth and surface-normal estimation from one image.
  - https://github.com/YvanYin/Metric3D
  - https://arxiv.org/abs/2404.15506

## Thermal candidates and constraints

- Self-Supervised Monocular Depth Estimation From Thermal Images via Adversarial
  Multi-Spectral Adaptation (WACV 2023) demonstrates thermal monocular depth using
  unpaired RGB/thermal video during training.
  - https://openaccess.thecvf.com/content/WACV2023/html/Shin_Self-Supervised_Monocular_Depth_Estimation_From_Thermal_Images_via_Adversarial_Multi-Spectral_WACV_2023_paper.html
- Deep Depth Estimation From Thermal Image (CVPR 2023) supports monocular or
  stereo thermal input and explicitly notes scale ambiguity and generalization
  issues for monocular depth.
  - https://openaccess.thecvf.com/content/CVPR2023/html/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.html
  - https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset
- ThermalMonoDepth provides an official self-supervised thermal depth/ego-motion
  implementation.
  - https://github.com/UkcheolShin/ThermalMonoDepth

## Fire360 implication

The existing Depth Anything V2 indoor metric PoC can run without a depth sensor,
but Fire360 thermal-display imagery is outside its normal RGB training domain.
Its output should be treated as estimated ordering/visualization until evaluated
against synchronized metric reference data. A thermal-specific model is a better
long-term path, but training still needs video geometry, paired depth, stereo, or
another supervision source even though deployment itself uses only one camera.
