# Code Map

This document groups the repository code by feature area and explains the main
runtime flows for the three demo modules.

## Repository Root

```text
configs/              Runtime configuration files
docs/images/          README and documentation preview images
docs/videos/          Small checked-in demo videos
scripts/              Command-line scripts for data, rendering, inference, and evaluation
src/parking_bev/      Core Python package
tests/                Unit tests
output/               Local generated artifacts, ignored by Git
data/                 External datasets and checkpoints, ignored by Git
```

## Configuration

| Path | Purpose |
|---|---|
| `configs/camera_only.yaml` | Synthetic four-camera BEV demo with LiDAR disabled. |
| `configs/demo.yaml` | Original synthetic camera+optional-LiDAR BEV demo. |
| `configs/fb_ssem_sample.yaml` | Static four-camera sample configuration for FB-SSEM-style inspection. |

## Core BEV Perception Code

| Path | Purpose |
|---|---|
| `src/parking_bev/app.py` | Main `parking-bev` CLI entry point for camera/LiDAR BEV rendering. |
| `src/parking_bev/config.py` | YAML configuration loading. |
| `src/parking_bev/camera.py` | Synthetic, static-image, and video camera rigs. |
| `src/parking_bev/bev.py` | Camera image warping into a BEV canvas. |
| `src/parking_bev/lidar.py` | LiDAR projection and BEV occupancy/height/density grids. |
| `src/parking_bev/voxelize.py` | Hard voxelization helper for LiDAR points. |
| `src/parking_bev/fusion.py` | Camera-only, LiDAR-only, and fused BEV visual rendering. |
| `src/parking_bev/metric_bev.py` | Metric BEV renderer for nuScenes objects, LiDAR, radar, and predictions. |
| `src/parking_bev/nuscenes_source.py` | nuScenes mini reader and sensor-to-ego coordinate transforms. |
| `src/parking_bev/predictions.py` | BEVFusion prediction loading and class/category conversion. |
| `src/parking_bev/evaluation.py` | Detection diagnostic metrics. |

## Tracking And Fusion Code

| Path | Purpose |
|---|---|
| `src/parking_bev/tracking.py` | Timestamp-aware Kalman tracker and prediction-to-track measurement conversion. |
| `src/parking_bev/tracking_evaluation.py` | Identity tracking diagnostics such as IDF1 and ID switches. |
| `src/parking_bev/radar_fusion.py` | Radar velocity association and velocity blending. |
| `src/parking_bev/appearance.py` | Camera crop histogram appearance features. |
| `src/parking_bev/learned_appearance.py` | Optional ResNet-18 learned appearance embeddings. |

## World Prediction Code

| Path | Purpose |
|---|---|
| `src/parking_bev/world_model.py` | `WorldModelLite`: 3-second constant-relative-velocity rollout and risk scoring. |

## Semantic Visualization Code

| Path | Purpose |
|---|---|
| `src/parking_bev/semantic_3d.py` | Stylized semantic driving-scene renderer, HUD, track stabilization, and future-path overlay. |
| `scripts/render_semantic_drive.py` | Main renderer for semantic surround, camera-BEV-style, and world-model videos. |
| `scripts/render_bevfusion_predictions.py` | One-frame BEVFusion prediction renderer. |
| `scripts/render_nuscenes_video.py` | nuScenes input preview video renderer. |
| `scripts/preview_nuscenes.py` | Single-frame nuScenes preview renderer. |

## BEVFusion Inference And Batch Evaluation Scripts

| Path | Purpose |
|---|---|
| `scripts/check_bevfusion_inputs.py` | Validates BEVFusion input preparation. |
| `scripts/run_bevfusion_inference.py` | Single-sample BEVFusion inference wrapper. |
| `scripts/run_bevfusion_scene.py` | Scene-level BEVFusion inference wrapper. |
| `scripts/run_bevfusion_batch.py` | Batch inference over nuScenes mini scenes. |
| `scripts/evaluate_bevfusion.py` | Single-frame detection diagnostics. |
| `scripts/evaluate_bevfusion_scene.py` | Scene-level detection diagnostics and comparison video. |
| `scripts/evaluate_bevfusion_batch.py` | Batch detection and tracking diagnostic aggregation. |
| `scripts/prepare_nuscenes_mmdet_infos.py` | nuScenes info-file preparation helper. |
| `scripts/wsl_bevfusion_smoke.py` | WSL BEVFusion smoke test helper. |
| `scripts/run_full_mini_after_wsl.ps1` | Windows helper to run full mini inference/evaluation after WSL setup. |
| `scripts/repair_wsl2_admin.ps1` | Windows admin helper for WSL2 repair. |

## Tracking Scripts

| Path | Purpose |
|---|---|
| `scripts/track_bevfusion_scene.py` | Renders tracked BEVFusion objects with stable IDs and trajectories. |
| `scripts/evaluate_tracking_ids.py` | Evaluates tracking identity consistency against nuScenes instance IDs. |
| `scripts/tune_motion_tracker.py` | Grid search helper for tracker parameters. |

## Tests

| Path | Purpose |
|---|---|
| `tests/test_bev.py` | Camera BEV projection and camera rig tests. |
| `tests/test_lidar.py` | LiDAR BEV projection tests. |
| `tests/test_nuscenes_source.py` | nuScenes, detection, tracking, radar, semantic rendering, and world-model tests. |

## Flow 1: Camera BEV / BEV Perception

Command:

```powershell
.\.venv\Scripts\python.exe scripts\render_semantic_drive.py `
  --predictions output\bevfusion_mini\scenes\07_scene-1077.json `
  --no-radar `
  --no-world-model `
  --no-smooth `
  --width 960 `
  --height 540 `
  --title "CAMERA BEV" `
  --sensor-label "CAMERA BEV ONLY" `
  --video output\camera_bev_semantic_style.mp4
```

Main logic:

```text
BEVFusion prediction JSON
  -> NuScenesSource reads frame pose and calibration
  -> predictions.py converts boxes from LiDAR frame to ego frame
  -> tracking.py keeps stable object IDs in global coordinates
  -> semantic_3d.py converts tracks back to current ego frame
  -> Semantic3DRenderer draws the camera-BEV-style semantic video
```

Notes:

- Radar velocity fusion is disabled by `--no-radar`.
- World prediction is disabled by `--no-world-model`.
- Current object locations still come from generated BEVFusion predictions. A
  strictly camera-only neural detector would require a camera-only checkpoint or
  model swap.

## Flow 2: World Prediction

Command:

```powershell
.\.venv\Scripts\python.exe scripts\render_semantic_drive.py `
  --predictions output\bevfusion_mini\scenes\07_scene-1077.json `
  --no-smooth `
  --min-visible-hits 2 `
  --fade-in-hits 4 `
  --fade-out-misses 4 `
  --video output\world_model_lite_stabilized.mp4
```

Main logic:

```text
BEVFusion predictions
  -> TimestampAwareTracker produces stable object tracks
  -> stabilized_snapshot_to_ego adds display alpha for fade-in/fade-out
  -> WorldModelLite estimates future object positions over 3 seconds
  -> risk scoring checks ego safety envelope and driving corridor
  -> Semantic3DRenderer overlays future paths and WORLD risk HUD
```

Risk outputs:

```text
clear -> no conflict over the prediction horizon
watch -> possible future corridor interaction
caution -> predicted conflict farther out in the horizon
critical -> predicted conflict soon
```

## Flow 3: Semantic Simulation / Visualization

Command:

```powershell
.\.venv\Scripts\python.exe scripts\render_semantic_drive.py `
  --predictions output\bevfusion_mini\scenes\02_scene-0553.json `
  --video output\semantic_surround_scene-0553.mp4
```

Main logic:

```text
BEVFusion predictions + nuScenes ego poses
  -> tracking in global coordinates
  -> interpolation between perception keyframes for smooth display
  -> ego-frame semantic tracks
  -> stylized road, ego vehicle, surrounding vehicles, pedestrians, HUD
  -> MP4 video and JSON report
```

Optional modes:

```text
--engineering-mode    Shows LiDAR points, radar points, IDs, distances, velocities, and histories.
--camera-comparison   Places raw nuScenes camera views next to the semantic rendering.
--no-smooth           Writes only raw perception keyframes without interpolation.
```
