# Multi-Camera and LiDAR BEV Perception System for Autonomous Parking

面向自动泊车的多相机与激光雷达 BEV 感知融合系统。

The project builds a metric Bird's-Eye View from four surround cameras and an
optional LiDAR stream. The first milestone focuses on deterministic geometry,
sensor switching, visualization, and testable interfaces before adding object
and parking-slot recognition.

## Current MVP

- Four inputs: front, rear, left, and right.
- Configurable perspective projection into one BEV canvas.
- Optional LiDAR branch controlled by `lidar.enabled`.
- LiDAR occupancy, height, and density grids in ego coordinates.
- Camera-only, LiDAR debug, and fused visualization modes.
- Built-in synthetic rig, so the pipeline runs before physical sensors arrive.
- Unit tests for BEV projection and metric LiDAR rasterization.

## Architecture

```text
4 cameras -> undistort/project -> RGB BEV -----------+
                                                      +-> fused BEV outputs
LiDAR (optional) -> ego transform -> occupancy BEV ---+
```

Recognition will be added after geometric calibration is validated:

```text
camera detections + parking-slot geometry + LiDAR occupancy
                         -> structured perception output -> parking planner
```

## Quick start

```powershell
python -m pip install -e .
parking-bev --config configs/demo.yaml --max-frames 120
```

The demo writes `output/demo_bev.mp4` and displays a live preview. Use
`--no-display` on a headless system.

```powershell
parking-bev --config configs/demo.yaml --max-frames 120 --no-display
python -m pytest
```

## LiDAR switch

Edit `configs/demo.yaml`:

```yaml
lidar:
  enabled: false  # camera-only mode
```

When disabled, the output schema remains unchanged; LiDAR grids are returned as
zero arrays and the fusion layer falls back to RGB BEV.

## Real sensor integration checklist

1. Capture synchronized frames from all four cameras.
2. Calibrate each camera's intrinsic and distortion parameters.
3. Estimate each camera-to-ego transform and ground-plane projection.
4. Estimate LiDAR-to-ego extrinsics and verify by camera reprojection.
5. Replace the demo destination quadrilaterals with calibrated homographies.
6. Measure BEV scale using known ground control points.

## Coordinate convention

- Ego `x`: forward, in metres.
- Ego `y`: left, in metres.
- Ego `z`: upward, in metres.
- BEV image origin: top-left.
- BEV image up: ego forward.

This repository is an early research prototype and is not a vehicle safety or
control system.

