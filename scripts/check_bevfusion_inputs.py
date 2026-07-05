from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from parking_bev.nuscenes_source import CAMERA_CHANNELS, NuScenesSource
from parking_bev.voxelize import HardVoxelizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one sample against BEVFusion input requirements")
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    args = parser.parse_args()

    source = NuScenesSource(args.dataroot, radar_enabled=False)
    ok, frame = source.read()
    if not ok or frame is None:
        raise RuntimeError("No nuScenes sample found")
    voxelized = HardVoxelizer()(frame.lidar_ego)

    report = {
        "sample_token": frame.token,
        "camera_count": len(frame.cameras),
        "camera_shapes": {name: list(frame.cameras[name].shape) for name in CAMERA_CHANNELS},
        "camera_intrinsics_valid": all(
            frame.calibrations[name].intrinsic is not None
            and frame.calibrations[name].intrinsic.shape == (3, 3)
            and np.isfinite(frame.calibrations[name].intrinsic).all()
            for name in CAMERA_CHANNELS
        ),
        "lidar_features": int(frame.lidar_ego.shape[1]),
        "lidar_input_points": voxelized.input_points,
        "lidar_retained_points": voxelized.retained_points,
        "voxel_count": int(len(voxelized.voxels)),
        "max_points_in_voxel": int(voxelized.points_per_voxel.max(initial=0)),
        "grid_size_xyz": HardVoxelizer().grid_size.tolist(),
        "ready": (
            len(frame.cameras) == 6
            and frame.lidar_ego.shape[1] == 5
            and len(voxelized.voxels) > 0
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
