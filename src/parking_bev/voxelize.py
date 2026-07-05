from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelizedPointCloud:
    voxels: np.ndarray
    coordinates_zyx: np.ndarray
    points_per_voxel: np.ndarray
    input_points: int
    retained_points: int


class HardVoxelizer:
    """Deterministic NumPy reference for BEVFusion hard voxelization."""

    def __init__(
        self,
        voxel_size: tuple[float, float, float] = (0.075, 0.075, 0.2),
        point_cloud_range: tuple[float, float, float, float, float, float] =
        (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        max_points_per_voxel: int = 10,
        max_voxels: int = 160_000,
    ) -> None:
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.minimum = np.asarray(point_cloud_range[:3], dtype=np.float32)
        self.maximum = np.asarray(point_cloud_range[3:], dtype=np.float32)
        self.grid_size = np.rint((self.maximum - self.minimum) / self.voxel_size).astype(np.int64)
        self.max_points = max_points_per_voxel
        self.max_voxels = max_voxels

    def __call__(self, points: np.ndarray) -> VoxelizedPointCloud:
        points = np.asarray(points, dtype=np.float32)
        valid = np.all((points[:, :3] >= self.minimum) & (points[:, :3] < self.maximum), axis=1)
        filtered = points[valid]
        if not len(filtered):
            return self._empty(points.shape[1], len(points))

        coordinates_xyz = np.floor((filtered[:, :3] - self.minimum) / self.voxel_size).astype(np.int64)
        keys = (coordinates_xyz[:, 0]
                + coordinates_xyz[:, 1] * self.grid_size[0]
                + coordinates_xyz[:, 2] * self.grid_size[0] * self.grid_size[1])
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        all_starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
        all_ends = np.r_[all_starts[1:], len(order)]
        starts = all_starts[:self.max_voxels]
        ends = all_ends[:self.max_voxels]

        voxels = np.zeros((len(starts), self.max_points, points.shape[1]), dtype=np.float32)
        counts = np.zeros(len(starts), dtype=np.int32)
        coordinates = np.zeros((len(starts), 3), dtype=np.int32)
        retained = 0
        for index, (start, end) in enumerate(zip(starts, ends)):
            selected = order[start:min(end, start + self.max_points)]
            count = len(selected)
            voxels[index, :count] = filtered[selected]
            counts[index] = count
            coordinates[index] = coordinates_xyz[selected[0], ::-1]
            retained += count

        return VoxelizedPointCloud(voxels, coordinates, counts, len(points), retained)

    def _empty(self, features: int, input_points: int) -> VoxelizedPointCloud:
        return VoxelizedPointCloud(
            np.empty((0, self.max_points, features), np.float32),
            np.empty((0, 3), np.int32),
            np.empty(0, np.int32),
            input_points,
            0,
        )
