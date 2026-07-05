from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarBEV:
    occupancy: np.ndarray
    height: np.ndarray
    density: np.ndarray


class SyntheticLidarSource:
    def __init__(self) -> None:
        self.index = 0

    def read(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        radius = 7.5 + 0.35 * np.sin(angles * 5 + self.index * 0.04)
        ring = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)))
        obstacle_x = 2.5 + 0.4 * np.sin(self.index * 0.05)
        rng = np.random.default_rng(self.index)
        obstacle = np.column_stack((
            rng.normal(obstacle_x, 0.35, 180),
            rng.normal(-2.0, 0.45, 180),
            rng.uniform(0.0, 1.5, 180),
        ))
        self.index += 1
        return np.vstack((ring, obstacle)).astype(np.float32)


class LidarBEVProjector:
    def __init__(self, bev_config: dict, lidar_config: dict) -> None:
        self.width = int(bev_config["width_px"])
        self.height_px = int(bev_config["height_px"])
        self.x_min = float(bev_config["x_min_m"])
        self.x_max = float(bev_config["x_max_m"])
        self.y_min = float(bev_config["y_min_m"])
        self.y_max = float(bev_config["y_max_m"])
        self.min_z = float(lidar_config["min_z_m"])
        self.max_z = float(lidar_config["max_z_m"])
        self.extrinsic = np.asarray(lidar_config["extrinsic_lidar_to_ego"], np.float32)

    def empty(self) -> LidarBEV:
        shape = (self.height_px, self.width)
        return LidarBEV(np.zeros(shape, np.uint8), np.zeros(shape, np.float32), np.zeros(shape, np.uint16))

    def project(self, points_lidar: np.ndarray) -> LidarBEV:
        if points_lidar.size == 0:
            return self.empty()
        points_h = np.column_stack((points_lidar[:, :3], np.ones(len(points_lidar), np.float32)))
        points = (self.extrinsic @ points_h.T).T[:, :3]
        valid = (
            (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.min_z) & (points[:, 2] <= self.max_z)
        )
        points = points[valid]
        if not len(points):
            return self.empty()

        u = ((self.y_max - points[:, 1]) / (self.y_max - self.y_min) * self.width).astype(np.int32)
        v = ((self.x_max - points[:, 0]) / (self.x_max - self.x_min) * self.height_px).astype(np.int32)
        u = np.clip(u, 0, self.width - 1)
        v = np.clip(v, 0, self.height_px - 1)

        occupancy = np.zeros((self.height_px, self.width), np.uint8)
        height = np.full((self.height_px, self.width), -np.inf, np.float32)
        density = np.zeros((self.height_px, self.width), np.uint16)
        occupancy[v, u] = 255
        np.maximum.at(height, (v, u), points[:, 2])
        np.add.at(density, (v, u), 1)
        height[~np.isfinite(height)] = 0.0
        return LidarBEV(occupancy, height, density)

