from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nuscenes_source import Object3D


@dataclass(frozen=True)
class RadarVelocityEstimate:
    velocity_ego: np.ndarray
    point_count: int
    confidence: float
    dispersion_mps: float


def estimate_radar_velocity(
    obj: Object3D,
    radar_ego: dict[str, np.ndarray],
    margin_m: float = 0.8,
    min_points: int = 2,
) -> RadarVelocityEstimate | None:
    clouds = [points for points in radar_ego.values() if len(points)]
    if not clouds:
        return None
    points = np.vstack(clouds)
    delta = points[:, :2] - obj.center_ego[:2]
    cosine, sine = np.cos(obj.yaw_ego), np.sin(obj.yaw_ego)
    local = delta @ np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    width, length = float(obj.size_wlh[0]), float(obj.size_wlh[1])
    associated = (
        (np.abs(local[:, 0]) <= length / 2 + margin_m)
        & (np.abs(local[:, 1]) <= width / 2 + margin_m)
        & np.isfinite(points[:, 3:5]).all(axis=1)
        & (np.linalg.norm(points[:, 3:5], axis=1) < 80.0)
    )
    velocities = points[associated, 3:5]
    if len(velocities) < min_points:
        return None

    median = np.median(velocities, axis=0)
    residuals = np.linalg.norm(velocities - median, axis=1)
    if len(velocities) >= 3:
        median_residual = float(np.median(residuals))
        keep = residuals <= max(2.0, 3.0 * median_residual)
        velocities = velocities[keep]
        if len(velocities) < min_points:
            return None
        median = np.median(velocities, axis=0)
        residuals = np.linalg.norm(velocities - median, axis=1)
    count = len(velocities)
    confidence = float(min(0.9, 1.0 - np.exp(-count / 3.0)))
    dispersion = float(np.median(residuals)) if len(residuals) else 0.0
    return RadarVelocityEstimate(median.astype(np.float32), count, confidence, dispersion)


def blend_object_and_radar_velocity(
    object_velocity: np.ndarray,
    estimate: RadarVelocityEstimate,
    weight_scale: float = 1.0,
) -> np.ndarray:
    weight = float(np.clip(estimate.confidence * weight_scale, 0.0, 1.0))
    return ((1.0 - weight) * np.asarray(object_velocity) + weight * estimate.velocity_ego).astype(np.float32)
