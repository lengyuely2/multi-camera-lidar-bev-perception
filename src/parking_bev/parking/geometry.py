"""Planar bicycle motion and exact distances between oriented rectangles."""
from __future__ import annotations

import numpy as np


def rectangle(center, length, width, yaw, margin=0.0):
    center, yaw = np.asarray(center, dtype=float), np.asarray(yaw, dtype=float)
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float)
    local *= [length / 2 + margin, width / 2 + margin]
    rotation = np.stack((np.stack((np.cos(yaw), -np.sin(yaw)), axis=-1),
                         np.stack((np.sin(yaw), np.cos(yaw)), axis=-1)), axis=-2)
    return np.einsum("...ij,kj->...ki", rotation, local) + center[..., None, :]


def bicycle_path(times, speed, steering_angle, wheelbase, rear_axle_x):
    """Base origin is body center. Signed speed is at rear axle; steering is road-wheel radians."""
    distance = np.asarray(times) * speed
    curvature = np.tan(steering_angle) / wheelbase
    yaw = distance * curvature
    if abs(curvature) < 1e-9:
        rear = np.column_stack((rear_axle_x + distance, np.zeros_like(distance)))
    else:
        rear = np.column_stack((rear_axle_x + np.sin(yaw) / curvature, (1 - np.cos(yaw)) / curvature))
    centers = rear - np.column_stack((rear_axle_x * np.cos(yaw), rear_axle_x * np.sin(yaw)))
    return centers, yaw


def polygon_clearance(first, second):
    """Batched convex rectangles (...,4,2); return exact nonnegative boundary gap and overlap."""
    first, second = np.broadcast_arrays(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    edges_first = np.roll(first, -1, axis=-2) - first
    edges_second = np.roll(second, -1, axis=-2) - second
    axes = np.concatenate((edges_first[..., :2, :], edges_second[..., :2, :]), axis=-2)
    axes = np.stack((-axes[..., 1], axes[..., 0]), axis=-1)
    projection_first = np.einsum("...pd,...ad->...pa", first, axes)
    projection_second = np.einsum("...pd,...ad->...pa", second, axes)
    separated = ((projection_first.max(axis=-2) < projection_second.min(axis=-2) - 1e-9) |
                 (projection_second.max(axis=-2) < projection_first.min(axis=-2) - 1e-9))
    overlap = ~separated.any(axis=-1)

    def vertex_edge_gap(vertices, polygon):
        edges = np.roll(polygon, -1, axis=-2) - polygon
        relative = vertices[..., :, None, :] - polygon[..., None, :, :]
        numerator = (relative * edges[..., None, :, :]).sum(axis=-1)
        denominator = (edges ** 2).sum(axis=-1)[..., None, :]
        fractions = np.clip(numerator / denominator, 0, 1)
        residual = relative - fractions[..., None] * edges[..., None, :, :]
        return np.sqrt((residual ** 2).sum(axis=-1).min(axis=(-1, -2)))
    gap = np.minimum(vertex_edge_gap(first, second), vertex_edge_gap(second, first))
    return np.where(overlap, 0.0, gap), overlap
