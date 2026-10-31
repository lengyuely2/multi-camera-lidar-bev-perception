"""Validated tracker snapshot interchange for the independent parking node."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perception.sensor_packet import validate_transform
from .assistance import ParkingObject


@dataclass(frozen=True)
class TrackFrame:
    timestamp_s: float
    frame_id: str
    objects: tuple[ParkingObject, ...]

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported tracking snapshot schema")
        stamp = float(payload["timestamp_s"])
        if not np.isfinite(stamp) or not payload["frame_id"]:
            raise ValueError("Tracking frame/time missing or invalid")
        objects = []
        identities = set()
        for item in payload["objects"]:
            identity = str(item["track_id"])
            if identity in identities:
                raise ValueError("Duplicate track identity")
            identities.add(identity)
            center = np.asarray(item["center"], dtype=float)
            size = np.asarray(item["size_lwh"], dtype=float)
            if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("Invalid 3D track geometry")
            if not isinstance(item["velocity_known"], bool):
                raise ValueError("velocity_known must be boolean")
            objects.append(ParkingObject(identity, str(item["class_name"]), center[:2], size[:2],
                float(item["yaw"]), np.asarray(item["velocity_xy"]), item["velocity_known"],
                float(item["score"]), int(item["hits"]), float(item["missed_s"]), float(center[2])))
        return cls(stamp, str(payload["frame_id"]), tuple(objects))

    def in_current_ego(self, transform, now_s):
        """Transform old frame -> current ego using tf2; propagate world object motion once."""
        transform = validate_transform(transform, "tracking_to_current_ego")
        age = now_s - self.timestamp_s
        if not np.isfinite(age) or age < -0.02:
            raise ValueError("Tracking frame is from the future")
        rotation, translation = transform[:3, :3], transform[:3, 3]
        output = []
        for item in self.objects:
            velocity = np.r_[item.velocity_xy if item.velocity_known else np.zeros(2), 0.]
            center = rotation @ (np.r_[item.center_xy, item.center_z_m] + velocity * max(age, 0.)) + translation
            heading = rotation @ np.array([np.cos(item.yaw_rad), np.sin(item.yaw_rad), 0.])
            output.append(ParkingObject(item.track_id, item.class_name, center[:2], item.size_lw,
                float(np.arctan2(heading[1], heading[0])), (rotation @ velocity)[:2], item.velocity_known,
                item.score, item.hits, item.missed_s, float(center[2])))
        return output
