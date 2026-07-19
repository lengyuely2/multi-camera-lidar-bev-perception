from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .tracking import TrackSnapshot

if TYPE_CHECKING:
    from .world_model import AgentFuture, WorldModelOutput


CLASS_COLORS = {
    "car": (205, 205, 205),
    "truck": (235, 175, 70),
    "construction_vehicle": (70, 155, 245),
    "bus": (225, 165, 65),
    "trailer": (190, 145, 65),
    "barrier": (60, 155, 245),
    "motorcycle": (115, 220, 145),
    "bicycle": (100, 225, 135),
    "pedestrian": (80, 185, 250),
    "traffic_cone": (55, 125, 245),
}

DRIVING_OBJECT_COLOR = (142, 145, 150)
DRIVING_PEDESTRIAN_COLOR = (105, 108, 113)
DRIVING_PRIORITY_COLOR = (225, 133, 38)
RISK_COLORS = {
    "clear": (118, 170, 105),
    "watch": (86, 174, 235),
    "caution": (54, 152, 238),
    "critical": (45, 68, 236),
}


@dataclass(frozen=True)
class SemanticTrack:
    track_id: int
    class_name: str
    score: float
    center_ego: np.ndarray
    size_wlh: np.ndarray
    yaw_ego: float
    velocity_ego: np.ndarray
    history_ego: np.ndarray
    missed: int
    display_alpha: float = 1.0

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.center_ego[:2]))

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_ego))


def interpolate_semantic_tracks(
    first: list[SemanticTrack],
    second: list[SemanticTrack],
    alpha: float,
) -> list[SemanticTrack]:
    """Interpolate matched tracker IDs for smooth display between sensor keyframes."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    first_by_id = {item.track_id: item for item in first}
    second_by_id = {item.track_id: item for item in second}
    output = []
    for track_id in sorted(first_by_id.keys() | second_by_id.keys()):
        before = first_by_id.get(track_id)
        after = second_by_id.get(track_id)
        if before is None:
            if alpha < 0.5:
                continue
            output.append(after)
            continue
        if after is None:
            if alpha >= 0.5:
                continue
            output.append(before)
            continue
        yaw_delta = float(np.arctan2(
            np.sin(after.yaw_ego - before.yaw_ego),
            np.cos(after.yaw_ego - before.yaw_ego),
        ))
        output.append(SemanticTrack(
            track_id=track_id,
            class_name=before.class_name,
            score=(1.0 - alpha) * before.score + alpha * after.score,
            center_ego=((1.0 - alpha) * before.center_ego + alpha * after.center_ego).astype(np.float32),
            size_wlh=((1.0 - alpha) * before.size_wlh + alpha * after.size_wlh).astype(np.float32),
            yaw_ego=before.yaw_ego + alpha * yaw_delta,
            velocity_ego=(
                (1.0 - alpha) * before.velocity_ego + alpha * after.velocity_ego
            ).astype(np.float32),
            history_ego=before.history_ego if alpha < 0.5 else after.history_ego,
            missed=before.missed if alpha < 0.5 else after.missed,
            display_alpha=float((1.0 - alpha) * before.display_alpha + alpha * after.display_alpha),
        ))
    return output


def snapshot_to_ego(snapshot: TrackSnapshot, ego_to_global: np.ndarray) -> SemanticTrack:
    """Transform one global tracker snapshot into the current ego frame."""
    global_to_ego = np.linalg.inv(ego_to_global)
    global_z = float(ego_to_global[2, 3])
    center_global = np.asarray([
        snapshot.position_global[0], snapshot.position_global[1], global_z, 1.0,
    ])
    center_ego = (global_to_ego @ center_global)[:3]
    center_ego[2] = max(float(snapshot.size_wlh[2]) * 0.5, 0.25)
    velocity_ego = (
        global_to_ego[:3, :3]
        @ np.asarray([snapshot.velocity_global[0], snapshot.velocity_global[1], 0.0])
    )[:2]
    heading_global = np.asarray([
        np.cos(snapshot.yaw_global), np.sin(snapshot.yaw_global), 0.0,
    ])
    heading_ego = global_to_ego[:3, :3] @ heading_global

    if len(snapshot.history_global):
        history_h = np.column_stack((
            snapshot.history_global,
            np.full(len(snapshot.history_global), global_z),
            np.ones(len(snapshot.history_global)),
        ))
        history_ego = (global_to_ego @ history_h.T).T[:, :3]
        history_ego[:, 2] = 0.08
    else:
        history_ego = np.empty((0, 3), dtype=np.float32)

    return SemanticTrack(
        track_id=snapshot.track_id,
        class_name=snapshot.class_name,
        score=snapshot.score,
        center_ego=center_ego.astype(np.float32),
        size_wlh=snapshot.size_wlh.astype(np.float32),
        yaw_ego=float(np.arctan2(heading_ego[1], heading_ego[0])),
        velocity_ego=velocity_ego.astype(np.float32),
        history_ego=history_ego.astype(np.float32),
        missed=snapshot.missed,
    )


def stabilized_snapshot_to_ego(
    snapshot: TrackSnapshot,
    ego_to_global: np.ndarray,
    min_visible_hits: int = 2,
    fade_in_hits: int = 4,
    fade_out_misses: int = 4,
) -> SemanticTrack | None:
    """Convert a tracker snapshot into a display track with fade-in/fade-out state."""
    if snapshot.hits < min_visible_hits:
        return None
    confirmed_steps = snapshot.hits - min_visible_hits + 1
    fade_in = confirmed_steps / max(fade_in_hits, 1)
    fade_out = (fade_out_misses + 1 - snapshot.missed) / max(fade_out_misses + 1, 1)
    alpha = float(np.clip(fade_in, 0.0, 1.0) * np.clip(fade_out, 0.0, 1.0))
    if alpha <= 0.02:
        return None
    return replace(snapshot_to_ego(snapshot, ego_to_global), display_alpha=alpha)


class Semantic3DRenderer:
    """Stylized driver visualization backed by metric 3D perception outputs."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self.camera_position = np.asarray([-22.0, 0.0, 18.0], dtype=np.float64)
        self.camera_target = np.asarray([18.0, 0.0, 0.0], dtype=np.float64)
        self.focal_px = width * 0.72
        self.principal = np.asarray([width * 0.5, height * 0.50], dtype=np.float64)
        forward = self.camera_target - self.camera_position
        self.camera_forward = forward / np.linalg.norm(forward)
        world_up = np.asarray([0.0, 0.0, 1.0])
        right = np.cross(self.camera_forward, world_up)
        self.camera_right = right / np.linalg.norm(right)
        self.camera_up = np.cross(self.camera_right, self.camera_forward)

    def project(self, points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
        relative = points - self.camera_position
        depth = relative @ self.camera_forward
        camera_x = relative @ self.camera_right
        camera_y = relative @ self.camera_up
        safe_depth = np.maximum(depth, 1e-4)
        pixels = np.column_stack((
            self.principal[0] + self.focal_px * camera_x / safe_depth,
            self.principal[1] - self.focal_px * camera_y / safe_depth,
        ))
        visible = (
            (depth > 0.3)
            & (pixels[:, 0] > -200)
            & (pixels[:, 0] < self.width + 200)
            & (pixels[:, 1] > -200)
            & (pixels[:, 1] < self.height + 200)
        )
        return pixels.astype(np.float32), depth.astype(np.float32), visible

    def render(
        self,
        tracks: list[SemanticTrack],
        elapsed_s: float,
        frame_index: int,
        radar_enabled: bool,
        radar_points_ego: np.ndarray | None = None,
        lidar_points_ego: np.ndarray | None = None,
        engineering_mode: bool = False,
        world_output: "WorldModelOutput | None" = None,
        title: str = "FUSION DRIVE",
        sensor_label: str | None = None,
    ) -> np.ndarray:
        image = self._background(engineering_mode)
        self._draw_ground(image, engineering_mode)
        if engineering_mode and lidar_points_ego is not None:
            self._draw_lidar(image, lidar_points_ego)
        if engineering_mode and radar_points_ego is not None:
            self._draw_radar(image, radar_points_ego)

        in_view = [
            item for item in tracks
            if -8.0 <= item.center_ego[0] <= 80.0 and abs(item.center_ego[1]) <= 35.0
        ]
        in_view.sort(key=lambda item: item.center_ego[0], reverse=True)
        priority = min(in_view, key=lambda item: item.distance_m, default=None)
        if engineering_mode:
            for track in in_view:
                self._draw_track_history(image, track)
        if world_output is not None:
            self._draw_world_predictions(image, world_output.predictions, engineering_mode)
        for track in in_view:
            self._draw_semantic_object(
                image, track, engineering_mode,
                priority is not None and track.track_id == priority.track_id,
            )
        self._draw_ego_vehicle(image, engineering_mode)
        self._draw_hud(
            image, in_view, elapsed_s, frame_index, radar_enabled,
            engineering_mode, world_output, title, sensor_label,
        )
        return image

    def _background(self, engineering_mode: bool) -> np.ndarray:
        if engineering_mode:
            top = np.asarray([20, 23, 29], dtype=np.float32)
            bottom = np.asarray([8, 10, 14], dtype=np.float32)
        else:
            top = np.asarray([226, 228, 231], dtype=np.float32)
            bottom = np.asarray([246, 247, 248], dtype=np.float32)
        blend = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
        row = top[None, None, :] * (1.0 - blend) + bottom[None, None, :] * blend
        return np.repeat(row, self.width, axis=1).astype(np.uint8)

    def _draw_ground(self, image: np.ndarray, engineering_mode: bool) -> None:
        road = np.asarray([
            [-6.0, -13.0, 0.0], [85.0, -13.0, 0.0],
            [85.0, 13.0, 0.0], [-6.0, 13.0, 0.0],
        ])
        pixels, _, visible = self.project(road)
        if visible.all():
            road_color = (37, 40, 46) if engineering_mode else (218, 220, 223)
            cv2.fillConvexPoly(image, pixels.astype(np.int32), road_color, cv2.LINE_AA)

        if engineering_mode:
            for x in range(0, 81, 10):
                self._world_line(image, (x, -13, 0.01), (x, 13, 0.01), (49, 53, 60), 1)
        for y in (-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5):
            if engineering_mode:
                color = (77, 80, 86) if abs(y) in (3.5, 10.5) else (48, 52, 58)
            else:
                color = (181, 184, 188) if abs(y) in (3.5, 10.5) else (207, 209, 212)
            thickness = 2 if abs(y) in (3.5, 10.5) else 1
            if not engineering_mode and abs(y) not in (3.5, 10.5):
                continue
            if abs(y) == 3.5:
                for start in np.arange(0.0, 82.0, 7.0):
                    self._world_line(
                        image, (start, y, 0.02), (min(start + 3.5, 85.0), y, 0.02),
                        color, thickness,
                    )
            else:
                self._world_line(image, (-5, y, 0.02), (85, y, 0.02), color, thickness)
        if not engineering_mode:
            self._draw_route_ribbon(image)

    def _draw_route_ribbon(self, image: np.ndarray) -> None:
        route = np.asarray([
            [2.3, -0.52, 0.045], [55.0, -0.52, 0.045],
            [55.0, 0.52, 0.045], [2.3, 0.52, 0.045],
        ])
        pixels, _, visible = self.project(route)
        if not visible.all():
            return
        overlay = image.copy()
        cv2.fillConvexPoly(overlay, pixels.astype(np.int32), (230, 145, 55), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
        self._world_line(image, (2.3, 0, 0.055), (55, 0, 0.055), (245, 171, 83), 2)

    def _draw_lidar(self, image: np.ndarray, points: np.ndarray) -> None:
        cloud = np.asarray(points)
        if not len(cloud):
            return
        stride = max(1, len(cloud) // 9000)
        xyz = cloud[::stride, :3]
        valid_range = (
            (xyz[:, 0] > -8) & (xyz[:, 0] < 80)
            & (np.abs(xyz[:, 1]) < 35) & (xyz[:, 2] > -2) & (xyz[:, 2] < 5)
        )
        xyz = xyz[valid_range]
        pixels, depth, visible = self.project(xyz)
        for pixel, z, is_visible in zip(pixels, xyz[:, 2], visible):
            if not is_visible:
                continue
            intensity = int(np.clip(115 + z * 28, 80, 230))
            cv2.circle(image, tuple(pixel.astype(int)), 1, (intensity, 145, 75), -1, cv2.LINE_AA)

    def _draw_radar(self, image: np.ndarray, points: np.ndarray) -> None:
        cloud = np.asarray(points)
        if not len(cloud):
            return
        pixels, _, visible = self.project(cloud[:, :3])
        for point, pixel, is_visible in zip(cloud, pixels, visible):
            if not is_visible:
                continue
            origin = tuple(pixel.astype(int))
            cv2.circle(image, origin, 3, (75, 90, 250), -1, cv2.LINE_AA)
            if point.shape[0] >= 5:
                end_world = point[:3].copy()
                end_world[:2] += np.clip(point[3:5], -5.0, 5.0)
                end_pixel, _, end_visible = self.project(end_world.reshape(1, 3))
                if end_visible[0]:
                    cv2.arrowedLine(
                        image, origin, tuple(end_pixel[0].astype(int)),
                        (75, 90, 250), 1, cv2.LINE_AA, tipLength=0.22,
                    )

    def _draw_track_history(self, image: np.ndarray, track: SemanticTrack) -> None:
        if len(track.history_ego) < 2:
            return
        pixels, _, visible = self.project(track.history_ego)
        points = pixels[visible].astype(np.int32)
        if len(points) >= 2:
            color = CLASS_COLORS.get(track.class_name, (190, 190, 190))
            cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)

    def _draw_world_predictions(
        self,
        image: np.ndarray,
        predictions: list["AgentFuture"],
        engineering_mode: bool,
    ) -> None:
        for prediction in predictions:
            path = prediction.centers_ego.copy()
            if len(path) < 2:
                continue
            if path[-1, 0] < -10.0 or path[0, 0] > 90.0:
                continue
            path[:, 2] = 0.10
            pixels, _, visible = self.project(path)
            points = pixels[visible].astype(np.int32)
            if len(points) < 2:
                continue
            alpha = float(np.clip(prediction.display_alpha, 0.0, 1.0))
            if alpha <= 0.03:
                continue
            target = image if alpha >= 0.98 else image.copy()
            color = RISK_COLORS.get(prediction.risk_level, (150, 150, 150))
            thickness = 3 if prediction.risk_level in {"caution", "critical"} else 2
            cv2.polylines(target, [points], False, color, thickness, cv2.LINE_AA)
            for index, point in enumerate(points[1:], 1):
                radius = 4 if index == len(points) - 1 else 2
                cv2.circle(target, tuple(point), radius, color, -1, cv2.LINE_AA)

            if prediction.risk_level in {"caution", "critical"}:
                label = (
                    f"{prediction.risk_level.upper()} "
                    f"{prediction.time_to_risk_s:.1f}s"
                    if prediction.time_to_risk_s is not None
                    else prediction.risk_level.upper()
                )
                anchor = tuple(points[min(len(points) - 1, 2)])
                scale = 0.42 if engineering_mode else 0.38
                self._outlined_text(target, label, (anchor[0] + 8, anchor[1] - 8),
                                    scale, color, 1)
            if target is not image:
                cv2.addWeighted(target, alpha, image, 1.0 - alpha, 0.0, image)

    def _draw_semantic_object(
        self,
        image: np.ndarray,
        track: SemanticTrack,
        engineering_mode: bool,
        priority: bool,
    ) -> None:
        alpha = float(np.clip(track.display_alpha, 0.0, 1.0))
        if alpha <= 0.03:
            return
        target = image if alpha >= 0.98 else image.copy()
        self._draw_semantic_object_body(target, track, engineering_mode, priority)
        if target is not image:
            cv2.addWeighted(target, alpha, image, 1.0 - alpha, 0.0, image)

    def _draw_semantic_object_body(
        self,
        image: np.ndarray,
        track: SemanticTrack,
        engineering_mode: bool,
        priority: bool,
    ) -> None:
        if engineering_mode:
            color = CLASS_COLORS.get(track.class_name, (190, 190, 190))
        elif priority and track.distance_m < 15.0:
            color = DRIVING_PRIORITY_COLOR
        elif track.class_name == "pedestrian":
            color = DRIVING_PEDESTRIAN_COLOR
        else:
            color = DRIVING_OBJECT_COLOR
        if track.class_name == "pedestrian":
            self._draw_pedestrian(image, track, color)
        elif track.class_name == "traffic_cone":
            self._draw_cone(image, track, color)
        elif not engineering_mode and track.class_name in {
            "car", "truck", "construction_vehicle", "bus", "trailer",
        }:
            self._draw_vehicle_model(
                image, track.center_ego, track.size_wlh, track.yaw_ego, color,
                long_body=track.class_name in {"truck", "bus", "trailer"},
            )
        else:
            self._draw_box(image, track.center_ego, track.size_wlh, track.yaw_ego, color)

        if not engineering_mode:
            return

        label_point = track.center_ego.copy()
        label_point[2] += max(float(track.size_wlh[2]) * 0.65, 1.2)
        pixels, _, visible = self.project(label_point.reshape(1, 3))
        if not visible[0]:
            return
        suffix = " P" if track.missed else ""
        label = (
            f"{track.class_name}  ID{track.track_id}{suffix}  "
            f"{track.distance_m:.0f}m  {track.speed_mps:.1f}m/s"
        )
        self._outlined_text(image, label, tuple(pixels[0].astype(int)), 0.42, color, 1)

    def _draw_pedestrian(
        self, image: np.ndarray, track: SemanticTrack, color: tuple[int, int, int],
    ) -> None:
        height = max(float(track.size_wlh[2]), 1.5)
        base = track.center_ego.copy()
        base[2] = 0.05
        hip = base.copy()
        hip[2] = height * 0.40
        shoulders = base.copy()
        shoulders[2] = height * 0.70
        head = base.copy()
        head[2] = height
        pixels, _, visible = self.project(np.vstack((base, hip, shoulders, head)))
        if not visible.all():
            return
        base_px, hip_px, shoulders_px, head_px = pixels.astype(int)
        radius = int(np.clip(105 / max(track.distance_m, 7.0), 3, 10))
        thickness = max(2, radius // 3)
        cv2.line(image, tuple(hip_px), tuple(shoulders_px), color, thickness + 1, cv2.LINE_AA)
        cv2.line(image, tuple(shoulders_px), tuple(shoulders_px + [-radius, radius]),
                 color, thickness, cv2.LINE_AA)
        cv2.line(image, tuple(shoulders_px), tuple(shoulders_px + [radius, radius]),
                 color, thickness, cv2.LINE_AA)
        cv2.line(image, tuple(hip_px), tuple(base_px + [-radius // 2, 0]),
                 color, thickness, cv2.LINE_AA)
        cv2.line(image, tuple(hip_px), tuple(base_px + [radius // 2, 0]),
                 color, thickness, cv2.LINE_AA)
        cv2.circle(image, tuple(head_px), radius, color, -1, cv2.LINE_AA)

    def _draw_cone(
        self, image: np.ndarray, track: SemanticTrack, color: tuple[int, int, int],
    ) -> None:
        center = track.center_ego.copy()
        half_width = max(float(track.size_wlh[0]) * 0.5, 0.18)
        height = max(float(track.size_wlh[2]), 0.7)
        points = np.asarray([
            [center[0], center[1] - half_width, 0.02],
            [center[0], center[1] + half_width, 0.02],
            [center[0], center[1], height],
        ])
        pixels, _, visible = self.project(points)
        if visible.all():
            cv2.fillConvexPoly(image, pixels.astype(np.int32), color, cv2.LINE_AA)

    def _draw_ego_vehicle(self, image: np.ndarray, engineering_mode: bool) -> None:
        center = np.asarray([0.0, 0.0, 0.78])
        if engineering_mode:
            self._draw_box(image, center, np.asarray([1.95, 4.7, 1.55]), 0.0, (235, 235, 238))
            self._world_line(image, (2.45, 0, 0.08), (10.0, 0, 0.08), (90, 165, 245), 2)
        else:
            self._draw_vehicle_model(
                image, center, np.asarray([1.95, 4.7, 1.55]), 0.0,
                (250, 250, 250), ego=True,
            )

    def _draw_vehicle_model(
        self,
        image: np.ndarray,
        center: np.ndarray,
        size_wlh: np.ndarray,
        yaw: float,
        color: tuple[int, int, int],
        long_body: bool = False,
        ego: bool = False,
    ) -> None:
        width, length, height = (float(value) for value in size_wlh)
        self._draw_vehicle_shadow(image, center, width, length, yaw)
        ground_center = center.copy()
        ground_center[2] = max(height * 0.30, 0.28)
        body_height = height * (0.60 if long_body else 0.48)
        self._draw_box(
            image, ground_center, np.asarray([width, length, body_height]), yaw, color,
            fill_strength=0.84, face_scale=0.80,
        )

        heading = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
        cabin_center = center.copy()
        cabin_center += heading * (-length * (0.02 if long_body else 0.08))
        cabin_height = height * (0.62 if long_body else 0.55)
        cabin_center[2] = body_height + cabin_height * 0.48
        cabin_length = length * (0.72 if long_body else 0.50)
        if long_body:
            self._draw_box(
                image,
                cabin_center,
                np.asarray([width * 0.84, cabin_length, cabin_height]),
                yaw,
                color,
                fill_strength=0.86,
                face_scale=0.86,
            )
        else:
            self._draw_tapered_cabin(
                image, cabin_center, width, cabin_length, cabin_height, yaw, color,
            )
        roof_center = cabin_center.copy()
        roof_center[2] = body_height + cabin_height * 0.96
        glass = (82, 93, 104) if not ego else (96, 111, 124)
        self._draw_box(
            image,
            roof_center,
            np.asarray([width * 0.64, cabin_length * 0.64, max(height * 0.055, 0.06)]),
            yaw,
            glass,
            fill_strength=0.90,
            face_scale=0.72,
        )

        wheel_offset_x = length * 0.33
        wheel_offset_y = width * 0.52
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
        wheel_local = np.asarray([
            [wheel_offset_x, wheel_offset_y], [wheel_offset_x, -wheel_offset_y],
            [-wheel_offset_x, wheel_offset_y], [-wheel_offset_x, -wheel_offset_y],
        ])
        wheel_xy = wheel_local @ rotation.T + center[:2]
        wheel_world = np.column_stack((wheel_xy, np.full(4, max(height * 0.17, 0.18))))
        pixels, _, visible = self.project(wheel_world)
        for pixel, is_visible in zip(pixels, visible):
            if is_visible:
                cv2.circle(image, tuple(pixel.astype(int)), 3 if ego else 2,
                           (48, 51, 55), -1, cv2.LINE_AA)
        self._draw_vehicle_lights(image, center, width, length, height, yaw, ego)

    def _draw_vehicle_shadow(
        self,
        image: np.ndarray,
        center: np.ndarray,
        width: float,
        length: float,
        yaw: float,
    ) -> None:
        local = np.asarray([
            [length * 0.52, width * 0.56],
            [length * 0.52, -width * 0.56],
            [-length * 0.52, -width * 0.56],
            [-length * 0.52, width * 0.56],
        ])
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
        xy = local @ rotation.T + center[:2]
        points = np.column_stack((xy, np.full(4, 0.018)))
        pixels, _, visible = self.project(points)
        if visible.all():
            overlay = image.copy()
            cv2.fillConvexPoly(overlay, pixels.astype(np.int32), (62, 64, 68), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.20, image, 0.80, 0.0, image)

    def _draw_vehicle_lights(
        self,
        image: np.ndarray,
        center: np.ndarray,
        width: float,
        length: float,
        height: float,
        yaw: float,
        ego: bool,
    ) -> None:
        local = np.asarray([
            [length * 0.505, width * 0.31], [length * 0.505, -width * 0.31],
            [-length * 0.505, width * 0.31], [-length * 0.505, -width * 0.31],
        ])
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
        xy = local @ rotation.T + center[:2]
        points = np.column_stack((xy, np.full(4, max(height * 0.28, 0.25))))
        pixels, _, visible = self.project(points)
        radius = 3 if ego else 2
        for index, (pixel, is_visible) in enumerate(zip(pixels, visible)):
            if not is_visible:
                continue
            color = (245, 245, 250) if index < 2 else (65, 72, 215)
            cv2.circle(image, tuple(pixel.astype(int)), radius, color, -1, cv2.LINE_AA)

    def _draw_tapered_cabin(
        self,
        image: np.ndarray,
        center: np.ndarray,
        vehicle_width: float,
        cabin_length: float,
        cabin_height: float,
        yaw: float,
        color: tuple[int, int, int],
    ) -> None:
        lower_width = vehicle_width * 0.84
        upper_width = vehicle_width * 0.62
        lower_length = cabin_length
        upper_length = cabin_length * 0.66
        local = np.asarray([
            [lower_length / 2, lower_width / 2, -cabin_height / 2],
            [lower_length / 2, -lower_width / 2, -cabin_height / 2],
            [-lower_length / 2, -lower_width / 2, -cabin_height / 2],
            [-lower_length / 2, lower_width / 2, -cabin_height / 2],
            [upper_length / 2, upper_width / 2, cabin_height / 2],
            [upper_length / 2, -upper_width / 2, cabin_height / 2],
            [-upper_length / 2, -upper_width / 2, cabin_height / 2],
            [-upper_length / 2, upper_width / 2, cabin_height / 2],
        ])
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        vertices = local @ rotation.T + center
        pixels, depth, visible = self.project(vertices)
        if visible.sum() < 4:
            return
        faces = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7))
        overlay = image.copy()
        face_order = sorted(faces, key=lambda face: float(np.mean(depth[list(face)])), reverse=True)
        for face in face_order:
            if not visible[list(face)].all():
                continue
            face_color = tuple(int(channel * 0.84) for channel in color)
            cv2.fillConvexPoly(overlay, pixels[list(face)].astype(np.int32), face_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0.0, image)
        for first, second in (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ):
            if visible[first] and visible[second]:
                cv2.line(
                    image, tuple(pixels[first].astype(int)), tuple(pixels[second].astype(int)),
                    color, 2, cv2.LINE_AA,
                )

    def _draw_box(
        self,
        image: np.ndarray,
        center: np.ndarray,
        size_wlh: np.ndarray,
        yaw: float,
        color: tuple[int, int, int],
        fill_strength: float = 0.72,
        face_scale: float = 0.55,
    ) -> None:
        width, length, height = (float(value) for value in size_wlh)
        width, length, height = max(width, 0.35), max(length, 0.45), max(height, 0.35)
        local = np.asarray([
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ])
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        corners = local @ rotation.T + center
        pixels, depth, visible = self.project(corners)
        if visible.sum() < 4:
            return
        faces = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7))
        overlay = image.copy()
        face_order = sorted(faces, key=lambda face: float(np.mean(depth[list(face)])), reverse=True)
        for face in face_order:
            if not visible[list(face)].all():
                continue
            face_color = tuple(int(channel * face_scale) for channel in color)
            cv2.fillConvexPoly(overlay, pixels[list(face)].astype(np.int32), face_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, fill_strength, image, 1.0 - fill_strength, 0.0, image)
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for first, second in edges:
            if visible[first] and visible[second]:
                cv2.line(
                    image, tuple(pixels[first].astype(int)), tuple(pixels[second].astype(int)),
                    color, 2, cv2.LINE_AA,
                )

    def _draw_hud(
        self,
        image: np.ndarray,
        tracks: list[SemanticTrack],
        elapsed_s: float,
        frame_index: int,
        radar_enabled: bool,
        engineering_mode: bool,
        world_output: "WorldModelOutput | None",
        title: str,
        sensor_label: str | None,
    ) -> None:
        default_sensor_label = (
            "CAMERA + LIDAR  |  RADAR ON" if engineering_mode and radar_enabled
            else "CAMERA + LIDAR  |  RADAR OFF" if engineering_mode
            else f"CAMERA  +  LIDAR  +  {'RADAR' if radar_enabled else 'NO RADAR'}"
        )
        displayed_sensor_label = sensor_label or default_sensor_label
        if engineering_mode:
            cv2.rectangle(image, (22, 20), (410, 150), (13, 16, 21), -1, cv2.LINE_AA)
            self._outlined_text(image, title, (42, 54), 0.78, (245, 245, 245), 2)
            cv2.putText(image, "ENGINEERING", (42, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (165, 170, 178), 1, cv2.LINE_AA)
            cv2.putText(image, displayed_sensor_label, (42, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 190, 245), 1, cv2.LINE_AA)
            cv2.putText(
                image, f"t {elapsed_s:05.1f}s   frame {frame_index:03d}   objects {len(tracks)}",
                (42, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 213, 218), 1, cv2.LINE_AA,
            )
        else:
            self._rounded_rect(image, (26, 24), (342, 128), 18, (250, 250, 250))
            cv2.putText(image, title, (48, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, (48, 50, 54), 2, cv2.LINE_AA)
            cv2.putText(image, displayed_sensor_label, (48, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (105, 109, 115), 1, cv2.LINE_AA)
            cv2.putText(image, f"{elapsed_s:04.1f}s     {len(tracks)} objects", (48, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (128, 132, 138), 1, cv2.LINE_AA)

        if world_output is not None:
            color = RISK_COLORS.get(world_output.risk_level, (130, 130, 130))
            text = (
                f"WORLD  {world_output.risk_level.upper()}  "
                f"{world_output.risk_score:.2f}"
            )
            if engineering_mode:
                cv2.rectangle(image, (22, 158), (500, 230), (13, 16, 21), -1, cv2.LINE_AA)
                cv2.putText(image, text, (42, 188), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, color, 2, cv2.LINE_AA)
                cv2.putText(image, world_output.advisory[:64], (42, 214),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 213, 218), 1, cv2.LINE_AA)
            else:
                self._rounded_rect(image, (26, 138), (442, 206), 16, (250, 250, 250))
                cv2.putText(image, text, (48, 166), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, color, 2, cv2.LINE_AA)
                cv2.putText(image, world_output.advisory[:54], (48, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (96, 101, 108), 1, cv2.LINE_AA)

        nearest = min(tracks, key=lambda item: item.distance_m, default=None)
        if nearest is not None:
            text = f"NEAREST  {nearest.class_name.upper()}  {nearest.distance_m:.1f} m"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
            x = self.width - size[0] - 38
            panel_color = (13, 16, 21) if engineering_mode else (250, 250, 250)
            self._rounded_rect(image, (x - 16, 24), (self.width - 22, 66), 14, panel_color)
            text_color = (95, 190, 245) if engineering_mode else (70, 76, 84)
            cv2.putText(image, text, (x, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, text_color, 1, cv2.LINE_AA)

        footer_color = (115, 120, 128) if engineering_mode else (120, 124, 130)
        cv2.putText(image, "Semantic display - not a safety control system", (25, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, footer_color, 1, cv2.LINE_AA)

    @staticmethod
    def _rounded_rect(
        image: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for center in (
            (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)

    def _world_line(
        self,
        image: np.ndarray,
        first: tuple[float, float, float],
        second: tuple[float, float, float],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        pixels, _, visible = self.project(np.asarray([first, second]))
        if visible.all():
            cv2.line(image, tuple(pixels[0].astype(int)), tuple(pixels[1].astype(int)),
                     color, thickness, cv2.LINE_AA)

    @staticmethod
    def _outlined_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (8, 10, 13), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)
