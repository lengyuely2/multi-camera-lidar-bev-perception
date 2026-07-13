from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .tracking import TrackSnapshot


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

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.center_ego[:2]))

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_ego))


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


class Semantic3DRenderer:
    """Stylized driver visualization backed by metric 3D perception outputs."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self.camera_position = np.asarray([-18.0, 0.0, 15.0], dtype=np.float64)
        self.camera_target = np.asarray([20.0, 0.0, 0.0], dtype=np.float64)
        self.focal_px = width * 0.76
        self.principal = np.asarray([width * 0.5, height * 0.51], dtype=np.float64)
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
    ) -> np.ndarray:
        image = self._background()
        self._draw_ground(image)
        if engineering_mode and lidar_points_ego is not None:
            self._draw_lidar(image, lidar_points_ego)
        if engineering_mode and radar_points_ego is not None:
            self._draw_radar(image, radar_points_ego)

        in_view = [
            item for item in tracks
            if -8.0 <= item.center_ego[0] <= 80.0 and abs(item.center_ego[1]) <= 35.0
        ]
        in_view.sort(key=lambda item: item.center_ego[0], reverse=True)
        if engineering_mode:
            for track in in_view:
                self._draw_track_history(image, track)
        for track in in_view:
            self._draw_semantic_object(image, track, engineering_mode)
        self._draw_ego_vehicle(image)
        self._draw_hud(image, in_view, elapsed_s, frame_index, radar_enabled, engineering_mode)
        return image

    def _background(self) -> np.ndarray:
        top = np.asarray([20, 23, 29], dtype=np.float32)
        bottom = np.asarray([8, 10, 14], dtype=np.float32)
        blend = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
        row = top[None, None, :] * (1.0 - blend) + bottom[None, None, :] * blend
        return np.repeat(row, self.width, axis=1).astype(np.uint8)

    def _draw_ground(self, image: np.ndarray) -> None:
        road = np.asarray([
            [-6.0, -13.0, 0.0], [85.0, -13.0, 0.0],
            [85.0, 13.0, 0.0], [-6.0, 13.0, 0.0],
        ])
        pixels, _, visible = self.project(road)
        if visible.all():
            cv2.fillConvexPoly(image, pixels.astype(np.int32), (37, 40, 46), cv2.LINE_AA)

        for x in range(0, 81, 10):
            self._world_line(image, (x, -13, 0.01), (x, 13, 0.01), (49, 53, 60), 1)
        for y in (-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5):
            color = (77, 80, 86) if abs(y) in (3.5, 10.5) else (48, 52, 58)
            thickness = 2 if abs(y) in (3.5, 10.5) else 1
            if abs(y) == 3.5:
                for start in np.arange(0.0, 82.0, 7.0):
                    self._world_line(
                        image, (start, y, 0.02), (min(start + 3.5, 85.0), y, 0.02),
                        color, thickness,
                    )
            else:
                self._world_line(image, (-5, y, 0.02), (85, y, 0.02), color, thickness)

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

    def _draw_semantic_object(
        self, image: np.ndarray, track: SemanticTrack, engineering_mode: bool,
    ) -> None:
        color = CLASS_COLORS.get(track.class_name, (190, 190, 190))
        if track.class_name == "pedestrian":
            self._draw_pedestrian(image, track, color)
        elif track.class_name == "traffic_cone":
            self._draw_cone(image, track, color)
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

    def _draw_ego_vehicle(self, image: np.ndarray) -> None:
        center = np.asarray([0.0, 0.0, 0.78])
        self._draw_box(image, center, np.asarray([1.95, 4.7, 1.55]), 0.0, (235, 235, 238))
        self._world_line(image, (2.45, 0, 0.08), (10.0, 0, 0.08), (90, 165, 245), 2)

    def _draw_box(
        self,
        image: np.ndarray,
        center: np.ndarray,
        size_wlh: np.ndarray,
        yaw: float,
        color: tuple[int, int, int],
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
            face_color = tuple(int(channel * 0.55) for channel in color)
            cv2.fillConvexPoly(overlay, pixels[list(face)].astype(np.int32), face_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
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
    ) -> None:
        cv2.rectangle(image, (22, 20), (410, 150), (13, 16, 21), -1, cv2.LINE_AA)
        self._outlined_text(image, "SURROUND MODEL", (42, 54), 0.78, (245, 245, 245), 2)
        mode = "ENGINEERING" if engineering_mode else "DRIVING VISUALIZATION"
        cv2.putText(image, mode, (42, 82), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (165, 170, 178), 1, cv2.LINE_AA)
        radar = "RADAR ON" if radar_enabled else "RADAR OFF"
        cv2.putText(image, f"CAMERA + LIDAR  |  {radar}", (42, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 190, 245), 1, cv2.LINE_AA)
        cv2.putText(image, f"t {elapsed_s:05.1f}s   frame {frame_index:03d}   objects {len(tracks)}",
                    (42, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 213, 218), 1, cv2.LINE_AA)

        nearest = min(tracks, key=lambda item: item.distance_m, default=None)
        if nearest is not None:
            text = f"NEAREST  {nearest.class_name.upper()}  {nearest.distance_m:.1f} m"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
            x = self.width - size[0] - 38
            cv2.rectangle(image, (x - 14, 26), (self.width - 22, 64), (13, 16, 21), -1)
            cv2.putText(image, text, (x, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (95, 190, 245), 1, cv2.LINE_AA)

        cv2.putText(image, "Semantic display - not a safety control system", (25, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (115, 120, 128), 1, cv2.LINE_AA)

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
