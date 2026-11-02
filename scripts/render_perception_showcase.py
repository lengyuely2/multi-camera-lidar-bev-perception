"""Render a recorded-sensor perception showcase; no simulated neural outputs.

Camera frames, point clouds and predictions are matched by nuScenes sample token.
Output frames hold the last keyframe: video FPS is not model throughput. Point
colors are projected camera RGB or predicted-box membership, not segmentation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parking_bev.perception.predictions import bevfusion_predictions_from_records, within_detection_range
from parking_bev.sensors.nuscenes_source import NuScenesSource
from parking_bev.tracking.tracker import TimestampAwareTracker, prediction_to_global_measurement
from parking_bev.visualization.metric_bev import MetricBEVRenderer


PALETTE = {
    "car": (176, 232, 85), "truck": (255, 184, 88), "bus": (255, 184, 88),
    "trailer": (242, 145, 114), "construction_vehicle": (245, 163, 110),
    "pedestrian": (80, 190, 255), "bicycle": (235, 136, 215),
    "motorcycle": (235, 136, 215), "barrier": (92, 142, 248),
    "traffic_cone": (92, 142, 248),
}
FOCUS = (65, 209, 255)  # OpenCV BGR
BG, PANEL = (25, 19, 15), (36, 28, 22)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def font(size):
    for path in [Path("C:/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyh.ttc"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONTS = {size: font(size) for size in [15, 17, 18, 20, 23, 31]}


def words(image, lines):
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for xy, text, size, color in lines:
        draw.text(xy, text, font=FONTS[size], fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def corners(obj):
    xy = MetricBEVRenderer.object_corners_xy(obj)
    z = float(obj.center_ego[2])
    half = float(obj.size_wlh[2]) / 2
    return np.vstack((np.column_stack((xy, np.full(4, z - half))),
                      np.column_stack((xy, np.full(4, z + half)))))


def apply_transform(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def calibrated_camera_transform(source, frame, channel):
    """Current LiDAR ego -> camera at its actual exposure time."""
    from pyquaternion import Quaternion
    sample = source.nusc.get("sample", frame.token)
    data = source.nusc.get("sample_data", sample["data"][channel])
    pose = source.nusc.get("ego_pose", data["ego_pose_token"])
    camera_ego = np.eye(4)
    camera_ego[:3, :3] = Quaternion(pose["rotation"]).rotation_matrix
    camera_ego[:3, 3] = pose["translation"]
    return np.linalg.inv(camera_ego @ frame.calibrations[channel].sensor_to_ego) @ frame.ego_to_global


def image_projection(points, transform, intrinsic):
    camera = apply_transform(points, transform)
    projection = camera @ intrinsic.T
    uv = projection[:, :2] / np.maximum(projection[:, 2:3], 1e-4)
    return uv, camera[:, 2]


def rgb_colors(points, frame, transforms):
    colors = np.full((len(points), 3), (78, 88, 100), np.uint8)
    best = np.full(len(points), np.inf)
    for channel, image in frame.cameras.items():
        uv, depth = image_projection(points, transforms[channel], frame.calibrations[channel].intrinsic)
        h, w = image.shape[:2]
        valid = ((depth > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
        # Prefer the camera whose optical axis is closest to the projected ray.
        rank = np.abs(uv[:, 0] - w / 2) / w
        indices = np.flatnonzero(valid & (rank < best))
        pixels = uv[indices].astype(int)
        colors[indices] = image[pixels[:, 1], pixels[:, 0]]
        best[indices] = rank[indices]
    return colors


def associated_colors(points, predictions):
    # Conservative box membership only. These are not learned per-point labels.
    colors = np.full((len(points), 3), (84, 91, 100), np.uint8)
    for item in sorted(predictions, key=lambda value: value.score):
        obj = item.object
        relative = points - obj.center_ego
        c, s = np.cos(obj.yaw_ego), np.sin(obj.yaw_ego)
        x, y = relative[:, 0] * c + relative[:, 1] * s, -relative[:, 0] * s + relative[:, 1] * c
        mask = ((np.abs(x) < obj.size_wlh[1] / 2) & (np.abs(y) < obj.size_wlh[0] / 2)
                & (np.abs(relative[:, 2]) < obj.size_wlh[2] / 2))
        colors[mask] = PALETTE[item.class_name]
    return colors


class CloudView:
    def __init__(self, width, height):
        self.width, self.height = width, height
        eye, target = np.array([-33., -43., 39.]), np.array([8., 0., 0.])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0., 0., 1.]))
        right /= np.linalg.norm(right)
        self.rotation = np.array([right, np.cross(right, forward), forward])
        self.eye = eye

    def project(self, points):
        p = (points - self.eye) @ self.rotation.T
        focal = self.width * 0.88
        uv = np.column_stack((self.width / 2 + focal * p[:, 0] / np.maximum(p[:, 2], .1),
                              self.height / 2 - focal * p[:, 1] / np.maximum(p[:, 2], .1)))
        return uv, p[:, 2]

    def render(self, points, colors, predictions=()):
        image = np.full((self.height, self.width, 3), PANEL, np.uint8)
        uv, depth = self.project(points)
        valid = ((depth > 1) & (uv[:, 0] >= 1) & (uv[:, 0] < self.width - 1)
                 & (uv[:, 1] >= 1) & (uv[:, 1] < self.height - 1))
        indices = np.flatnonzero(valid)
        indices = indices[np.argsort(-depth[indices], kind="stable")]
        pixels = uv[indices].astype(int)
        # Far-to-near splats; a few adjacent pixels make sparse scans legible at 1080p.
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            image[pixels[:, 1] + dy, pixels[:, 0] + dx] = colors[indices]
        for item in predictions:
            uv, depth = self.project(corners(item.object))
            if np.all(depth > 1):
                color = PALETTE[item.class_name]
                for a, b in EDGES:
                    cv2.line(image, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), color, 1, cv2.LINE_AA)
        return image


@dataclass
class Keyframe:
    token: str
    timestamp: float
    predictions: list
    tracks: list
    pose: np.ndarray
    inference_seconds: float | None


def make_keyframes(source, payload, score):
    tracker = TimestampAwareTracker(association_distance_m=4., max_missed_seconds=.8, appearance_weight=0.)
    result = []
    for record in payload["frames"]:
        frame = source.read_token(record["sample_token"])
        predictions = [p for p in bevfusion_predictions_from_records(record["predictions"],
                       frame.calibrations["LIDAR_TOP"].sensor_to_ego, score)
                       if within_detection_range(p.class_name, p.object.center_ego)]
        timestamp = frame.timestamp_us / 1e6
        tracks = tracker.update(timestamp, [prediction_to_global_measurement(p, frame.ego_to_global) for p in predictions])
        result.append(Keyframe(frame.token, timestamp, predictions, tracks, frame.ego_to_global, record.get("inference_seconds")))
    return result


def choose_focus(keyframes):
    """Select a visible vehicle with measured heading change for presentation.

    This is an offline highlight heuristic, not a trained turning classifier.
    Heading changes are measured in the fixed world frame, excluding ego yaw.
    """
    candidates = []
    for end, key in enumerate(keyframes):
        for track in key.tracks:
            if track.class_name not in {"car", "truck", "bus"} or track.missed:
                continue
            history = [(i, next((t for t in keyframes[i].tracks if t.track_id == track.track_id and not t.missed), None))
                       for i in range(max(0, end - 8), end + 1)]
            history = [(i, t) for i, t in history if t is not None]
            if len(history) < 5:
                continue
            global_to_ego = np.linalg.inv(key.pose)
            center = apply_transform(np.array([[*track.position_global, track.z_global]]), global_to_ego)[0]
            if not (2 < center[0] < 40 and abs(center[1]) < 22):
                continue
            # Modulo pi avoids treating a box-direction flip as a 180-degree turn.
            angles = np.unwrap(np.array([t.yaw_global for _, t in history]) * 2) / 2
            if np.max(np.abs(np.diff(angles))) > np.deg2rad(35):
                continue
            change = abs(float(angles[-1] - angles[0]))
            displacement = float(np.linalg.norm(history[-1][1].position_global - history[0][1].position_global))
            if change > np.deg2rad(20) and displacement > 2:
                candidates.append((change - .004 * np.linalg.norm(center[:2]), track.track_id, end,
                                   float(np.rad2deg(change)), history[0][0]))
    if candidates:
        _, identifier, index, change, start = max(candidates)
        return dict(track_id=identifier, screenshot_index=index, heading_change_deg=round(change, 1),
                    window_start_index=start, turning_candidate=True,
                    selection="offline predicted world-heading change; not a learned turn label")
    # Honest fallback: a tracked vehicle, without asserting that it turns.
    best = [(t.hits, t.track_id, i) for i, k in enumerate(keyframes) for t in k.tracks
            if t.class_name == "car" and not t.missed]
    _, identifier, index = max(best) if best else (0, -1, 0)
    return dict(track_id=identifier, screenshot_index=index, turning_candidate=False,
                selection="stable vehicle track; no qualifying turn found")


def track_geometry(track, pose):
    from parking_bev.sensors.nuscenes_source import Object3D
    inverse = np.linalg.inv(pose)
    center = apply_transform(np.array([[*track.position_global, track.z_global]]), inverse)[0]
    heading = inverse[:3, :3] @ [np.cos(track.yaw_global), np.sin(track.yaw_global), 0.]
    return Object3D(str(track.track_id), track.class_name, center, track.size_wlh,
                    float(np.arctan2(heading[1], heading[0])), np.zeros(2))


def camera_panel(frame, channel, transform, key, focus, width, height):
    image = frame.cameras[channel].copy()
    track = next((t for t in key.tracks if t.track_id == focus["track_id"] and not t.missed), None)
    if track:
        obj = track_geometry(track, key.pose)
        uv, depth = image_projection(corners(obj), transform, frame.calibrations[channel].intrinsic)
        if np.all(depth > .5):
            lo, hi = uv.min(axis=0), uv.max(axis=0)
            h, w = image.shape[:2]
            if hi[0] > 0 and lo[0] < w and hi[1] > 0 and lo[1] < h:
                for a, b in EDGES:
                    cv2.line(image, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), FOCUS, 4, cv2.LINE_AA)
                cv2.putText(image, f'TRACK #{track.track_id}', (max(0, int(lo[0])), max(35, int(lo[1]) - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, FOCUS, 3, cv2.LINE_AA)
    output = np.full((height, width, 3), PANEL, np.uint8)
    scale = min(width / image.shape[1], (height - 34) / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    x, y = (width - resized.shape[1]) // 2, 34 + (height - 34 - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return words(output, [((12, 5), channel.replace("CAM_", ""), 17, (210, 221, 232)),
                           ((width - 160, 7), "RECORDED CAMERA", 15, (140, 156, 173))])


def bev_panel(key, points, focus, width, height, tracking=False):
    image = np.full((height, width, 3), PANEL, np.uint8)
    scale = min(width / 80, height / 56)
    def pixel(xy):
        xy = np.asarray(xy)
        return np.stack((width / 2 - xy[..., 1] * scale, height * .78 - xy[..., 0] * scale), axis=-1).astype(int)
    for x in range(-10, 51, 10):
        cv2.line(image, tuple(pixel([x, -60])), tuple(pixel([x, 60])), (40, 47, 58), 1)
        cv2.putText(image, f'{x}m', (8, int(pixel([x, 0])[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, .35, (112, 128, 146), 1)
    for y in range(-60, 61, 10):
        cv2.line(image, tuple(pixel([-20, y])), tuple(pixel([60, y])), (40, 47, 58), 1)
    uv = pixel(points[:, :2])
    valid = ((uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    image[uv[valid, 1], uv[valid, 0]] = (72, 83, 95)
    if tracking:
        items = [(track_geometry(t, key.pose), PALETTE[t.class_name], t.track_id, t)
                 for t in key.tracks if not t.missed and t.hits >= 2]
    else:
        items = [(p.object, PALETTE[p.class_name], None, None) for p in key.predictions]
    for obj, color, identifier, track in items:
        selected = identifier == focus["track_id"]
        color = FOCUS if selected else color
        outline = pixel(MetricBEVRenderer.object_corners_xy(obj))
        if tracking and track is not None and len(track.history_global) > 1 and track.class_name not in {'barrier', 'traffic_cone'}:
            history = np.column_stack((track.history_global, np.full(len(track.history_global), track.z_global)))
            history = apply_transform(history, np.linalg.inv(key.pose))
            cv2.polylines(image, [pixel(history[:, :2])], False, color, 3 if selected else 1, cv2.LINE_AA)
        cv2.polylines(image, [outline], True, color, 3 if selected else 1, cv2.LINE_AA)
        center = pixel(obj.center_ego[:2])
        heading = obj.center_ego[:2] + np.array([np.cos(obj.yaw_ego), np.sin(obj.yaw_ego)]) * 2.5
        cv2.arrowedLine(image, tuple(center), tuple(pixel(heading)), color, 2 if selected else 1, tipLength=.3)
        if (tracking and identifier is not None and obj.category not in {'barrier', 'traffic_cone'}
                and 0 < center[0] < width - 45 and 12 < center[1] < height - 15):
            cv2.putText(image, f'#{identifier}', tuple(center + [6, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                        .53 if selected else .35, color, 2 if selected else 1, cv2.LINE_AA)
    ego = pixel(np.array([[2.35, .975], [2.35, -.975], [-2.35, -.975], [-2.35, .975]]))
    cv2.fillPoly(image, [ego], (235, 238, 242))
    cv2.arrowedLine(image, tuple(pixel([0, 0])), tuple(pixel([5, 0])), (235, 238, 242), 2, tipLength=.2)
    cv2.putText(image, 'EGO', tuple(pixel([-3, 0]) + [-12, 14]), cv2.FONT_HERSHEY_SIMPLEX, .4, (210, 220, 232), 1)
    return image


def panel(content, title, subtitle):
    w = content.shape[1]
    output = np.full((content.shape[0] + 68, w, 3), PANEL, np.uint8)
    output[68:] = content
    return words(output, [((16, 8), title, 20, (235, 242, 248)),
                          ((16, 38), subtitle, 15, (151, 168, 187))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'output/showcase/asyncbev/scene.json')
    parser.add_argument('--dataroot', type=Path, default=ROOT / 'data/external/nuscenes')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/showcase')
    parser.add_argument('--score', type=float, default=.35)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--screenshot-frame', type=int, help='Zero-based cover keyframe; default is selected turn window end')
    parser.add_argument('--h264', action='store_true', help='Encode shareable H.264 using optional imageio-ffmpeg')
    args = parser.parse_args()
    if args.fps < 1:
        parser.error('--fps must be positive')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if args.h264:
        try:
            import imageio_ffmpeg
        except ImportError:
            parser.error('H.264 export requires: python -m pip install imageio-ffmpeg')
    payload = json.loads(args.predictions.read_text(encoding='utf-8'))
    if payload.get('box_center_convention') != 'gravity':
        parser.error('Use gravity-center predictions (the AsyncBEV scene exporter provides these).')
    source = NuScenesSource(args.dataroot, cameras_enabled=False, lidar_enabled=False,
                           radar_enabled=False, annotations_enabled=False)
    keys = make_keyframes(source, payload, args.score)
    if args.max_frames:
        keys = keys[:args.max_frames]
    if not keys:
        parser.error('No prediction frames')
    focus = choose_focus(keys)
    cover = args.screenshot_frame if args.screenshot_frame is not None else focus['screenshot_index']
    if not 0 <= cover < len(keys):
        parser.error('--screenshot-frame must be within the selected keyframes')
    source.cameras_enabled = source.lidar_enabled = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / 'perception_showcase.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create {video}')
    cloud_view = CloudView(682, 350)
    frames_written, audit = 0, []
    intervals = np.diff([k.timestamp for k in keys])
    if len(intervals) and np.any(intervals <= 0):
        raise ValueError('Prediction timestamps must increase')
    last_dt = float(np.median(intervals)) if len(intervals) else .5
    end_time = keys[-1].timestamp - keys[0].timestamp + last_dt
    source_hz = 1 / last_dt
    try:
        for i, key in enumerate(keys):
            frame = source.read_token(key.token)
            transforms = {name: calibrated_camera_transform(source, frame, name) for name in frame.cameras}
            packet = args.predictions.parent / f'packet_{i:03d}.npz'
            if packet.exists():
                with np.load(packet, allow_pickle=False) as p:
                    if abs(float(p['timestamp_s']) - key.timestamp) > .001 or abs(float(p['lidar_timestamp_s']) - key.timestamp) > .001:
                        raise ValueError('Showcase packets must be matched, nominal-time LiDAR inputs')
                    points = p['points']
                    # Short history gives useful density without rendering the full 10-sweep trail.
                    points = points[points[:, 4] <= .16, :3]
                    points = apply_transform(points, frame.calibrations['LIDAR_TOP'].sensor_to_ego)
            else:
                points = frame.lidar_ego[:, :3]
            valid = (np.linalg.norm(points[:, :2], axis=1) < 48) & (points[:, 2] > -2) & (points[:, 2] < 6)
            points = points[valid]
            canvas = np.full((1080, 1920, 3), BG, np.uint8)
            for row, channel in enumerate(['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']):
                canvas[118 + row * 282:118 + row * 282 + 272, 24:520] = camera_panel(
                    frame, channel, transforms[channel], key, focus, 496, 272)
            rgb = cloud_view.render(points, rgb_colors(points, frame, transforms))
            classes = cloud_view.render(points, associated_colors(points, key.predictions), key.predictions)
            canvas[118:536, 534:1216] = panel(rgb, '01 / CAMERA-COLORED POINT CLOUD', 'Calibrated RGB on real LiDAR | Not dense reconstruction')
            canvas[118:536, 1230:1912] = panel(classes, '02 / FUSION 3D DETECTIONS', 'Box-associated colors | Gray: unmatched | Not segmentation')
            canvas[550:968, 534:1216] = panel(bev_panel(key, points, focus, 682, 350),
                '03 / METRIC BEV DETECTIONS', 'Metres | Class-colored boxes | Arrows: object heading')
            canvas[550:968, 1230:1912] = panel(bev_panel(key, points, focus, 682, 350, True),
                '04 / TEMPORAL OBJECT TRACKING', 'World-frame tracking | History trails | Yellow: focus ID')
            t = key.timestamp - keys[0].timestamp
            timing = f'{key.inference_seconds:.2f}s' if key.inference_seconds is not None else 'n/a'
            subtitle = f"{payload.get('model', 'Fusion')}  /  {payload.get('scene_name', '')}  /  six cameras + LiDAR"
            focus_label = f"FOCUS #{focus['track_id']}"
            if focus['turning_candidate']:
                focus_label += f"  |  estimated heading change {focus['heading_change_deg']:.0f} deg"
            canvas = words(canvas, [
                ((24, 16), 'MULTI-SENSOR BEV PERCEPTION', 31, (238, 244, 250)),
                ((24, 65), subtitle, 18, (150, 172, 193)),
                ((1470, 18), 'OFFLINE REPLAY', 23, (95, 226, 194)),
                ((1470, 63), f't + {t:05.2f}s   |   frame {i + 1:02d}/{len(keys)}', 18, (186, 203, 220)),
                ((24, 985), f'{focus_label}   |   detections {len(key.predictions)}   |   model {timing}/frame', 18, (239, 207, 104)),
                ((24, 1022), f'Recorded sensors + pretrained model | Source keyframes: ~{source_hz:.1f} Hz (held) | Video FPS is not inference throughput', 18, (163, 181, 200)),
                ((24, 1050), 'Offline prototype | No vehicle deployment, parking-line or free-space segmentation | Box colors are for visualization.', 15, (130, 148, 168)),
            ])
            for j, (name, label) in enumerate([('car', 'VEHICLE'), ('pedestrian', 'PERSON'),
                                               ('bicycle', 'CYCLE'), ('barrier', 'BARRIER')]):
                x = 1280 + j * 150
                cv2.circle(canvas, (x, 1002), 5, PALETTE[name], -1, cv2.LINE_AA)
                cv2.putText(canvas, label, (x + 12, 1007), cv2.FONT_HERSHEY_SIMPLEX, .4, (184, 198, 214), 1, cv2.LINE_AA)
            if i == cover:
                cv2.imwrite(str(args.output_dir / 'perception_showcase.png'), canvas)
            if i == 0:
                cv2.imwrite(str(args.output_dir / 'perception_showcase_first.png'), canvas)
            next_t = keys[i + 1].timestamp - keys[0].timestamp if i + 1 < len(keys) else end_time
            target = round(next_t * args.fps)
            while frames_written < target:
                writer.write(canvas)
                frames_written += 1
            audit.append(dict(sample_token=key.token, timestamp_s=key.timestamp, point_count=len(points),
                              prediction_count=len(key.predictions), inference_seconds=key.inference_seconds))
            print(f'Rendered {i + 1}/{len(keys)}: {len(points)} points', flush=True)
    finally:
        writer.release()
    if args.h264:
        encoded = args.output_dir / 'perception_showcase.h264.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(video), '-an', '-c:v', 'libx264', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(encoded)], check=True)
        encoded.replace(video)
    report = dict(model=payload.get('model'), scene=payload.get('scene_name'), offline_replay=True,
                  display_language='en',
                  synthetic_inputs=False, annotation_boxes_used=False, point_semantics='predicted_box_membership_only',
                  rgb='calibrated camera projection onto real LiDAR points', focus=focus,
                  keyframes=len(keys), video_fps=args.fps, written_frames=frames_written,
                  screenshot_keyframe=cover,
                  video_codec='H.264' if args.h264 else 'MPEG-4 Part 2',
                  duration_s=frames_written / args.fps, playback='hold recorded keyframes; no interpolation',
                  score_threshold=args.score, frames=audit)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
