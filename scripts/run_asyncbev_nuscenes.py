"""Run label-free AsyncBEV on nuScenes; export predictions usable by existing renderers.

Run with the model environment's Python and PYTHONPATH=src. No annotations or flow
supplements are read. LiDAR history uses measured poses, not object motion labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parking_bev.perception.sensor_packet import CAMERA_ORDER, compensate_sweep, validate_packet
from parking_bev.perception.backend import ModelWorker


class NuScenesSensorTables:
    """Read only sensor/pose tables; never open sample_annotation.json."""
    def __init__(self, dataroot, version="v1.0-mini"):
        self.tables = {}
        for name in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor"):
            records = json.loads((dataroot / version / (name + ".json")).read_text())
            self.tables[name] = {record["token"]: record for record in records}
        self.scene = list(self.tables["scene"].values())
        for record in self.tables["sample_data"].values():
            if record["is_key_frame"]:
                calibration = self.get("calibrated_sensor", record["calibrated_sensor_token"])
                sensor = self.get("sensor", calibration["sensor_token"])
                self.get("sample", record["sample_token"]).setdefault("data", {})[sensor["channel"]] = record["token"]

    def get(self, table, token):
        return self.tables[table][token]


def pose(translation, rotation):
    from pyquaternion import Quaternion
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Quaternion(rotation).rotation_matrix
    result[:3, 3] = translation
    return result


def sensor_pose(nusc, sample_data):
    ego = nusc.get("ego_pose", sample_data["ego_pose_token"])
    calibration = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    return pose(ego["translation"], ego["rotation"]), pose(calibration["translation"], calibration["rotation"])


def build_packet(nusc, sample, dataroot, delay_frames=0, sweeps=10):
    target = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    source = target
    for _ in range(delay_frames):
        if not source["prev"]:
            raise ValueError("Insufficient LiDAR history for requested delay")
        source = nusc.get("sample_data", source["prev"])
    target_ego, extrinsic = sensor_pose(nusc, target)
    source_ego, source_extrinsic = sensor_pose(nusc, source)
    if not np.allclose(extrinsic, source_extrinsic):
        raise ValueError("Changing LiDAR extrinsics are unsupported by this checkpoint")
    packet = dict(timestamp_s=np.array(target["timestamp"] / 1e6),
                  lidar_timestamp_s=np.array(source["timestamp"] / 1e6), frame_id=np.array("lidar"),
                  ego_to_global=target_ego, source_ego_to_global=source_ego, lidar_to_ego=extrinsic)
    projections, intrinsics, camera_stamps = [], [], []
    for index, name in enumerate(CAMERA_ORDER):
        camera = nusc.get("sample_data", sample["data"][name])
        camera_ego, camera_extrinsic = sensor_pose(nusc, camera)
        calibration = nusc.get("calibrated_sensor", camera["calibrated_sensor_token"])
        image = cv2.imread(str(dataroot / camera["filename"]))
        if image is None:
            raise FileNotFoundError(camera["filename"])
        packet["image_" + str(index)] = image
        projections.append(np.linalg.inv(camera_ego @ camera_extrinsic) @ target_ego @ extrinsic)
        intrinsics.append(calibration["camera_intrinsic"])
        camera_stamps.append(camera["timestamp"] / 1e6)
    packet.update(lidar2cam=np.array(projections), intrinsics=np.array(intrinsics),
                  camera_timestamps_s=np.array(camera_stamps))
    scans = []
    scan = source
    for _ in range(sweeps + 1):
        points = np.fromfile(dataroot / scan["filename"], dtype=np.float32).reshape(-1, 5)
        scan_ego, scan_extrinsic = sensor_pose(nusc, scan)
        points = points[(np.abs(points[:, 0]) >= 1) | (np.abs(points[:, 1]) >= 1)]
        scans.append(compensate_sweep(points, scan_ego @ scan_extrinsic, source_ego @ source_extrinsic,
                                      (source["timestamp"] - scan["timestamp"]) / 1e6))
        if not scan["prev"]:
            break
        scan = nusc.get("sample_data", scan["prev"])
    packet["points"] = np.concatenate(scans)
    validate_packet(packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=ROOT / "data/external/nuscenes")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--delay-frames", type=int, default=0)
    parser.add_argument("--model-python", default=sys.executable)
    parser.add_argument("--endpoint", help="Reuse an already running loopback model service")
    parser.add_argument("--packet-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "output/asyncbev/scene.json")
    args = parser.parse_args()
    if args.delay_frames < 0 or args.start_frame < 0 or args.max_frames < 1:
        parser.error("delay/start frames must be nonnegative and max-frames must be positive")
    nusc = NuScenesSensorTables(args.dataroot)
    scene = nusc.scene[args.scene_index]
    token = scene["first_sample_token"]
    for _ in range(args.start_frame):
        token = nusc.get("sample", token)["next"]
        if not token:
            raise ValueError("start-frame exceeds scene length")
    payload = dict(model="CMT+AsyncBEV", scene_name=scene["name"], scene_index=args.scene_index,
                   score_threshold=0.2, box_center_convention="gravity", frames=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.model_python, str(ROOT / "integrations/asyncbev/worker.py")]
    worker = None
    try:
        if not args.packet_only:
            endpoint = (args.endpoint.rsplit(":", 1)[0], int(args.endpoint.rsplit(":", 1)[1])) if args.endpoint else None
            worker = ModelWorker(None if endpoint else command, endpoint=endpoint, startup_timeout_s=300, timeout_s=120,
                                 packet_directory=str(ROOT / "output/asyncbev/ipc"))
        for index in range(args.max_frames):
            sample = nusc.get("sample", token)
            packet = build_packet(nusc, sample, args.dataroot, args.delay_frames)
            np.savez(args.output.parent / ("packet_%03d.npz" % index), **packet)
            if worker:
                batch = worker.infer(packet)
                records = []
                for detection in batch.predictions:
                    obj = detection.object
                    records.append({"class": detection.class_name, "score": detection.score,
                                    "box_lidar": [*obj.center_ego.tolist(), *obj.size_wlh[[1, 0, 2]].tolist(),
                                                  obj.yaw_ego, *obj.velocity_ego.tolist()]})
                payload["frames"].append(dict(frame_index=index, sample_token=token,
                    timestamp_s=batch.timestamp_s, inference_seconds=batch.inference_seconds,
                    peak_gpu_memory_gib=batch.peak_gpu_memory_gib,
                    prediction_count=len(records), predictions=records))
                print("Frame %d: %d detections, %.3f seconds" % (index, len(records), batch.inference_seconds), flush=True)
            else:
                print("Exported label-free sensor packet", index, packet["points"].shape, flush=True)
            token = sample["next"]
            if not token:
                break
        if worker:
            timings = [frame["inference_seconds"] for frame in payload["frames"]]
            payload["summary"] = dict(frames=len(timings), mean_inference_seconds=float(np.mean(timings)),
                                      p95_inference_seconds=float(np.percentile(timings, 95)))
        args.output.write_text(json.dumps(payload, indent=2))
    finally:
        if worker:
            worker.close()


if __name__ == "__main__":
    main()
