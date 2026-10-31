"""Render configurable parking examples, with explicitly synthetic sensor/vehicle states."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parking_bev.parking import ParkingAssistant, ParkingConfig, ParkingObject, VehicleGeometry, VehicleState
from parking_bev.parking.geometry import bicycle_path
from parking_bev.runtime.config import load_config
from parking_bev.visualization.parking import ParkingRenderer


@dataclass(frozen=True)
class Scenario:
    title: str
    speed: float
    steering: float
    obstacles: tuple
    perception_age_s: float = 0.05


SCENARIOS = {
    "reverse_pedestrian": Scenario("合成示例 · 倒车时行人从左后方接近", -.35, 0.,
        (("ped-1", "pedestrian", [-3.8, 2.2], [.65, .65], [0., -.55], 0.),
         ("parked-1", "car", [0., -3.2], [4.4, 1.85], [0., 0.], 0.))),
    "turning_corner": Scenario("合成示例 · 倒车转向时前右角外摆", -.3, .55,
        (("cone-1", "traffic_cone", [1.8, -1.9], [.35, .35], [0., 0.], 0.),)),
    "side_clearance": Scenario("合成示例 · 车侧近距离障碍", 0., 0.,
        (("parked-1", "car", [0., 2.2], [4.2, 1.8], [0., 0.], 0.),
         ("cone-1", "traffic_cone", [-3.1, -.3], [.35, .35], [0., 0.], 0.))),
    "stale_perception": Scenario("合成示例 · 感知过期，不再显示正常提示", -.3, .55,
        (("cone-1", "traffic_cone", [1.8, -1.9], [.35, .35], [0., 0.], 0.),), 1.6),
}


def scene_at(scenario, elapsed_s, vehicle):
    centers, yaws = bicycle_path(np.array([elapsed_s]), scenario.speed, scenario.steering,
                                vehicle.wheelbase_m, vehicle.rear_axle_x_m)
    yaw = float(yaws[0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    objects = []
    for identity, name, initial, size, velocity, object_yaw in scenario.obstacles:
        velocity = np.asarray(velocity)
        center = rotation.T @ (np.asarray(initial) + velocity * elapsed_s - centers[0])
        objects.append(ParkingObject(identity, name, center, np.asarray(size), object_yaw - yaw,
                                     rotation.T @ velocity, True, .95, 4))
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/parking_assistance.yaml")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/parking_assistance")
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("seconds and fps must be positive")
    config = load_config(args.config)
    vehicle = VehicleGeometry(**config["vehicle"])
    assistant = ParkingAssistant(vehicle, ParkingConfig(**config["parking"]))
    renderer = ParkingRenderer(vehicle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    video_path = args.output_dir / "parking_assistance.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 860))
    if not writer.isOpened():
        raise RuntimeError("Cannot create parking demo video")
    summaries = {}
    try:
        for name, scenario in selected.items():
            reports, best, best_level = [], None, -1
            priorities = {"unknown": 4, "stop": 3, "caution": 2, "watch": 1, "no_conflict_detected": 0}
            for frame in range(max(1, round(args.seconds * args.fps))):
                elapsed = frame / args.fps
                now = 100. + elapsed
                objects = scene_at(scenario, elapsed, vehicle)
                report = assistant.assess(objects, VehicleState(now, scenario.speed, scenario.steering),
                                          now - scenario.perception_age_s, now)
                image = renderer.render(report, objects, scenario.speed, scenario.steering, scenario.title)
                writer.write(image)
                reports.append(report.to_dict())
                if priorities[report.level] > best_level:
                    best, best_level = image, priorities[report.level]
            cv2.imwrite(str(args.output_dir / (name + ".png")), best)
            summaries[name] = dict(title=scenario.title, synthetic_inputs=True, reports=reports)
            print(name, {level: sum(item["level"] == level for item in reports) for level in priorities}, flush=True)
    finally:
        writer.release()
    (args.output_dir / "report.json").write_text(json.dumps(dict(vehicle=config["vehicle"],
        parking=config["parking"], synthetic_inputs=True, scenarios=summaries), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Video:", video_path)


if __name__ == "__main__":
    main()
