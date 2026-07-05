from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import load_bevfusion_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BEVFusion predictions on the metric BEV")
    parser.add_argument("--predictions", type=Path, default=Path("output/bevfusion_predictions.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--output", type=Path, default=Path("output/bevfusion_predictions.jpg"))
    args = parser.parse_args()

    source = NuScenesSource(args.dataroot, radar_enabled=False, annotations_enabled=False)
    ok, frame = source.read()
    if not ok or frame is None:
        raise RuntimeError("No nuScenes sample found")
    predictions = load_bevfusion_predictions(
        args.predictions, frame.calibrations["LIDAR_TOP"].sensor_to_ego)
    objects = [item.object for item in predictions]

    predicted_frame = type(frame)(
        token=frame.token,
        timestamp_us=frame.timestamp_us,
        cameras=frame.cameras,
        lidar_ego=frame.lidar_ego,
        radar_ego={},
        objects=tuple(objects),
        calibrations=frame.calibrations,
        ego_to_global=frame.ego_to_global,
    )
    image = MetricBEVRenderer().render(predicted_frame)
    cv2.putText(image, "AI PREDICTIONS (not ground truth)", (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 240, 255), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"Wrote {len(objects)} predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()
