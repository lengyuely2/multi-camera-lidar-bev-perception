from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import numpy as np

from parking_bev.nuscenes_source import NuScenesSource, Object3D
from parking_bev.predictions import (
    bevfusion_predictions_from_records,
    detection_class,
    within_detection_range,
)
from parking_bev.tracking import TrackMeasurement, TimestampAwareTracker, prediction_to_global_measurement
from parking_bev.tracking_evaluation import TrackingIdentityEvaluator


@dataclass(frozen=True)
class PreparedFrame:
    timestamp_s: float
    ego_to_global: np.ndarray
    measurements: tuple[TrackMeasurement, ...]
    ground_truth: tuple[Object3D, ...]


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one number is required")
    return values


def _prepare_scene(path: Path, source: NuScenesSource) -> tuple[str, list[PreparedFrame]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = []
    for record in payload["frames"]:
        frame = source.read_token(record["sample_token"])
        predictions = [
            item for item in bevfusion_predictions_from_records(
                record["predictions"],
                frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                score_threshold=0.0,
            )
            if within_detection_range(item.class_name, item.object.center_ego)
        ]
        measurements = tuple(
            prediction_to_global_measurement(item, frame.ego_to_global)
            for item in predictions
        )
        ground_truth = tuple(
            obj for obj in frame.objects
            if (class_name := detection_class(obj.category)) is not None
            and within_detection_range(class_name, obj.center_ego)
        )
        frames.append(PreparedFrame(
            timestamp_s=frame.timestamp_us / 1_000_000.0,
            ego_to_global=frame.ego_to_global,
            measurements=measurements,
            ground_truth=ground_truth,
        ))
    return payload["scene_name"], frames


def _evaluate_scene(
    frames: list[PreparedFrame],
    score: float,
    association_distance: float,
    max_missed_seconds: float,
) -> dict:
    tracker = TimestampAwareTracker(
        history_size=10,
        association_distance_m=association_distance,
        max_missed_seconds=max_missed_seconds,
        appearance_weight=0.0,
    )
    evaluator = TrackingIdentityEvaluator(center_threshold_m=2.0)
    for frame in frames:
        measurements = [item for item in frame.measurements if item.score >= score]
        snapshots = tracker.update(frame.timestamp_s, measurements)
        confirmed = [item for item in snapshots if item.hits >= 2]
        evaluator.update(confirmed, list(frame.ground_truth), frame.ego_to_global)
    return evaluator.finalize()


def _aggregate(reports: list[dict]) -> dict:
    keys = (
        "frames", "unique_ground_truth_instances", "unique_track_ids",
        "ground_truth_detections", "track_outputs", "spatial_matches",
        "identity_true_positives", "identity_false_positives", "identity_false_negatives",
        "id_switches", "fragments",
    )
    result = {key: sum(report[key] for report in reports) for key in keys}
    gt = result["ground_truth_detections"]
    outputs = result["track_outputs"]
    idtp = result["identity_true_positives"]
    result["detection_coverage"] = result["spatial_matches"] / gt if gt else 0.0
    result["track_output_precision"] = result["spatial_matches"] / outputs if outputs else 0.0
    result["id_precision"] = idtp / outputs if outputs else 0.0
    result["id_recall"] = idtp / gt if gt else 0.0
    result["id_f1"] = 2 * idtp / (outputs + gt) if outputs + gt else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search the lightweight motion tracker using cached BEVFusion predictions")
    parser.add_argument("--prediction-dir", type=Path, default=Path("output/bevfusion_mini/scenes"))
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--scores", type=_float_list, default=_float_list("0.1,0.2,0.3"))
    parser.add_argument("--association-distances", type=_float_list, default=_float_list("1.5,2,3,4"))
    parser.add_argument("--max-missed-seconds", type=_float_list, default=_float_list("0.6,1.2,2.0"))
    parser.add_argument("--output", type=Path, default=Path("output/bevfusion_mini/tracker_tuning.json"))
    args = parser.parse_args()

    paths = sorted(args.prediction_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files matched {args.prediction_dir / args.pattern}")

    source = NuScenesSource(args.dataroot, cameras_enabled=False, radar_enabled=False)
    scenes = []
    for path in paths:
        name, frames = _prepare_scene(path, source)
        scenes.append((name, frames))
        print(f"Prepared {name}: {len(frames)} frames", flush=True)

    trials = []
    combinations = list(itertools.product(
        args.scores, args.association_distances, args.max_missed_seconds))
    for index, (score, association_distance, missed_seconds) in enumerate(combinations, 1):
        scene_reports = [
            {
                "scene_name": name,
                **_evaluate_scene(frames, score, association_distance, missed_seconds),
            }
            for name, frames in scenes
        ]
        aggregate = _aggregate(scene_reports)
        trial = {
            "score_threshold": score,
            "association_distance_m": association_distance,
            "max_missed_seconds": missed_seconds,
            "aggregate": aggregate,
            "scenes": scene_reports,
        }
        trials.append(trial)
        print(
            f"[{index:02d}/{len(combinations)}] score={score:.2f} distance={association_distance:.1f} "
            f"missed={missed_seconds:.1f}s IDF1={aggregate['id_f1']:.4f}",
            flush=True,
        )

    trials.sort(key=lambda item: item["aggregate"]["id_f1"], reverse=True)
    baseline = next((item for item in trials if
                     item["score_threshold"] == 0.2
                     and item["association_distance_m"] == 2.0
                     and item["max_missed_seconds"] == 1.2), None)
    report = {
        "scope": "motion-only tracker grid search; BEVFusion inference is not rerun",
        "scene_count": len(scenes),
        "trial_count": len(trials),
        "best": trials[0],
        "baseline": baseline,
        "idf1_improvement": (trials[0]["aggregate"]["id_f1"]
                              - baseline["aggregate"]["id_f1"] if baseline else None),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_parameters": {
            key: report["best"][key]
            for key in ("score_threshold", "association_distance_m", "max_missed_seconds")
        },
        "best_metrics": report["best"]["aggregate"],
        "baseline_id_f1": baseline["aggregate"]["id_f1"] if baseline else None,
        "idf1_improvement": report["idf1_improvement"],
    }, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
