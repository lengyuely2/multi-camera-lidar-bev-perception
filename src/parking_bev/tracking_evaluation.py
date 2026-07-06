from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .nuscenes_source import Object3D
from .predictions import detection_class
from .tracking import TrackSnapshot

TRACKING_CLASSES = {
    "bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"
}


class TrackingIdentityEvaluator:
    """Framewise spatial matching followed by sequence-level identity scoring."""

    def __init__(self, center_threshold_m: float = 2.0) -> None:
        self.center_threshold_m = center_threshold_m
        self.total_ground_truth = 0
        self.total_track_outputs = 0
        self.spatial_matches = 0
        self.id_switches = 0
        self.fragments = 0
        self.frames = 0
        self.associations: Counter[tuple[str, int]] = Counter()
        self.ground_truth_instances: set[str] = set()
        self.track_ids: set[int] = set()
        self.last_track_by_instance: dict[str, int] = {}
        self.previously_matched: set[str] = set()
        self.ever_matched: set[str] = set()

    def update(
        self,
        tracks: list[TrackSnapshot],
        ground_truth: list[Object3D],
        ego_to_global: np.ndarray,
    ) -> None:
        tracks = [item for item in tracks if item.class_name in TRACKING_CLASSES]
        expected = [
            item for item in ground_truth
            if item.instance_token is not None and detection_class(item.category) in TRACKING_CLASSES
        ]
        self.frames += 1
        self.total_ground_truth += len(expected)
        self.total_track_outputs += len(tracks)
        self.ground_truth_instances.update(item.instance_token for item in expected if item.instance_token)
        self.track_ids.update(item.track_id for item in tracks)
        rotation = ego_to_global[:3, :3]
        translation = ego_to_global[:3, 3]
        current_matched: set[str] = set()

        classes = set(track.class_name for track in tracks) | {
            class_name for item in expected
            if (class_name := detection_class(item.category)) is not None
        }
        for class_name in classes:
            class_tracks = [item for item in tracks if item.class_name == class_name]
            class_expected = [item for item in expected if detection_class(item.category) == class_name]
            if not class_tracks or not class_expected:
                continue
            track_xy = np.asarray([item.position_global[:2] for item in class_tracks])
            expected_xy = np.asarray([
                (rotation @ item.center_ego + translation)[:2] for item in class_expected
            ])
            distances = np.linalg.norm(track_xy[:, None, :] - expected_xy[None, :, :], axis=2)
            rows, columns = linear_sum_assignment(distances)
            for row, column in zip(rows, columns):
                if distances[row, column] > self.center_threshold_m:
                    continue
                track_id = class_tracks[row].track_id
                instance_token = class_expected[column].instance_token
                assert instance_token is not None
                self.spatial_matches += 1
                self.associations[(instance_token, track_id)] += 1
                current_matched.add(instance_token)
                previous_track = self.last_track_by_instance.get(instance_token)
                if previous_track is not None and previous_track != track_id:
                    self.id_switches += 1
                if instance_token in self.ever_matched and instance_token not in self.previously_matched:
                    self.fragments += 1
                self.last_track_by_instance[instance_token] = track_id
                self.ever_matched.add(instance_token)

        self.previously_matched = current_matched

    def finalize(self) -> dict:
        instance_ids = sorted(self.ground_truth_instances)
        track_ids = sorted(self.track_ids)
        identity_true_positives = 0
        if instance_ids and track_ids:
            counts = np.zeros((len(instance_ids), len(track_ids)), dtype=np.int32)
            instance_index = {token: index for index, token in enumerate(instance_ids)}
            track_index = {track_id: index for index, track_id in enumerate(track_ids)}
            for (instance_token, track_id), count in self.associations.items():
                counts[instance_index[instance_token], track_index[track_id]] = count
            rows, columns = linear_sum_assignment(-counts)
            identity_true_positives = int(counts[rows, columns].sum())

        identity_false_positives = self.total_track_outputs - identity_true_positives
        identity_false_negatives = self.total_ground_truth - identity_true_positives
        id_precision = (identity_true_positives / self.total_track_outputs
                        if self.total_track_outputs else 0.0)
        id_recall = (identity_true_positives / self.total_ground_truth
                     if self.total_ground_truth else 0.0)
        id_f1 = (2 * identity_true_positives
                 / (2 * identity_true_positives + identity_false_positives + identity_false_negatives)
                 if self.total_track_outputs + self.total_ground_truth else 0.0)
        return {
            "frames": self.frames,
            "unique_ground_truth_instances": len(instance_ids),
            "unique_track_ids": len(track_ids),
            "ground_truth_detections": self.total_ground_truth,
            "track_outputs": self.total_track_outputs,
            "spatial_matches": self.spatial_matches,
            "detection_coverage": self.spatial_matches / self.total_ground_truth
            if self.total_ground_truth else 0.0,
            "track_output_precision": self.spatial_matches / self.total_track_outputs
            if self.total_track_outputs else 0.0,
            "identity_true_positives": identity_true_positives,
            "identity_false_positives": identity_false_positives,
            "identity_false_negatives": identity_false_negatives,
            "id_precision": id_precision,
            "id_recall": id_recall,
            "id_f1": id_f1,
            "id_switches": self.id_switches,
            "fragments": self.fragments,
            "scope": "single-scene diagnostic; not official nuScenes AMOTA/AMOTP",
        }
