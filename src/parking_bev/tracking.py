from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .predictions import Prediction3D


@dataclass(frozen=True)
class TrackMeasurement:
    class_name: str
    score: float
    position_global: np.ndarray
    velocity_global: np.ndarray
    size_wlh: np.ndarray
    yaw_global: float


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    class_name: str
    score: float
    position_global: np.ndarray
    velocity_global: np.ndarray
    size_wlh: np.ndarray
    yaw_global: float
    hits: int
    missed: int
    history_global: np.ndarray


@dataclass
class _Track:
    track_id: int
    class_name: str
    score: float
    state: np.ndarray
    covariance: np.ndarray
    size_wlh: np.ndarray
    yaw_global: float
    last_timestamp_s: float
    last_measurement_timestamp_s: float
    hits: int = 1
    missed: int = 0
    history: list[np.ndarray] = field(default_factory=list)


class TimestampAwareTracker:
    """Class-aware constant-velocity Kalman tracker using real timestamps."""

    def __init__(
        self,
        association_distance_m: float = 3.0,
        max_missed_seconds: float = 1.2,
        acceleration_noise: float = 3.0,
        measurement_noise_m: float = 0.6,
        history_size: int = 20,
    ) -> None:
        self.association_distance_m = association_distance_m
        self.max_missed_seconds = max_missed_seconds
        self.acceleration_noise = acceleration_noise
        self.measurement_noise_m = measurement_noise_m
        self.history_size = history_size
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, timestamp_s: float, measurements: list[TrackMeasurement]) -> list[TrackSnapshot]:
        for track in self._tracks:
            self._predict(track, timestamp_s)

        matched_tracks: set[int] = set()
        matched_measurements: set[int] = set()
        classes = set(track.class_name for track in self._tracks) | set(item.class_name for item in measurements)
        for class_name in classes:
            track_indices = [i for i, track in enumerate(self._tracks) if track.class_name == class_name]
            measurement_indices = [i for i, item in enumerate(measurements) if item.class_name == class_name]
            if not track_indices or not measurement_indices:
                continue
            distances = np.asarray([
                [np.linalg.norm(self._tracks[track_index].state[:2]
                                - measurements[measurement_index].position_global[:2])
                 for measurement_index in measurement_indices]
                for track_index in track_indices
            ])
            rows, columns = linear_sum_assignment(distances)
            for row, column in zip(rows, columns):
                if distances[row, column] > self.association_distance_m:
                    continue
                track_index = track_indices[row]
                measurement_index = measurement_indices[column]
                self._correct(self._tracks[track_index], measurements[measurement_index])
                matched_tracks.add(track_index)
                matched_measurements.add(measurement_index)

        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.missed += 1
                self._append_history(track)

        for index, measurement in enumerate(measurements):
            if index not in matched_measurements:
                self._tracks.append(self._new_track(measurement, timestamp_s))

        self._tracks = [
            track for track in self._tracks
            if timestamp_s - track.last_measurement_timestamp_s <= self.max_missed_seconds
        ]
        return [self._snapshot(track) for track in self._tracks]

    def _predict(self, track: _Track, timestamp_s: float) -> None:
        dt = max(0.0, timestamp_s - track.last_timestamp_s)
        transition = np.asarray([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        process_noise = self.acceleration_noise**2 * np.asarray([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ])
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process_noise
        track.last_timestamp_s = timestamp_s

    def _correct(self, track: _Track, measurement: TrackMeasurement) -> None:
        observation = measurement.position_global[:2].astype(np.float64)
        observation_matrix = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        measurement_noise = np.eye(2) * self.measurement_noise_m**2
        innovation = observation - observation_matrix @ track.state
        innovation_covariance = observation_matrix @ track.covariance @ observation_matrix.T + measurement_noise
        kalman_gain = track.covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + kalman_gain @ innovation
        track.covariance = (np.eye(4) - kalman_gain @ observation_matrix) @ track.covariance
        track.score = measurement.score
        track.size_wlh = measurement.size_wlh.copy()
        track.yaw_global = measurement.yaw_global
        track.hits += 1
        track.missed = 0
        track.last_measurement_timestamp_s = track.last_timestamp_s
        self._append_history(track)

    def _new_track(self, measurement: TrackMeasurement, timestamp_s: float) -> _Track:
        state = np.r_[measurement.position_global[:2], measurement.velocity_global[:2]].astype(np.float64)
        track = _Track(
            track_id=self._next_id,
            class_name=measurement.class_name,
            score=measurement.score,
            state=state,
            covariance=np.diag([1.0, 1.0, 9.0, 9.0]),
            size_wlh=measurement.size_wlh.copy(),
            yaw_global=measurement.yaw_global,
            last_timestamp_s=timestamp_s,
            last_measurement_timestamp_s=timestamp_s,
            history=[state[:2].copy()],
        )
        self._next_id += 1
        return track

    def _append_history(self, track: _Track) -> None:
        track.history.append(track.state[:2].copy())
        del track.history[:-self.history_size]

    @staticmethod
    def _snapshot(track: _Track) -> TrackSnapshot:
        return TrackSnapshot(
            track.track_id, track.class_name, track.score,
            track.state[:2].copy(), track.state[2:].copy(), track.size_wlh.copy(),
            track.yaw_global, track.hits, track.missed, np.asarray(track.history).copy(),
        )


def prediction_to_global_measurement(
    prediction: Prediction3D,
    ego_to_global: np.ndarray,
) -> TrackMeasurement:
    rotation = ego_to_global[:3, :3]
    translation = ego_to_global[:3, 3]
    center_global = rotation @ prediction.object.center_ego + translation
    velocity_global = (rotation @ np.r_[prediction.object.velocity_ego, 0.0])[:2]
    heading_global = rotation @ np.asarray([
        np.cos(prediction.object.yaw_ego), np.sin(prediction.object.yaw_ego), 0.0
    ])
    return TrackMeasurement(
        prediction.class_name,
        prediction.score,
        center_global,
        velocity_global,
        prediction.object.size_wlh,
        float(np.arctan2(heading_global[1], heading_global[0])),
    )
