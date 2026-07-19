from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..visualization.semantic_3d import SemanticTrack


RISK_LEVELS = ("clear", "watch", "caution", "critical")


@dataclass(frozen=True)
class AgentFuture:
    track_id: int
    class_name: str
    score: float
    times_s: np.ndarray
    centers_ego: np.ndarray
    relative_velocity_ego: np.ndarray
    display_alpha: float
    risk_score: float
    risk_level: str
    time_to_risk_s: float | None
    min_clearance_m: float
    min_longitudinal_gap_m: float
    min_lateral_gap_m: float
    explanation: str


@dataclass(frozen=True)
class WorldModelOutput:
    horizon_s: float
    step_s: float
    ego_velocity_ego: np.ndarray
    predictions: list[AgentFuture]
    risk_level: str
    risk_score: float
    advisory: str

    @property
    def highest_risk(self) -> AgentFuture | None:
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda item: item.risk_score)


class WorldModelLite:
    """Constant-velocity short-horizon world model for driving risk triage.

    The module predicts object motion in the current ego frame. Track velocities
    are treated as world velocities expressed in ego axes, so callers should pass
    the ego vehicle velocity when available to get relative future positions.
    """

    def __init__(
        self,
        horizon_s: float = 3.0,
        step_s: float = 0.5,
        ego_width_m: float = 1.95,
        ego_length_m: float = 4.7,
        lateral_margin_m: float = 0.45,
        front_buffer_m: float = 1.5,
        rear_buffer_m: float = 0.8,
        watch_distance_m: float = 18.0,
        caution_distance_m: float = 10.0,
        risk_activation_alpha: float = 0.75,
    ) -> None:
        if horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if step_s <= 0:
            raise ValueError("step_s must be positive")
        self.horizon_s = float(horizon_s)
        self.step_s = float(step_s)
        self.ego_width_m = float(ego_width_m)
        self.ego_length_m = float(ego_length_m)
        self.lateral_margin_m = float(lateral_margin_m)
        self.front_buffer_m = float(front_buffer_m)
        self.rear_buffer_m = float(rear_buffer_m)
        self.watch_distance_m = float(watch_distance_m)
        self.caution_distance_m = float(caution_distance_m)
        self.risk_activation_alpha = float(risk_activation_alpha)

    def assess(
        self,
        tracks: list[SemanticTrack],
        ego_velocity_ego: np.ndarray | None = None,
    ) -> WorldModelOutput:
        ego_velocity = (
            np.zeros(2, dtype=np.float32)
            if ego_velocity_ego is None
            else np.asarray(ego_velocity_ego, dtype=np.float32)[:2]
        )
        predictions = [
            self._predict_track(track, ego_velocity)
            for track in tracks
        ]
        predictions.sort(
            key=lambda item: (
                -item.risk_score,
                item.time_to_risk_s if item.time_to_risk_s is not None else self.horizon_s + 1.0,
                item.track_id,
            )
        )
        highest = predictions[0] if predictions else None
        risk_level = highest.risk_level if highest is not None else "clear"
        risk_score = highest.risk_score if highest is not None else 0.0
        advisory = self._advisory(highest)
        return WorldModelOutput(
            horizon_s=self.horizon_s,
            step_s=self.step_s,
            ego_velocity_ego=ego_velocity.astype(np.float32),
            predictions=predictions,
            risk_level=risk_level,
            risk_score=risk_score,
            advisory=advisory,
        )

    def _predict_track(self, track: SemanticTrack, ego_velocity_ego: np.ndarray) -> AgentFuture:
        times = self._time_grid()
        relative_velocity = np.asarray(track.velocity_ego, dtype=np.float32)[:2] - ego_velocity_ego
        centers = np.repeat(np.asarray(track.center_ego, dtype=np.float32).reshape(1, 3), len(times), axis=0)
        centers[:, :2] += times[:, None] * relative_velocity.reshape(1, 2)

        width = max(float(track.size_wlh[0]), 0.35)
        length = max(float(track.size_wlh[1]), 0.45)
        ego_half_width = self.ego_width_m * 0.5
        ego_half_length = self.ego_length_m * 0.5
        inflated_half_width = ego_half_width + width * 0.5 + self.lateral_margin_m
        inflated_front_x = ego_half_length + length * 0.5 + self.front_buffer_m
        inflated_rear_x = -(ego_half_length + length * 0.5 + self.rear_buffer_m)

        x = centers[:, 0]
        y = centers[:, 1]
        lateral_gap = np.abs(y) - inflated_half_width
        longitudinal_gap = x - inflated_front_x
        behind_gap = inflated_rear_x - x
        dx_outside = np.maximum.reduce((
            longitudinal_gap,
            behind_gap,
            np.zeros_like(x),
        ))
        dy_outside = np.maximum(lateral_gap, 0.0)
        clearance = np.sqrt(dx_outside**2 + dy_outside**2)

        in_corridor = lateral_gap <= 0.0
        in_envelope = in_corridor & (x >= inflated_rear_x) & (x <= inflated_front_x)
        ahead_in_corridor = in_corridor & (x > inflated_front_x)

        min_index = int(np.argmin(clearance))
        min_clearance = float(clearance[min_index])
        min_longitudinal_gap = float(longitudinal_gap[min_index])
        min_lateral_gap = float(lateral_gap[min_index])

        risky_indices = np.flatnonzero(in_envelope)
        risk_time = float(times[int(risky_indices[0])]) if len(risky_indices) else None
        risk_level, risk_score = self._risk_from_geometry(
            risk_time=risk_time,
            centers=centers,
            in_corridor=in_corridor,
            ahead_in_corridor=ahead_in_corridor,
            clearance=clearance,
            relative_velocity=relative_velocity,
            inflated_front_x=inflated_front_x,
        )
        display_alpha = float(np.clip(getattr(track, "display_alpha", 1.0), 0.0, 1.0))
        risk_level, risk_score = self._stabilized_risk(risk_level, risk_score, risk_time, display_alpha)
        explanation = self._explanation(track, risk_level, risk_time, centers, in_corridor)
        return AgentFuture(
            track_id=track.track_id,
            class_name=track.class_name,
            score=track.score,
            times_s=times.astype(np.float32),
            centers_ego=centers.astype(np.float32),
            relative_velocity_ego=relative_velocity.astype(np.float32),
            display_alpha=display_alpha,
            risk_score=risk_score,
            risk_level=risk_level,
            time_to_risk_s=risk_time,
            min_clearance_m=min_clearance,
            min_longitudinal_gap_m=min_longitudinal_gap,
            min_lateral_gap_m=min_lateral_gap,
            explanation=explanation,
        )

    def _time_grid(self) -> np.ndarray:
        steps = int(np.floor(self.horizon_s / self.step_s))
        times = np.arange(0, steps + 1, dtype=np.float32) * self.step_s
        if float(times[-1]) < self.horizon_s:
            times = np.append(times, self.horizon_s).astype(np.float32)
        return times

    def _risk_from_geometry(
        self,
        risk_time: float | None,
        centers: np.ndarray,
        in_corridor: np.ndarray,
        ahead_in_corridor: np.ndarray,
        clearance: np.ndarray,
        relative_velocity: np.ndarray,
        inflated_front_x: float,
    ) -> tuple[str, float]:
        if risk_time is not None:
            urgency = 1.0 - min(risk_time, self.horizon_s) / max(self.horizon_s, 1e-6)
            score = float(np.clip(0.72 + 0.25 * urgency, 0.0, 1.0))
            level = "critical" if risk_time <= 1.5 else "caution"
            return level, score

        closest_index = int(np.argmin(clearance))
        closest = centers[closest_index]
        closest_x = float(closest[0])
        closest_distance = max(closest_x - inflated_front_x, 0.0)
        closing_fast = bool(relative_velocity[0] < -1.0)
        if bool(ahead_in_corridor[closest_index]) and closest_distance <= self.caution_distance_m:
            score = 0.58 + 0.16 * (1.0 - closest_distance / max(self.caution_distance_m, 1e-6))
            if closing_fast:
                score += 0.08
            return "caution", float(np.clip(score, 0.0, 0.88))
        if bool(np.any(in_corridor & (centers[:, 0] > 0.0))) and closest_distance <= self.watch_distance_m:
            score = 0.32 + 0.20 * (1.0 - closest_distance / max(self.watch_distance_m, 1e-6))
            if closing_fast:
                score += 0.08
            return "watch", float(np.clip(score, 0.0, 0.62))
        return "clear", 0.0

    def _stabilized_risk(
        self,
        risk_level: str,
        risk_score: float,
        risk_time: float | None,
        display_alpha: float,
    ) -> tuple[str, float]:
        if risk_level == "clear" or display_alpha >= self.risk_activation_alpha:
            return risk_level, risk_score
        score = float(risk_score * display_alpha / max(self.risk_activation_alpha, 1e-6))
        if score >= 0.64:
            return ("critical" if risk_time is not None and risk_time <= 1.5 else "caution"), score
        if score >= 0.18:
            return "watch", score
        return "clear", score

    def _explanation(
        self,
        track: SemanticTrack,
        risk_level: str,
        risk_time: float | None,
        centers: np.ndarray,
        in_corridor: np.ndarray,
    ) -> str:
        nearest_index = int(np.argmin(np.linalg.norm(centers[:, :2], axis=1)))
        nearest = centers[nearest_index]
        nearest_distance = float(np.linalg.norm(nearest[:2]))
        label = f"{track.class_name} ID{track.track_id}"
        if risk_level == "critical":
            return (
                f"{label} is predicted to enter the ego safety envelope "
                f"in {risk_time:.1f}s"
            )
        if risk_level == "caution" and risk_time is not None:
            return (
                f"{label} may conflict with the ego path within "
                f"{risk_time:.1f}s"
            )
        if risk_level == "caution":
            return (
                f"{label} remains in the ego lane near {nearest_distance:.1f}m "
                f"within {self.horizon_s:.1f}s"
            )
        if risk_level == "watch":
            crossing = bool(np.any(in_corridor & (centers[:, 0] > 0.0)))
            if crossing:
                return (
                    f"{label} is projected into the ego corridor "
                    f"within {self.horizon_s:.1f}s"
                )
            return f"{label} is near the planned corridor"
        return f"{label} has no path conflict within {self.horizon_s:.1f}s"

    @staticmethod
    def _advisory(highest: AgentFuture | None) -> str:
        if highest is None or highest.risk_level == "clear":
            return "Path clear over the prediction horizon"
        if highest.risk_level == "critical":
            return f"Brake or yield: {highest.explanation}"
        if highest.risk_level == "caution":
            return f"Prepare to slow: {highest.explanation}"
        return f"Monitor: {highest.explanation}"


def world_model_output_to_dict(output: WorldModelOutput) -> dict:
    return {
        "horizon_s": output.horizon_s,
        "step_s": output.step_s,
        "ego_velocity_ego_mps": output.ego_velocity_ego.tolist(),
        "risk_level": output.risk_level,
        "risk_score": output.risk_score,
        "advisory": output.advisory,
        "predictions": [
            {
                "track_id": item.track_id,
                "class_name": item.class_name,
                "score": item.score,
                "risk_level": item.risk_level,
                "risk_score": item.risk_score,
                "display_alpha": item.display_alpha,
                "time_to_risk_s": item.time_to_risk_s,
                "min_clearance_m": item.min_clearance_m,
                "min_longitudinal_gap_m": item.min_longitudinal_gap_m,
                "min_lateral_gap_m": item.min_lateral_gap_m,
                "relative_velocity_ego_mps": item.relative_velocity_ego.tolist(),
                "future_centers_ego_m": item.centers_ego.tolist(),
                "explanation": item.explanation,
            }
            for item in output.predictions
        ],
    }
