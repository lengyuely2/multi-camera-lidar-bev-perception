"""Parking-specific measured gaps, swept-envelope conflicts and stale-data states.

Object velocities are world velocities expressed in the current ego axes, not
relative velocities. x forward, y left, yaw positive counterclockwise. Only
detected objects are evaluated; an empty list is never a free-space certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import bicycle_path, polygon_clearance, rectangle

SECTORS = ("front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right")
SECTOR_TEXT = dict(zip(SECTORS, ("前方", "左前方", "左侧", "左后方", "后方", "右后方", "右侧", "右前方")))
CLASS_TEXT = {"pedestrian": "行人", "car": "车辆", "truck": "卡车", "bus": "客车", "barrier": "障碍物",
              "traffic_cone": "锥桶", "bicycle": "自行车", "motorcycle": "摩托车", "unknown": "障碍物"}
LEVELS = {"no_conflict_detected": 0, "watch": 1, "caution": 2, "stop": 3, "unknown": 4}


@dataclass(frozen=True)
class VehicleGeometry:
    length_m: float = 4.7
    width_m: float = 1.95
    wheelbase_m: float = 2.8
    rear_axle_x_m: float = -1.35
    profile_name: str = "example_passenger_car"

    def __post_init__(self):
        values = [self.length_m, self.width_m, self.wheelbase_m, self.rear_axle_x_m]
        if not np.isfinite(values).all() or min(values[:3]) <= 0:
            raise ValueError("Vehicle dimensions must be finite and positive")
        if not -self.length_m / 2 <= self.rear_axle_x_m <= self.length_m / 2 - self.wheelbase_m:
            raise ValueError("Both axles must lie inside the configured body footprint")


@dataclass(frozen=True)
class ParkingConfig:
    horizon_s: float = 3.0
    step_s: float = 0.1
    safety_margin_m: float = 0.25
    watch_gap_m: float = 0.8
    caution_gap_m: float = 0.45
    stop_gap_m: float = 0.15
    max_speed_mps: float = 1.4
    max_steering_rad: float = 0.65
    max_perception_age_s: float = 0.5
    max_vehicle_age_s: float = 0.3
    max_object_missed_s: float = 0.2
    min_score: float = 0.3
    min_track_hits: int = 2
    roi_m: float = 12.0
    max_prediction_displacement_m: float = 0.1

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0 or not self.min_score <= 1:
            raise ValueError("Parking configuration values must be finite and positive; min_score <= 1")
        if not self.stop_gap_m <= self.caution_gap_m <= self.watch_gap_m:
            raise ValueError("Parking gap thresholds must be ordered stop <= caution <= watch")
        if self.max_steering_rad >= np.pi / 2 or self.step_s > self.horizon_s:
            raise ValueError("Invalid steering limit or prediction step")


@dataclass(frozen=True)
class VehicleState:
    timestamp_s: float
    speed_mps: float
    steering_angle_rad: float


@dataclass(frozen=True)
class ParkingObject:
    track_id: str
    class_name: str
    center_xy: np.ndarray
    size_lw: np.ndarray
    yaw_rad: float
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity_known: bool = False
    score: float = 1.0
    hits: int = 1
    missed_s: float = 0.0
    center_z_m: float = 0.0

    def __post_init__(self):
        for name in ("center_xy", "size_lw", "velocity_xy"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(name + " must be a finite XY vector")
            object.__setattr__(self, name, value)
        if np.any(self.size_lw <= 0) or not np.isfinite([self.yaw_rad, self.score, self.missed_s, self.center_z_m]).all():
            raise ValueError("Invalid object geometry/metadata")
        if not 0 <= self.score <= 1 or self.missed_s < 0 or self.hits < 1 or not self.track_id:
            raise ValueError("Invalid object confidence, age or track identity")


@dataclass(frozen=True)
class ParkingAlert:
    track_id: str
    class_name: str
    sector: str
    level: str
    current_gap_m: float
    predicted_gap_m: float
    envelope_entry_s: float | None
    closing_speed_mps: float
    motion_confirmed: bool
    provisional: bool
    explanation: str


@dataclass(frozen=True)
class ParkingReport:
    timestamp_s: float
    status: str
    level: str
    advisory: str
    direction: str
    profile_name: str
    perception_age_s: float | None
    vehicle_state_age_s: float | None
    sector_gaps_m: dict[str, float | None]
    alerts: tuple[ParkingAlert, ...]
    times_s: np.ndarray
    body_polygons: np.ndarray
    envelope_polygons: np.ndarray
    ignored_objects: int = 0

    def to_dict(self):
        return dict(schema_version=1, timestamp_s=self.timestamp_s, status=self.status, level=self.level,
                    advisory=self.advisory, direction=self.direction, vehicle_profile=self.profile_name,
                    perception_age_s=self.perception_age_s, vehicle_state_age_s=self.vehicle_state_age_s,
                    sector_gaps_m=self.sector_gaps_m, alerts=[asdict(item) for item in self.alerts],
                    times_s=self.times_s.tolist(), body_polygons=self.body_polygons.tolist(),
                    envelope_polygons=self.envelope_polygons.tolist(), ignored_objects=self.ignored_objects,
                    coverage="detected_objects_only", free_space_verified=False)


class ParkingAssistant:
    def __init__(self, vehicle: VehicleGeometry | None = None, config: ParkingConfig | None = None):
        self.vehicle, self.config = vehicle or VehicleGeometry(), config or ParkingConfig()

    def unavailable(self, now_s, status, message, perception_age=None, vehicle_age=None):
        return ParkingReport(float(now_s), status, "unknown", message, "unknown", self.vehicle.profile_name,
            perception_age, vehicle_age, {key: None for key in SECTORS}, (),
            np.empty(0), np.empty((0, 4, 2)), np.empty((0, 4, 2)))

    def assess(self, objects: list[ParkingObject], vehicle_state: VehicleState | None,
               observation_timestamp_s: float | None, now_s: float) -> ParkingReport:
        if not np.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if observation_timestamp_s is None:
            return self.unavailable(now_s, "missing_perception", "等待周边感知数据，无法评估停车间距")
        age = now_s - observation_timestamp_s
        if not np.isfinite(age) or age < -0.02:
            return self.unavailable(now_s, "clock_mismatch", "感知时间戳异常，无法评估停车风险")
        age = max(0., age)
        if age > self.config.max_perception_age_s:
            return self.unavailable(now_s, "stale_perception", "周边感知已过期，当前间距与轨迹提示不可用", age)
        if vehicle_state is None:
            return self.unavailable(now_s, "missing_vehicle_state", "等待实测车速与前轮转角", age)
        vehicle_age = now_s - vehicle_state.timestamp_s
        if not np.isfinite([vehicle_age, vehicle_state.speed_mps, vehicle_state.steering_angle_rad]).all() or vehicle_age < -0.02:
            return self.unavailable(now_s, "invalid_vehicle_state", "车辆状态或时间戳异常", age)
        vehicle_age = max(0., vehicle_age)
        if vehicle_age > self.config.max_vehicle_age_s:
            return self.unavailable(now_s, "stale_vehicle_state", "实测车速或转向数据已过期", age, vehicle_age)
        speed, steering = vehicle_state.speed_mps, vehicle_state.steering_angle_rad
        if abs(speed) > self.config.max_speed_mps or abs(steering) > self.config.max_steering_rad:
            return self.unavailable(now_s, "out_of_scope", "车速或前轮转角超出当前停车模型范围", age, vehicle_age)
        eligible = [item for item in objects if item.score >= self.config.min_score and
                    item.missed_s <= self.config.max_object_missed_s and
                    np.linalg.norm(item.center_xy) <= self.config.roi_m + max(item.size_lw)]
        if objects and not eligible and any(item.score < self.config.min_score or item.missed_s > self.config.max_object_missed_s for item in objects):
            return self.unavailable(now_s, "insufficient_tracks", "目标信息过旧或置信度不足，无法评估当前周边间距", age, vehicle_age)
        maximum_relative_speed = abs(speed) + max((np.linalg.norm(item.velocity_xy) if item.velocity_known else 0 for item in eligible), default=0.)
        step = min(self.config.step_s, self.config.max_prediction_displacement_m / max(maximum_relative_speed, 1e-6))
        count = int(np.ceil(self.config.horizon_s / step))
        if count > 2000:
            return self.unavailable(now_s, "out_of_scope", "目标运动超出当前停车预测模型范围", age, vehicle_age)
        times = np.linspace(0., self.config.horizon_s, count + 1)
        centers, yaws = bicycle_path(times, speed, steering, self.vehicle.wheelbase_m, self.vehicle.rear_axle_x_m)
        bodies = rectangle(centers, self.vehicle.length_m, self.vehicle.width_m, yaws)
        envelopes = rectangle(centers, self.vehicle.length_m, self.vehicle.width_m, yaws, self.config.safety_margin_m)
        gaps = {key: None for key in SECTORS}
        alerts = []
        for item in eligible:
            # Positions were expressed in the current ego frame by the caller. Their
            # age is checked here but must not be applied a second time to position.
            velocity = item.velocity_xy if item.velocity_known else np.zeros(2)
            positions = item.center_xy + times[:, None] * velocity
            polygons = rectangle(positions, *item.size_lw, item.yaw_rad)
            clearance, overlap = polygon_clearance(bodies, polygons)
            _, in_envelope = polygon_clearance(envelopes, polygons)
            current_gap, future_gap = float(clearance[0]), float(clearance.min())
            sector = self.sector(item.center_xy)
            gaps[sector] = current_gap if gaps[sector] is None else min(gaps[sector], current_gap)
            indices = np.flatnonzero(in_envelope)
            entry = float(times[indices[0]]) if len(indices) else None
            provisional = item.hits < self.config.min_track_hits
            level = "no_conflict_detected"
            if current_gap <= self.config.stop_gap_m or overlap[0]:
                level = "stop"
            elif current_gap <= self.config.caution_gap_m:
                level = "caution"
            elif entry is not None:
                level = "stop" if entry <= 1.0 else "caution"
            elif current_gap <= self.config.watch_gap_m or future_gap <= self.config.watch_gap_m:
                level = "watch"
            if provisional and level == "stop" and current_gap > self.config.caution_gap_m:
                level = "caution"
            if level == "no_conflict_detected":
                continue
            closing = max(0., float((clearance[0] - clearance[min(1, len(times) - 1)]) / (times[1] - times[0])))
            location, name = SECTOR_TEXT[sector], CLASS_TEXT.get(item.class_name, "目标")
            if entry is not None and entry > 0:
                explanation = f"{location}{name}可能在 {entry:.1f} 秒内进入车身预留区域"
            elif level in ("stop", "caution"):
                explanation = f"{location}{name}距离车身约 {current_gap:.2f} 米"
            elif closing > 0.1:
                explanation = f"{location}{name}正在接近车身预测区域"
            else:
                explanation = f"请留意{location}{name}，车身间距约 {current_gap:.2f} 米"
            if not item.velocity_known:
                explanation += "；目标运动尚未确认"
            if provisional:
                explanation += "；新目标待连续观测确认"
            alerts.append(ParkingAlert(item.track_id, item.class_name, sector, level, current_gap, future_gap,
                entry, closing, item.velocity_known, provisional, explanation))
        alerts.sort(key=lambda item: (-LEVELS[item.level], item.envelope_entry_s if item.envelope_entry_s is not None else 99, item.current_gap_m))
        level = alerts[0].level if alerts else "no_conflict_detected"
        advisory = alerts[0].explanation if alerts else "暂未检测到目标框与预测区域冲突；未验证空地和未知障碍"
        if level == "stop":
            advisory = "建议停止当前泊车动作。" + advisory
        direction = "forward" if speed > 0.03 else "reverse" if speed < -0.03 else "stationary"
        return ParkingReport(float(now_s), "active", level, advisory, direction, self.vehicle.profile_name,
            age, vehicle_age, gaps, tuple(alerts), times, bodies, envelopes, len(objects) - len(eligible))

    def sector(self, center):
        x, y = center
        longitudinal = "front" if x > self.vehicle.length_m / 2 else "rear" if x < -self.vehicle.length_m / 2 else ""
        lateral = "left" if y > self.vehicle.width_m / 2 else "right" if y < -self.vehicle.width_m / 2 else ""
        if longitudinal and lateral:
            return longitudinal + "_" + lateral
        if longitudinal or lateral:
            return longitudinal or lateral
        return "left" if y >= 0 else "right"
