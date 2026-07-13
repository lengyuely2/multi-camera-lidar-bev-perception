import numpy as np

from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import CAMERA_CHANNELS, Object3D, RADAR_CHANNELS
from parking_bev.evaluation import evaluate_detections
from parking_bev.predictions import Prediction3D, detection_class
from parking_bev.voxelize import HardVoxelizer
from parking_bev.tracking import TimestampAwareTracker, TrackMeasurement
from parking_bev.tracking_evaluation import TrackingIdentityEvaluator
from parking_bev.appearance import extract_object_appearance
from parking_bev.radar_fusion import estimate_radar_velocity
from parking_bev.semantic_3d import (
    Semantic3DRenderer,
    SemanticTrack,
    interpolate_semantic_tracks,
    snapshot_to_ego,
)


def test_nuscenes_sensor_layout():
    assert len(CAMERA_CHANNELS) == 6
    assert len(RADAR_CHANNELS) == 5
    assert "CAM_FRONT" in CAMERA_CHANNELS
    assert "RADAR_FRONT" in RADAR_CHANNELS
    assert np.dtype(np.float32).itemsize == 4


def test_metric_bev_coordinate_convention():
    renderer = MetricBEVRenderer(width=120, height=120)
    pixels = renderer.ego_to_pixel(np.asarray([[0, 0], [10, 0], [0, 10]], np.float32))
    np.testing.assert_allclose(pixels, [[60, 60], [60, 50], [50, 60]])


def test_object_box_uses_length_along_forward_axis():
    obj = Object3D(
        token="test",
        category="vehicle.car",
        center_ego=np.asarray([10, 2, 0], np.float32),
        size_wlh=np.asarray([2, 4, 1.5], np.float32),
        yaw_ego=0.0,
        velocity_ego=np.zeros(2, np.float32),
    )
    corners = MetricBEVRenderer.object_corners_xy(obj)
    np.testing.assert_allclose(corners, [[12, 3], [12, 1], [8, 1], [8, 3]])


def test_velocity_visualization_is_bounded():
    displacement = MetricBEVRenderer._limited_velocity(np.asarray([100, 0], np.float32))
    np.testing.assert_allclose(displacement, [[6, 0]])


def test_hard_voxelizer_filters_and_caps_points():
    points = np.asarray([
        [0.1, 0.1, 0.1, 1, 0],
        [0.2, 0.1, 0.1, 2, 0],
        [1.1, 0.1, 0.1, 3, 0],
        [9.0, 0.1, 0.1, 4, 0],
    ], np.float32)
    voxelizer = HardVoxelizer(
        voxel_size=(1, 1, 1),
        point_cloud_range=(0, 0, 0, 2, 2, 2),
        max_points_per_voxel=1,
        max_voxels=10,
    )
    result = voxelizer(points)
    assert result.input_points == 4
    assert result.retained_points == 2
    assert result.voxels.shape == (2, 1, 5)
    np.testing.assert_array_equal(result.coordinates_zyx, [[0, 0, 0], [0, 0, 1]])


def test_nuscenes_categories_map_to_detection_classes():
    assert detection_class("human.pedestrian.adult") == "pedestrian"
    assert detection_class("vehicle.bus.rigid") == "bus"
    assert detection_class("movable_object.pushable_pullable") is None


def test_detection_evaluation_matches_by_class_and_distance():
    expected = Object3D("gt", "vehicle.car", np.asarray([5, 0, 0], np.float32),
                        np.asarray([2, 4, 1.5], np.float32), 0.0, np.zeros(2, np.float32))
    predicted_object = Object3D("pred", "vehicle.car", np.asarray([5.5, 0, 0], np.float32),
                                np.asarray([2, 4, 1.5], np.float32), 0.1, np.zeros(2, np.float32))
    prediction = Prediction3D(predicted_object, "car", 0.9)
    overall, _ = evaluate_detections([prediction], [expected], center_threshold_m=2.0)
    assert overall.true_positives == 1
    assert overall.false_positives == 0
    assert overall.false_negatives == 0
    assert overall.mean_center_error_m == 0.5


def test_timestamp_tracker_keeps_id_and_uses_real_dt():
    tracker = TimestampAwareTracker(association_distance_m=3.0)
    first = TrackMeasurement("car", 0.9, np.asarray([0, 0, 0], np.float32),
                             np.asarray([1, 0], np.float32), np.asarray([2, 4, 1.5], np.float32), 0.0)
    first_tracks = tracker.update(10.0, [first])
    second = TrackMeasurement("car", 0.9, np.asarray([2, 0, 0], np.float32),
                              np.asarray([1, 0], np.float32), np.asarray([2, 4, 1.5], np.float32), 0.0)
    second_tracks = tracker.update(12.0, [second])
    assert first_tracks[0].track_id == second_tracks[0].track_id
    assert second_tracks[0].hits == 2
    assert second_tracks[0].position_global[0] > 1.5


def test_timestamp_tracker_expires_tracks_by_elapsed_time():
    tracker = TimestampAwareTracker(max_missed_seconds=1.0)
    measurement = TrackMeasurement("car", 0.9, np.asarray([0, 0, 0], np.float32),
                                   np.zeros(2, np.float32), np.asarray([2, 4, 1.5], np.float32), 0.0)
    assert len(tracker.update(1.0, [measurement])) == 1
    assert len(tracker.update(2.1, [])) == 0


def test_semantic_3d_projection_places_forward_center_on_screen():
    renderer = Semantic3DRenderer(width=800, height=450)
    pixels, depth, visible = renderer.project(np.asarray([[20, 0, 0], [20, 4, 0]], np.float32))
    assert visible.all()
    assert depth[0] > 0
    assert abs(pixels[0, 0] - 400) < 1
    assert pixels[1, 0] < pixels[0, 0]


def test_semantic_3d_driving_mode_is_lighter_than_engineering_mode():
    renderer = Semantic3DRenderer(width=640, height=360)
    driving = renderer.render([], 0.0, 0, radar_enabled=True, engineering_mode=False)
    engineering = renderer.render([], 0.0, 0, radar_enabled=True, engineering_mode=True)
    assert driving.shape == (360, 640, 3)
    assert float(driving.mean()) > float(engineering.mean()) + 100.0


def test_semantic_tracks_interpolate_positions_and_wrapped_yaw():
    common = {
        "track_id": 3,
        "class_name": "car",
        "score": 0.9,
        "size_wlh": np.asarray([2.0, 4.0, 1.5], np.float32),
        "velocity_ego": np.asarray([2.0, 0.0], np.float32),
        "history_ego": np.empty((0, 3), np.float32),
        "missed": 0,
    }
    before = SemanticTrack(
        center_ego=np.asarray([0.0, 0.0, 0.75], np.float32),
        yaw_ego=np.deg2rad(179.0),
        **common,
    )
    after = SemanticTrack(
        center_ego=np.asarray([10.0, 2.0, 0.75], np.float32),
        yaw_ego=np.deg2rad(-179.0),
        **common,
    )
    middle = interpolate_semantic_tracks([before], [after], 0.5)[0]
    np.testing.assert_allclose(middle.center_ego, [5.0, 1.0, 0.75])
    assert abs(abs(np.rad2deg(middle.yaw_ego)) - 180.0) < 0.01


def test_real_camera_montage_contains_all_six_views():
    from scripts.render_semantic_drive import _camera_montage

    names = (
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    )
    cameras = {
        name: np.full((90, 160, 3), index * 30, dtype=np.uint8)
        for index, name in enumerate(names, 1)
    }
    montage = _camera_montage(cameras, 640, 720)
    assert montage.shape == (720, 640, 3)
    assert montage[180, 320, 0] == 30
    assert montage[450, 160, 0] == 60
    assert montage[450, 480, 0] == 90
    assert montage[630, 100, 0] == 120
    assert montage[630, 320, 0] == 150
    assert montage[630, 540, 0] == 180


def test_tracker_snapshot_transforms_to_semantic_ego_track():
    from parking_bev.tracking import TrackSnapshot

    snapshot = TrackSnapshot(
        7, "car", 0.9, np.asarray([10.0, 2.0]), np.asarray([3.0, 0.0]),
        np.asarray([2.0, 4.0, 1.5]), 0.0, 4, 0,
        np.asarray([[8.0, 2.0], [10.0, 2.0]]),
    )
    track = snapshot_to_ego(snapshot, np.eye(4))
    np.testing.assert_allclose(track.center_ego, [10.0, 2.0, 0.75])
    np.testing.assert_allclose(track.velocity_ego, [3.0, 0.0])
    assert track.track_id == 7
    assert track.distance_m > 10.0


def test_tracking_identity_evaluator_detects_id_switch():
    from parking_bev.tracking import TrackSnapshot

    evaluator = TrackingIdentityEvaluator(center_threshold_m=2.0)
    expected = Object3D(
        "ann", "vehicle.car", np.asarray([1, 0, 0], np.float32),
        np.asarray([2, 4, 1.5], np.float32), 0.0, np.zeros(2, np.float32), "instance-1")
    first = TrackSnapshot(1, "car", 0.9, np.asarray([1, 0]), np.zeros(2),
                          np.asarray([2, 4, 1.5]), 0.0, 2, 0, np.asarray([[1, 0]]))
    second = TrackSnapshot(2, "car", 0.9, np.asarray([1, 0]), np.zeros(2),
                           np.asarray([2, 4, 1.5]), 0.0, 2, 0, np.asarray([[1, 0]]))
    evaluator.update([first], [expected], np.eye(4))
    evaluator.update([second], [expected], np.eye(4))
    result = evaluator.finalize()
    assert result["spatial_matches"] == 2
    assert result["id_switches"] == 1
    assert result["id_f1"] == 0.5
    assert result["mean_velocity_error_mps"] == 0.0


def test_camera_appearance_histogram_is_normalized():
    from parking_bev.nuscenes_source import SensorCalibration

    image = np.zeros((100, 100, 3), np.uint8)
    image[:] = (0, 0, 255)
    intrinsic = np.asarray([[50, 0, 50], [0, 50, 50], [0, 0, 1]], np.float32)
    calibration = SensorCalibration(np.eye(4, dtype=np.float32), intrinsic)
    obj = Object3D("box", "vehicle.car", np.asarray([0, 0, 8], np.float32),
                   np.asarray([2, 4, 2], np.float32), 0.0, np.zeros(2, np.float32))
    feature = extract_object_appearance(obj, {"CAM_FRONT": image}, {"CAM_FRONT": calibration})
    assert feature is not None
    assert feature.shape == (128,)
    assert np.isclose(np.linalg.norm(feature), 1.0)


def test_radar_velocity_is_associated_inside_oriented_box():
    obj = Object3D("box", "vehicle.car", np.asarray([10, 0, 0], np.float32),
                   np.asarray([2, 4, 1.5], np.float32), 0.0, np.zeros(2, np.float32))
    radar = {"RADAR_FRONT": np.asarray([
        [10, 0, 0, 5, 1],
        [11, 0.5, 0, 5.2, 0.8],
        [30, 30, 0, -20, 0],
    ], np.float32)}
    estimate = estimate_radar_velocity(obj, radar)
    assert estimate is not None
    assert estimate.point_count == 2
    np.testing.assert_allclose(estimate.velocity_ego, [5.1, 0.9], atol=1e-5)
