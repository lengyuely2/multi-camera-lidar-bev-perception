import numpy as np

from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import CAMERA_CHANNELS, Object3D, RADAR_CHANNELS
from parking_bev.evaluation import evaluate_detections
from parking_bev.predictions import Prediction3D, detection_class
from parking_bev.voxelize import HardVoxelizer


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
