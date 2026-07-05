import numpy as np

from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import CAMERA_CHANNELS, Object3D, RADAR_CHANNELS


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
