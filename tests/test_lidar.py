import numpy as np

from parking_bev.lidar import LidarBEVProjector


BEV_CONFIG = {
    "width_px": 200,
    "height_px": 200,
    "x_min_m": -10.0,
    "x_max_m": 10.0,
    "y_min_m": -10.0,
    "y_max_m": 10.0,
}

LIDAR_CONFIG = {
    "min_z_m": -1.5,
    "max_z_m": 2.5,
    "extrinsic_lidar_to_ego": np.eye(4).tolist(),
}


def test_lidar_origin_projects_to_bev_center():
    projector = LidarBEVProjector(BEV_CONFIG, LIDAR_CONFIG)
    result = projector.project(np.array([[0.0, 0.0, 1.25]], np.float32))
    assert result.occupancy[100, 100] == 255
    assert np.isclose(result.height[100, 100], 1.25)
    assert result.density[100, 100] == 1


def test_empty_lidar_keeps_stable_schema():
    projector = LidarBEVProjector(BEV_CONFIG, LIDAR_CONFIG)
    result = projector.empty()
    assert result.occupancy.shape == (200, 200)
    assert np.count_nonzero(result.occupancy) == 0
