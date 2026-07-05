import numpy as np

from parking_bev.nuscenes_source import CAMERA_CHANNELS, RADAR_CHANNELS


def test_nuscenes_sensor_layout():
    assert len(CAMERA_CHANNELS) == 6
    assert len(RADAR_CHANNELS) == 5
    assert "CAM_FRONT" in CAMERA_CHANNELS
    assert "RADAR_FRONT" in RADAR_CHANNELS
    assert np.dtype(np.float32).itemsize == 4
