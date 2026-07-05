import numpy as np

from parking_bev.bev import CameraBEVProjector
from parking_bev.camera import SyntheticCameraRig


def test_camera_bev_shape_and_content():
    rig = SyntheticCameraRig(80, 60)
    ok, frames = rig.read()
    assert ok
    quads = {
        "front": [[40, 0], [120, 0], [110, 80], [50, 80]],
        "rear": [[50, 80], [110, 80], [120, 160], [40, 160]],
        "left": [[0, 40], [50, 50], [50, 110], [0, 120]],
        "right": [[110, 50], [160, 40], [160, 120], [110, 110]],
    }
    result = CameraBEVProjector(160, 160, quads).project(frames)
    assert result.shape == (160, 160, 3)
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) > 0

