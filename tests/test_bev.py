import cv2
import numpy as np

from parking_bev.bev import CameraBEVProjector
from parking_bev.camera import CAMERA_NAMES, StaticImageCameraRig, SyntheticCameraRig


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


def test_static_image_rig_repeats_synchronized_sample(tmp_path):
    sources = {}
    for index, name in enumerate(CAMERA_NAMES):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), np.full((12, 16, 3), index * 40 + 20, np.uint8))
        sources[name] = str(path)
    rig = StaticImageCameraRig(sources)
    ok, frames = rig.read()
    assert ok
    assert set(frames) == set(CAMERA_NAMES)
    assert all(frame.shape == (12, 16, 3) for frame in frames.values())
