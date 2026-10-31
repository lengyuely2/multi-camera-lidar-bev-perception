import json
import sys

import numpy as np
import pytest

from parking_bev.perception.backend import DetectionBatch, ModelWorker, transform_predictions
from parking_bev.perception.sensor_packet import compensate_sweep, pointcloud_matrix, validate_packet


def payload():
    return dict(timestamp_s=1.0, frame_id="lidar", model="fixture", inference_seconds=0.1,
                detections=[dict(center=[2, 0, 1], size_lwh=[4, 2, 2], yaw=0,
                                 velocity_xy=[3, 0], score=0.9, class_name="car")])


def packet():
    result = dict(points=np.array([[2, 0, 0, 1, 0]], dtype=np.float32), timestamp_s=np.array(1.),
                  lidar_timestamp_s=np.array(0.8), camera_timestamps_s=np.ones(6), frame_id=np.array("lidar"),
                  lidar_to_ego=np.eye(4), ego_to_global=np.eye(4), source_ego_to_global=np.eye(4),
                  lidar2cam=np.repeat(np.eye(4)[None], 6, axis=0), intrinsics=np.repeat(np.eye(3)[None], 6, axis=0))
    result.update({"image_" + str(index): np.zeros((8, 8, 3), dtype=np.uint8) for index in range(6)})
    return result


def test_metric_contract_and_transform_preserve_center_size_and_velocity():
    batch = DetectionBatch.from_payload(payload())
    np.testing.assert_allclose(batch.predictions[0].object.size_wlh, [2, 4, 2])
    transform = np.array([[0, -1, 0, 10], [1, 0, 0, -1], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    result = transform_predictions(batch.predictions, transform)[0].object
    np.testing.assert_allclose(result.center_ego, [10, 1, 1.5])
    np.testing.assert_allclose(result.velocity_ego, [0, 3])
    assert result.yaw_ego == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("field,value", [("score", float("nan")), ("size_lwh", [4, 0, 2]), ("center", [1, 2])])
def test_invalid_model_output_is_rejected(field, value):
    data = payload()
    data["detections"][0][field] = value
    with pytest.raises(ValueError):
        DetectionBatch.from_payload(data)


def test_historical_scan_compensation_and_time_channel():
    source = np.eye(4)
    source[0, 3] = 2
    target = np.eye(4)
    target[0, 3] = 5
    result = compensate_sweep(np.array([[10, 1, 0, 42]]), source, target, 0.25)
    np.testing.assert_allclose(result, [[7, 1, 0, 42, 0.25]])


def test_ros_structured_pointcloud_and_older_tuple_iterator_agree():
    values = [(2, 3, 4, 42), (5, 6, 7, 11)]
    structured = np.array(values, dtype=[(name, "<f4") for name in ("x", "y", "z", "intensity")])
    np.testing.assert_allclose(pointcloud_matrix(structured), pointcloud_matrix(iter(values)))
    assert pointcloud_matrix(structured[:0]).shape == (0, 4)


def test_planar_tracker_preserves_measured_height_for_ros_boxes():
    from parking_bev.tracking.tracker import TimestampAwareTracker, prediction_to_global_measurement
    detection = DetectionBatch.from_payload(payload()).predictions[0]
    tracker = TimestampAwareTracker()
    transform = np.eye(4)
    transform[2, 3] = 4
    measured = tracker.update(1., [prediction_to_global_measurement(detection, transform)])
    predicted = tracker.update(1.1, [])
    assert measured[0].z_global == predicted[0].z_global == 5
    assert predicted[0].position_global.shape == (2,)


def test_packet_accepts_calibrated_raw_sensors_without_annotations():
    validate_packet(packet())


@pytest.mark.parametrize("change", ["time", "camera", "transform", "labels", "sweep_age"])
def test_packet_rejects_wrong_units_stale_cameras_and_annotations(change):
    data = packet()
    if change == "time":
        data["lidar_timestamp_s"] = np.array(800000.)
    elif change == "camera":
        data["camera_timestamps_s"][2] += 0.2
    elif change == "transform":
        data["lidar_to_ego"][0, 0] = -1
    elif change == "labels":
        data["gt_flow"] = np.zeros(3)
    else:
        data["points"][0, 4] = -1
    with pytest.raises(ValueError):
        validate_packet(data)


def write_worker(path, reply, stall=False):
    path.write_text("import json,sys,time\nprint(json.dumps({'status':'ready'}),flush=True)\n"
                    "for line in sys.stdin:\n"
                    + (" time.sleep(10)\n" if stall else " print(" + repr(json.dumps(reply)) + ",flush=True)\n"))


def test_worker_runs_resident_process_and_validates_frame_identity(tmp_path):
    script = tmp_path / "worker.py"
    write_worker(script, payload())
    with ModelWorker([sys.executable, "-u", str(script)], timeout_s=2) as worker:
        assert len(worker.infer(packet()).predictions) == 1
        data = packet()
        data["timestamp_s"] = np.array(2.)
        with pytest.raises(RuntimeError, match="wrong sensor"):
            worker.infer(data)
        assert not worker.alive


def test_worker_timeout_terminates_instead_of_returning_stale_detections(tmp_path):
    script = tmp_path / "worker.py"
    write_worker(script, payload(), stall=True)
    with ModelWorker([sys.executable, "-u", str(script)], timeout_s=0.05) as worker:
        with pytest.raises(TimeoutError):
            worker.infer(packet())
        assert not worker.alive
