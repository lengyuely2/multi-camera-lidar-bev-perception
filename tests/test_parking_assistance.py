import numpy as np
import pytest

from parking_bev.parking import ParkingAssistant, ParkingConfig, ParkingObject, VehicleGeometry, VehicleState
from parking_bev.parking.geometry import bicycle_path, polygon_clearance, rectangle
from parking_bev.parking.inputs import TrackFrame


def obstacle(center, size=(.5, .5), velocity=(0, 0), yaw=0, hits=3, missed=0):
    return ParkingObject("1", "pedestrian", np.array(center), np.array(size), yaw,
                         np.array(velocity), True, .95, hits, missed)


def assess(objects, speed=0., steering=0., now=10., observed=10.):
    return ParkingAssistant().assess(objects, VehicleState(now, speed, steering), observed, now)


def test_bicycle_straight_forward_reverse_and_reference_offset():
    for speed in (-1., 0., 1.):
        centers, yaws = bicycle_path(np.array([0., 1., 2.]), speed, 0., 2.8, -1.35)
        np.testing.assert_allclose(centers[:, 0], np.array([0., 1., 2.]) * speed)
        np.testing.assert_allclose(centers[:, 1], 0)
        np.testing.assert_allclose(yaws, 0)


def test_bicycle_turns_are_mirrored_and_reverse_yaw_sign_is_correct():
    left, left_yaw = bicycle_path(np.array([0., 3.]), .5, .5, 2.8, -1.35)
    right, right_yaw = bicycle_path(np.array([0., 3.]), .5, -.5, 2.8, -1.35)
    np.testing.assert_allclose(left[:, 0], right[:, 0])
    np.testing.assert_allclose(left[:, 1], -right[:, 1])
    np.testing.assert_allclose(left_yaw, -right_yaw)
    _, reversed_yaw = bicycle_path(np.array([3.]), -.5, .5, 2.8, -1.35)
    assert reversed_yaw[0] < 0


def test_rotated_rectangles_use_surface_gap_instead_of_center_distance():
    ego = rectangle([0, 0], 4., 2., 0.)
    adjacent = rectangle([0, 3.], 1., 2., np.pi / 2)
    gap, overlap = polygon_clearance(ego, adjacent)
    assert gap == pytest.approx(1.5)
    assert not overlap
    diagonal = rectangle([3, 2], 1., 1., 0)
    gap, overlap = polygon_clearance(ego, diagonal)
    assert gap == pytest.approx(np.sqrt(.5))
    assert not overlap


def test_touching_and_overlap_have_zero_gap():
    ego = rectangle([0, 0], 4., 2., .2)
    gap, overlap = polygon_clearance(ego, ego)
    assert gap == 0 and overlap
    gap, overlap = polygon_clearance(rectangle([0, 0], 2, 2, 0), rectangle([2, 0], 2, 2, 0))
    assert gap == 0 and overlap


def test_reverse_path_detects_obstacle_behind_vehicle():
    report = assess([obstacle([-4, 0])], speed=-.6)
    assert report.direction == "reverse"
    assert report.level in ("caution", "stop")
    assert report.alerts[0].envelope_entry_s is not None
    assert report.alerts[0].sector == "rear"
    straight_forward = assess([obstacle([-4, 0])], speed=.6)
    assert straight_forward.level == "no_conflict_detected"


def test_reversing_turn_sweeps_front_corner_toward_side_obstacle():
    objects = [obstacle([1.8, -1.9], size=(.35, .35))]
    turned = assess(objects, speed=-.3, steering=.55)
    straight = assess(objects, speed=-.3, steering=0.)
    assert turned.alerts[0].envelope_entry_s is not None
    assert straight.alerts[0].envelope_entry_s is None
    assert turned.level == "caution"
    assert straight.level == "watch"


def test_dynamic_pedestrian_crossing_and_provisional_target_handling():
    moving = assess([obstacle([-3.8, 2.2], velocity=(0, -.55))], speed=-.35)
    assert moving.alerts[0].envelope_entry_s is not None
    assert moving.alerts[0].motion_confirmed
    provisional = assess([obstacle([-3, 0], hits=1)], speed=-.6)
    assert provisional.level == "caution"
    close = assess([obstacle([0, 1.25], hits=1)])
    assert close.level == "stop"


def test_stationary_vehicle_reports_side_body_clearance():
    report = assess([obstacle([0, 2.2], size=(4.2, 1.8))])
    assert report.direction == "stationary"
    assert report.sector_gaps_m["left"] == pytest.approx(.325)
    assert report.level == "caution"


@pytest.mark.parametrize("observed,status", [(8., "stale_perception"), (10.5, "clock_mismatch"), (None, "missing_perception")])
def test_expired_missing_future_perception_never_reports_clear(observed, status):
    report = ParkingAssistant().assess([], VehicleState(10., 0, 0), observed, 10.)
    assert report.status == status and report.level == "unknown"
    assert not len(report.envelope_polygons)
    assert all(value is None for value in report.sector_gaps_m.values())


def test_missing_stale_vehicle_state_and_excess_speed_are_unknown():
    assistant = ParkingAssistant()
    for state, expected in [(None, "missing_vehicle_state"), (VehicleState(9, 0, 0), "stale_vehicle_state"),
                             (VehicleState(10, 3, 0), "out_of_scope")]:
        report = assistant.assess([], state, 10, 10)
        assert report.status == expected and report.level == "unknown"


def test_empty_detection_set_is_not_a_free_space_claim():
    report = assess([])
    assert report.level == "no_conflict_detected"
    assert report.to_dict()["free_space_verified"] is False
    assert report.to_dict()["coverage"] == "detected_objects_only"


def test_only_old_tracks_produce_insufficient_information():
    report = assess([obstacle([0, 1.2], missed=.5)])
    assert report.status == "insufficient_tracks" and report.level == "unknown"


def test_different_vehicle_dimensions_change_clearance():
    short = ParkingAssistant(VehicleGeometry(length_m=4.7))
    long = ParkingAssistant(VehicleGeometry(length_m=6.5, wheelbase_m=3.8, rear_axle_x_m=-2.0))
    item = obstacle([4.1, 0])
    state = VehicleState(10, .1, 0)
    assert short.assess([item], state, 10, 10).level == "no_conflict_detected"
    assert long.assess([item], state, 10, 10).level in ("watch", "caution", "stop")


def test_invalid_vehicle_and_object_geometry_are_rejected():
    with pytest.raises(ValueError):
        VehicleGeometry(wheelbase_m=10)
    with pytest.raises(ValueError):
        ParkingConfig(stop_gap_m=1., caution_gap_m=.5)
    with pytest.raises(ValueError):
        obstacle([np.nan, 1])


def test_tracker_frame_prediction_and_ego_transform_do_not_double_compensate():
    frame = TrackFrame.from_dict(dict(schema_version=1, timestamp_s=10., frame_id="odom", objects=[
        dict(track_id="1", class_name="car", center=[10, 0, 2], size_lwh=[4, 2, 1.5], yaw=0,
             velocity_xy=[2, 0], velocity_known=True, score=.9, hits=3, missed_s=0.)]))
    transform = np.array([[0, -1, 0, 5], [1, 0, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1.]])
    obj = frame.in_current_ego(transform, 10.25)[0]
    np.testing.assert_allclose(obj.center_xy, [5, 9.5])
    np.testing.assert_allclose(obj.velocity_xy, [0, 2])
    assert obj.yaw_rad == pytest.approx(np.pi / 2)
    assert obj.center_z_m == 2
