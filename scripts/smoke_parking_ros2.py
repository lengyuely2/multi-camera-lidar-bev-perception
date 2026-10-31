"""Check parking ROS callbacks, vehicle telemetry, TF, outputs and stale-state clearing.

Inputs are explicitly synthetic. No model/GPU or physical vehicle is required.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from parking_bev_ros.parking_node import ParkingNode


class Probe(Node):
    def __init__(self):
        super().__init__("parking_synthetic_probe")
        self.mode = "turn"
        self.reports, self.images, self.markers = [], [], []
        self.tracks = self.create_publisher(String, "/parking_bev/track_states", 1)
        self.state = self.create_publisher(AckermannDriveStamped, "/vehicle/measured_drive", 1)
        self.create_subscription(String, "/parking_assistance/report", self.on_report, 10)
        self.create_subscription(Image, "/parking_assistance/image", self.images.append, 1)
        self.create_subscription(MarkerArray, "/parking_assistance/swept_path", self.markers.append, 1)
        self.tf = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.frame_id, transform.child_frame_id = "odom", "base_link"
        transform.transform.rotation.w = 1.
        self.tf.sendTransform(transform)
        self.timer = self.create_timer(.05, self.tick)

    def on_report(self, message):
        self.reports.append(json.loads(message.data))

    def tick(self):
        now = self.get_clock().now()
        now_s = now.nanoseconds / 1e9
        if self.mode != "stale_vehicle":
            state = AckermannDriveStamped()
            state.header.stamp, state.header.frame_id = now.to_msg(), "base_link"
            state.drive.speed = -.3 if self.mode in ("turn", "stale_perception") else 0.
            state.drive.steering_angle = .55 if self.mode in ("turn", "stale_perception") else 0.
            self.state.publish(state)
        if self.mode == "invalid":
            self.tracks.publish(String(data='{"schema_version": 99}'))
            return
        side = self.mode == "side"
        obj = dict(track_id=self.mode, class_name="car" if side else "traffic_cone", score=.95,
                   center=[0., 2.2, .8] if side else [1.8, -1.9, .2],
                   size_lwh=[4.2, 1.8, 1.6] if side else [.35, .35, .4], yaw=0.,
                   velocity_xy=[0., 0.], velocity_known=True, hits=4, missed_s=0.)
        payload = dict(schema_version=1, timestamp_s=now_s - (1.6 if self.mode == "stale_perception" else .02),
                       frame_id="odom", objects=[obj])
        self.tracks.publish(String(data=json.dumps(payload)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/parking_assistance"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=[])
    parking = ParkingNode()
    probe = Probe()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(parking)
    executor.add_node(probe)
    results = {}
    try:
        def wait_for(condition):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=.05)
                if condition():
                    return
            raise AssertionError("Parking ROS condition failed. Latest reports: " + repr(probe.reports[-2:]))
        wait_for(lambda: bool(probe.reports) and probe.reports[-1]["status"] == "active" and
                 probe.reports[-1]["level"] == "caution" and probe.reports[-1]["alerts"][0]["track_id"] == "turn")
        results["turning_envelope"] = probe.reports[-1]
        assert results["turning_envelope"]["alerts"][0]["envelope_entry_s"] is not None
        assert len(results["turning_envelope"]["envelope_polygons"]) > 1
        wait_for(lambda: bool(probe.images) and bool(probe.markers) and len(probe.markers[-1].markers) > 1 and
                 probe.images[-1].header.stamp.sec + probe.images[-1].header.stamp.nanosec / 1e9 >= results["turning_envelope"]["timestamp_s"])
        cv2.imwrite(str(args.output_dir / "ros_parking.png"), CvBridge().imgmsg_to_cv2(probe.images[-1], "bgr8"))
        probe.mode = "side"
        wait_for(lambda: probe.reports[-1]["status"] == "active" and probe.reports[-1]["sector_gaps_m"]["left"] is not None)
        results["side_gap"] = probe.reports[-1]
        assert abs(results["side_gap"]["sector_gaps_m"]["left"] - .325) < 1e-4
        probe.mode = "stale_perception"
        wait_for(lambda: probe.reports[-1]["status"] == "stale_perception")
        results["stale_perception"] = probe.reports[-1]
        assert results["stale_perception"]["level"] == "unknown"
        assert not results["stale_perception"]["envelope_polygons"]
        wait_for(lambda: len(probe.markers[-1].markers) == 1)
        probe.mode = "stale_vehicle"
        wait_for(lambda: probe.reports[-1]["status"] == "stale_vehicle_state")
        results["stale_vehicle"] = probe.reports[-1]
        probe.mode = "invalid"
        wait_for(lambda: probe.reports[-1]["status"] == "invalid_perception")
        results["invalid_input"] = probe.reports[-1]
        assert results["invalid_input"]["level"] == "unknown"
        report = dict(synthetic_inputs=True, cases=results, received_reports=len(probe.reports),
                      received_images=len(probe.images), received_marker_arrays=len(probe.markers))
        (args.output_dir / "ros_parking_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(passed=list(results), received_reports=len(probe.reports),
                             received_images=len(probe.images)), ensure_ascii=False))
    finally:
        executor.shutdown()
        parking.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
