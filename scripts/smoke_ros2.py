"""Real ROS 2 transport checks. Run after sourcing ROS and the colcon workspace.

--predictions checks saved-result replay. --packet with --endpoint checks the
complete sensor/TF -> actual model -> detections/tracks/diagnostics path.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointField, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
from parking_bev.perception.sensor_packet import CAMERA_ORDER
from parking_bev.parking.inputs import TrackFrame


def diagnostic_level(status):
    return status.level[0] if isinstance(status.level, bytes) else int(status.level)


class SensorReplay(Node):
    def __init__(self, path):
        super().__init__("asyncbev_sensor_smoke")
        self.packet = dict(np.load(path, allow_pickle=False))
        self.bridge = CvBridge()
        self.image_pubs = [self.create_publisher(Image, "/sensing/camera/" + name.lower() + "/image_rect", qos_profile_sensor_data) for name in CAMERA_ORDER]
        self.info_pubs = [self.create_publisher(CameraInfo, "/sensing/camera/" + name.lower() + "/camera_info", qos_profile_sensor_data) for name in CAMERA_ORDER]
        self.cloud_pub = self.create_publisher(PointCloud2, "/sensing/lidar/points", qos_profile_sensor_data)
        self.tf = StaticTransformBroadcaster(self)
        transforms = []
        def add(parent, child, matrix):
            message = TransformStamped(header=Header(frame_id=parent), child_frame_id=child)
            message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = map(float, matrix[:3, 3])
            q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            message.transform.rotation.x, message.transform.rotation.y, message.transform.rotation.z, message.transform.rotation.w = map(float, q)
            transforms.append(message)
        add("odom", "base_link", self.packet["ego_to_global"])
        add("base_link", "lidar", self.packet["lidar_to_ego"])
        for index in range(6):
            add("base_link", "camera_" + str(index), self.packet["lidar_to_ego"] @ np.linalg.inv(self.packet["lidar2cam"][index]))
        self.tf.sendTransform(transforms)
        self.timer = self.create_timer(0.5, self.tick)

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        reference = float(self.packet["timestamp_s"])
        stamp = lambda value: Time(nanoseconds=round((now + value - reference) * 1e9)).to_msg()
        for index in range(6):
            array = self.packet["image_" + str(index)]
            header = Header(frame_id="camera_" + str(index), stamp=stamp(float(self.packet["camera_timestamps_s"][index])))
            info = CameraInfo(header=header, height=array.shape[0], width=array.shape[1], distortion_model="plumb_bob")
            intrinsic = self.packet["intrinsics"][index]
            info.k = intrinsic.ravel().tolist()
            projection = np.zeros((3, 4))
            projection[:3, :3] = intrinsic
            info.p = projection.ravel().tolist()
            info.r = np.eye(3).ravel().tolist()
            self.info_pubs[index].publish(info)
            image = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
            image.header = header
            self.image_pubs[index].publish(image)
        fields = [PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                  for index, name in enumerate(("x", "y", "z", "intensity"))]
        points = self.packet["points"]
        points = points[points[:, 4] == 0, :4]
        cloud = point_cloud2.create_cloud(Header(frame_id="lidar", stamp=stamp(float(self.packet["lidar_timestamp_s"]))), fields, points)
        self.cloud_pub.publish(cloud)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=Path)
    group.add_argument("--packet", type=Path)
    parser.add_argument("--endpoint", default="127.0.0.1:18765")
    parser.add_argument("--packet-directory", default="/mnt/d/my_project/multi-camera-lidar-bev-perception/output/asyncbev/ipc")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--with-parking", action="store_true", help="Also verify stale model output is rejected by parking assistance")
    args = parser.parse_args()
    parameters = ["--ros-args"]
    if args.predictions:
        parameters += ["-p", "predictions:=" + str(args.predictions), "-p", "loop:=true"]
    else:
        parameters += ["-p", "worker_endpoint:=" + args.endpoint,
                       "-p", "packet_directory:=" + args.packet_directory, "-p", "history_scans:=0"]
    rclpy.init(args=parameters)
    observer = Node("parking_bev_smoke_observer")
    received, tracked, diagnostic_messages, snapshots, parking_reports = [], [], [], [], []
    prefix = "/parking_bev_replay" if args.predictions else "/parking_bev"
    observer.create_subscription(Detection3DArray, prefix + "/detections", received.append, 1)
    observer.create_subscription(DiagnosticArray, prefix + "/diagnostics", diagnostic_messages.append, 1)
    observer.create_subscription(Detection3DArray, prefix + "/tracks", tracked.append, 1)
    observer.create_subscription(String, prefix + "/track_states", lambda message: snapshots.append(TrackFrame.from_dict(json.loads(message.data))), 1)
    if args.with_parking:
        observer.create_subscription(String, "/parking_assistance/report", lambda message: parking_reports.append(json.loads(message.data)), 1)
    executor = MultiThreadedExecutor(num_threads=4)
    nodes = [observer]
    try:
        if args.predictions:
            from parking_bev_ros.replay_node import ReplayNode
            node = ReplayNode()
        else:
            from parking_bev_ros.perception_node import PerceptionNode
            node = PerceptionNode()
            nodes.append(SensorReplay(args.packet))
            if args.with_parking:
                from parking_bev_ros.parking_node import ParkingNode
                nodes.append(ParkingNode())
        nodes.append(node)
        for item in nodes:
            executor.add_node(item)
        deadline = time.monotonic() + args.timeout
        progress_at = time.monotonic() + 5
        def complete():
            return len(received) >= 2 and (args.predictions or (
                len(tracked) >= 2 and len(snapshots) >= 2 and
                (not args.with_parking or any(item["status"] == "stale_perception" for item in parking_reports)) and
                any(diagnostic_level(status) == 0 for message in diagnostic_messages for status in message.status)))
        while not complete() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if time.monotonic() >= progress_at:
                print("ROS check: detections=%d tracks=%d diagnostics=%s" % (
                    len(received), len(tracked), [(status.level, status.message) for message in diagnostic_messages[-2:] for status in message.status]), flush=True)
                progress_at = time.monotonic() + 10
        if not complete():
            errors = [status.message for message in diagnostic_messages for status in message.status]
            raise RuntimeError("ROS transport timed out: " + repr(errors[-5:]))
        message = received[-1]
        assert message.header.frame_id == ("lidar" if args.predictions else "base_link")
        for detection in message.detections:
            assert detection.bbox.size.x > 0 and detection.bbox.size.y > 0 and detection.bbox.size.z > 0
            assert 0 <= detection.results[0].hypothesis.score <= 1
        if not args.predictions:
            assert tracked[-1].header.frame_id == "odom"
            assert len(tracked[-1].detections) > 0
            assert all(item.id for item in tracked[-1].detections)
            assert snapshots[-1].frame_id == "odom" and len(snapshots[-1].objects) > 0
            assert not any(diagnostic_level(status) == 2 for message in diagnostic_messages for status in message.status)
        report = dict(mode="saved_prediction_replay" if args.predictions else "raw_sensors_to_asyncbev",
                      received_messages=len(received), detections=len(message.detections), frame_id=message.header.frame_id,
                      tracked_messages=len(tracked),
                      track_state_messages=len(snapshots),
                      parking_statuses=sorted({item["status"] for item in parking_reports}),
                      diagnostics=[dict(level=diagnostic_level(status), message=status.message,
                                        values={value.key: value.value for value in status.values})
                                   for diagnostic in diagnostic_messages for status in diagnostic.status])
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        executor.shutdown()
        for item in reversed(nodes):
            item.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
