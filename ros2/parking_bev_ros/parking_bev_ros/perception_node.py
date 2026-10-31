"""Six-camera + LiDAR ROS input, bounded latest-frame inference, tf2 and tracking."""
from collections import deque
from copy import deepcopy
import json
import shlex
import threading
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

from parking_bev.perception.backend import ModelWorker, transform_predictions
from parking_bev.perception.predictions import Prediction3D, category_for_detection_class
from parking_bev.perception.sensor_packet import CAMERA_ORDER, compensate_sweep, pointcloud_matrix, validate_packet
from parking_bev.sensors.nuscenes_source import Object3D
from parking_bev.tracking.tracker import TimestampAwareTracker, prediction_to_global_measurement
from .messages import box_markers, detections_message


def timestamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec / 1e9


def transform_matrix(message):
    value = message.transform
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w]).as_matrix()
    matrix[:3, 3] = [value.translation.x, value.translation.y, value.translation.z]
    return matrix


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("parking_bev")
        self.declare_parameter("worker_command", "")
        self.declare_parameter("worker_endpoint", "")
        self.declare_parameter("lidar_topic", "/sensing/lidar/points")
        self.declare_parameter("camera_topics", ["/sensing/camera/" + name.lower() + "/image_rect" for name in CAMERA_ORDER])
        self.declare_parameter("camera_info_topics", ["/sensing/camera/" + name.lower() + "/camera_info" for name in CAMERA_ORDER])
        self.declare_parameter("ego_frame", "base_link")
        self.declare_parameter("fixed_frame", "odom")
        self.declare_parameter("max_delay_s", 0.6)
        self.declare_parameter("max_camera_skew_s", 0.08)
        self.declare_parameter("max_input_age_s", 1.0)
        self.declare_parameter("worker_timeout_s", 60.0)
        self.declare_parameter("packet_directory", "")
        self.declare_parameter("images_rectified", True)
        self.declare_parameter("intensity_scale", 1.0)
        self.declare_parameter("history_scans", 10)
        def parameter(name):
            return self.get_parameter(name).value
        self.parameter = parameter
        camera_topics, info_topics = parameter("camera_topics"), parameter("camera_info_topics")
        command = shlex.split(parameter("worker_command"))
        endpoint = parameter("worker_endpoint")
        endpoint = (endpoint.rsplit(":", 1)[0], int(endpoint.rsplit(":", 1)[1])) if endpoint else None
        if not command and not endpoint:
            raise ValueError("Set worker_command or worker_endpoint for the isolated model")
        if len(camera_topics) != 6 or len(info_topics) != 6:
            raise ValueError("Exactly six camera/image-info topics in CMT camera order are required")
        self.bridge = CvBridge()
        self.tf = Buffer(cache_time=Duration(seconds=15))
        self.listener = TransformListener(self.tf, self, spin_thread=False)
        self.lock = threading.Lock()
        self.pending = None
        self.clouds = deque(maxlen=int(parameter("history_scans")) + 1)
        self.infos = [None] * 6
        self.tracker = TimestampAwareTracker()
        self.last_timestamp = None
        self.dropped = 0
        self._worker_closed = False
        self.worker = ModelWorker(command or None, startup_timeout_s=300, timeout_s=parameter("worker_timeout_s"),
                                  packet_directory=parameter("packet_directory") or None, endpoint=endpoint)
        self.detections = self.create_publisher(Detection3DArray, "~/detections", 1)
        self.tracks = self.create_publisher(Detection3DArray, "~/tracks", 1)
        self.track_states = self.create_publisher(String, "~/track_states", 1)
        self.markers = self.create_publisher(MarkerArray, "~/markers", 1)
        self.diagnostics = self.create_publisher(DiagnosticArray, "~/diagnostics", 1)
        self.info_subscribers = [self.create_subscription(CameraInfo, topic,
            lambda message, index=index: self.on_info(index, message), qos_profile_sensor_data)
            for index, topic in enumerate(info_topics)]
        self.cloud_history = self.create_subscription(PointCloud2, parameter("lidar_topic"), self.on_cloud, qos_profile_sensor_data)
        self.sensor_subscribers = [message_filters.Subscriber(self, PointCloud2, parameter("lidar_topic"), qos_profile=qos_profile_sensor_data)]
        self.sensor_subscribers += [message_filters.Subscriber(self, Image, topic, qos_profile=qos_profile_sensor_data) for topic in camera_topics]
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(self.sensor_subscribers, queue_size=12, slop=parameter("max_delay_s"))
        self.synchronizer.registerCallback(self.on_bundle)
        self.inference_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(0.02, self.process, callback_group=self.inference_group)
        self.get_logger().info("AsyncBEV model ready; waiting for calibrated sensors and tf2")

    def on_info(self, index, message):
        with self.lock:
            self.infos[index] = message

    def on_cloud(self, message):
        with self.lock:
            if self.clouds and timestamp(message) < timestamp(self.clouds[-1]):
                self.clouds.clear()
            self.clouds.append(message)

    def on_bundle(self, cloud, *images):
        camera_times = np.array([timestamp(image) for image in images])
        if np.ptp(camera_times) > self.parameter("max_camera_skew_s"):
            self.dropped += 1
            return
        with self.lock:
            if self.pending is not None:
                self.dropped += 1
            self.pending = (cloud, images, list(self.infos), list(self.clouds), time.perf_counter())

    def lookup(self, target, source, stamp):
        return transform_matrix(self.tf.lookup_transform(target, source, Time.from_msg(stamp), timeout=Duration(seconds=0.15)))

    def make_packet(self, cloud, images, infos, clouds):
        if any(info is None for info in infos):
            raise ValueError("CameraInfo is missing")
        reference = images[4].header.stamp
        reference_s = timestamp(images[4])
        lidar_frame = cloud.header.frame_id
        fixed, ego = self.parameter("fixed_frame"), self.parameter("ego_frame")
        lidar_to_ego = self.lookup(ego, lidar_frame, cloud.header.stamp)
        target_ego = self.lookup(fixed, ego, reference)
        source_ego = self.lookup(fixed, ego, cloud.header.stamp)
        packet = dict(timestamp_s=np.array(reference_s), lidar_timestamp_s=np.array(timestamp(cloud)),
                      frame_id=np.array(lidar_frame), lidar_to_ego=lidar_to_ego,
                      ego_to_global=target_ego, source_ego_to_global=source_ego,
                      camera_timestamps_s=np.array([timestamp(image) for image in images]))
        intrinsics, projections = [], []
        for index, (image, info) in enumerate(zip(images, infos)):
            if info.header.frame_id != image.header.frame_id or info.width != image.width or info.height != image.height:
                raise ValueError("CameraInfo and Image frame/dimensions differ")
            matrix = np.array(info.p).reshape(3, 4)[:, :3] if self.parameter("images_rectified") else np.array(info.k).reshape(3, 3)
            bgr = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            if not self.parameter("images_rectified") and len(info.d) and np.any(info.d):
                if info.distortion_model not in ("plumb_bob", "rational_polynomial"):
                    raise ValueError("Use externally rectified images for this distortion model")
                bgr = cv2.undistort(bgr, matrix, np.array(info.d))
            packet["image_" + str(index)] = bgr
            intrinsics.append(matrix)
            projection = self.tf.lookup_transform_full(image.header.frame_id, Time.from_msg(image.header.stamp),
                lidar_frame, Time.from_msg(reference), fixed, timeout=Duration(seconds=0.15))
            projections.append(transform_matrix(projection))
        packet.update(intrinsics=np.array(intrinsics), lidar2cam=np.array(projections))
        # Consume historical scans at or before the selected source LiDAR time.
        selected = [item for item in clouds if 0 < timestamp(cloud) - timestamp(item) <= 0.6]
        count = int(self.parameter("history_scans"))
        selected = selected[-count:] if count > 0 else []
        selected.append(cloud)
        scans = []
        for item in selected:
            if item.header.frame_id != lidar_frame:
                raise ValueError("LiDAR frame changed")
            fields = {field.name for field in item.fields}
            if not {"x", "y", "z", "intensity"}.issubset(fields):
                raise ValueError("PointCloud2 requires x/y/z/intensity fields")
            points = pointcloud_matrix(point_cloud2.read_points(item, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            points[:, 3] *= float(self.parameter("intensity_scale"))
            points = points[(np.abs(points[:, 0]) >= 1) | (np.abs(points[:, 1]) >= 1)]
            world_sensor = self.lookup(fixed, lidar_frame, item.header.stamp)
            scans.append(compensate_sweep(points, world_sensor, source_ego @ lidar_to_ego, timestamp(cloud) - timestamp(item)))
        packet["points"] = np.concatenate(scans)
        validate_packet(packet, self.parameter("max_delay_s"), self.parameter("max_camera_skew_s"))
        return packet, Header(stamp=reference, frame_id=ego)

    def report(self, level, message, values=None):
        status = DiagnosticStatus(level=level, name="parking_bev/asyncbev", message=message, hardware_id="gpu")
        status.values = [KeyValue(key=str(key), value=str(value)) for key, value in (values or {}).items()]
        self.diagnostics.publish(DiagnosticArray(header=Header(stamp=self.get_clock().now().to_msg()), status=[status]))

    def process(self):
        if self._worker_closed:
            return
        with self.lock:
            bundle, self.pending = self.pending, None
        if bundle is None:
            return
        cloud, images, infos, clouds, received_at = bundle
        try:
            sensor_age = self.get_clock().now().nanoseconds / 1e9 - timestamp(images[4])
            if abs(sensor_age) > self.parameter("max_input_age_s"):
                raise ValueError("Sensor data is stale or clock differs; set use_sim_time for bag replay")
            packet, header = self.make_packet(cloud, images, infos, clouds)
            batch = self.worker.infer(packet)
            predictions = transform_predictions(batch.predictions, packet["lidar_to_ego"])
            detections = detections_message(predictions, header)
            if self.last_timestamp is not None and batch.timestamp_s <= self.last_timestamp:
                self.tracker = TimestampAwareTracker()
            self.last_timestamp = batch.timestamp_s
            measurements = [prediction_to_global_measurement(item, packet["ego_to_global"]) for item in predictions]
            tracks = self.tracker.update(batch.timestamp_s, measurements)
            track_predictions = [Prediction3D(Object3D(str(item.track_id), category_for_detection_class(item.class_name),
                np.r_[item.position_global, item.z_global], item.size_wlh, item.yaw_global, item.velocity_global), item.class_name, item.score) for item in tracks]
            track_header = deepcopy(header)
            track_header.frame_id = self.parameter("fixed_frame")
            tracked_message = detections_message(track_predictions, track_header, [item.track_id for item in tracks])
            self.detections.publish(detections)
            self.tracks.publish(tracked_message)
            self.track_states.publish(String(data=json.dumps(dict(schema_version=1, timestamp_s=batch.timestamp_s,
                frame_id=track_header.frame_id, objects=[dict(track_id=str(item.track_id), class_name=item.class_name,
                    score=float(item.score), center=[*item.position_global.tolist(), item.z_global],
                    size_lwh=item.size_wlh[[1, 0, 2]].tolist(), yaw=float(item.yaw_global),
                    velocity_xy=item.velocity_global.tolist(), velocity_known=item.hits >= 2,
                    hits=item.hits, missed_s=max(0., batch.timestamp_s - item.last_measurement_timestamp_s))
                    for item in tracks]), allow_nan=False)))
            self.markers.publish(box_markers(detections))
            self.report(DiagnosticStatus.OK, "inference complete", dict(
                model=batch.model, detections=len(predictions), tracks=len(tracks),
                model_seconds=batch.inference_seconds, callback_seconds=time.perf_counter() - received_at,
                sensor_to_output_seconds=self.get_clock().now().nanoseconds / 1e9 - batch.timestamp_s,
                dropped=self.dropped))
        except Exception as exc:
            self.report(DiagnosticStatus.ERROR, str(exc))
            self.get_logger().error(str(exc))
            if not self.worker.alive:
                self._worker_closed = True
                self.get_logger().error("Model worker stopped; restart this node after resolving the error")

    def destroy_node(self):
        self.worker.close()
        self.listener.unregister()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        node = PerceptionNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
