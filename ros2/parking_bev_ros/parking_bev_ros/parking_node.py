"""Independent parking advisory node; consumes measured state, never publishes drive commands."""
from dataclasses import asdict
import json
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from parking_bev.parking import ParkingAssistant, ParkingConfig, VehicleGeometry, VehicleState
from parking_bev.parking.inputs import TrackFrame
from parking_bev.runtime.config import load_config
from parking_bev.visualization.parking import ParkingRenderer
from .perception_node import transform_matrix


class ParkingNode(Node):
    def __init__(self):
        super().__init__("parking_assistance")
        self.declare_parameter("profile_config", "")
        self.declare_parameter("tracks_topic", "/parking_bev/track_states")
        self.declare_parameter("vehicle_state_topic", "/vehicle/measured_drive")
        self.declare_parameter("ego_frame", "base_link")
        self.declare_parameter("fixed_frame", "odom")
        self.declare_parameter("update_hz", 10.0)
        self.declare_parameter("image_hz", 3.0)
        self.declare_parameter("publish_image", True)
        self.declare_parameter("font_path", "")
        profile = self.get_parameter("profile_config").value
        values = load_config(profile) if profile else {}
        vehicle_defaults = asdict(VehicleGeometry(**values.get("vehicle", {})))
        config_defaults = asdict(ParkingConfig(**values.get("parking", {})))
        for prefix, defaults in (("vehicle", vehicle_defaults), ("parking", config_defaults)):
            for key, value in defaults.items():
                self.declare_parameter(prefix + "." + key, value)
        vehicle = VehicleGeometry(**{key: self.get_parameter("vehicle." + key).value for key in vehicle_defaults})
        config = ParkingConfig(**{key: self.get_parameter("parking." + key).value for key in config_defaults})
        self.assistant = ParkingAssistant(vehicle, config)
        self.tf = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.tf, self, spin_thread=False)
        self.frame, self.vehicle_state, self.input_error = None, None, None
        self.bridge = CvBridge()
        self.renderer = ParkingRenderer(vehicle, font_path=self.get_parameter("font_path").value or None) if self.get_parameter("publish_image").value else None
        self.report_publisher = self.create_publisher(String, "~/report", 1)
        self.advisory = self.create_publisher(String, "~/advisory", 1)
        self.markers = self.create_publisher(MarkerArray, "~/swept_path", 1)
        self.image_publisher = self.create_publisher(Image, "~/image", 1)
        self.create_subscription(String, self.get_parameter("tracks_topic").value, self.on_tracks, 1)
        self.create_subscription(AckermannDriveStamped, self.get_parameter("vehicle_state_topic").value, self.on_vehicle, 1)
        rate, image_rate = float(self.get_parameter("update_hz").value), float(self.get_parameter("image_hz").value)
        if rate <= 0 or image_rate <= 0:
            raise ValueError("Parking node rates must be positive")
        self.image_period = 1 / image_rate
        self.last_image = -float("inf")
        self.timer = self.create_timer(1 / rate, self.tick)
        self.get_logger().info("Parking assistance ready; geometry profile=" + vehicle.profile_name)

    def on_tracks(self, message):
        try:
            self.frame = TrackFrame.from_dict(json.loads(message.data))
            self.input_error = None
        except Exception as exc:
            self.frame = None
            self.input_error = "目标输入异常：" + str(exc)

    def on_vehicle(self, message):
        if message.header.frame_id != self.get_parameter("ego_frame").value:
            self.vehicle_state = None
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        self.vehicle_state = VehicleState(stamp, message.drive.speed, message.drive.steering_angle)

    def tick(self):
        now = self.get_clock().now()
        now_s = now.nanoseconds / 1e9
        frame = self.frame
        state = self.vehicle_state
        objects = []
        report = self.assistant.assess([], state, frame.timestamp_s if frame else None, now_s)
        if self.input_error:
            report = self.assistant.unavailable(now_s, "invalid_perception", self.input_error)
        elif report.status == "active":
            try:
                transform = self.tf.lookup_transform_full(self.get_parameter("ego_frame").value, now,
                    frame.frame_id, Time(nanoseconds=round(frame.timestamp_s * 1e9)),
                    self.get_parameter("fixed_frame").value, timeout=Duration(seconds=.03))
                objects = frame.in_current_ego(transform_matrix(transform), now_s)
                report = self.assistant.assess(objects, state, frame.timestamp_s, now_s)
            except Exception as exc:
                report = self.assistant.unavailable(now_s, "transform_unavailable", "当前车辆位姿不可用：" + str(exc))
        header = Header(stamp=now.to_msg(), frame_id=self.get_parameter("ego_frame").value)
        payload = report.to_dict()
        payload.update(vehicle_state_source=self.get_parameter("vehicle_state_topic").value,
                       speed_mps=state.speed_mps if state and np.isfinite(state.speed_mps) else None,
                       steering_angle_rad=state.steering_angle_rad if state and np.isfinite(state.steering_angle_rad) else None)
        self.report_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False, allow_nan=False)))
        self.advisory.publish(String(data=report.advisory))
        self.markers.publish(self.swept_markers(report, header))
        if self.renderer and time.monotonic() - self.last_image >= self.image_period:
            image = self.renderer.render(report, objects, state.speed_mps if state and np.isfinite(state.speed_mps) else 0.,
                state.steering_angle_rad if state and np.isfinite(state.steering_angle_rad) else 0., "ROS 2 · 实时数据时效检查")
            message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            message.header = header
            self.image_publisher.publish(message)
            self.last_image = time.monotonic()

    @staticmethod
    def swept_markers(report, header):
        clear = Marker(header=header, action=Marker.DELETEALL)
        output = MarkerArray(markers=[clear])
        colors = {"stop": (1., .2, .2), "caution": (1., .65, .1), "watch": (.2, .65, 1.),
                  "no_conflict_detected": (.3, .65, .75), "unknown": (.5, .5, .5)}
        step = max(1, len(report.envelope_polygons) // 15)
        for index, polygon in enumerate(report.envelope_polygons[::step]):
            marker = Marker(header=header, ns="parking_swept_envelope", id=index, type=Marker.LINE_STRIP, action=Marker.ADD)
            marker.pose.orientation.w = 1.
            marker.scale.x = .025
            marker.color.r, marker.color.g, marker.color.b = colors[report.level]
            marker.color.a = .7
            marker.points = [Point(x=float(x), y=float(y), z=.06) for x, y in [*polygon, polygon[0]]]
            marker.lifetime.nanosec = 300_000_000
            output.markers.append(marker)
        return output

    def destroy_node(self):
        self.listener.unregister()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = ParkingNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
