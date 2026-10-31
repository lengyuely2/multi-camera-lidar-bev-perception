"""Publish actual saved detections in ROS; replay validates messaging, not live inference."""
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray
from parking_bev.perception.backend import DetectionBatch
from parking_bev.perception.predictions import bevfusion_predictions_from_records
from .messages import box_markers, detections_message


class ReplayNode(Node):
    def __init__(self):
        super().__init__("parking_bev_replay")
        self.declare_parameter("predictions", "")
        self.declare_parameter("frame_id", "lidar")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("loop", False)
        path = Path(self.get_parameter("predictions").value)
        payload = json.loads(path.read_text())
        self.frames = payload.get("frames", [payload])
        if not self.frames:
            raise ValueError("Prediction file contains no frames")
        self.index = 0
        self.publisher = self.create_publisher(Detection3DArray, "~/detections", 1)
        self.markers = self.create_publisher(MarkerArray, "~/markers", 1)
        rate = float(self.get_parameter("rate_hz").value)
        if rate <= 0:
            raise ValueError("rate_hz must be positive")
        self.timer = self.create_timer(1 / rate, self.tick)
        self.get_logger().info("Replaying saved model predictions; this node does not run inference")

    def tick(self):
        frame = self.frames[self.index]
        if "detections" in frame:
            predictions = DetectionBatch.from_payload(frame).predictions
        else:
            predictions = bevfusion_predictions_from_records(frame["predictions"], np.eye(4), 0.2)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.get_parameter("frame_id").value)
        output = detections_message(predictions, header)
        self.publisher.publish(output)
        self.markers.publish(box_markers(output))
        self.index += 1
        if self.index >= len(self.frames):
            if self.get_parameter("loop").value:
                self.index = 0
            else:
                self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
