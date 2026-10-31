"""Standard ROS messages; all centers are gravity centers, all sizes are L/W/H."""
import math

from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


def detections_message(predictions, header, ids=None):
    output = Detection3DArray()
    output.header = header
    for index, prediction in enumerate(predictions):
        obj = prediction.object
        item = Detection3D()
        item.header = header
        item.id = str(ids[index]) if ids is not None else ""
        item.bbox.center.position.x, item.bbox.center.position.y, item.bbox.center.position.z = map(float, obj.center_ego)
        item.bbox.center.orientation.z = math.sin(obj.yaw_ego / 2)
        item.bbox.center.orientation.w = math.cos(obj.yaw_ego / 2)
        item.bbox.size.x, item.bbox.size.y, item.bbox.size.z = map(float, obj.size_wlh[[1, 0, 2]])
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = prediction.class_name
        hypothesis.hypothesis.score = float(prediction.score)
        hypothesis.pose.pose = item.bbox.center
        item.results = [hypothesis]
        output.detections.append(item)
    return output


def box_markers(detections):
    clear = Marker()
    clear.header = detections.header
    clear.action = Marker.DELETEALL
    output = MarkerArray(markers=[clear])
    for index, detection in enumerate(detections.detections):
        marker = Marker()
        marker.header = detections.header
        marker.ns = "parking_bev_boxes"
        marker.id = index
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = detection.bbox.center
        marker.scale = detection.bbox.size
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.8, 0.9, 0.45
        marker.lifetime.sec = 1
        output.markers.append(marker)
    return output
