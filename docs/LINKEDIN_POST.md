# LinkedIn Post Draft (English)

I'm building a ROS 2 perception-to-parking prototype using six cameras and LiDAR.

My latest update integrates CMT + AsyncBEV alongside the existing BEVFusion backend, with a synchronized visual walkthrough of camera inputs, colored point clouds, 3D detections and BEV tracking.

What I've focused on:

• Spatiotemporal consistency: calibration, timestamp checks, TF transforms and ego-motion-compensated LiDAR history.

• ROS 2 integration: an isolated GPU inference worker, detection and tracking topics, RViz markers and latency diagnostics.

• Parking-oriented geometry: configurable vehicle dimensions, oriented-box clearance and swept-body prediction, including front-corner swing during reversing turns.

• Freshness-aware behavior: latest-frame processing and explicit stale/unknown states when perception or vehicle feedback is too old.

In the video, the yellow #13 links the same tracked vehicle across camera and BEV views as it turns. These are real nuScenes recordings and pretrained-model outputs. Point colors illustrate camera projection or predicted-box membership, not semantic segmentation.

Current status: offline replay and ROS 2 data-path validation, with parking logic tested on synthetic cases. Runtime optimization and real-vehicle testing are next.

My contribution is deployment adaptation, system integration, parking application logic and visualization. Credit to the CMT and AsyncBEV authors for the underlying models and pretrained weights.

Code, demo and setup:
https://github.com/lengyuely2/multi-camera-lidar-bev-perception

#ROS2 #SensorFusion #LiDAR #ComputerVision #Robotics #AutonomousDriving
