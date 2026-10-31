from glob import glob
from setuptools import setup

setup(
    name="parking_bev_ros", version="0.1.0", packages=["parking_bev_ros"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/parking_bev_ros"]),
                ("share/parking_bev_ros", ["package.xml"]),
                ("share/parking_bev_ros/launch", glob("launch/*.launch.py")),
                ("share/parking_bev_ros/config", glob("config/*.yaml")),
                ("share/parking_bev_ros/rviz", glob("rviz/*.rviz"))],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="parking-bev contributors", maintainer_email="maintainer@example.com",
    description="ROS 2 AsyncBEV sensor integration", license="Apache-2.0",
    entry_points={"console_scripts": ["perception_node = parking_bev_ros.perception_node:main",
                                       "replay_node = parking_bev_ros.replay_node:main",
                                       "parking_node = parking_bev_ros.parking_node:main"]},
)
